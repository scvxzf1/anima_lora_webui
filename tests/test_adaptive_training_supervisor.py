from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from library.training.adaptive_runtime import supervisor
from library.training.adaptive_runtime.retry import MemoryPlan, RetryLimits, run_recovery
from library.training.adaptive_runtime.training_config import require_training_contract
from library.training.adaptive_runtime.training_runtime import TrainingRuntime, validate_entry
from library.training.adaptive_runtime.training_state import save_training_state
from library.training.adaptive_runtime.training_worker import execute


def configured(**changes):
    return Namespace(**{**dict(adaptive_precision="fp16_fp32", adaptive_fp32_modules=["blocks.0.*"],
                              adaptive_loss_scale=1024, adaptive_oom_retry=True, seed=42,
                              attn_mode="torch",
                              blocks_to_swap=24, adaptive_oom_retry_max_swap=26,
                              adaptive_oom_retry_swap_increment=2, adaptive_oom_retry_max_attempts=3,
                              adaptive_oom_retry_timeout=10), **changes})


def worker_config(directory, **changes):
    return configured(adaptive_oom_retry=False,
                      _adaptive_training_worker={"directory": str(directory)}, **changes)


def test_frozen_merged_worker_args_preserve_precision_and_override_swap(tmp_path):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"plan": {"blocks_to_swap": 26}, "resume": None}))
    args = configured(config_file="do-not-reload.toml", save_state=False)
    before = deepcopy(vars(args))
    command = supervisor.worker_arguments(args, request)
    assert command[1:3] == ["-m", "library.training.adaptive_runtime.training_worker"]
    path = Path(command[-1])
    result = json.loads(path.read_text())
    assert result["blocks_to_swap"] == 26 and not result["adaptive_oom_retry"]
    assert result["adaptive_fp32_modules"] == before["adaptive_fp32_modules"]
    assert result["adaptive_loss_scale"] == 1024 and result["seed"] == 42
    assert result["resume"] is None and not result["save_state"]
    assert path.stat().st_mode & 0o777 == 0o600
    assert vars(args) == before
    request.write_text(json.dumps({"plan": {"blocks_to_swap": 26}, "resume": "unverified"}))
    with pytest.raises(ValueError, match="data-cursor"):
        supervisor.worker_arguments(args, request)


@pytest.mark.parametrize("changes", [
    {"seed": None}, {"seed": True}, {"gradient_accumulation_steps": 2},
    {"adaptive_oom_retry_max_swap": 28}, {"adaptive_oom_retry_max_attempts": 0},
    {"adaptive_oom_retry_swap_increment": True}, {"blocks_to_swap": 27},
    {"adaptive_oom_retry_timeout": float("nan")}, {"adaptive_oom_retry_timeout": True},
    {"resume": "old"}, {"save_every_n_steps": 1}, {"scale_weight_norms": 1.0},
    {"adaptive_precision": "off"},
])
def test_invalid_retry_config_rejected(changes):
    with pytest.raises(ValueError):
        require_training_contract(configured(**changes))


def test_direct_training_cannot_bypass_supervisor():
    with pytest.raises(ValueError, match="supervised"):
        validate_entry(SimpleNamespace(), configured())
    with pytest.raises(ValueError, match="isolated"):
        validate_entry(SimpleNamespace(), worker_config("."))
    validate_entry(SimpleNamespace(), Namespace())


@pytest.mark.parametrize("status,exit_code", [("ok", 0), ("failed", 1)])
@pytest.mark.parametrize("model_family", ["anima", "z_image"])
def test_cli_dispatch_uses_merged_config_not_parsed_defaults(
    monkeypatch, status, exit_code, model_family
):
    from library.training import cli_entry

    merged = configured(model_family=model_family)
    parser = SimpleNamespace(parse_args=lambda argv: configured(adaptive_oom_retry=False))
    monkeypatch.setattr(cli_entry._config_schema, "populate_schema", lambda *a, **k: None)
    monkeypatch.setattr(cli_entry, "verify_command_line_training_args", lambda args: None)
    monkeypatch.setattr(cli_entry, "read_config_from_file", lambda args, parser: merged)
    seen = []
    monkeypatch.setattr(supervisor, "run_supervised", lambda args: seen.append(args) or {"status": status})

    def unexpected_trainer():
        raise AssertionError("Supervisor must launch the only trainer")

    with pytest.raises(SystemExit) as raised:
        cli_entry.run_training_cli(setup_parser=lambda: parser, trainer_factory=unexpected_trainer,
                                    install_stop_signal_handlers=lambda: None,
                                    install_crash_reporter=lambda argv: None, argv=["train.py"])
    assert raised.value.code == exit_code and seen == [merged]
    assert seen[0].model_family == model_family


@pytest.mark.parametrize("model_family", ["anima", "krea2_raw", "z_image"])
def test_supervisor_retries_structured_startup_oom_for_each_dit_family(
    tmp_path, monkeypatch, model_family
):
    plans = []
    worker_configs = []

    class Runner:
        def __init__(self, directory, command, **kwargs):
            self.directory, self.command = directory, command

        def __call__(self, plan, *, attempt, resume):
            plans.append(plan)
            directory = self.directory / f"attempt-{attempt:03d}"
            directory.mkdir()
            request = directory / "request.json"
            request.write_text(json.dumps({"plan": {"blocks_to_swap": plan.blocks_to_swap}, "resume": resume}))
            command = self.command(request)
            worker_configs.append(json.loads(Path(command[-1]).read_text()))
            if attempt == 0:
                return {"status": "cuda_oom", "stage": "backward", "optimizer_started": False}
            return {"status": "ok", "completed_steps": 2}

    monkeypatch.setattr(supervisor, "IsolatedRunner", Runner)
    result = supervisor.run_supervised(
        configured(output_dir=str(tmp_path), model_family=model_family)
    )
    assert result["status"] == "ok" and [p.blocks_to_swap for p in plans] == [24, 26]
    assert [item["blocks_to_swap"] for item in worker_configs] == [24, 26]
    assert all(item["model_family"] == model_family for item in worker_configs)
    assert all(item["adaptive_precision"] == "fp16_fp32" for item in worker_configs)
    assert all(item["adaptive_fp32_modules"] == ["blocks.0.*"] for item in worker_configs)
    assert all(item["attn_mode"] == "torch" for item in worker_configs)
    report = json.loads((tmp_path / ".adaptive-recovery/summary.json").read_text())
    assert not report["production_ready"] and not report["data_cursor_resume_supported"]
    assert not list(tmp_path.rglob("training-args.json"))
    with pytest.raises(FileExistsError):
        supervisor.run_supervised(
            configured(output_dir=str(tmp_path), model_family=model_family)
        )


@pytest.mark.parametrize("exception,status", [
    (torch.cuda.OutOfMemoryError("CUDA allocation"), "cuda_oom"),
    (RuntimeError("CUDA out of memory"), "error"),
    (MemoryError("host out of memory"), "error"),
])
def test_worker_classifies_exception_not_log_text(tmp_path, exception, status):
    class Trainer:
        def train(self, args):
            self._adaptive_training_runtime.phase("forward")
            raise exception

    result = execute(worker_config(tmp_path), tmp_path, Trainer)
    assert result["status"] == status and result["stage"] == "forward"
    assert result["optimizer_started"] is False
    assert result == json.loads((tmp_path / "result.json").read_text())


def test_auto_resolved_z_image_worker_keeps_precision_contract(tmp_path):
    config = worker_config(
        tmp_path,
        adaptive_precision="auto",
        adaptive_resolved_mode="fp16_fp32",
        adaptive_candidate="fp16",
        model_family="z_image",
        mixed_precision="fp16",
        attn_mode="torch",
        block_swap_restore_mode="foreach",
    )

    class Trainer:
        def train(self, args):
            self._adaptive_training_runtime.phase("model_load")
            raise torch.cuda.OutOfMemoryError("CUDA allocation")

    result = execute(config, tmp_path, Trainer)
    assert result["status"] == "cuda_oom" and result["stage"] == "model_load"


def test_worker_early_return_is_not_success(tmp_path):
    result = execute(worker_config(tmp_path), tmp_path,
                     lambda: SimpleNamespace(train=lambda args: None))
    assert result["status"] == "error" and result["completed_steps"] == 0


def test_private_worker_still_validates_retry_restrictions(tmp_path):
    def unexpected_trainer():
        raise AssertionError("Invalid worker must fail before constructing trainer")

    result = execute(worker_config(tmp_path, resume="unverified"), tmp_path, unexpected_trainer)
    assert result["status"] == "error" and result["error_type"] == "ValueError"


@pytest.mark.parametrize("status", ["error", "host_limit", "timeout", "cancelled"])
def test_non_cuda_failure_is_not_retried(status):
    calls = []

    def runner(*args, **kwargs):
        calls.append(kwargs)
        return {"status": status, "stage": "forward", "optimizer_started": False}

    report = run_recovery(MemoryPlan(), RetryLimits(26), runner, record=lambda r: None)
    assert report["status"] == "failed" and len(calls) == 1


def test_post_update_oom_stops_even_with_saved_snapshot(tmp_path):
    runtime = TrainingRuntime(tmp_path)
    runtime.phase("optimizer")
    runtime.saved_state = str(tmp_path / "state-00000001")
    runtime.completed_steps = 1
    runtime.phase("backward")
    report = run_recovery(MemoryPlan(), RetryLimits(26),
                          lambda *a, **k: runtime.result("cuda_oom"), record=lambda r: None)
    assert len(report["attempts"]) == 1
    assert report["attempts"][0]["stop_reason"] == "no_committed_checkpoint"
    assert report["attempts"][0]["result"]["saved_state"] == runtime.saved_state


def fake_state(step=1, missing=None, fail=False):
    def save(directory, **kwargs):
        path = Path(directory)
        for name in ("model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl",
                     "scaler.pt", "adaptive_precision.json"):
            if name != missing:
                (path / name).write_text("test-fixture")
        (path / "train_state.json").write_text(json.dumps({"current_step": step}))
        if fail:
            raise OSError("disk full")

    return SimpleNamespace(global_step=step, args=Namespace(max_train_steps=step),
                           accelerator=SimpleNamespace(save_state=save, optimizer_step_was_skipped=False))


@pytest.mark.parametrize("missing", ["model.safetensors", "optimizer.bin", "scheduler.bin", "scaler.pt",
                                     "random_states_0.pkl", "adaptive_precision.json"])
def test_incomplete_state_is_not_published(tmp_path, missing):
    with pytest.raises(ValueError, match="Incomplete"):
        save_training_state(fake_state(missing=missing), tmp_path)
    assert not list(tmp_path.glob("state-*"))


def test_atomic_save_failure_preserves_previous_and_only_latest_retained(tmp_path):
    runtime = TrainingRuntime(tmp_path)
    runtime.phase("optimizer")
    runtime.commit(fake_state())
    previous = runtime.saved_state
    assert Path(previous).is_dir()
    runtime.commit(fake_state(step=2))
    assert not Path(previous).exists() and Path(runtime.saved_state).is_dir()
    previous = runtime.saved_state
    with pytest.raises(OSError, match="disk full"):
        runtime.commit(fake_state(step=3, fail=True))
    assert runtime.saved_state == previous and runtime.completed_steps == 2
    assert not (tmp_path / "state-00000003").exists()
    assert json.loads((tmp_path / "latest-state.json").read_text())["saved_state"] == previous


def test_worker_success_requires_real_boundary_and_completed_target(tmp_path):
    class Trainer:
        def train(self, args):
            runtime = self._adaptive_training_runtime
            state = fake_state()
            runtime.phase("optimizer")
            runtime.after_optimizer(state)
            runtime.commit(state)
            runtime.finish(state)

    result = execute(worker_config(tmp_path), tmp_path, Trainer)
    assert result["status"] == "ok" and result["completed_steps"] == 1
    assert result["committed_checkpoint"] is None
    assert json.loads((Path(result["saved_state"]) / "snapshot.json").read_text())["global_step"] == 1


def test_scaler_skipped_update_cannot_commit(tmp_path):
    state = fake_state()
    state.accelerator.optimizer_step_was_skipped = True
    runtime = TrainingRuntime(tmp_path)
    runtime.phase("optimizer")
    with pytest.raises(RuntimeError, match="skipped"):
        runtime.after_optimizer(state)
    assert runtime.completed_steps == 0 and runtime.saved_state is None


def test_actual_accelerate_cpu_snapshot_inventory(tmp_path):
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState
    from library.training.adaptive_runtime.training_precision import register_precision_checkpoint
    from library.training.checkpoints import CheckpointSaver

    # Accelerate keeps a process-global state. Other adaptive tests may have
    # initialized a CUDA accelerator before this CPU-only serialization check,
    # so isolate both sides of the fixture explicitly.
    AcceleratorState._reset_state(reset_partial_state=True)
    accelerator = None
    try:
        accelerator = Accelerator(cpu=True)
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
        model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
        loss = model(torch.ones(1, 2)).square().mean()
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        # CPU-only serialization exercise; CUDA scaling is tested separately.
        accelerator.scaler = torch.amp.GradScaler("cpu")
        register_precision_checkpoint(accelerator, {"schema": "test-fixture"})

        saver = CheckpointSaver(args=Namespace(), accelerator=accelerator, save_dtype=torch.float32,
                                metadata={}, minimum_metadata={},
                                get_sai_model_spec_fn=lambda args: {},
                                current_step=SimpleNamespace(value=0), current_epoch=SimpleNamespace(value=1))
        saver.register_hooks(model)
        saved = save_training_state(SimpleNamespace(global_step=1, accelerator=accelerator), tmp_path)
        assert (saved / "snapshot.json").is_file()
        assert torch.load(saved / "scaler.pt", weights_only=True)["scale"] > 0
    finally:
        if accelerator is not None:
            accelerator.end_training()
        AcceleratorState._reset_state(reset_partial_state=True)
