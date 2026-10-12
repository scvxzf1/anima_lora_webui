"""CPU boundaries for native startup-only fresh-process supervision."""
from argparse import Namespace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from library.training.adaptive_runtime.contract import freeze_native_sources, precision_contract_id
from library.training.adaptive_runtime.native_retry_config import native_retry_configuration_errors
from library.training.adaptive_runtime.process import IsolatedRunner
from library.training.adaptive_runtime.retry import MemoryPlan, RetryLimits, run_recovery
from library.training.adaptive_runtime.supervisor import worker_arguments
from library.training.adaptive_runtime.training_config import require_training_contract
from library.training.adaptive_runtime.training_worker import execute, _native_cuda_oom


def configured(tmp_path, **changes):
    source = tmp_path / "base.bin"
    if not source.exists():
        source.write_bytes(b"frozen source")
    args = Namespace(adaptive_precision="native_fp16", mixed_precision="fp16", seed=42,
                     blocks_to_swap=0, adaptive_oom_retry=True, output_dir=str(tmp_path),
                     pretrained_model_name_or_path=str(source), **changes)
    return args


@pytest.mark.parametrize("mode", ["native_fp16", "auto"])
def test_native_contract_accepts_frozen_modes_and_output_behaviour(tmp_path, mode):
    args = configured(tmp_path, save_state=True, save_state_on_train_end=True,
                      save_every_n_steps=1, log_with="tensorboard", sample_every_n_steps=1)
    args.adaptive_precision = mode
    if mode == "auto":
        args.adaptive_resolved_mode = "native_fp16"
    require_training_contract(args)
    freeze_native_sources(args)
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"resume": None, "plan": {"blocks_to_swap": 2}}))
    command = worker_arguments(args, request)
    child = json.loads(Path(command[-1]).read_text())
    for key in ("save_state", "save_state_on_train_end", "save_every_n_steps", "log_with", "sample_every_n_steps"):
        assert child[key] == getattr(args, key)
    assert child["blocks_to_swap"] == 2 and child["resume"] is None
    assert precision_contract_id(Namespace(**child)) == precision_contract_id(args)


@pytest.mark.parametrize("key,value", [("resume", "saved"), ("auto_resume", True),
    ("checkpointing_epochs", 1), ("skip_until_initial_step", True), ("sample_at_first", True),
    ("auto_block_swap", True), ("base_compute", "w8a16_convrot"),
    ("block_swap_restore_mode", "slab"), ("block_swap_transfer_dtype", "int8"),
    ("max_data_loader_n_workers", 2), ("seed", True), ("adaptive_oom_retry_max_attempts", 17)])
def test_native_retry_rejects_unsafe_invocations(tmp_path, key, value):
    args = configured(tmp_path)
    setattr(args, key, value)
    assert native_retry_configuration_errors(args)


def test_source_drift_and_missing_capacity_stop_before_trainer(tmp_path):
    args = configured(tmp_path, _adaptive_training_worker={"directory": str(tmp_path)})
    args.adaptive_oom_retry = False
    freeze_native_sources(args)
    args._adaptive_precision_contract_id = precision_contract_id(args)
    Path(args.pretrained_model_name_or_path).write_bytes(b"changed")
    result = execute(args, tmp_path, lambda: pytest.fail("trainer started"))
    assert result["status"] == "error" and result["failure_kind"] == "precision_contract_mismatch"


@pytest.mark.parametrize("message,expected", [("CUDA allocation", True),
    ("CPU out of memory", False), ("DefaultCPUAllocator allocation", False),
    ("out of memory", False), ("host out of memory", False)])
def test_native_typed_oom_requires_cuda_evidence(message, expected):
    assert _native_cuda_oom(torch.cuda.OutOfMemoryError(message), SimpleNamespace()) is expected


def recover(result):
    calls = []
    def runner(plan, **kwargs):
        calls.append(plan.blocks_to_swap)
        return result
    report = run_recovery(MemoryPlan(), RetryLimits(4, max_attempts=2), runner,
                          record=lambda report: None, expected_precision_contract="H", startup_only=True)
    return calls, report


@pytest.mark.parametrize("change,reason", [({"optimizer_started": True, "committed_checkpoint": "fake"}, "native_startup_only_optimizer_started"),
    ({"optimizer_started": 0}, "unknown_optimizer_progress"),
    ({"stage": "setup"}, "unsupported_startup_stage"),
    ({"stage": "unknown"}, "unsupported_startup_stage"),
    ({"precision_contract_id": "drift"}, "precision_contract_mismatch")])
def test_startup_rejects_partial_optimizer_unknown_and_drift(change, reason):
    calls, report = recover({"status": "cuda_oom", "stage": "backward", "optimizer_started": False,
                             "precision_contract_id": "H", "precision_manifest_id": "b" * 64, **change})
    assert calls == [0] and report["attempts"][-1]["stop_reason"] == reason


def test_native_success_requires_finished_boundary():
    calls, report = recover({"status": "ok", "stage": "finished", "optimizer_started": True,
                             "precision_contract_id": "H", "precision_manifest_id": "b" * 64, "completed_steps": 2})
    assert report["status"] == "failed" and calls == [0]


@pytest.mark.parametrize("option", ["allow_batch_change", "allow_checkpoint_change"])
def test_native_retry_never_changes_other_memory_semantics(option):
    with pytest.raises(ValueError, match="only increase block swap"):
        run_recovery(MemoryPlan(), RetryLimits(4, **{option: True}),
                     lambda *a, **k: pytest.fail("worker started"),
                     record=lambda value: None, startup_only=True)


@pytest.mark.parametrize("merge", ["base_weights", "lora_path"])
def test_native_retry_rejects_unfrozen_base_merge_sources(tmp_path, merge):
    args = configured(tmp_path, **{merge: ["extra.safetensors"]})
    with pytest.raises(ValueError, match=merge):
        require_training_contract(args)


def test_real_cpu_subprocess_native_startup_retry_keeps_h_and_distinct_pid(tmp_path):
    script = tmp_path / "fixture.py"
    script.write_text('''import json, os, sys
from pathlib import Path
p=Path(sys.argv[1]); r=json.loads(p.read_text()); n=r['plan']['blocks_to_swap']
v={'status':'cuda_oom' if n == 0 else 'ok','stage':'model_load' if n == 0 else 'finished',
'optimizer_started':n > 0, 'precision_contract_id':'H','precision_manifest_id':None if n == 0 else 'b'*64,
'finished':n > 0,'completed_steps':2 if n > 0 else 0,'pid':os.getpid()}
(p.parent/'result.json').write_text(json.dumps(v))
''')
    runner = IsolatedRunner(tmp_path / "recovery", lambda request: [sys.executable, str(script), str(request)],
                            resources=lambda: SimpleNamespace(available=10**10, reserve=1))
    result = run_recovery(MemoryPlan(), RetryLimits(2), runner, record=lambda report: None,
                          expected_precision_contract="H", startup_only=True)
    assert result["status"] == "ok"
    assert [item["plan"]["blocks_to_swap"] for item in result["attempts"]] == [0, 2]
    assert len({item["result"]["pid"] for item in result["attempts"]}) == 2


def tiny_native_parts():
    from networks.lora_modules.lora import LoRAModule
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2).half() for _ in range(28)])
    model.final_layer = torch.nn.Linear(2, 2).half()
    model.requires_grad_(False)
    for block in model.blocks:
        block.fp32_residual = True
        block.gradient_checkpointing = True
    model.final_layer.fp32_residual = True
    network = torch.nn.Module()
    network.unet_loras = torch.nn.ModuleList([LoRAModule("tiny", model.blocks[0], lora_dim=1)])
    network.unet_loras[0].fp32_compute = False
    network.text_encoder_loras = []
    scaler = SimpleNamespace(is_enabled=lambda: True, get_growth_factor=lambda: 2.,
                             get_backoff_factor=lambda: .5, get_growth_interval=lambda: 2000,
                             state_dict=lambda: {"scale": 1024., "growth_factor": 2., "_growth_tracker": 0})
    accelerator = SimpleNamespace(unwrap_model=lambda model: model, native_amp=True, mixed_precision="fp16",
                                  scaler=scaler, optimizer_step_was_skipped=False)
    return model, network, accelerator


def test_ambiguous_typed_oom_stops_even_with_cuda_context(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    runtime = SimpleNamespace(_accelerator=SimpleNamespace(device="cuda"))
    assert not _native_cuda_oom(torch.OutOfMemoryError("out of memory"), runtime)


def private_native_args(tmp_path):
    args = configured(tmp_path, _adaptive_training_worker={"directory": str(tmp_path)})
    args.adaptive_oom_retry = False
    args.max_train_steps = 3
    args._adaptive_swap_capacity = {"model_limit": 26, "effective_cap": 26, "requested_cap": 0}
    freeze_native_sources(args)
    args._adaptive_precision_contract_id = precision_contract_id(args)
    return args


def test_native_worker_scalar_boundaries_allow_two_updates_and_a_skip(tmp_path):
    class Trainer:
        def train(self, args):
            runtime = self._adaptive_training_runtime
            model, network, accelerator = tiny_native_parts()
            runtime.capture_model(model)
            runtime.prepared(model, network, accelerator)
            state = SimpleNamespace(args=args, accelerator=accelerator, global_step=0)
            for step in range(1, 4):
                runtime.phase("forward")
                runtime.phase("backward")
                runtime.phase("optimizer")
                accelerator.optimizer_step_was_skipped = step == 2
                runtime.after_optimizer(state)
                state.global_step = step
                runtime.commit(state)
            runtime.finish(state)
    result = execute(private_native_args(tmp_path), tmp_path, Trainer)
    assert result["status"] == "ok" and result["finished"] is True
    assert result["effective_steps"] == 2 and result["skipped_steps"] == 1
    scalar = json.loads((tmp_path / "latest-state.json").read_text())
    assert scalar["completed_steps"] == 3 and not list(tmp_path.glob("*.pt"))


@pytest.mark.parametrize("mutation", ["dtype", "guard"])
def test_native_worker_actual_metadata_drift_is_never_oom(tmp_path, mutation):
    class Trainer:
        def train(self, args):
            runtime = self._adaptive_training_runtime
            model, network, accelerator = tiny_native_parts()
            runtime.capture_model(model)
            runtime.prepared(model, network, accelerator)
            runtime.phase("forward")
            if mutation == "dtype":
                model.blocks[0].float()
            else:
                model.blocks[0].fp32_residual = False
            raise torch.cuda.OutOfMemoryError("CUDA allocation")
    result = execute(private_native_args(tmp_path), tmp_path, Trainer)
    assert result["status"] == "error" and result["failure_kind"] == "precision_contract_mismatch"
