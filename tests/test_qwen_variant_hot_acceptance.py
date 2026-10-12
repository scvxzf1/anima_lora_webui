"""CPU-only tests for the opt-in Qwen adapter hot acceptance harness."""

import json
import os
import signal
import subprocess
import sys

import pytest
import toml
import torch
from safetensors.torch import save_file

from scripts.qwen_image_2_1 import variant_hot_acceptance as harness


def test_variant_configs_preserve_stage3_contract(tmp_path):
    base = harness.load_reference()
    assert base["gradient_checkpointing"] is True
    assert base["selective_checkpoint"] == "off"
    assert base["cpu_offload_checkpointing"] is False
    assert base["unsloth_offload_checkpointing"] is False
    for variant in harness.VARIANTS:
        for phase in ("smoke", "hot"):
            config = harness.make_config(base, variant, phase, tmp_path / variant / phase)
            assert config["network_args"] == (
                [] if variant == "lora" else [f"use_{variant}=true"]
            )
            assert config["max_train_steps"] == (2 if phase == "smoke" else 20)
            for key in ("seed", "pretrained_model_name_or_path",
                        "qwen3", "vae", "mixed_precision", "base_compute",
                        "blocks_to_swap", "torch_compile", "attn_mode",
                        "qwen_fused_projections"):
                assert config[key] == base[key]
            assert config["dataset_config"] == str(tmp_path / variant / "dataset.toml")
            assert config["output_dir"].startswith(str(tmp_path))


def test_train_cli_passes_variant_network_args_to_bootstrap(tmp_path):
    from library.config.io import read_config_from_file
    from library.config.schema import populate_schema
    from library.training.bootstrap import TrainingBootstrap
    from library.training.cli_args import setup_parser
    from library.training.cli_entry import build_network_extras

    parser = setup_parser()
    populate_schema(parser, extras=build_network_extras())
    base = harness.load_reference()
    for variant in harness.VARIANTS:
        config = harness.make_config(base, variant, "smoke", tmp_path / variant)
        path = tmp_path / f"{variant}.toml"
        path.write_text(toml.dumps(config))
        argv = ["--config_file", str(path), "--no-config-snapshot"]
        parsed = read_config_from_file(parser.parse_args(argv), parser, argv=argv)
        kwargs = TrainingBootstrap.build_net_kwargs(parsed)
        assert kwargs.get("use_lokr") == ("true" if variant == "lokr" else None)
        assert kwargs.get("use_loha") == ("true" if variant == "loha" else None)


def test_hot_summary_excludes_warmup_and_requires_all_steps(tmp_path):
    progress = tmp_path / "progress.jsonl"
    rows = [{"ev": "step", "global_step": step, "ts": 100 + 8 * step,
             "loss/current": 0.5, "cuda/max_memory_allocated_gb": 6.0}
            for step in range(1, 21)]
    progress.write_text("\n".join(json.dumps(row) for row in rows))
    summary = harness.summarize(progress)
    assert summary["included_steps"] == list(range(6, 21))
    assert summary["median_s"] == 8
    progress.write_text("\n".join(json.dumps(row) for row in rows[:-1]))
    with pytest.raises(RuntimeError, match="missing progress"):
        harness.summarize(progress)


def test_smoke_requires_both_steps(tmp_path):
    progress = tmp_path / "progress.jsonl"
    first = {"ev": "step", "global_step": 1, "loss/current": 0.5}
    second = {"ev": "step", "global_step": 2, "loss/current": 0.4}
    end = {"ev": "run_end", "status": "ok", "final_step": 2}
    progress.write_text(json.dumps(first) + "\n")
    with pytest.raises(RuntimeError, match="steps 1..2"):
        harness.verify_smoke(progress)
    progress.write_text("\n".join(json.dumps(row) for row in (first, second, end)))
    harness.verify_smoke(progress)
    progress.write_text("\n".join(json.dumps(row) for row in
                                     (first, second, {"ev": "step", "global_step": 2}, end)))
    harness.verify_smoke(progress)
    progress.write_text("\n".join(json.dumps(row) for row in
                                     (first, second, {"ev": "step", "global_step": 1}, end)))
    with pytest.raises(RuntimeError, match="status-only"):
        harness.verify_smoke(progress)


def test_progress_rejects_bad_exit_loss_and_steps(tmp_path):
    path = tmp_path / "progress.jsonl"
    rows = [{"ev": "step", "global_step": step, "loss/current": 0.5}
            for step in (1, 2)]
    rows.append({"ev": "run_end", "status": "ok", "final_step": 2})
    for broken in (
        rows[:-1],
        [*rows[:-1], {**rows[-1], "status": "error"}],
        [{**rows[0], "loss/current": float("nan")}, *rows[1:]],
        [rows[0], {**rows[1], "global_step": 1}, rows[2]],
    ):
        path.write_text("\n".join(json.dumps(row) for row in broken))
        with pytest.raises(RuntimeError):
            harness.verify_progress(path, 2)


@pytest.mark.parametrize("variant,keys", [
    ("lora", ("block.lora_down.weight", "block.lora_up.weight")),
    ("lokr", ("block.lokr_w1", "block.lokr_w2")),
    ("loha", ("block.hada_w1_a", "block.hada_w1_b", "block.hada_w2_a", "block.hada_w2_b")),
])
def test_checkpoint_variant_and_tensor_validation(tmp_path, variant, keys):
    path = tmp_path / "adapter.safetensors"
    tensors = {key: torch.ones(2) for key in keys}
    save_file(tensors, str(path), {"ss_network_spec": variant})
    assert harness.verify_checkpoint(path, variant)["tensor_count"] == len(keys)
    with pytest.raises(RuntimeError, match="network spec"):
        harness.verify_checkpoint(path, "lora" if variant != "lora" else "lokr")
    tensors[keys[0]] = torch.zeros(2)
    save_file(tensors, str(path), {"ss_network_spec": variant})
    with pytest.raises(RuntimeError, match="Invalid trainable"):
        harness.verify_checkpoint(path, variant)
    save_file({keys[0]: torch.ones(2)}, str(path), {"ss_network_spec": variant})
    with pytest.raises(RuntimeError, match="Missing"):
        harness.verify_checkpoint(path, variant)
    mixed = {key: torch.ones(2) for key in keys}
    mixed["block.hada_w1_a" if variant != "loha" else "block.lokr_w1"] = torch.ones(2)
    save_file(mixed, str(path), {"ss_network_spec": variant})
    with pytest.raises(RuntimeError, match="another adapter"):
        harness.verify_checkpoint(path, variant)


def test_gpu_telemetry_requires_continuous_valid_samples(tmp_path):
    path = tmp_path / "gpu.csv"
    rows = [f"2026/09/29 12:00:{second:02d}.000, 70, 1200, 5000, 6000, 120, 95"
            for second in range(10)]
    path.write_text("\n".join(rows))
    assert harness.verify_telemetry(path, "hot")["samples"] == 10
    path.write_text("\n".join(rows[:3]))
    with pytest.raises(RuntimeError, match="Insufficient"):
        harness.verify_telemetry(path, "hot")
    path.write_text("\n".join([*rows[:3], rows[9]] * 3))
    with pytest.raises(RuntimeError, match="missing or non-monotonic"):
        harness.verify_telemetry(path, "hot")


def test_wait_trainer_rejects_timeout_and_foreign_gpu_pid(monkeypatch):
    class Running:
        pid = 111
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(harness, "_compute_pids", lambda: set())
    monkeypatch.setattr(harness.time, "monotonic", iter((0, 99999)).__next__)
    with pytest.raises(TimeoutError):
        harness._wait_trainer(Running(), "smoke")
    monkeypatch.setattr(harness.time, "monotonic", lambda: 0)
    monkeypatch.setattr(harness, "_compute_pids", lambda: {222})
    monkeypatch.setattr(harness.os, "getpgid", lambda _pid: 222)
    with pytest.raises(RuntimeError, match="Unexpected GPU"):
        harness._wait_trainer(Running(), "smoke")


def test_compute_pid_query_fails_closed_on_malformed_output(monkeypatch):
    result = subprocess.CompletedProcess([], 0, stdout="unexpected\n", stderr="")
    monkeypatch.setattr(harness.subprocess, "run", lambda *_args, **_kwargs: result)
    with pytest.raises(RuntimeError, match="Cannot parse"):
        harness._compute_pids()


def test_stop_group_reaps_real_cpu_process():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                               start_new_session=True)
    harness._stop_group(process)
    assert process.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)


def test_interrupt_during_cleanup_is_raised_after_cleanup():
    cleaned = []
    with pytest.raises(KeyboardInterrupt, match="during cleanup"):
        with harness._defer_interrupts():
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            cleaned.append(True)
    assert cleaned == [True]


@pytest.mark.parametrize("outcome", ("ok", "exit", "timeout", "interrupt"))
def test_run_one_cleans_both_processes_and_accepts_only_complete_run(tmp_path, monkeypatch, outcome):
    root = tmp_path / "runs"
    root.mkdir()
    (root / "source_sha256.json").write_text("{}")
    base = harness.load_reference()
    monkeypatch.setattr(harness, "source_manifest", lambda: {})
    monkeypatch.setattr(harness, "input_manifest", lambda _base: {})
    monkeypatch.setattr(harness, "gpu_preflight", lambda: {})
    monkeypatch.setattr(harness, "disk_preflight", lambda _root: {})
    created = []
    stopped = []

    class FakeProcess:
        returncode = 0

        def __init__(self, *_args, **kwargs):
            assert kwargs["start_new_session"] is True
            created.append(self)

    monkeypatch.setattr(harness.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(harness, "_stop_group", lambda process: stopped.append(process))

    def wait(_trainer, _phase):
        if outcome == "timeout":
            raise TimeoutError("expired")
        if outcome == "interrupt":
            raise KeyboardInterrupt()
        directory = root / "lora-smoke"
        rows = [{"ev": "step", "global_step": step, "loss/current": 0.5}
                for step in (1, 2)]
        rows.append({"ev": "run_end", "status": "ok", "final_step": 2})
        (directory / "progress.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows))
        (directory / "gpu.csv").write_text("\n".join(
            f"2026/09/29 12:00:0{second}.000, 70, 1200, 5000, 6000, 120, 95"
            for second in range(3)))
        checkpoint = directory / "checkpoints"
        checkpoint.mkdir()
        save_file({"block.lora_down.weight": torch.ones(2),
                   "block.lora_up.weight": torch.ones(2)},
                  str(checkpoint / "qwen21_lora_smoke.safetensors"),
                  {"ss_network_spec": "lora"})
        return 1 if outcome == "exit" else 0

    monkeypatch.setattr(harness, "_wait_trainer", wait)
    if outcome == "ok":
        harness.run_one(base, "lora", "smoke", root, {}, {})
        assert (root / "lora-smoke" / "acceptance.json").is_file()
    else:
        with pytest.raises((RuntimeError, TimeoutError, KeyboardInterrupt)):
            harness.run_one(base, "lora", "smoke", root, {}, {})
        assert not (root / "lora-smoke" / "acceptance.json").exists()
    assert len(created) == 2
    assert stopped == [created[1], created[0]]


def test_plan_is_read_only_and_run_requires_explicit_flag(tmp_path, capsys):
    target = tmp_path / "new-run"
    assert harness.main(["--plan", "--output", str(target)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert [job["variant"] for job in plan["jobs"]] == [
        "lora", "lora", "lokr", "lokr", "loha", "loha"
    ]
    assert "scripts/qwen_image_2_1/variant_hot_acceptance.py" in plan["source_sha256"]
    assert "networks/registry_api.py" in plan["source_sha256"]
    assert "library/training/loop.py" in plan["source_sha256"]
    assert plan["input_fingerprints"]
    assert "not proof" in plan["input_fingerprint_limit"]
    assert not target.exists()
    assert harness.main(["--plan", "--output", str(target), "--variants", "lokr",
                         "--phases", "smoke"]) == 0
    filtered = json.loads(capsys.readouterr().out)
    assert [(job["variant"], job["phase"]) for job in filtered["jobs"]] == [("lokr", "smoke")]
    with pytest.raises(SystemExit):
        harness.main([])
    with pytest.raises(FileExistsError):
        harness.main(["--plan", "--output", str(tmp_path)])
    with pytest.raises(ValueError, match="Stage-3 history"):
        harness.main(["--plan", "--output", str(harness.REFERENCE / "new")])
