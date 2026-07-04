from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import toml

from bench.mfu import flops
from bench.mfu import run_training

pytestmark = pytest.mark.fast


GPU_ROWS = [
    {
        "index": "0",
        "name": "NVIDIA GeForce GTX 1050",
        "memory_total_mb": "4096",
        "memory_used_mb": "100",
        "utilization_gpu_pct": "0",
    },
    {
        "index": "1",
        "name": "NVIDIA GeForce RTX 3080 Ti Laptop GPU",
        "memory_total_mb": "16384",
        "memory_used_mb": "100",
        "utilization_gpu_pct": "0",
    },
]


def _args(**overrides):
    base = dict(
        gpu_index="1",
        allow_gpu0=False,
        min_vram_mb=12000,
        allow_low_vram=False,
        python=Path(".venv/bin/python"),
        steps=80,
        output_root="output/bench/mfu",
        launch_mode="tasks-gui",
        sample_prompts=run_training.DEFAULT_PROMPTS,
        dataset_config=run_training.DEFAULT_DATASET_CONFIG,
        extra=[],
        peak_probe_level="block",
        peak_probe_max_steps=1,
        metric_step_window=(10, 60),
        peak_tflops=181.0,
        dry_run=True,
        stop_on_fail=True,
        train_timeout_sec=3600,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_parse_step_shape_from_peak_probe_event():
    event = {"tensor_shape": [1, 1, 63, 64, 2048]}
    shape = flops.parse_step_shape_from_peak_probe_event(event)
    assert shape.batch_size == 1
    assert shape.token_count == 4032


def test_forward_flops_grow_with_token_count():
    spec = flops.AnimaModelSpec()
    small = flops.StepShape(batch_size=1, time_patches=1, height_patches=63, width_patches=64)
    large = flops.StepShape(batch_size=1, time_patches=1, height_patches=60, width_patches=70)
    assert flops.total_forward_flops(large, spec) > flops.total_forward_flops(small, spec)


def test_estimate_mfu_returns_fraction():
    spec = flops.AnimaModelSpec()
    shape = flops.StepShape(batch_size=1, time_patches=1, height_patches=63, width_patches=64)
    mfu = flops.estimate_mfu(shape=shape, avg_step_sec=0.5, peak_flops=181e12, spec=spec)
    assert mfu is not None
    assert 0 < mfu < 10


def test_parse_suite_compile():
    assert run_training._parse_arms(None, "compile") == ["baseline", "no_compile"]


def test_parse_suite_plain_lora():
    assert run_training._parse_arms(None, "plain_lora") == ["plain_lora_ckpt"]


def test_mfu_defaults_do_not_reference_local_rokkotsu_configs():
    values = [
        run_training.DEFAULT_VARIANT,
        run_training.DEFAULT_DATASET_CONFIG,
        run_training.DEFAULT_SINGLE_DATASET_CONFIG,
        run_training.DEFAULT_PROMPTS,
        *(arm.variant for arm in run_training.ARMS.values()),
    ]
    assert all("mfu_rokkotsu" not in value for value in values)
    default_variant_path = (
        run_training.REPO_ROOT / "configs" / "gui-methods" / f"{run_training.DEFAULT_VARIANT}.toml"
    )
    assert default_variant_path.exists()
    assert (run_training.REPO_ROOT / run_training.DEFAULT_DATASET_CONFIG).exists()
    assert (run_training.REPO_ROOT / run_training.DEFAULT_PROMPTS).exists()


def test_gpu_guard_refuses_gpu0_without_override(monkeypatch):
    monkeypatch.setattr(run_training, "_gpu_rows", lambda: GPU_ROWS)
    with pytest.raises(SystemExit, match="physical GPU 0"):
        run_training._check_gpu(_args(gpu_index="0"))


def test_torch_mapping_check_sets_pci_order(monkeypatch):
    seen_env = {}

    def fake_run(cmd, cwd, env, capture_output, text, timeout):
        seen_env.update(env)
        payload = {"count": 1, "name": "NVIDIA GeForce RTX 3080 Ti Laptop GPU", "memory_total_mb": 15982}
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload) + "\n", stderr="")

    monkeypatch.setattr(run_training.subprocess, "run", fake_run)

    info = run_training._verify_torch_mapping(_args(gpu_index="1"))

    assert seen_env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert seen_env["CUDA_VISIBLE_DEVICES"] == "1"
    assert "3080 Ti" in info["name"]


def test_build_train_cmd_enables_peak_probe_and_compile(tmp_path):
    args = argparse.Namespace(
        python=Path(".venv/bin/python"),
        steps=80,
        gpu_index="1",
        launch_mode="tasks-gui",
        sample_prompts=run_training.DEFAULT_PROMPTS,
        dataset_config=run_training.DEFAULT_DATASET_CONFIG,
        extra=[],
        peak_probe_level="block",
        peak_probe_max_steps=1,
    )
    cmd, env, paths = run_training._build_train_cmd(args, run_training.ARMS["baseline"], 42, tmp_path / "run", "baseline_s42_80step")

    assert cmd[:4] == [".venv/bin/python", "tasks.py", "lora-gui", run_training.DEFAULT_VARIANT]
    assert "--peak_probe_jsonl" in cmd
    assert "--peak_probe_level" in cmd and "block" in cmd
    assert "--torch_compile" in cmd
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert paths["peak_probe_jsonl"].endswith("baseline_s42_80step.peak_probe.jsonl")


def test_build_train_cmd_can_disable_compile_via_arm(tmp_path):
    args = argparse.Namespace(
        python=Path(".venv/bin/python"),
        steps=80,
        gpu_index="1",
        launch_mode="tasks-gui",
        sample_prompts=run_training.DEFAULT_PROMPTS,
        dataset_config=run_training.DEFAULT_DATASET_CONFIG,
        extra=[],
        peak_probe_level="block",
        peak_probe_max_steps=1,
    )
    cmd, _, paths = run_training._build_train_cmd(args, run_training.ARMS["no_compile"], 42, tmp_path / "run", "no_compile_s42_80step")
    assert "--torch_compile" not in cmd
    assert "--no-torch_compile" not in cmd
    assert "--config_file" in cmd
    config_path = Path(paths["config_file"])
    assert config_path.exists()
    payload = toml.load(config_path)
    assert payload["torch_compile"] is False


def test_build_train_cmd_direct_mode_materializes_config(tmp_path):
    args = argparse.Namespace(
        python=Path(".venv/bin/python"),
        steps=4,
        gpu_index="1",
        launch_mode="direct",
        sample_prompts=run_training.DEFAULT_PROMPTS,
        dataset_config=run_training.DEFAULT_SINGLE_DATASET_CONFIG,
        extra=["--max_data_loader_n_workers", "0"],
        peak_probe_level="block",
        peak_probe_max_steps=1,
    )
    cmd, _, paths = run_training._build_train_cmd(
        args,
        run_training.ARMS["baseline"],
        42,
        tmp_path / "run",
        "baseline_s42_4step",
    )

    assert cmd[:2] == [".venv/bin/python", "train.py"]
    assert "tasks.py" not in cmd
    assert "--config_file" in cmd
    config_path = Path(paths["config_file"])
    payload = toml.load(config_path)
    assert payload["torch_compile"] is True
    assert "--torch_compile" in cmd


def test_build_train_cmd_direct_mode_no_compile_materializes_false(tmp_path):
    args = argparse.Namespace(
        python=Path(".venv/bin/python"),
        steps=4,
        gpu_index="1",
        launch_mode="direct",
        sample_prompts=run_training.DEFAULT_PROMPTS,
        dataset_config=run_training.DEFAULT_SINGLE_DATASET_CONFIG,
        extra=["--max_data_loader_n_workers", "0"],
        peak_probe_level="block",
        peak_probe_max_steps=1,
    )
    cmd, _, paths = run_training._build_train_cmd(
        args,
        run_training.ARMS["no_compile"],
        42,
        tmp_path / "run",
        "no_compile_s42_4step",
    )

    config_path = Path(paths["config_file"])
    payload = toml.load(config_path)
    assert payload["torch_compile"] is False
    assert "--torch_compile" not in cmd


def test_summarize_progress_step_window_and_vram(tmp_path):
    path = tmp_path / "progress.jsonl"
    events = [
        {"ev": "run_start", "ts": 0.0},
        {"ev": "step", "ts": 1.0, "global_step": 1, "avr_loss": 0.5, "cuda/max_memory_reserved_gb": 8.0},
        {"ev": "step", "ts": 3.0, "global_step": 2, "avr_loss": 0.4, "cuda/max_memory_reserved_gb": 9.0},
        {"ev": "step", "ts": 6.0, "global_step": 3, "avr_loss": 0.3, "cuda/max_memory_reserved_gb": 10.0},
        {"ev": "run_end", "ts": 7.0, "status": "ok", "final_step": 3, "error": None},
    ]
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")

    summary = run_training._summarize_progress(path, metric_window=(2, 3))

    assert summary["steps_completed"] == 3
    assert summary["avg_step_sec"] == 2.5
    assert summary["median_step_sec"] == 2.5
    assert summary["peak_reserved_gb"] == 10.0
    assert summary["avr_loss"] == 0.3
    assert summary["run_end_status"] == "ok"


def test_first_block_probe_shape_reads_block_zero(tmp_path):
    path = tmp_path / "peak_probe.jsonl"
    events = [
        {"ev": "peak_probe", "label": "step_begin", "step": 0},
        {"ev": "peak_probe", "label": "block_before", "block_idx": 0, "tensor_shape": [1, 1, 63, 64, 2048]},
        {"ev": "peak_probe", "label": "block_after", "block_idx": 1, "tensor_shape": [1, 1, 63, 64, 2048]},
    ]
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")

    shape = run_training._first_block_probe_shape(path)

    assert shape is not None
    assert shape.token_count == 4032


def test_estimate_metrics_combines_progress_and_peak_probe(tmp_path):
    progress_path = tmp_path / "progress.jsonl"
    peak_path = tmp_path / "peak_probe.jsonl"
    progress_events = [
        {"ev": "step", "ts": 1.0, "global_step": 1, "avr_loss": 0.5},
        {"ev": "step", "ts": 3.0, "global_step": 2, "avr_loss": 0.4},
        {"ev": "step", "ts": 5.0, "global_step": 3, "avr_loss": 0.3},
    ]
    peak_events = [
        {"ev": "peak_probe", "label": "block_before", "block_idx": 0, "tensor_shape": [1, 1, 63, 64, 2048]},
    ]
    progress_path.write_text("\n".join(json.dumps(e) for e in progress_events), encoding="utf-8")
    peak_path.write_text("\n".join(json.dumps(e) for e in peak_events), encoding="utf-8")

    metrics = run_training._estimate_metrics(
        progress_path=progress_path,
        peak_probe_path=peak_path,
        metric_window=(2, 3),
        peak_tflops=181.0,
        spec=flops.AnimaModelSpec(),
    )

    assert metrics["shape"]["token_count"] == 4032
    assert metrics["forward_flops"] is not None
    assert metrics["train_step_flops"] is not None
    assert metrics["achieved_tflops"] is not None
    assert metrics["mfu"] is not None


def test_dry_run_default_output_root_is_isolated():
    assert (
        run_training._resolve_output_root(
            run_training.DEFAULT_ROOT,
            dry_run=True,
            output_root_explicit=False,
        )
        == run_training.DEFAULT_DRY_RUN_ROOT
    )
    assert (
        run_training._resolve_output_root(
            run_training.DEFAULT_ROOT,
            dry_run=True,
            output_root_explicit=True,
        )
        == run_training.DEFAULT_ROOT
    )


def test_run_one_passes_train_timeout_to_subprocess(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd, cwd, env, stdout, stderr, timeout):
        seen["timeout"] = timeout
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_training.subprocess, "run", fake_run)
    args = _args(output_root=str(tmp_path / "out"), dry_run=False, train_timeout_sec=13)

    record = run_training._run_one(
        args,
        run_training.ARMS["baseline"],
        42,
        gpu_rows=[],
        spec=flops.AnimaModelSpec(),
    )

    assert seen["timeout"] == 13
    assert record["metrics"]["returncode"] == 0
    assert record["metrics"]["timed_out"] is False
    assert record["metrics"]["train_timeout_sec"] == 13


def test_run_one_records_train_timeout(monkeypatch, tmp_path):
    def fake_run(cmd, cwd, env, stdout, stderr, timeout):
        raise run_training.subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(run_training.subprocess, "run", fake_run)
    args = _args(
        output_root=str(tmp_path / "out"),
        dry_run=False,
        stop_on_fail=False,
        train_timeout_sec=9,
    )

    record = run_training._run_one(
        args,
        run_training.ARMS["baseline"],
        42,
        gpu_rows=[],
        spec=flops.AnimaModelSpec(),
    )

    assert record["metrics"]["returncode"] == 124
    assert record["metrics"]["timed_out"] is True
    assert record["metrics"]["train_timeout_sec"] == 9
    assert record["metrics"]["run_end_status"] == "timeout"
    assert record["metrics"]["run_end_error"] == "timeout after 9s"
    assert "timeout after 9s" in Path(record["paths"]["stdout"]).read_text(encoding="utf-8")
