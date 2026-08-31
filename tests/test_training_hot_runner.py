from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.mfu import run_training as mfu_run_training
from bench.plain_lora_speed import run_matrix as plain_lora_run_matrix
from bench.signal_probe import run_training as signal_run_training
from bench.training_hot import run_matrix

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


def _gpu_guard_args(gpu_index: str) -> argparse.Namespace:
    return argparse.Namespace(
        gpu_index=gpu_index,
        allow_gpu0=False,
        min_vram_mb=12000,
        allow_low_vram=False,
        python=Path(".venv/bin/python"),
    )


@pytest.mark.parametrize(
    "runner",
    [plain_lora_run_matrix, mfu_run_training, signal_run_training],
    ids=["plain-lora", "mfu", "signal-probe"],
)
def test_runner_gpu_guard_refuses_gpu0_without_override(monkeypatch, runner) -> None:
    monkeypatch.setattr(runner, "_gpu_rows", lambda: GPU_ROWS)

    with pytest.raises(SystemExit, match="physical GPU 0"):
        runner._check_gpu(_gpu_guard_args("0"))


@pytest.mark.parametrize(
    "runner",
    [plain_lora_run_matrix, mfu_run_training, signal_run_training],
    ids=["plain-lora", "mfu", "signal-probe"],
)
def test_runner_torch_mapping_sets_pci_order(monkeypatch, runner) -> None:
    seen_env = {}

    def fake_run(cmd, cwd, env, capture_output, text, timeout):
        seen_env.update(env)
        payload = {
            "count": 1,
            "name": "NVIDIA GeForce RTX 3080 Ti Laptop GPU",
            "memory_total_mb": 15982,
        }
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(payload) + "\n",
            stderr="",
        )

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    info = runner._verify_torch_mapping(_gpu_guard_args("1"))

    assert seen_env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert seen_env["CUDA_VISIBLE_DEVICES"] == "1"
    assert "3080 Ti" in info["name"]


def _args(**overrides):
    base = dict(
        python=Path(".venv/bin/python"),
        steps=12,
        gpu_index="1",
        sample_prompts="configs/bench/signal_probe_prompts.txt",
        dataset_config="configs/bench/signal_probe_dataset.toml",
        extra=[],
        profile_steps=None,
        memory_probe=False,
        memory_probe_max_steps=2,
        block_swap_profile=False,
        metric_step_window=None,
        images_per_step=1.0,
        output_root="output/bench/training_hot",
        dry_run=True,
        stop_on_fail=True,
        train_timeout_sec=3600,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_default_suite_excludes_lokr() -> None:
    cases = run_matrix._parse_cases(None, "all_nonlokr")

    assert cases
    assert all("lokr" not in case.target.lower() for case in cases)


def test_compat_runtime_suite_covers_blockswap_compile_and_checkpoint_modes() -> None:
    cases = run_matrix._parse_cases(None, "compat_runtime")
    names = {case.name for case in cases}

    assert names == {
        "compat_blockswap_grad_ckpt",
        "compat_blockswap_selective_mlp",
        "compat_blockswap_cudagraphs",
        "compat_blockswap_max_autotune",
    }
    assert all(case.target == "lora" for case in cases)
    assert all("lokr" not in case.name.lower() for case in cases)


def test_parse_custom_case_specs() -> None:
    cases = run_matrix._parse_cases(
        ["gui:loha:balanced_16g", "method:lora:default:methods", "config:output/runs/x/config.runtime.toml"],
        None,
    )

    assert [(case.mode, case.target, case.preset) for case in cases] == [
        ("variant", "loha", "balanced_16g"),
        ("method", "lora", "default"),
        ("config", "output/runs/x/config.runtime.toml", "default"),
    ]
    assert cases[1].methods_subdir == "methods"


def test_build_train_cmd_for_gui_variant(tmp_path: Path) -> None:
    args = _args()
    case = run_matrix._case_from_spec("gui:lora")

    cmd, env, paths = run_matrix._build_train_cmd(args, case, 42, tmp_path / "run", "gui_lora_s42")

    assert cmd[:4] == [".venv/bin/python", "tasks.py", "lora-gui", "lora"]
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["PRESET"] == "default"
    assert "--progress_jsonl" in cmd
    assert cmd[cmd.index("--sample_every_n_steps") + 1] == "0"
    assert cmd[cmd.index("--validation_split_num") + 1] == "0"
    assert paths["progress_jsonl"].endswith("gui_lora_s42.progress.jsonl")


def test_build_train_cmd_for_method_and_config_file(tmp_path: Path) -> None:
    args = _args(extra=["--torch_compile"])
    method = run_matrix._case_from_spec("method:lora:balanced_16g")
    config = run_matrix._case_from_spec("config:output/runs/demo/config.runtime.toml")

    method_cmd, method_env, _ = run_matrix._build_train_cmd(args, method, 7, tmp_path / "m", "method_lora_s7")
    config_cmd, _config_env, _ = run_matrix._build_train_cmd(args, config, 7, tmp_path / "c", "config_demo_s7")

    assert method_cmd[:8] == [
        ".venv/bin/python",
        "train.py",
        "--method",
        "lora",
        "--preset",
        "balanced_16g",
        "--methods_subdir",
        "methods",
    ]
    assert method_env["PRESET"] == "balanced_16g"
    assert "--torch_compile" in method_cmd
    assert config_cmd[:4] == [".venv/bin/python", "train.py", "--config_file", "output/runs/demo/config.runtime.toml"]


def test_build_train_cmd_for_compat_runtime_case(tmp_path: Path) -> None:
    args = _args()
    case = run_matrix.CASES["compat_blockswap_max_autotune"]

    cmd, _env, _paths = run_matrix._build_train_cmd(
        args,
        case,
        42,
        tmp_path / "run",
        "compat_blockswap_max_autotune_s42",
    )

    assert cmd[:4] == [".venv/bin/python", "tasks.py", "lora-gui", "lora"]
    assert "--blocks_to_swap" in cmd and cmd[cmd.index("--blocks_to_swap") + 1] == "8"
    assert "--torch_compile" in cmd
    assert "--compile_inductor_mode" in cmd
    assert cmd[cmd.index("--compile_inductor_mode") + 1] == "max-autotune"


def test_run_one_dry_run_writes_summary_and_index(tmp_path: Path) -> None:
    args = _args(output_root=str(tmp_path / "out"))
    case = run_matrix.CASES["gui_lora"]

    record = run_matrix._run_one(args, case, 42, gpu_rows=[], cache_rows=[])
    csv_path = run_matrix._write_index(args.output_root, [record], args)

    summary = json.loads(Path(record["paths"]["summary"]).read_text(encoding="utf-8"))
    assert summary["metrics"]["returncode"] == 0
    assert summary["metrics"]["checkpoint_bytes"] == 0
    assert summary["metrics"]["checkpoint_file_count"] == 0
    assert Path(record["paths"]["stdout"]).read_text(encoding="utf-8").startswith("# dry-run")

    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert rows[0]["case_name"] == "gui_lora"
    assert rows[0]["mode"] == "variant"
    assert rows[0]["checkpoint_bytes"] == "0"
    assert rows[0]["checkpoint_file_count"] == "0"


def test_dry_run_default_output_root_is_isolated() -> None:
    assert (
        run_matrix._resolve_output_root(
            run_matrix.DEFAULT_ROOT,
            dry_run=True,
            output_root_explicit=False,
        )
        == run_matrix.DEFAULT_DRY_RUN_ROOT
    )
    assert (
        run_matrix._resolve_output_root(
            run_matrix.DEFAULT_ROOT,
            dry_run=True,
            output_root_explicit=True,
        )
        == run_matrix.DEFAULT_ROOT
    )
    assert (
        run_matrix._resolve_output_root(
            run_matrix.DEFAULT_ROOT,
            dry_run=False,
            output_root_explicit=False,
        )
        == run_matrix.DEFAULT_ROOT
    )


def test_run_one_passes_train_timeout_to_subprocess(monkeypatch, tmp_path: Path) -> None:
    seen = {}

    def fake_run(cmd, cwd, env, stdout, stderr, timeout):
        seen["timeout"] = timeout
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_matrix.subprocess, "run", fake_run)
    args = _args(output_root=str(tmp_path / "out"), dry_run=False, train_timeout_sec=7)

    record = run_matrix._run_one(args, run_matrix.CASES["gui_lora"], 42, gpu_rows=[], cache_rows=[])

    assert seen["timeout"] == 7
    assert record["metrics"]["returncode"] == 0
    assert record["metrics"]["timed_out"] is False
    assert record["metrics"]["train_timeout_sec"] == 7


def test_run_one_records_train_timeout(monkeypatch, tmp_path: Path) -> None:
    def fake_run(cmd, cwd, env, stdout, stderr, timeout):
        raise run_matrix.subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(run_matrix.subprocess, "run", fake_run)
    args = _args(
        output_root=str(tmp_path / "out"),
        dry_run=False,
        stop_on_fail=False,
        train_timeout_sec=3,
    )

    record = run_matrix._run_one(args, run_matrix.CASES["gui_lora"], 42, gpu_rows=[], cache_rows=[])

    assert record["metrics"]["returncode"] == 124
    assert record["metrics"]["timed_out"] is True
    assert record["metrics"]["train_timeout_sec"] == 3
    assert "timeout after 3s" in Path(record["paths"]["stdout"]).read_text(encoding="utf-8")


def test_tasks_test_fast_help_is_lightweight() -> None:
    cp = subprocess.run(
        [sys.executable, "tasks.py", "test-fast", "--help"],
        cwd=run_matrix.REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert cp.returncode == 0
    assert "test-fast" in cp.stdout
    assert "fast smoke" in cp.stdout
