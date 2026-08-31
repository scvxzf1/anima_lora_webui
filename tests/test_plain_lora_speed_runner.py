from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.plain_lora_speed import run_matrix

pytestmark = pytest.mark.fast


def _args(**overrides):
    base = dict(
        gpu_index="1",
        allow_gpu0=False,
        min_vram_mb=12000,
        allow_low_vram=False,
        python=Path(".venv/bin/python"),
        steps=80,
        output_root="output/bench/plain_lora_speed",
        sample_prompts="configs/bench/signal_probe_prompts.txt",
        dataset_config="configs/bench/signal_probe_dataset.toml",
        extra=[],
        profile_steps=None,
        metric_step_window=(10, 60),
        images_per_step=1.0,
        dry_run=True,
        stop_on_fail=True,
        train_timeout_sec=3600,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_parse_suite_rank():
    assert run_matrix._parse_arms(None, "rank") == ["rank16", "rank32", "rank64"]


def test_parse_arms_csv():
    assert run_matrix._parse_arms(["baseline,rank16", "workers4"], None) == [
        "baseline",
        "rank16",
        "workers4",
    ]


def test_build_train_cmd_uses_plain_gui_lora_and_slim_flags(tmp_path):
    args = argparse.Namespace(
        python=Path(".venv/bin/python"),
        steps=80,
        gpu_index="1",
        sample_prompts="configs/bench/signal_probe_prompts.txt",
        dataset_config="configs/bench/signal_probe_dataset.toml",
        extra=[],
        profile_steps=None,
    )
    arm = run_matrix.ARMS["baseline"]

    cmd, env, paths = run_matrix._build_train_cmd(args, arm, 42, tmp_path / "run", "baseline_s42_80step")

    assert cmd[:4] == [".venv/bin/python", "tasks.py", "lora-gui", "lora"]
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["PRESET"] == "default"
    assert "--method" not in cmd  # tasks.py lora-gui injects gui-methods; no heavy methods/lora.toml path.
    for flag in ["--no-use_cmmd", "--torch_compile", "--dataloader_pin_memory", "--persistent_data_loader_workers"]:
        assert flag in cmd
    assert cmd[cmd.index("--validation_split_num") + 1] == "0"
    assert cmd[cmd.index("--sample_every_n_steps") + 1] == "0"
    assert paths["progress_jsonl"].endswith("baseline_s42_80step.progress.jsonl")


def test_build_train_cmd_sets_profile_env(tmp_path):
    args = argparse.Namespace(
        python=Path(".venv/bin/python"),
        steps=80,
        gpu_index="1",
        sample_prompts="configs/bench/signal_probe_prompts.txt",
        dataset_config="configs/bench/signal_probe_dataset.toml",
        extra=[],
        profile_steps="10-60",
    )
    cmd, env, paths = run_matrix._build_train_cmd(args, run_matrix.ARMS["rank16"], 42, tmp_path / "run", "rank16_s42_80step")

    assert env["PROFILE_STEPS"] == "10-60"
    assert env["NSYS_OUT"].endswith("rank16_s42_80step.nsys-rep")
    assert "--network_dim" in cmd and "16" in cmd
    assert paths["nsys_report"].endswith("rank16_s42_80step.nsys-rep")


def test_summarize_progress_step_window_and_vram(tmp_path):
    path = tmp_path / "progress.jsonl"
    events = [
        {"ev": "run_start", "ts": 0.0},
        {"ev": "step", "ts": 1.0, "global_step": 1, "avr_loss": 0.5, "cuda/max_memory_reserved_gb": 8.0},
        {"ev": "step", "ts": 3.0, "global_step": 2, "avr_loss": 0.4, "cuda/max_memory_reserved_gb": 9.0},
        {"ev": "step", "ts": 6.0, "global_step": 3, "avr_loss": 0.3, "cuda/max_memory_reserved_gb": 10.0},
    ]
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")

    summary = run_matrix._summarize_progress(path, metric_window=(2, 3), images_per_step=1.0)

    assert summary["steps_completed"] == 3
    assert summary["avg_step_sec"] == 2.5
    assert summary["median_step_sec"] == 2.5
    assert summary["images_per_hour"] == 1440.0
    assert summary["peak_reserved_gb"] == 10.0
    assert summary["avr_loss"] == 0.3


def test_dataset_cache_preflight_counts_ready_subset(tmp_path):
    image_dir = tmp_path / "images"
    cache_dir = tmp_path / "cache"
    image_dir.mkdir()
    cache_dir.mkdir()
    for i in range(2):
        (image_dir / f"img{i}.png").write_bytes(b"x")
        (cache_dir / f"img{i}_1024x1024_anima.npz").write_bytes(b"x")
        (cache_dir / f"img{i}_anima_te.safetensors").write_bytes(b"x")
    cfg = tmp_path / "dataset.toml"
    cfg.write_text(
        f'''
[general]
caption_extension = ".txt"

[[datasets]]
batch_size = 1

  [[datasets.subsets]]
  image_dir = "{image_dir}"
  cache_dir = "{cache_dir}"
  num_repeats = 1
''',
        encoding="utf-8",
    )

    rows = run_matrix._dataset_cache_summary(cfg)

    assert len(rows) == 1
    assert rows[0].image_count == 2
    assert rows[0].vae_cache_count == 2
    assert rows[0].text_cache_count == 2
    assert rows[0].ready is True


def test_dry_run_default_output_root_is_isolated():
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


def test_run_one_passes_train_timeout_to_subprocess(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd, cwd, env, stdout, stderr, timeout):
        seen["timeout"] = timeout
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_matrix.subprocess, "run", fake_run)
    args = _args(output_root=str(tmp_path / "out"), dry_run=False, train_timeout_sec=11)

    record = run_matrix._run_one(args, run_matrix.ARMS["baseline"], 42, gpu_rows=[], cache_rows=[])

    assert seen["timeout"] == 11
    assert record["metrics"]["returncode"] == 0
    assert record["metrics"]["timed_out"] is False
    assert record["metrics"]["train_timeout_sec"] == 11


def test_run_one_records_train_timeout(monkeypatch, tmp_path):
    def fake_run(cmd, cwd, env, stdout, stderr, timeout):
        raise run_matrix.subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(run_matrix.subprocess, "run", fake_run)
    args = _args(
        output_root=str(tmp_path / "out"),
        dry_run=False,
        stop_on_fail=False,
        train_timeout_sec=5,
    )

    record = run_matrix._run_one(args, run_matrix.ARMS["baseline"], 42, gpu_rows=[], cache_rows=[])

    assert record["metrics"]["returncode"] == 124
    assert record["metrics"]["timed_out"] is True
    assert record["metrics"]["train_timeout_sec"] == 5
    assert "timeout after 5s" in Path(record["paths"]["stdout"]).read_text(encoding="utf-8")
