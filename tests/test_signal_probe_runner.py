from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.signal_probe import run_training


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
    )
    base.update(overrides)
    return argparse.Namespace(**base)


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
