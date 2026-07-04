from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.mfu import gpu_theoretical

pytestmark = pytest.mark.fast


def test_fp32_peak_tflops_formula():
    info = gpu_theoretical.GpuStaticInfo(
        index=0,
        name="Test GPU",
        compute_capability="8.6",
        major=8,
        minor=6,
        sm_count=68,
        sm_clock_mhz=1710.0,
        memory_clock_mhz=9501.0,
        memory_bus_width_bits=320,
        total_memory_bytes=10 * 1024**3,
    )
    tflops = gpu_theoretical.fp32_peak_tflops(info)
    assert round(tflops, 3) == round(68 * 64 * 2 * 1710e6 / 1e12, 3)


def test_tensor_bf16_peak_tflops_formula_for_ampere_consumer():
    info = gpu_theoretical.GpuStaticInfo(
        index=0,
        name="RTX 3080",
        compute_capability="8.6",
        major=8,
        minor=6,
        sm_count=68,
        sm_clock_mhz=1710.0,
        memory_clock_mhz=9501.0,
        memory_bus_width_bits=320,
        total_memory_bytes=10 * 1024**3,
    )
    tflops = gpu_theoretical.tensor_bf16_peak_tflops(info)
    assert tflops is not None
    assert round(tflops, 3) == round(68 * 1024 * 1710e6 / 1e12, 3)


def test_memory_bandwidth_formula():
    info = gpu_theoretical.GpuStaticInfo(
        index=0,
        name="RTX 3080",
        compute_capability="8.6",
        major=8,
        minor=6,
        sm_count=68,
        sm_clock_mhz=1710.0,
        memory_clock_mhz=9501.0,
        memory_bus_width_bits=320,
        total_memory_bytes=10 * 1024**3,
    )
    gbps = gpu_theoretical.memory_bandwidth_gbps(info)
    assert gbps is not None
    assert round(gbps, 3) == round((320 / 8) * 9501e6 * 2 / 1e9, 3)


def test_recommended_peak_prefers_tensor_core_peak():
    info = gpu_theoretical.GpuStaticInfo(
        index=0,
        name="RTX 3080",
        compute_capability="8.6",
        major=8,
        minor=6,
        sm_count=68,
        sm_clock_mhz=1710.0,
        memory_clock_mhz=9501.0,
        memory_bus_width_bits=320,
        total_memory_bytes=10 * 1024**3,
    )
    assert gpu_theoretical.recommended_peak_tflops(info) == gpu_theoretical.tensor_bf16_peak_tflops(info)


def test_build_metrics_contains_expected_keys():
    info = gpu_theoretical.GpuStaticInfo(
        index=0,
        name="RTX 3080",
        compute_capability="8.6",
        major=8,
        minor=6,
        sm_count=68,
        sm_clock_mhz=1710.0,
        memory_clock_mhz=9501.0,
        memory_bus_width_bits=320,
        total_memory_bytes=10 * 1024**3,
    )
    metrics = gpu_theoretical.build_metrics(info)
    assert metrics["gpu"]["name"] == "RTX 3080"
    assert "recommended_peak_tflops" in metrics["theoretical"]
    assert "memory_bandwidth_gbps" in metrics["theoretical"]


def test_query_gpu_info_parses_json(monkeypatch):
    payload = {
        "index": 0,
        "name": "RTX 3080",
        "major": 8,
        "minor": 6,
        "compute_capability": "8.6",
        "sm_count": 68,
        "sm_clock_mhz": 1710.0,
        "memory_clock_mhz": 9501.0,
        "memory_bus_width_bits": 320,
        "total_memory_bytes": 10614407168,
        "uuid": "GPU-123",
        "pci_bus_id": "0000:04:00.0",
    }

    def fake_run(cmd, cwd, capture_output, text, timeout, env):
        assert env["ANIMA_GPU_INDEX"] == "0"
        assert env["CUDA_VISIBLE_DEVICES"] == "7"
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload) + "\n", stderr="")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setattr(gpu_theoretical.subprocess, "run", fake_run)
    info = gpu_theoretical._query_gpu_info(Path(".venv/bin/python"), 0)
    assert info.name == "RTX 3080"
    assert info.sm_count == 68
