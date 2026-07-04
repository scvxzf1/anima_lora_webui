#!/usr/bin/env python
"""Probe GPU theoretical ceilings for MFU normalization.

Outputs:
- FP32 theoretical throughput
- estimated Tensor Core BF16/FP16 throughput
- theoretical memory bandwidth
- a recommended `peak_tflops` value for `bench.mfu.run_training`

The probe prefers runtime-readable hardware properties over hardcoded SKU
tables, but still uses a small architecture mapping for Tensor Core throughput
per SM because CUDA runtime does not expose that directly.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench._common import REPO_ROOT, make_run_dir, write_result

DEFAULT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_GPU_INDEX = 0


@dataclass(frozen=True)
class GpuStaticInfo:
    index: int
    name: str
    compute_capability: str
    major: int
    minor: int
    sm_count: int
    sm_clock_mhz: float
    memory_clock_mhz: float
    memory_bus_width_bits: int
    total_memory_bytes: int
    uuid: str | None = None
    pci_bus_id: str | None = None


def _query_gpu_info(python: Path, gpu_index: int) -> GpuStaticInfo:
    code = r"""
import json, torch
idx = int(__import__("os").environ["ANIMA_GPU_INDEX"])
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
count = torch.cuda.device_count()
if idx < 0 or idx >= count:
    raise SystemExit(f"GPU index {idx} out of range for visible device count {count}")
p = torch.cuda.get_device_properties(idx)
out = {
    "index": idx,
    "name": p.name,
    "major": p.major,
    "minor": p.minor,
    "compute_capability": f"{p.major}.{p.minor}",
    "sm_count": p.multi_processor_count,
    "sm_clock_mhz": getattr(p, "clock_rate", 0) / 1000.0,
    "memory_clock_mhz": getattr(p, "memory_clock_rate", 0) / 1000.0,
    "memory_bus_width_bits": getattr(p, "memory_bus_width", 0),
    "total_memory_bytes": p.total_memory,
    "uuid": str(getattr(p, "uuid", None)) if getattr(p, "uuid", None) is not None else None,
    "pci_bus_id": str(getattr(p, "pci_bus_id", None)) if getattr(p, "pci_bus_id", None) is not None else None,
}
print(json.dumps(out, ensure_ascii=False))
"""
    env = {**os.environ, "ANIMA_GPU_INDEX": str(gpu_index)}
    cp = subprocess.run(
        [str(python), "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    if cp.returncode != 0:
        raise SystemExit(f"failed to query torch CUDA properties:\n{cp.stdout}\n{cp.stderr}")
    lines = [ln for ln in cp.stdout.splitlines() if ln.strip().startswith("{")]
    if not lines:
        raise SystemExit(f"torch CUDA query produced no JSON:\n{cp.stdout}\n{cp.stderr}")
    data = json.loads(lines[-1])
    return GpuStaticInfo(
        index=int(data["index"]),
        name=str(data["name"]),
        compute_capability=str(data["compute_capability"]),
        major=int(data["major"]),
        minor=int(data["minor"]),
        sm_count=int(data["sm_count"]),
        sm_clock_mhz=float(data["sm_clock_mhz"]),
        memory_clock_mhz=float(data["memory_clock_mhz"]),
        memory_bus_width_bits=int(data["memory_bus_width_bits"]),
        total_memory_bytes=int(data["total_memory_bytes"]),
        uuid=data.get("uuid"),
        pci_bus_id=data.get("pci_bus_id"),
    )


def _per_sm_tensor_ops_per_cycle_bf16(major: int, minor: int) -> int | None:
    """Estimated BF16/FP16 tensor FMA ops per SM per cycle.

    Returns FLOP operations counted with the common "FMA = 2 FLOPs" convention.
    These values are intended for MFU normalization and match the scale used by
    public GPU TFLOPS specs.
    """

    cc = (int(major), int(minor))
    table: dict[tuple[int, int], int] = {
        (7, 5): 1024,   # Turing
        (8, 0): 2048,   # A100 / GA100
        (8, 6): 1024,   # GA10x consumer Ampere
        (8, 9): 2048,   # Ada datacenter-like estimate
        (9, 0): 2048,   # Hopper
        (10, 0): 2048,  # Blackwell+ conservative carry-forward
        (12, 0): 2048,
    }
    if cc in table:
        return table[cc]
    if major >= 9:
        return 2048
    if major == 8:
        return 1024
    if major == 7:
        return 1024
    return None


def fp32_peak_tflops(info: GpuStaticInfo) -> float:
    # Assume 64 FP32 CUDA cores per SM for NVIDIA post-Pascal consumer/datacenter GPUs.
    # The common spec formula is:
    #   SM_count * cores_per_sm * 2 FLOPs/FMA * clock_hz
    cores_per_sm = 64
    return info.sm_count * cores_per_sm * 2.0 * (info.sm_clock_mhz * 1e6) / 1e12


def tensor_bf16_peak_tflops(info: GpuStaticInfo) -> float | None:
    ops_per_sm_per_cycle = _per_sm_tensor_ops_per_cycle_bf16(info.major, info.minor)
    if ops_per_sm_per_cycle is None or info.sm_clock_mhz <= 0:
        return None
    return info.sm_count * ops_per_sm_per_cycle * (info.sm_clock_mhz * 1e6) / 1e12


def memory_bandwidth_gbps(info: GpuStaticInfo, *, ddr_factor: float = 2.0) -> float | None:
    if info.memory_clock_mhz <= 0 or info.memory_bus_width_bits <= 0:
        return None
    bytes_per_cycle = info.memory_bus_width_bits / 8.0
    hz = info.memory_clock_mhz * 1e6
    return bytes_per_cycle * hz * ddr_factor / 1e9


def recommended_peak_tflops(info: GpuStaticInfo) -> float:
    tensor = tensor_bf16_peak_tflops(info)
    if tensor is not None:
        return tensor
    return fp32_peak_tflops(info)


def build_metrics(info: GpuStaticInfo) -> dict[str, Any]:
    tensor = tensor_bf16_peak_tflops(info)
    bandwidth = memory_bandwidth_gbps(info)
    fp32 = fp32_peak_tflops(info)
    return {
        "gpu": {
            "index": info.index,
            "name": info.name,
            "compute_capability": info.compute_capability,
            "sm_count": info.sm_count,
            "sm_clock_mhz": round(info.sm_clock_mhz, 3),
            "memory_clock_mhz": round(info.memory_clock_mhz, 3),
            "memory_bus_width_bits": info.memory_bus_width_bits,
            "total_memory_gib": round(info.total_memory_bytes / (1024 ** 3), 3),
            "uuid": info.uuid,
            "pci_bus_id": info.pci_bus_id,
        },
        "theoretical": {
            "fp32_peak_tflops": round(fp32, 6),
            "bf16_peak_tflops": None if tensor is None else round(tensor, 6),
            "memory_bandwidth_gbps": None if bandwidth is None else round(bandwidth, 6),
            "recommended_peak_tflops": round(recommended_peak_tflops(info), 6),
        },
        "formula_notes": {
            "fp32": "sm_count * 64 cores_per_sm * 2 flop_per_fma * sm_clock_hz",
            "bf16": "sm_count * tensor_ops_per_sm_per_cycle * sm_clock_hz",
            "memory_bandwidth": "(bus_width_bits / 8) * memory_clock_hz * 2 / 1e9",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--gpu-index", type=int, default=DEFAULT_GPU_INDEX)
    parser.add_argument("--label", default=None, help="Optional run-dir label")
    args = parser.parse_args()

    if not args.python.exists():
        raise SystemExit(f"python executable not found: {args.python}")

    info = _query_gpu_info(args.python, args.gpu_index)
    metrics = build_metrics(info)
    run_dir = make_run_dir("mfu", label=args.label or f"gpu-theoretical-gpu{args.gpu_index}")
    out = run_dir / "gpu_theoretical.json"
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        artifacts=[out],
        label="gpu-theoretical",
        device=f"cuda:{args.gpu_index}",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\nartifact: {out}")


if __name__ == "__main__":
    main()
