"""Reproducibility and strict target-device identity."""

from __future__ import annotations
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GPU_UUID = "GPU-806debd9-d254-da57-d515-ccae15b4ac6a"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def software_manifest() -> dict:
    packages = {}
    for name in ("torch", "triton", "flash_attn", "bitsandbytes", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    sources = {}
    for directory in ("library", "networks", "bench/convrot_90hx"):
        for path in sorted((ROOT / directory).rglob("*.py")):
            sources[str(path.relative_to(ROOT))] = file_hash(path)
    probe = ROOT / "scripts/experiments/int8_linear_equivalence_probe.py"
    sources[str(probe.relative_to(ROOT))] = file_hash(probe)
    return {
        "packages": packages,
        "source_hashes": sources,
        "source_hash": object_hash(sources),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ),
    }


def validate_device(expected_uuid: str = GPU_UUID) -> dict:
    import torch

    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_uuid:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES must equal target UUID {expected_uuid}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device must be visible")
    props = torch.cuda.get_device_properties(0)
    actual = "GPU-" + str(props.uuid).removeprefix("GPU-")
    if actual != expected_uuid:
        raise RuntimeError(f"CUDA device UUID mismatch: {actual} != {expected_uuid}")
    return {
        "uuid": actual,
        "name": props.name,
        "sm": [props.major, props.minor],
        "total_memory_bytes": props.total_memory,
        "torch_cuda": torch.version.cuda,
        "logical_device": "cuda:0",
        "int_mm_available": hasattr(torch, "_int_mm"),
    }
