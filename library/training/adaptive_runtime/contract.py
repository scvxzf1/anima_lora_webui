"""Stable identity for the precision request shared by OOM retry workers.

The request identity is intentionally smaller than a model checkpoint.  It
does not certify numerical safety; it proves that a retry did not silently
change the numerical policy while changing only the memory plan.
"""

from __future__ import annotations

import hashlib
import json
import os

import torch

from .training_config import resolved_mode


def _backend_flag(backend, name):
    return getattr(backend, name, None)


def precision_environment() -> dict:
    """Capture process-level settings that change floating-point execution.

    A retry on a different visible device or with different backend flags is
    not the same numerical experiment, even when the TOML fields are equal.
    CPU-only static validation does not need a device probe. If CUDA reports
    available, an unreadable device identity cannot establish a contract.
    """
    cuda = {
        "available": bool(torch.cuda.is_available()),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if cuda["available"]:
        try:
            index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            cuda.update(
                {
                    "index": index,
                    "count": torch.cuda.device_count(),
                    "name": properties.name,
                    "major": properties.major,
                    "minor": properties.minor,
                    "total_memory": properties.total_memory,
                    "uuid": str(getattr(properties, "uuid", "")) or None,
                }
            )
        except Exception as exc:
            raise ValueError(
                "precision_contract_mismatch: CUDA device identity unavailable"
            ) from exc
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "hip_version": getattr(torch.version, "hip", None),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": _backend_flag(torch.backends.cuda.matmul, "allow_tf32"),
        "matmul_allow_fp16_reduced_precision_reduction": _backend_flag(
            torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"
        ),
        "matmul_allow_bf16_reduced_precision_reduction": _backend_flag(
            torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"
        ),
        "cudnn_allow_tf32": _backend_flag(torch.backends.cudnn, "allow_tf32"),
        "cudnn_benchmark": _backend_flag(torch.backends.cudnn, "benchmark"),
        "cudnn_deterministic": _backend_flag(torch.backends.cudnn, "deterministic"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "sdp_flash": getattr(torch.backends.cuda, "flash_sdp_enabled", lambda: None)(),
        "sdp_mem_efficient": getattr(
            torch.backends.cuda, "mem_efficient_sdp_enabled", lambda: None
        )(),
        "sdp_math": getattr(torch.backends.cuda, "math_sdp_enabled", lambda: None)(),
        "sdp_cudnn": getattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: None)(),
        "cuda": cuda,
    }


def precision_request(args) -> dict:
    """Return the precision-affecting part of a merged training config.

    Memory controls and output paths are deliberately absent.  The ordered
    FP32 pattern list is retained because it is part of the user's explicit
    request, even when two orderings happen to produce the same assignments.
    """
    patterns = getattr(args, "adaptive_fp32_modules", ()) or ()
    if not isinstance(patterns, (list, tuple)):
        patterns = (str(patterns),)
    return {
        "schema": "adaptive_precision_request_v1",
        "mode": resolved_mode(args),
        "requested_mode": str(getattr(args, "adaptive_precision", "off") or "off"),
        "candidate": getattr(args, "adaptive_candidate", None),
        "mixed_precision": str(getattr(args, "mixed_precision", "bf16") or "bf16"),
        "model_family": str(getattr(args, "model_family", "anima") or "anima"),
        "adaptive_fp32_modules": [str(pattern) for pattern in patterns],
        "adaptive_loss_scale": float(getattr(args, "adaptive_loss_scale", 1024.0)),
        "base_compute": str(getattr(args, "base_compute", "bf16") or "bf16"),
        "attn_mode": str(getattr(args, "attn_mode", "torch") or "torch"),
        "environment": precision_environment(),
    }


def precision_contract_id(args) -> str:
    """Hash the canonical precision request for cross-process comparison."""
    payload = json.dumps(
        precision_request(args), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
