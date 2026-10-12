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
    if resolved_mode(args) == "native_fp16":
        return _native_precision_request(args)
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


_NATIVE_FIELDS = ("base_compute", "network_module", "network_dim", "network_alpha", "network_dropout",
                  "network_train_unet_only", "network_train_text_encoder_only",
                  "gradient_checkpointing", "selective_checkpoint", "torch_compile",
                  "cpu_offload_checkpointing", "unsloth_offload_checkpointing",
                  "attn_mode", "attn_softmax_scale", "v100_flash_stability",
                  "block_swap_transfer_dtype", "block_swap_restore_mode", "full_fp16", "full_bf16")


def _native_precision_request(args):
    from networks import all_network_kwargs

    kwargs = {key: str(getattr(args, key))
              for key in (*all_network_kwargs(), "gradient_accumulation_steps")
              if getattr(args, key, None) is not None}
    for argument in getattr(args, "network_args", None) or ():
        key, value = argument.split("=", 1)
        kwargs[key] = value
    kwargs["model_family"] = str(getattr(args, "model_family", "anima") or "anima")
    return {"schema": "native_precision_request_v1", "mode": "native_fp16",
            "requested_mode": str(getattr(args, "adaptive_precision", "off")),
            "mixed_precision": getattr(args, "mixed_precision", "fp16"),
            "adaptive_loss_scale": float(getattr(args, "adaptive_loss_scale", 1024.0)),
            "model_family": kwargs["model_family"], "network_kwargs": kwargs,
            "execution": {key: (getattr(args, key, None) or "bf16") if key == "base_compute"
                          else getattr(args, key, None) for key in _NATIVE_FIELDS},
            "sources": getattr(args, "_native_source_identities", None),
            "source_paths": {key: getattr(args, key, None) for key in
                             ("pretrained_model_name_or_path", "network_weights")},
            "environment": precision_environment()}


def _native_source_identity(path):
    from pathlib import Path

    source = Path(os.path.expandvars(str(path))).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("Native startup retry requires local source files")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        before = _native_source_stat(os.fstat(stream.fileno()))
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        after = _native_source_stat(os.fstat(stream.fileno()))
    if before != after or after != _native_source_stat(source.stat()):
        raise ValueError("precision_contract_mismatch: native source changed while hashing")
    return {"path": str(source), "sha256": digest.hexdigest(), "size": after["size"], "stat": after}


def _native_source_stat(stat):
    return {"device": stat.st_dev, "inode": stat.st_ino, "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}


def validate_native_source_stats(args):
    """Detect replacement during actual loading without re-hashing per step."""
    from pathlib import Path

    for expected in args._native_source_identities.values():
        try:
            actual = _native_source_stat(Path(expected["path"]).stat())
        except OSError as exc:
            raise ValueError("precision_contract_mismatch: native source disappeared during loading") from exc
        if actual != expected["stat"]:
            raise ValueError("precision_contract_mismatch: native source changed during loading")


def freeze_native_sources(args):
    """Supervisor freezes local content once, before any worker is launched."""
    identities = {}
    for key in ("pretrained_model_name_or_path", "network_weights"):
        path = getattr(args, key, None)
        if path:
            identities[key] = _native_source_identity(path)
        elif key == "pretrained_model_name_or_path":
            raise ValueError("Native startup retry requires a local base checkpoint")
    args._native_source_identities = identities
    return identities


def validate_native_sources(args):
    """Worker startup rechecks content once; never call on per-step paths."""
    expected = getattr(args, "_native_source_identities", None)
    if not expected:
        raise ValueError("precision_contract_mismatch: native source identities missing")
    actual = {}
    for key in ("pretrained_model_name_or_path", "network_weights"):
        path = getattr(args, key, None)
        if path:
            actual[key] = _native_source_identity(path)
    if actual != expected:
        raise ValueError("precision_contract_mismatch: native source contents changed")
    return actual
