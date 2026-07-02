"""Precision helpers for Anima inference runtime."""

from __future__ import annotations

from typing import Any

import torch

RUNTIME_DTYPE_CHOICES = ("bf16", "fp16", "fp32")
TEXT_ENCODER_DTYPE_CHOICES = ("same", "bf16", "fp16", "fp32")

_TORCH_DTYPE_BY_NAME = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def normalize_runtime_dtype(value: Any, *, default: str = "bf16") -> str:
    raw = str(value or "").strip().lower()
    normalized = raw or default
    if normalized not in RUNTIME_DTYPE_CHOICES:
        raise ValueError(f"不支持的推理精度: {value}")
    return normalized


def normalize_text_encoder_dtype(value: Any, *, default: str = "same") -> str:
    raw = str(value or "").strip().lower()
    normalized = raw or default
    if normalized not in TEXT_ENCODER_DTYPE_CHOICES:
        raise ValueError(f"不支持的文本编码器精度: {value}")
    return normalized


def runtime_dtype_to_torch(value: Any) -> torch.dtype:
    normalized = normalize_runtime_dtype(value)
    return _TORCH_DTYPE_BY_NAME[normalized]


def text_encoder_dtype_to_torch(value: Any, runtime_dtype: Any) -> torch.dtype:
    normalized = normalize_text_encoder_dtype(value)
    if normalized == "same":
        return runtime_dtype_to_torch(runtime_dtype)
    return _TORCH_DTYPE_BY_NAME[normalized]


def resolve_runtime_dtype(args: Any) -> torch.dtype:
    return runtime_dtype_to_torch(getattr(args, "runtime_dtype", "bf16"))


def resolve_text_encoder_dtype(args: Any) -> torch.dtype:
    return text_encoder_dtype_to_torch(
        getattr(args, "text_encoder_dtype", "same"),
        getattr(args, "runtime_dtype", "bf16"),
    )
