"""Z-Image attention backend selection for Diffusers transformers."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from enum import Enum

import torch


Z_IMAGE_ATTENTION_MODES = frozenset({"torch", "flash"})


def normalize_z_image_attention_mode(value: object) -> str:
    mode = str(value or "flash").strip().lower().replace("-", "_")
    if mode in {"sdpa", "native"}:
        return "torch"
    if mode not in Z_IMAGE_ATTENTION_MODES:
        raise ValueError(
            "Z-Image attn_mode supports only 'torch' (native SDPA) and "
            f"'flash' (Diffusers FlashAttention varlen); got {value!r}"
        )
    return mode


def _load_diffusers_attention_api() -> tuple[
    type[Enum],
    Callable[[object], AbstractContextManager[None]],
    type[object],
]:
    try:
        from diffusers.models import AttentionBackendName, attention_backend
        from diffusers.models.transformers.transformer_z_image import (
            ZSingleStreamAttnProcessor,
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "Z-Image attn_mode='flash' requires a Diffusers build with the "
            "flash_varlen attention backend"
        ) from exc
    return AttentionBackendName, attention_backend, ZSingleStreamAttnProcessor


def prepare_z_image_attention(
    model: object,
    mode: object,
    *,
    dtype: torch.dtype | None,
) -> str:
    """Install the requested backend without changing Diffusers global state.

    Z-Image supplies a padding mask, so its public ``flash`` mode maps to
    Diffusers ``flash_varlen`` rather than the fixed-length ``flash`` backend.
    """

    normalized = normalize_z_image_attention_mode(mode)
    if normalized == "flash" and dtype != torch.bfloat16:
        raise RuntimeError(
            f"Z-Image attn_mode='flash' is validated only for bf16 compute; got {dtype}"
        )

    backend_names, backend_context, processor_type = _load_diffusers_attention_api()
    flash_backend = backend_names.FLASH_VARLEN
    if normalized == "flash":
        try:
            # The public context performs Diffusers' dependency/kernel checks
            # and restores the previous global backend on exit.
            with backend_context(flash_backend):
                pass
        except (ImportError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "Z-Image attn_mode='flash' requires the Diffusers "
                "flash_varlen backend and a compatible FlashAttention 2 provider"
            ) from exc

    configured = 0
    modules = getattr(model, "modules", None)
    if not callable(modules):
        raise TypeError("Z-Image transformer does not expose modules()")
    for module in modules():
        processor = getattr(module, "processor", None)
        if not isinstance(processor, processor_type):
            continue
        if normalized == "flash":
            setter = getattr(module, "set_attention_backend", None)
            if callable(setter):
                setter(flash_backend.value)
            else:
                processor._attention_backend = flash_backend
        else:
            processor._attention_backend = None
        configured += 1

    if configured == 0:
        raise RuntimeError(
            "Z-Image transformer has no ZSingleStreamAttnProcessor instances; "
            "cannot configure its attention backend"
        )
    return normalized


__all__ = [
    "Z_IMAGE_ATTENTION_MODES",
    "normalize_z_image_attention_mode",
    "prepare_z_image_attention",
]
