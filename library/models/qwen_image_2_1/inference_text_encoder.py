"""Sequential standalone prompt encoding using the shared low-memory runtime."""

from __future__ import annotations

import gc
import sys
from collections.abc import Callable
from typing import Any, TypeVar

import torch

from library.inference.precision import resolve_text_encoder_dtype
from library.models.qwen_image_2_1.text_encoder_runtime import loaded_text_encoder_for_cache
from library.runtime.device import clean_memory_on_device

_Conditions = TypeVar("_Conditions")


def _release_encoding_stage(device: torch.device, stage_error: BaseException | None) -> None:
    primary = stage_error
    try:
        for cleanup in (gc.collect, lambda: clean_memory_on_device(device)):
            try:
                cleanup()
            except BaseException as error:
                if primary is None:
                    primary = error
                elif error is not primary:
                    primary.add_note(f"Qwen inference encoder cleanup failed: {type(error).__name__}: {error}")
        if stage_error is None and primary is not None:
            raise primary
    finally:
        # Exception tracebacks retain this frame. Avoid a frame -> exception ->
        # traceback cycle after allocator cleanup fails, without clearing the
        # caller's original traceback.
        del primary, stage_error


def encode_with_text_encoder(
    args: Any,
    device: torch.device,
    encode: Callable[[torch.nn.Module], _Conditions],
    *,
    loader: Callable[..., torch.nn.Module],
) -> _Conditions:
    """Keep the encoder inside this stage, including hook teardown on errors."""
    dtype = resolve_text_encoder_dtype(args)
    execution_device = torch.device("cpu") if getattr(args, "text_encoder_cpu", False) else device
    # Strict CPU loading avoids placing the full 8B encoder on a small GPU.
    # The shared runtime moves/offloads only its core, leaving lm_head on CPU.
    encoder = loader(args.text_encoder, dtype=dtype, device="cpu")
    try:
        with loaded_text_encoder_for_cache(
            encoder,
            dtype=dtype,
            device=str(execution_device),
            offload="auto" if execution_device.type == "cuda" else "off",
            cache_policy="auto",
        ):
            return encode(encoder)
    finally:
        del encoder
        _release_encoding_stage(device, sys.exception())
