"""Low-memory runtime for Qwen Image 2.1's frozen Qwen3-VL encoder.

The ComfyUI text encoder checkpoint is much larger than the 10 GB training
GPU used by the WebUI.  This module keeps the strict project loader as the
single source of model construction, then optionally installs Accelerate's
CPU-offload hooks on ``model.model`` (the vision and language core).  The
language-model cache path never needs ``lm_head`` or all intermediate hidden
states, so those remain on CPU and are not involved in the cache forward.
"""

from __future__ import annotations

import contextlib
import gc
from collections.abc import Iterator

import torch

from .cache_policy import resolve_cache_policy


def _parameter_bytes(module: torch.nn.Module) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in module.parameters())


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Qwen Image 2.1 text encoder requested CUDA but CUDA is unavailable")
    return device


def _should_offload(module: torch.nn.Module, device: torch.device, mode: str) -> bool:
    if mode not in {"auto", "on", "off"}:
        raise ValueError(f"Qwen text encoder offload must be auto, on, or off; got {mode!r}")
    if device.type != "cuda":
        if mode == "on":
            raise ValueError("Qwen text encoder CPU offload requires a CUDA execution device")
        return False
    if mode == "on":
        return True
    if mode == "off":
        return False
    # Leave a substantial activation/workspace reserve.  A Qwen3-VL 8B
    # encoder is ~17 GB in BF16, so this selects offload on 10/12/16 GB cards
    # while allowing a sufficiently large card to use eager CUDA placement.
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    except RuntimeError:
        return True
    return _parameter_bytes(module) > int(free_bytes * 0.60)


@contextlib.contextmanager
def text_encoder_for_cache(
    path: str,
    *,
    dtype: torch.dtype,
    device: str = "auto",
    offload: str = "auto",
    cache_policy: str = "auto",
) -> Iterator[torch.nn.Module]:
    """Load a frozen Qwen3-VL encoder for cache generation.

    The yielded object is the normal ``Qwen3VLForConditionalGeneration``
    instance, so existing caching strategies can keep using ``model.model``
    and ``model.parameters()``.  In offload mode its parameters are meta by
    design; ``strategy.py`` reads ``_qwen_execution_device`` instead of
    inspecting those parameters.
    """
    from library.models.qwen_image_2_1.weights import load_qwen_image_2_1_text_encoder

    # Reject policy/device conflicts before loading a multi-GB checkpoint.
    device, offload = resolve_cache_policy(cache_policy, device=device, offload=offload)
    _resolve_device(device)
    model = load_qwen_image_2_1_text_encoder(path, dtype=dtype, device="cpu")
    with loaded_text_encoder_for_cache(
        model, dtype=dtype, device=device, offload=offload, cache_policy=cache_policy
    ) as prepared:
        yield prepared


@contextlib.contextmanager
def loaded_text_encoder_for_cache(
    model: torch.nn.Module,
    *,
    dtype: torch.dtype,
    device: str = "auto",
    offload: str = "auto",
    cache_policy: str = "auto",
) -> Iterator[torch.nn.Module]:
    """Prepare an already-loaded Qwen3-VL encoder for cache generation."""
    device, offload = resolve_cache_policy(cache_policy, device=device, offload=offload)
    execution_device = _resolve_device(device)
    core = model.model
    installed_offload = False
    try:
        if _should_offload(core, execution_device, offload):
            from accelerate import cpu_offload

            installed_offload = True
            cpu_offload(core, execution_device=execution_device, offload_buffers=False)
        else:
            # Only this core participates in caching; keep unused lm_head on CPU.
            core.to(device=execution_device, dtype=dtype)
        model._qwen_execution_device = execution_device
        model._qwen_cache_offloaded = installed_offload
        print(
            f"Qwen3-VL cache policy: requested={cache_policy}, "
            f"execution_device={execution_device}, cpu_offload={str(installed_offload).lower()}, "
            f"core_parameters_gib={_parameter_bytes(core) / 1024**3:.2f}",
            flush=True,
        )
        yield model
    finally:
        if installed_offload:
            from accelerate.hooks import remove_hook_from_module

            # Detaching recursively restores the CPU parameter placement held
            # by Accelerate.  Calling model.to('cpu') while hooks are active
            # would attempt to materialize meta tensors and can OOM.
            remove_hook_from_module(core, recurse=True)
        model.to("cpu")
        if hasattr(model, "_qwen_execution_device"):
            delattr(model, "_qwen_execution_device")
        if hasattr(model, "_qwen_cache_offloaded"):
            delattr(model, "_qwen_cache_offloaded")
        del model
        gc.collect()
        if execution_device.type == "cuda":
            torch.cuda.empty_cache()
