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
from collections.abc import Callable, Iterator

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


def _cleanup_text_encoder(model: torch.nn.Module, installed_offload: bool) -> list[BaseException]:
    """Attempt every cleanup step, including after a partial hook installation."""
    errors: list[BaseException] = []

    def attempt(action: Callable[[], object]) -> None:
        try:
            action()
        except BaseException as error:
            errors.append(error)

    def remove_hooks() -> None:
        from accelerate.hooks import remove_hook_from_module

        core = model.model
        attempt(lambda: remove_hook_from_module(core, recurse=True))
        if errors:
            # A failed detach must not prevent cleanup of the remaining children.
            for module in core.modules():
                attempt(lambda: remove_hook_from_module(module, recurse=False))

                def discard_hook() -> None:
                    if hasattr(module, "_old_forward"):
                        module.forward = module._old_forward
                        delattr(module, "_old_forward")
                    module.__dict__.pop("_hf_hook", None)

                attempt(discard_hook)

    if installed_offload:
        attempt(remove_hooks)

    def move_to_cpu() -> None:
        # init_hook can replace parameters before _hf_hook is attached. There
        # is no saved weight on the module to restore in that case; this failed
        # encoder must be discarded rather than materialized with to_empty.
        if any(tensor.is_meta for tensors in (model.parameters(), model.buffers()) for tensor in tensors):
            raise RuntimeError("Qwen text encoder cleanup left meta tensors; encoder is unusable and must be discarded")
        model.to("cpu")

    attempt(move_to_cpu)
    for attribute in ("_qwen_execution_device", "_qwen_cache_offloaded"):
        attempt(lambda: model.__dict__.pop(attribute, None))
    return errors


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
    stage_error: BaseException | None = None
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
    except BaseException as error:
        stage_error = error
        raise
    finally:
        cleanup_errors = _cleanup_text_encoder(model, installed_offload)
        del model, core
        try:
            gc.collect()
        except BaseException as error:
            cleanup_errors.append(error)
        if execution_device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except BaseException as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            primary = stage_error if stage_error is not None else cleanup_errors[0]
            for error in cleanup_errors:
                if error is not primary:
                    primary.add_note(f"Qwen text encoder cleanup failed: {type(error).__name__}: {error}")
            if stage_error is None:
                raise primary
