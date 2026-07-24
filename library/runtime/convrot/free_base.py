"""Helpers to drop dual-resident bf16 base weights after ConvRot quant."""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn


_ATTR_FREED = "_convrot_weight_freed"
_ATTR_FREED_BYTES = "_convrot_weight_freed_bytes"
_ATTR_SHAPE = "_convrot_orig_weight_shape"
_ATTR_DTYPE = "_convrot_orig_weight_dtype"


def is_base_weight_freed(linear: nn.Module) -> bool:
    return bool(getattr(linear, _ATTR_FREED, False))


def free_linear_weight_storage(linear: nn.Linear) -> int:
    """Replace ``linear.weight`` storage with a meta tensor (0 GPU/CPU bytes).

    Keeps ``in_features`` / ``out_features`` and the Parameter slot so module
    structure stays intact. Accidental calls to the original ``nn.Linear.forward``
    will fail loudly on meta tensors — ConvRot always replaces ``org_forward``.

    Returns approximate bytes previously occupied by the dense weight.
    """
    if not isinstance(linear, nn.Linear):
        raise TypeError(f"expected nn.Linear, got {type(linear).__name__}")
    if is_base_weight_freed(linear):
        return 0
    weight = linear.weight
    if weight.device.type == "meta":
        setattr(linear, _ATTR_FREED, True)
        return 0
    nbytes = int(weight.numel()) * int(weight.element_size())
    shape = tuple(int(d) for d in weight.shape)
    dtype = weight.dtype
    meta = torch.empty(shape, dtype=dtype, device="meta")
    linear.weight = nn.Parameter(meta, requires_grad=False)
    setattr(linear, _ATTR_FREED, True)
    setattr(linear, _ATTR_FREED_BYTES, nbytes)
    setattr(linear, _ATTR_SHAPE, shape)
    setattr(linear, _ATTR_DTYPE, dtype)
    return nbytes


def free_base_weights_for_patches(
    network: nn.Module,
    patches: Iterable[Any],
) -> dict[str, int]:
    """Free base Linear weights for successfully patched modules.

    Dedupes by module id so a shared Linear is only freed once.
    """
    lora_by_name: dict[str, nn.Module] = {}
    for lora in getattr(network, "unet_loras", []) or []:
        name = str(getattr(lora, "lora_name", "") or "")
        if name:
            lora_by_name[name] = lora

    freed_modules = 0
    freed_bytes = 0
    seen: set[int] = set()
    for patch in patches:
        lora_name = str(getattr(patch, "lora_name", "") or "")
        lora = lora_by_name.get(lora_name)
        if lora is None:
            continue
        refs = getattr(lora, "org_module_ref", None)
        if not refs:
            continue
        base = refs[0]
        if not isinstance(base, nn.Linear):
            continue
        mid = id(base)
        if mid in seen:
            continue
        seen.add(mid)
        freed_bytes += free_linear_weight_storage(base)
        freed_modules += 1
    return {
        "freed_modules": freed_modules,
        "freed_bytes": freed_bytes,
        "unique_linears": len(seen),
    }
