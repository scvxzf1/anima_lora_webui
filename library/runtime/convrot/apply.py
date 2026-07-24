"""Apply ConvRot W8A* base forwards onto LoRA-wrapped frozen Linears."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from library.runtime.convrot.checks import (
    is_dora_module,
    raise_if_compiled,
    validate_base_linear_for_convrot,
)
from library.runtime.convrot.free_base import free_base_weights_for_patches
from library.runtime.convrot.linear_w8a16 import w8a16_forward_from_buffers
from library.runtime.convrot.linear_w8a8 import w8a8_forward_from_buffers
from library.runtime.convrot.prequant import (
    PrequantCheckpoint,
    load_prequant_checkpoint,
    resolve_effective_group_size,
)
from library.runtime.convrot.quant import rotate_and_quantize_weight
from library.runtime.convrot.rht import assert_group_divides
from library.runtime.convrot.scope import classify_convrot_linear_module

logger = logging.getLogger(__name__)

ConvRotMode = Literal["w8a16", "w8a8"]

_BUFFER_Q = "_convrot_quantized_weight"
_BUFFER_SCALE = "_convrot_scale"
_ATTR_GROUP = "_convrot_group_size"
_ATTR_MODE = "_convrot_mode"
_ATTR_WEIGHT_SOURCE = "_convrot_weight_source"


@dataclass(frozen=True)
class ConvRotLoRABaseForwardPatch:
    lora_name: str
    name: str
    family: str
    block_idx: int
    shape: tuple[int, ...]
    group_size: int
    mode: str
    bf16_bytes: int
    payload_bytes: int
    weight_source: str = "online_from_bf16"
    skipped_reason: str | None = None


@dataclass(frozen=True)
class ConvRotApplyResult:
    patches: list[ConvRotLoRABaseForwardPatch]
    skipped: list[ConvRotLoRABaseForwardPatch]
    mode: str
    scope: str
    group_size: int
    weight_source: str = "online_from_bf16"
    prequant_path: str | None = None
    freed_modules: int = 0
    freed_bytes: int = 0

    @property
    def patched_count(self) -> int:
        return len(self.patches)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)


def _set_buffer(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    if name in module._buffers:
        module._buffers[name] = tensor
    else:
        module.register_buffer(name, tensor, persistent=False)


def _resolve_base_linear(lora: nn.Module) -> nn.Linear | None:
    refs = getattr(lora, "org_module_ref", None)
    if refs:
        base = refs[0]
        if isinstance(base, nn.Linear):
            return base
        return None
    org_forward = getattr(lora, "org_forward", None)
    owner = getattr(org_forward, "__self__", None)
    if isinstance(owner, nn.Linear):
        return owner
    return None


def _install_org_forward(
    lora: nn.Module,
    *,
    mode: ConvRotMode,
    group_size: int,
) -> None:
    if mode == "w8a16":

        def _base_forward(
            x: torch.Tensor,
            *,
            _lora=lora,
            _gs=group_size,
        ) -> torch.Tensor:
            return w8a16_forward_from_buffers(
                x,
                _lora._convrot_quantized_weight,
                _lora._convrot_scale,
                group_size=_gs,
                hadamard=None,
            )

    else:

        def _base_forward(
            x: torch.Tensor,
            *,
            _lora=lora,
            _gs=group_size,
        ) -> torch.Tensor:
            return w8a8_forward_from_buffers(
                x,
                _lora._convrot_quantized_weight,
                _lora._convrot_scale,
                group_size=_gs,
                hadamard=None,
            )

    lora.org_forward = _base_forward


def _resolve_payload(
    *,
    weight_source: str,
    base_module: nn.Linear,
    original_name: str,
    group_size: int,
    prequant: PrequantCheckpoint | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight_source == "online_from_bf16":
        return rotate_and_quantize_weight(base_module.weight.detach(), group_size)

    assert prequant is not None
    layer = prequant.get(str(original_name))
    if layer is None:
        raise KeyError(
            f"prequant checkpoint missing layer {original_name!r} "
            f"(file has {prequant.layer_count} layers; path={prequant.path})"
        )
    q = layer.quantized_weight
    scale = layer.scale
    if tuple(q.shape) != tuple(base_module.weight.shape):
        raise ValueError(
            f"prequant shape mismatch for {original_name!r}: "
            f"file {tuple(q.shape)} vs live Linear {tuple(base_module.weight.shape)}"
        )
    assert_group_divides(int(q.shape[1]), group_size)
    # Keep buffers on the same device as the live base (usually CUDA).
    device = base_module.weight.device
    if device.type == "meta":
        # Already freed somehow; stay on CPU and let forward move as needed.
        return q.contiguous(), scale.to(torch.float32).contiguous()
    return (
        q.to(device=device).contiguous(),
        scale.to(device=device, dtype=torch.float32).contiguous(),
    )


def apply_convrot_to_lora_network(
    network: nn.Module,
    *,
    mode: ConvRotMode = "w8a16",
    scope: str = "mlp",
    group_size: int = 256,
    weight_source: str = "online_from_bf16",
    prequant_path: str | None = None,
    dry_run: bool = False,
    allow_zero_patches: bool = False,
    free_base_weights: bool = True,
    unet: nn.Module | None = None,
    prequant_group_size_strict: bool = True,
) -> ConvRotApplyResult:
    """Patch LoRA ``org_forward`` with ConvRot W8A* base path.

    Only replaces ``lora.org_forward``; never swaps child ``nn.Linear`` modules.
    When ``free_base_weights=True`` (default), dense bf16 ``Linear.weight``
    storage for patched modules is released after quant so GPU no longer holds
    dual-resident bf16+int8 payloads.

    ``weight_source``:

    * ``online_from_bf16`` — RHT+quant live frozen weights (default).
    * ``prequant_checkpoint`` — load rotated int8 + scale from ``prequant_path``
      (native ``anima_lora_convrot_prequant_v1`` or weight/weight_scale pairs).
    """
    if mode not in {"w8a16", "w8a8"}:
        raise ValueError(f"unsupported convrot mode={mode!r}")
    if weight_source not in {"online_from_bf16", "prequant_checkpoint"}:
        raise ValueError(f"unsupported convrot_weight_source={weight_source!r}")

    prequant: PrequantCheckpoint | None = None
    effective_group = int(group_size)
    if weight_source == "prequant_checkpoint":
        if not prequant_path:
            raise ValueError(
                "convrot_weight_source=prequant_checkpoint requires convrot_prequant_path"
            )
        prequant = load_prequant_checkpoint(prequant_path)
        effective_group = resolve_effective_group_size(
            prequant,
            requested_group_size=int(group_size),
            strict=prequant_group_size_strict,
        )
    elif prequant_path:
        # Explicit path with online source is confusing; refuse.
        raise ValueError(
            "convrot_prequant_path is only valid with "
            "convrot_weight_source=prequant_checkpoint"
        )

    raise_if_compiled(network, context="apply_convrot_to_lora_network")
    if unet is not None:
        raise_if_compiled(unet, context="apply_convrot_to_lora_network(unet)")

    patches: list[ConvRotLoRABaseForwardPatch] = []
    skipped: list[ConvRotLoRABaseForwardPatch] = []
    group_size = int(effective_group)

    for lora in getattr(network, "unet_loras", []) or []:
        original_name = getattr(lora, "original_name", None)
        if not original_name:
            continue
        classified = classify_convrot_linear_module(str(original_name), scope=scope)
        if classified is None:
            continue
        block_idx, family = classified
        lora_name = str(getattr(lora, "lora_name", original_name))

        if is_dora_module(lora):
            skipped.append(
                ConvRotLoRABaseForwardPatch(
                    lora_name=lora_name,
                    name=str(original_name),
                    family=family,
                    block_idx=block_idx,
                    shape=(),
                    group_size=group_size,
                    mode=mode,
                    bf16_bytes=0,
                    payload_bytes=0,
                    weight_source=weight_source,
                    skipped_reason="dora_unsupported",
                )
            )
            continue

        base_module = _resolve_base_linear(lora)
        if base_module is None:
            skipped.append(
                ConvRotLoRABaseForwardPatch(
                    lora_name=lora_name,
                    name=str(original_name),
                    family=family,
                    block_idx=block_idx,
                    shape=(),
                    group_size=group_size,
                    mode=mode,
                    bf16_bytes=0,
                    payload_bytes=0,
                    weight_source=weight_source,
                    skipped_reason="no_org_module_ref",
                )
            )
            continue

        try:
            validate_base_linear_for_convrot(
                base_module, group_size=group_size, name=str(original_name)
            )
        except (TypeError, ValueError) as exc:
            skipped.append(
                ConvRotLoRABaseForwardPatch(
                    lora_name=lora_name,
                    name=str(original_name),
                    family=family,
                    block_idx=block_idx,
                    shape=tuple(int(d) for d in base_module.weight.shape),
                    group_size=group_size,
                    mode=mode,
                    bf16_bytes=int(base_module.weight.numel()) * 2,
                    payload_bytes=0,
                    weight_source=weight_source,
                    skipped_reason=str(exc),
                )
            )
            continue

        weight = base_module.weight.detach()
        bf16_bytes = int(weight.numel()) * 2
        payload_bytes = int(weight.numel()) + int(weight.shape[0]) * 4
        patch = ConvRotLoRABaseForwardPatch(
            lora_name=lora_name,
            name=str(original_name),
            family=family,
            block_idx=block_idx,
            shape=tuple(int(d) for d in weight.shape),
            group_size=group_size,
            mode=mode,
            bf16_bytes=bf16_bytes,
            payload_bytes=payload_bytes,
            weight_source=weight_source,
        )

        if dry_run:
            # For prequant dry_run still verify the layer exists / shape matches.
            if weight_source == "prequant_checkpoint":
                try:
                    _resolve_payload(
                        weight_source=weight_source,
                        base_module=base_module,
                        original_name=str(original_name),
                        group_size=group_size,
                        prequant=prequant,
                    )
                except (KeyError, ValueError) as exc:
                    skipped.append(
                        ConvRotLoRABaseForwardPatch(
                            lora_name=lora_name,
                            name=str(original_name),
                            family=family,
                            block_idx=block_idx,
                            shape=patch.shape,
                            group_size=group_size,
                            mode=mode,
                            bf16_bytes=bf16_bytes,
                            payload_bytes=0,
                            weight_source=weight_source,
                            skipped_reason=f"prequant:{exc}",
                        )
                    )
                    continue
            patches.append(patch)
            continue

        try:
            q, scale = _resolve_payload(
                weight_source=weight_source,
                base_module=base_module,
                original_name=str(original_name),
                group_size=group_size,
                prequant=prequant,
            )
        except (KeyError, ValueError) as exc:
            skipped.append(
                ConvRotLoRABaseForwardPatch(
                    lora_name=lora_name,
                    name=str(original_name),
                    family=family,
                    block_idx=block_idx,
                    shape=patch.shape,
                    group_size=group_size,
                    mode=mode,
                    bf16_bytes=bf16_bytes,
                    payload_bytes=0,
                    weight_source=weight_source,
                    skipped_reason=f"prequant:{exc}",
                )
            )
            continue

        patches.append(patch)
        _set_buffer(lora, _BUFFER_Q, q.contiguous())
        _set_buffer(lora, _BUFFER_SCALE, scale.to(torch.float32).contiguous())
        setattr(lora, _ATTR_GROUP, group_size)
        setattr(lora, _ATTR_MODE, mode)
        setattr(lora, _ATTR_WEIGHT_SOURCE, weight_source)
        _install_org_forward(lora, mode=mode, group_size=group_size)

    freed_modules = 0
    freed_bytes = 0
    if not dry_run and free_base_weights and patches:
        free_stats = free_base_weights_for_patches(network, patches)
        freed_modules = int(free_stats.get("freed_modules", 0))
        freed_bytes = int(free_stats.get("freed_bytes", 0))

    result = ConvRotApplyResult(
        patches=patches,
        skipped=skipped,
        mode=mode,
        scope=scope,
        group_size=group_size,
        weight_source=weight_source,
        prequant_path=str(prequant_path) if prequant_path else None,
        freed_modules=freed_modules,
        freed_bytes=freed_bytes,
    )

    if not dry_run and result.patched_count == 0 and not allow_zero_patches:
        skip_reasons = sorted(
            {item.skipped_reason or "unknown" for item in skipped}
        ) or ["no_matching_scope"]
        raise RuntimeError(
            "apply_convrot_to_lora_network: patched=0 "
            f"(mode={mode} scope={scope} group={group_size} "
            f"weight_source={weight_source}); "
            f"skipped={result.skipped_count} reasons={skip_reasons}. "
            "Phase 1 requires plain LoRA modules with org_module_ref on scope targets."
        )

    logger.info(
        "[convrot] mode=%s scope=%s group=%s source=%s patched=%d skipped=%d "
        "freed_modules=%d freed_mb=%.1f dry_run=%s prequant=%s",
        mode,
        scope,
        group_size,
        weight_source,
        result.patched_count,
        result.skipped_count,
        freed_modules,
        freed_bytes / (1024 * 1024),
        dry_run,
        prequant_path,
    )
    return result
