"""Experimental int8 storage wrappers for frozen Anima base Linear weights."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

INT8_MAX = 127.0
SCALE_EPS = 1e-12

BLOCK_MODULE_RE = re.compile(r"^(?:net\.)?blocks\.(?P<block>\d+)\.(?P<name>.+)$")

MLP_LINEAR_MODULES = {
    "mlp.layer1",
    "mlp.layer2",
}

ATTENTION_LINEAR_MODULES = {
    "self_attn.qkv_proj",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.kv_proj",
    "self_attn.output_proj",
    "cross_attn.q_proj",
    "cross_attn.k_proj",
    "cross_attn.v_proj",
    "cross_attn.kv_proj",
    "cross_attn.output_proj",
}


@dataclass(frozen=True)
class Int8LinearReplacement:
    name: str
    family: str
    block_idx: int
    shape: tuple[int, ...]
    bf16_bytes: int
    payload_bytes: int


def _canonical_scope(scope: str) -> set[str]:
    normalized = {item.strip().lower() for item in scope.split(",") if item.strip()}
    if not normalized:
        return {"mlp"}
    if "all" in normalized:
        return {"mlp", "attention"}
    aliases = {"mlp": "mlp", "attn": "attention", "attention": "attention"}
    unknown = normalized - set(aliases)
    if unknown:
        raise ValueError(f"unknown int8 linear scope: {', '.join(sorted(unknown))}")
    return {aliases[item] for item in normalized}


def classify_frozen_linear_module(name: str, *, scope: str = "mlp") -> tuple[int, str] | None:
    match = BLOCK_MODULE_RE.match(name)
    if match is None:
        return None
    module_name = match.group("name")
    selected = _canonical_scope(scope)
    if module_name in MLP_LINEAR_MODULES and "mlp" in selected:
        return int(match.group("block")), "mlp"
    if module_name in ATTENTION_LINEAR_MODULES and "attention" in selected:
        return int(match.group("block")), "attention"
    return None


def quantize_weight_per_channel(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.dim() != 2:
        raise ValueError(f"expected a 2D Linear weight, got shape={tuple(weight.shape)}")
    rows = weight.detach().to(torch.float32)
    amax = rows.abs().amax(dim=1)
    scale = (amax / INT8_MAX).clamp_min(SCALE_EPS)
    quantized = (rows / scale[:, None]).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return quantized, scale


class Int8FrozenLinear(nn.Module):
    """Store a frozen Linear weight as int8 + per-output-channel scale."""

    def __init__(self, quantized_weight: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        if quantized_weight.dtype is not torch.int8:
            raise TypeError("quantized_weight must be torch.int8")
        if quantized_weight.dim() != 2:
            raise ValueError("quantized_weight must be 2D")
        if scale.shape != (quantized_weight.shape[0],):
            raise ValueError("scale must have one value per output channel")
        self.register_buffer("quantized_weight", quantized_weight.contiguous())
        self.register_buffer("scale", scale.to(torch.float32).contiguous())

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "Int8FrozenLinear":
        if linear.bias is not None:
            raise ValueError("Int8FrozenLinear only supports bias=False Linear modules")
        if linear.weight.requires_grad:
            raise ValueError("Int8FrozenLinear only supports frozen Linear weights")
        quantized, scale = quantize_weight_per_channel(linear.weight.detach())
        return cls(quantized, scale)

    @property
    def in_features(self) -> int:
        return int(self.quantized_weight.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.quantized_weight.shape[0])

    def dequantized_weight(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        weight = self.quantized_weight.to(torch.float32) * self.scale[:, None]
        return weight.to(dtype) if dtype is not None else weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantized_weight(dtype=x.dtype if x.is_floating_point() else None)
        return F.linear(x, weight, None)


def _named_module_parents(root: nn.Module) -> Iterable[tuple[str, nn.Module, str, nn.Module]]:
    for parent_name, parent in root.named_modules():
        for child_name, child in parent.named_children():
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            yield full_name, parent, child_name, child


def replace_frozen_base_linears_with_int8(
    root: nn.Module,
    *,
    scope: str = "mlp",
    dry_run: bool = False,
) -> list[Int8LinearReplacement]:
    """Replace selected frozen base Linear modules under ``root.blocks``.

    The function deliberately ignores trainable Linears and anything outside
    the Anima block MLP/attention projection paths. It is experimental and is
    intended for small-batch equivalence probes before any training integration.
    """
    replacements: list[Int8LinearReplacement] = []
    for name, parent, child_name, child in list(_named_module_parents(root)):
        if not isinstance(child, nn.Linear):
            continue
        classified = classify_frozen_linear_module(name, scope=scope)
        if classified is None:
            continue
        block_idx, family = classified
        if child.bias is not None or child.weight.requires_grad:
            continue

        bf16_bytes = int(child.weight.numel()) * 2
        payload_bytes = int(child.weight.numel()) + int(child.weight.shape[0]) * 4
        replacements.append(
            Int8LinearReplacement(
                name=name,
                family=family,
                block_idx=block_idx,
                shape=tuple(int(dim) for dim in child.weight.shape),
                bf16_bytes=bf16_bytes,
                payload_bytes=payload_bytes,
            )
        )
        if not dry_run:
            setattr(parent, child_name, Int8FrozenLinear.from_linear(child))
    return replacements
