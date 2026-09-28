"""Opt-in frozen, packed SwiGLU for Qwen Image 2.1 training blocks.

Install after checkpoint loading and before block-swap CPU masters or block
compilation. This changes neither the attention-only LoRA target selection nor
the checkpoint loader's on-disk format.
"""

from __future__ import annotations

import torch
from torch import nn


class FrozenQwenImage21SwiGLU(nn.Module):
    """One gate/up projection, with gate rows preceding up rows."""

    def __init__(self, gate_up: nn.Linear, out: nn.Linear) -> None:
        super().__init__()
        self.gate_up = gate_up
        self.out = out

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(hidden_states).chunk(2, dim=-1)
        return self.out(torch.nn.functional.silu(gate) * up)


def _validate_source(mlp: nn.Module, *, freeze: bool) -> tuple[nn.Linear, nn.Linear, nn.Linear]:
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21SwiGLUFeedForward,
    )

    if type(mlp) is not QwenImage21SwiGLUFeedForward:
        raise TypeError("expected an unmodified QwenImage21SwiGLUFeedForward")
    if set(mlp._modules) != {"gate_layer", "proj", "out", "activation_fn"}:
        raise TypeError("Qwen Image 2.1 MLP has unexpected child modules")
    if "forward" in mlp.__dict__ or any(
        getattr(mlp, name) for name in ("_forward_pre_hooks", "_forward_hooks", "_backward_hooks")
    ):
        raise RuntimeError("cannot replace an MLP with patched forward or hooks")
    if type(mlp.activation_fn) is not nn.SiLU:
        raise TypeError("Qwen Image 2.1 MLP activation must be SiLU")
    if any(
        getattr(mlp.activation_fn, name)
        for name in ("_forward_pre_hooks", "_forward_hooks", "_backward_hooks")
    ):
        raise RuntimeError("cannot discard activation hooks")
    gate, up, out = mlp.gate_layer, mlp.proj, mlp.out
    if any(type(linear) is not nn.Linear for linear in (gate, up, out)):
        raise TypeError("Qwen Image 2.1 MLP requires plain Linear projections")
    for linear in (gate, up, out):
        if linear.bias is not None or "forward" in linear.__dict__ or any(
            getattr(linear, name)
            for name in ("_forward_pre_hooks", "_forward_hooks", "_backward_hooks")
        ) or set(linear._modules):
            raise RuntimeError("cannot discard bias, hooks, or an attached MLP adapter")
        if linear.weight.is_meta:
            raise RuntimeError("materialize MLP checkpoint weights before fusion")
    if gate.in_features != up.in_features or gate.out_features != up.out_features:
        raise ValueError("gate and up projection shapes differ")
    if out.in_features != gate.out_features or out.out_features != gate.in_features:
        raise ValueError("MLP output projection has incompatible shape")
    if len({linear.weight.device for linear in (gate, up, out)}) != 1 or len(
        {linear.weight.dtype for linear in (gate, up, out)}
    ) != 1:
        raise ValueError("MLP projection device and dtype must match")
    if not freeze and any(linear.weight.requires_grad for linear in (gate, up, out)):
        raise RuntimeError("MLP weights are trainable; pass freeze=True only for frozen-base training")
    return gate, up, out


def fuse_qwen_image_2_1_mlp(
    mlp: nn.Module, *, freeze: bool = False
) -> FrozenQwenImage21SwiGLU:
    """Replace split weights once; explicit ``freeze`` opts into base freezing."""
    gate, up, out = _validate_source(mlp, freeze=freeze)
    packed = torch.empty(
        (2 * gate.out_features, gate.in_features),
        device=gate.weight.device,
        dtype=gate.weight.dtype,
    )
    with torch.no_grad():
        packed[: gate.out_features].copy_(gate.weight)
        packed[gate.out_features :].copy_(up.weight)
    with torch.device("meta"):
        gate_up = nn.Linear(gate.in_features, 2 * gate.out_features, bias=False)
    gate_up.weight = nn.Parameter(packed, requires_grad=False)
    if freeze:
        out.requires_grad_(False)
    fused = FrozenQwenImage21SwiGLU(gate_up, out)
    fused.training = mlp.training
    gate_up.training = gate.training
    return fused


def install_qwen_image_2_1_fused_mlps(model: nn.Module, *, freeze: bool = False) -> int:
    """Fuse every block atomically after preflight, before swap masters/compile."""
    blocks = getattr(model, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise TypeError("Qwen Image 2.1 fusion requires transformer_blocks ModuleList")
    if getattr(model, "_qwen_image_2_1_blocks_compiled", False) or any(
        hasattr(block, "_orig_mod") for block in blocks
    ):
        raise RuntimeError("install fused MLPs before block compilation")
    offloader = getattr(model, "offloader", None)
    if offloader is not None:
        if not hasattr(offloader, "_cpu_weight_masters") or offloader._cpu_weight_masters is not None:
            raise RuntimeError("install fused MLPs before block-swap CPU masters")
        if any(module_map is not None for module_map in getattr(offloader, "_block_module_maps", ())):
            raise RuntimeError("install fused MLPs before block-swap module maps")
    if getattr(model, "blocks_to_swap", 0) and offloader is None:
        raise RuntimeError("block swap state is not inspectable")
    for block in blocks:
        _validate_source(getattr(block, "img_mlp", None), freeze=freeze)
    for block in blocks:
        block.img_mlp = fuse_qwen_image_2_1_mlp(block.img_mlp, freeze=freeze)
    return len(blocks)
