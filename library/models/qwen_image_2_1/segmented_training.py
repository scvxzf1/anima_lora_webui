"""Budgeted projection retention for Qwen Image 2.1 training blocks."""

from __future__ import annotations

from types import MethodType

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from library.models.qwen_image_2_1.packed_qkv import prepare_projected_qkv


class SegmentedBlockForward:
    def __init__(self, block: nn.Module, original_forward, selected_count: int, budget_mib: int):
        self.block = block
        self.original_forward = original_forward
        self.selected_count = selected_count
        self.budget_bytes = budget_mib * 1024 * 1024
        self.project_qkv = block.attn.packed_qkv
        self.project_gate_up = block.img_mlp.gate_up
        self.attention_tail = self._attention_tail
        self.mlp_tail = self._mlp_tail

    def compile_segments(self, *, backend: str, mode: str | None, dynamic: bool) -> None:
        options = dict(backend=backend, mode=mode, dynamic=dynamic)
        self.project_qkv = torch.compile(self.project_qkv, **options)
        self.project_gate_up = torch.compile(self.project_gate_up, **options)
        self.attention_tail = torch.compile(self._attention_tail, **options)
        self.mlp_tail = torch.compile(self._mlp_tail, **options)

    def _attention_tail(self, projected, rotary_emb, attention_mask, segments, key_valid):
        query, key, value, seq_len = prepare_projected_qkv(
            self.block.attn, projected, rotary_emb, None, None, None,
        )
        return self.block.attn.processor.attend_prepared(
            self.block.attn, query, key, value, seq_len,
            attention_mask, segments, key_valid,
        )

    def _mlp_tail(self, gate_up):
        gate, up = gate_up.chunk(2, dim=-1)
        return self.block.img_mlp.out(torch.nn.functional.silu(gate) * up)

    def __call__(
        self, hidden_states, modulation, rotary_emb=None, attention_mask=None,
        target_token_mask=None, layer_cache=None, kv_cache_mode=None,
        cache_write_slice=None, segments=None, key_valid=None,
    ):
        if not torch.is_grad_enabled():
            return self.original_forward(
                hidden_states, modulation, rotary_emb, attention_mask,
                target_token_mask, layer_cache, kv_cache_mode,
                cache_write_slice, segments, key_valid,
            )
        if layer_cache is not None or kv_cache_mode is not None or cache_write_slice is not None:
            raise ValueError("segmented Qwen training does not support KV cache")
        # Only the two raw projection outputs are budgeted. Other saved tensors
        # and allocator/workspace costs require measured headroom.
        tokens = hidden_states.numel() // hidden_states.shape[-1]
        projected_width = (
            self.block.attn.packed_qkv.weight.shape[0]
            + self.block.img_mlp.gate_up.weight.shape[0]
        )
        payload = self.selected_count * tokens * projected_width * hidden_states.element_size()
        if payload > self.budget_bytes:
            raise RuntimeError(
                f"Qwen saved projection payload {payload / 1048576:.2f} MiB exceeds "
                f"qwen_projection_budget_mib={self.budget_bytes / 1048576:g}"
            )
        block = self.block
        mod1, mod2 = modulation.chunk(2, dim=-1)
        img_modulated, gate1 = block._modulate(
            block.img_norm1(hidden_states), mod1, target_token_mask,
        )
        projected = self.project_qkv(img_modulated)
        attn_output = checkpoint(
            self.attention_tail, projected, rotary_emb, attention_mask,
            segments, key_valid, use_reentrant=False,
        )
        hidden_states = hidden_states + gate1.tanh() * attn_output
        img_modulated2, gate2 = block._modulate(
            block.img_norm2(hidden_states), mod2, target_token_mask,
        )
        gate_up = self.project_gate_up(img_modulated2)
        mlp_output = checkpoint(self.mlp_tail, gate_up, use_reentrant=False)
        hidden_states = hidden_states + gate2.tanh() * mlp_output
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states


def install_segmented_training(model: nn.Module, *, count: int, budget_mib: int) -> int:
    """Select the final logical blocks, retaining their QKV and gate/up outputs."""
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock
    from library.models.qwen_image_2_1.fused_mlp import FrozenQwenImage21SwiGLU
    from library.models.qwen_image_2_1.packed_qkv import FrozenPackedQKV

    blocks = getattr(model, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList) or count < 1 or count > len(blocks):
        raise ValueError("qwen_saved_projection_blocks must be within 1..block_count")
    if budget_mib <= 0:
        raise ValueError("qwen_projection_budget_mib must be positive")
    if not getattr(model, "gradient_checkpointing", False):
        raise ValueError("segmented Qwen training requires gradient checkpointing")
    if getattr(model, "_qwen_image_2_1_blocks_compiled", False):
        raise RuntimeError("install segmented training before compile")
    selected = blocks[-count:]
    for block in selected:
        if type(block) is not QwenImage21TransformerBlock:
            raise TypeError("segmented training requires native Qwen Image 2.1 blocks")
        if not isinstance(getattr(block.attn, "packed_qkv", None), FrozenPackedQKV):
            raise TypeError("segmented training requires packed QKV")
        if not isinstance(block.img_mlp, FrozenQwenImage21SwiGLU):
            raise TypeError("segmented training requires fused MLP")
        if not hasattr(block.attn.processor, "attend_prepared"):
            raise TypeError("segmented training requires a prepared-attention processor")

    adapter = getattr(model, "_qwen_image_2_1_block_swap_adapter", None)
    for index in range(len(blocks) - count, len(blocks)):
        block = blocks[index]
        original = adapter.inner_forwards[index] if adapter is not None else block.forward
        segmented = SegmentedBlockForward(block, original, count, budget_mib)
        block._qwen_segmented_forward = segmented
        if adapter is not None:
            adapter.inner_forwards[index] = segmented
        else:
            def forward_segmented(_block, *args, **kwargs):
                return _block._qwen_segmented_forward(*args, **kwargs)
            block.forward = MethodType(forward_segmented, block)

    if adapter is None:
        original_checkpoint = model._gradient_checkpointing_func
        selected_ids = {id(block) for block in selected}

        def checkpoint_selected(function, *args, **kwargs):
            if id(function) in selected_ids:
                return function(*args, **kwargs)
            return original_checkpoint(function, *args, **kwargs)

        model._gradient_checkpointing_func = checkpoint_selected
    return count
