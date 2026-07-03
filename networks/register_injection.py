"""Shared mid-stack register-token injection for the Anima DiT."""

from __future__ import annotations

from typing import Callable, Optional

import torch


class RegisterInjector:
    """Wrap ``anima._run_blocks`` to inject register tokens mid-stack."""

    def __init__(
        self,
        *,
        num_registers: int,
        insert_block: int,
        get_scaled_tokens: Callable[[], torch.Tensor],
    ) -> None:
        self.K = int(num_registers)
        self.insert_block = int(insert_block)
        self._get_tokens = get_scaled_tokens
        self._applied = False
        self._anima = None
        self._orig_run_blocks = None
        self._orig_native_flatten = None
        self._hook_handles: list = []
        self._inject = None
        self.last_reg_ratio: Optional[float] = None
        self.last_patch_sink_ratio: Optional[float] = None

    @property
    def extra_seq_tokens(self) -> int:
        return self.K

    def apply(self, anima) -> None:
        if self._applied:
            return
        n_blocks = len(anima.blocks)
        if not (0 <= self.insert_block < n_blocks):
            raise ValueError(
                f"insert_block must be in [0, {n_blocks}), got {self.insert_block}"
            )

        self._anima = anima
        self._orig_native_flatten = anima._native_flatten
        anima._native_flatten = True

        injector = self
        self._orig_run_blocks = anima._run_blocks

        def wrapped_run_blocks(
            x_padded,
            t_embedding_B_T_D,
            crossattn_emb,
            attn_params,
            **block_kwargs,
        ):
            batch = x_padded.shape[0]
            seq = x_padded.shape[2]
            x_ext = x_padded
            if injector.K > 0:
                reg = injector._get_tokens().to(
                    dtype=x_padded.dtype, device=x_padded.device
                )
                embed_dim = reg.shape[-1]
                reg = reg.view(1, 1, injector.K, 1, embed_dim).expand(
                    batch, -1, -1, -1, -1
                )
                rope_ext = None
                rope = block_kwargs.get("rope_cos_sin")
                if rope is not None:
                    cos, sin = rope
                    pad_shape = (injector.K,) + tuple(cos.shape[1:])
                    rope_ext = (
                        torch.cat([cos, cos.new_ones(pad_shape)], dim=0),
                        torch.cat([sin, sin.new_zeros(pad_shape)], dim=0),
                    )
                if injector.insert_block == 0:
                    x_ext = torch.cat([x_padded, reg], dim=2)
                    if rope_ext is not None:
                        block_kwargs = {**block_kwargs, "rope_cos_sin": rope_ext}
                else:
                    injector._inject = (reg, rope_ext)

            try:
                out = injector._orig_run_blocks(
                    x_ext,
                    t_embedding_B_T_D,
                    crossattn_emb,
                    attn_params,
                    **block_kwargs,
                )
            finally:
                injector._inject = None

            with torch.no_grad():
                patch_tokens = out[:, :, :seq, :, :].float().norm(dim=-1).flatten()
                med = patch_tokens.median().clamp_min(1e-6)
                topk = max(1, int(0.002 * patch_tokens.numel()))
                injector.last_patch_sink_ratio = (
                    patch_tokens.topk(topk).values.mean() / med
                ).item()
                if injector.K > 0 and out.shape[2] > seq:
                    reg_tokens = out[:, :, seq:, :, :].float().norm(dim=-1).flatten()
                    injector.last_reg_ratio = (reg_tokens.max() / med).item()

            return out[:, :, :seq, :, :]

        anima._run_blocks = wrapped_run_blocks

        if self.K > 0 and self.insert_block > 0:
            insert_at = self.insert_block

            def make_pre_hook(block_idx):
                def pre_hook(module, args, kwargs):
                    pending = injector._inject
                    if pending is None:
                        return None
                    reg, rope_ext = pending
                    x = args[0]
                    if block_idx == insert_at:
                        x = torch.cat([x, reg], dim=2)
                    if rope_ext is not None and kwargs.get("rope_cos_sin") is not None:
                        kwargs = {**kwargs, "rope_cos_sin": rope_ext}
                    return (x,) + tuple(args[1:]), kwargs

                return pre_hook

            for block_idx in range(insert_at, n_blocks):
                self._hook_handles.append(
                    anima.blocks[block_idx].register_forward_pre_hook(
                        make_pre_hook(block_idx), with_kwargs=True
                    )
                )

        self._applied = True

    def remove(self) -> None:
        if not self._applied:
            return
        anima = self._anima
        anima._run_blocks = self._orig_run_blocks
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []
        anima._native_flatten = self._orig_native_flatten
        self._anima = None
        self._applied = False
