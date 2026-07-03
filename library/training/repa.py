"""Minimal REPA utilities shared by register-token interop tests.

This repository does not yet carry the full upstream REPA training stack, but
LoRA-family register tokens already need the token-grid pooling helper so a
future REPA capture can trim trailing register tokens before patch-grid checks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pool_dit_tokens_to_grid(
    captured: torch.Tensor,
    latent_hw: tuple[int, int],
    patch: int,
    gh: int,
    gw: int,
    trim_tail: int = 0,
) -> torch.Tensor:
    """Captured block output -> ``(B, gh*gw, D)`` fp32 tokens on the target grid.

    Supports both eager ``(B,1,H,W,D)`` and native-flatten ``(B,1,seq,1,D)``
    block outputs by flattening the token axis first. ``trim_tail`` drops
    trailing non-patch tokens such as register tokens before the patch-grid
    validation.
    """

    batch = captured.shape[0]
    dit_dim = captured.shape[-1]
    tokens = captured.reshape(batch, -1, dit_dim)
    if trim_tail > 0:
        tokens = tokens[:, :-trim_tail]

    h_lat, w_lat = latent_hw
    h_dit, w_dit = h_lat // patch, w_lat // patch
    expected = h_dit * w_dit
    if tokens.shape[1] != expected:
        raise RuntimeError(
            f"REPA: captured {tokens.shape[1]} DiT tokens but latent grid is "
            f"{h_dit}x{w_dit}={expected} (patch={patch}). "
            "Block/patch-grid mismatch."
        )

    dit_grid = tokens.reshape(batch, h_dit, w_dit, dit_dim).permute(0, 3, 1, 2)
    dit_pooled = F.adaptive_avg_pool2d(dit_grid.float(), (gh, gw))
    return dit_pooled.flatten(2).transpose(1, 2)
