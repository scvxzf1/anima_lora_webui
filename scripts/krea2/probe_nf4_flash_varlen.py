#!/usr/bin/env python3
"""Run the unified NF4 probe with experimental FlashAttention varlen GQA.

This is a research wrapper, not the production attention backend. It packs the
valid part of Krea-2's dense padding mask before calling FlashAttention 2 and
then re-expands the result so the surrounding DiT contract stays unchanged.

Example::

    K2_ABL_GPU=1 K2_ABL_IMG=1024 K2_ABL_STEPS=20 \
      K2_ABL_NF4=1 K2_ABL_SWAP=0 K2_ABL_GRAD_CKPT=full \
      K2_ABL_COMPILE=1 K2_ABL_TE_CPU=1 \
      .venv/bin/python scripts/krea2/probe_nf4_flash_varlen.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("K2_ABL_GPU", "1"))

import torch
from einops import rearrange
from flash_attn import flash_attn_varlen_func

torch._dynamo.config.capture_dynamic_output_shape_ops = True

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import library.models.krea2_raw.dit as krea_dit  # noqa: E402


def attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    scale: float | None = None,
    gqa: bool = False,
) -> torch.Tensor:
    del gqa
    batch, _heads, sequence, _dim = q.shape
    if mask is None:
        valid = torch.ones(batch, sequence, device=q.device, dtype=torch.bool)
    else:
        valid = mask[:, 0].diagonal(dim1=-2, dim2=-1).to(torch.bool)

    lengths = valid.sum(dim=1, dtype=torch.int32)
    cu_seqlens = torch.zeros(batch + 1, device=q.device, dtype=torch.int32)
    cu_seqlens[1:] = lengths.cumsum(dim=0)
    q_packed = rearrange(q, "b h l d -> b l h d")[valid].contiguous()
    k_packed = rearrange(k, "b h l d -> b l h d")[valid].contiguous()
    v_packed = rearrange(v, "b h l d -> b l h d")[valid].contiguous()
    output_packed = flash_attn_varlen_func(
        q_packed,
        k_packed,
        v_packed,
        cu_seqlens,
        cu_seqlens,
        sequence,
        sequence,
        softmax_scale=scale,
    )
    output = torch.zeros(
        batch, sequence, q.shape[1], q.shape[-1],
        device=q.device, dtype=q.dtype,
    )
    output[valid] = output_packed
    return rearrange(output, "b l h d -> b l (h d)")


krea_dit.attention = attention_varlen

from probe_nf4_ablation import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
