"""Compile Qwen Image 2.1 blocks after LoRA has patched their Linear forwards."""

from __future__ import annotations

import logging

import torch
from torch import nn


logger = logging.getLogger(__name__)


def compile_qwen_image_2_1_blocks(
    model: nn.Module,
    *,
    backend: str = "inductor",
    mode: str | None = None,
    dynamic_seq: bool = True,
    scope: str = "all",
) -> int:
    """Compile inner block forwards, leaving swap dispatch outside the graph."""
    if backend == "cudagraphs" or mode not in (None, "default"):
        raise ValueError("Qwen Image 2.1 compile supports only non-CUDAGraph default mode")
    if scope not in ("resident", "all"):
        raise ValueError(f"unsupported Qwen Image 2.1 compile scope: {scope}")
    if getattr(model, "_qwen_image_2_1_blocks_compiled", False):
        raise RuntimeError("Qwen Image 2.1 blocks are already compiled")

    blocks = getattr(model, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError("Qwen Image 2.1 compile requires transformer_blocks ModuleList")
    adapter = getattr(model, "_qwen_image_2_1_block_swap_adapter", None)
    count = len(blocks)
    if adapter is not None and scope == "resident":
        count -= model.blocks_to_swap
    source = adapter.inner_forwards if adapter is not None else [block.forward for block in blocks]
    compiled = [
        torch.compile(forward, backend=backend, mode=mode, dynamic=dynamic_seq)
        for forward in source[:count]
    ]
    if adapter is not None:
        adapter.inner_forwards[:count] = compiled
    else:
        for block, forward in zip(blocks, compiled):
            block.forward = forward
    model._qwen_image_2_1_blocks_compiled = True
    logger.info(
        "Qwen Image 2.1 compiled %s/%s block forwards (backend=%s, dynamic=%s)",
        count,
        len(blocks),
        backend,
        dynamic_seq,
    )
    return count
