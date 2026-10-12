"""Compile Qwen Image 2.1 blocks after LoRA has patched their Linear forwards."""

from __future__ import annotations

from collections.abc import Mapping
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
    experiment_args=None,
    network=None,
) -> int:
    """Compile inner block forwards, leaving swap dispatch outside the graph."""
    if backend == "cudagraphs" or mode not in (None, "default"):
        raise ValueError("Qwen Image 2.1 compile supports only non-CUDAGraph default mode")
    if scope not in ("resident", "all"):
        raise ValueError(f"unsupported Qwen Image 2.1 compile scope: {scope}")
    mixed_nf4 = getattr(model, "_mixed_nf4_family", None) == "qwen_image_2_1"
    if mixed_nf4 and backend != "inductor":
        raise ValueError("Qwen Image 2.1 NF4 compile requires the inductor backend")
    # BF16 fusion can remove eager rounding boundaries and amplify LoRA
    # gradient differences. Preserve them for the packed NF4 training path.
    compile_options = {"options": {"emulate_precision_casts": True}} if mixed_nf4 else {}
    if getattr(model, "_qwen_image_2_1_blocks_compiled", False):
        raise RuntimeError("Qwen Image 2.1 blocks are already compiled")

    experiment_values = (dict(experiment_args) if isinstance(experiment_args, Mapping)
                         else vars(experiment_args)) if experiment_args is not None else {}
    experiment = experiment_values.get("qwen_nf4_compile_experiment") is True
    if experiment:
        from .nf4_compile_runtime import prepare_qwen_nf4_compile_experiment
        if backend != "inductor" or dynamic_seq or scope != "all":
            raise ValueError("Qwen NF4 experiment requires static all-scope inductor compile")
        prepare_qwen_nf4_compile_experiment(model, network, experiment_args)
        compile_options = {"options": {"emulate_precision_casts": True, "triton.persistent_reductions": False}}
    # torch.compile rejects even mode="default" together with custom options.
    compile_mode = None if compile_options else mode

    blocks = getattr(model, "transformer_blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError("Qwen Image 2.1 compile requires transformer_blocks ModuleList")
    adapter = getattr(model, "_qwen_image_2_1_block_swap_adapter", None)
    count = len(blocks)
    if adapter is not None and scope == "resident":
        count -= model.blocks_to_swap
    source = adapter.inner_forwards if adapter is not None else [block.forward for block in blocks]
    compiled = []
    for block, forward in zip(blocks[:count], source[:count]):
        segmented = getattr(block, "_qwen_segmented_forward", None)
        if segmented is not None:
            segmented.compile_segments(backend=backend, mode=mode, dynamic=dynamic_seq)
            compiled.append(forward)
        else:
            compiled.append(torch.compile(
                forward, backend=backend, mode=compile_mode, dynamic=dynamic_seq, **compile_options,
            ))
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
