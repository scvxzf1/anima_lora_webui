"""Reconfigure frozen-weight residency only at a completed optimizer boundary."""

from __future__ import annotations

import torch

from .block_swap_masters import Params4bitBlockSwapCpuMaster, _bind_params4bit_master
from .device import synchronize_device
from .block_swap_sparse_masters import capture_missing, trim_masters


def _place_frozen_weights(offloader, blocks, count):
    resident = len(blocks) - count
    device = offloader.device
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    # Release the departing tail before allocating any promoted weights.
    for index in (*range(resident, len(blocks)), *range(resident)):
        target = device if index < resident else torch.device("cpu")
        modules = offloader._get_block_module_map(index, blocks[index])
        for name, master in offloader._cpu_weight_masters[index].items():
            module = modules[name]
            weight = module.weight
            if weight.requires_grad:
                raise RuntimeError("Dynamic swap cannot migrate trainable weights")
            if weight.device == target:
                continue
            if isinstance(master, Params4bitBlockSwapCpuMaster):
                _bind_params4bit_master(module, master, target, parked=True)
            else:
                weight.data = master.to(device=target, non_blocking=True)


def reconfigure_block_swap(offloader, blocks, count: int) -> bool:
    """Keep adapters/optimizer identities intact; recover migration-only OOM.

    No forward graph or accumulated gradients may be live. A training-step OOM
    is not handled here: only allocations made by this boundary transaction.
    """
    if not isinstance(count, int) or not 0 <= count <= len(blocks) - 2:
        raise ValueError("Dynamic swap count must be in 0..len(blocks)-2")
    if offloader.forward_only or not offloader.supports_backward:
        raise ValueError("Dynamic swap requires the training offloader")
    if offloader.transfer_dtype != "bf16":
        raise ValueError("Dynamic swap requires lossless BF16 transfers")
    if any(
        p.grad is not None
        for block in blocks
        for p in block.parameters()
        if p.requires_grad
    ):
        raise RuntimeError("Dynamic swap requires cleared optimizer gradients")
    previous = offloader.blocks_to_swap
    if count == previous:
        return False
    for index in list(offloader.futures):
        offloader._wait_blocks_move(index, phase="reconfigure")
    synchronize_device(offloader.device)
    offloader.flush_profile_events(blocking=True)
    offloader._ensure_cpu_weight_masters(blocks)
    offloader._swap_gpu_slab_cache.clear()
    offloader._slot_assignments.clear()
    try:
        capture_missing(offloader, blocks, count)
        _place_frozen_weights(offloader, blocks, count)
    except torch.cuda.OutOfMemoryError:
        try:
            _place_frozen_weights(offloader, blocks, previous)
            synchronize_device(offloader.device)
            trim_masters(offloader, blocks, previous)
        except Exception as rollback_error:
            raise RuntimeError(
                "Dynamic swap migration rollback failed"
            ) from rollback_error
        if offloader.cuda_available:
            torch.cuda.empty_cache()
        raise
    synchronize_device(offloader.device)
    for handle in offloader.remove_handles:
        handle.remove()
    offloader.remove_handles = []
    offloader.blocks_to_swap = count
    trim_masters(offloader, blocks, count)
    offloader._swap_plan_cache.clear()
    offloader._swap_slab_plan_cache.clear()
    offloader._reset_slab_slot_ownership()
    offloader._warm_swap_plan_cache(blocks)
    for index, block in enumerate(blocks):
        hook = offloader.create_backward_hook(blocks, index)
        if hook is not None:
            offloader.remove_handles.append(block.register_full_backward_hook(hook))
    if count > previous and offloader.cuda_available:
        torch.cuda.empty_cache()
    return True
