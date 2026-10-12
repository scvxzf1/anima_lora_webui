"""Offloader-owned CPU parking between completed training and VAE decode.

Callers must release completed autograd graphs before parking. CPU masters and
Parameter identities survive; execution slabs and mixed-representation storage
do not. This API never releases the managed-payload placement protection.
"""

from __future__ import annotations

import torch

from .block_swap_masters import (
    Params4bitBlockSwapCpuMaster,
    _bind_params4bit_master,
    _can_swap_frozen_weight_to_cpu,
    _parked_cpu_master_tensor,
    _restore_cpu_master_tensor,
)
from .device import synchronize_device


def model_swap_blocks(model):
    """Resolve the registered family's block collection without guessing."""
    offloader = model.offloader
    candidates = [
        blocks for name in ("blocks", "layers", "transformer_blocks")
        if (blocks := getattr(model, name, None)) is not None
    ]
    if len(candidates) != 1 or len(candidates[0]) != offloader.num_blocks:
        raise ValueError("Expected one model block collection matching the offloader")
    return candidates[0]


def _drain(offloader):
    errors = []
    # A failed future must not leave other transfers writing into parked data.
    for index in list(offloader.futures):
        try:
            offloader._wait_blocks_move(index, phase="cpu_park")
        except Exception as exc:
            errors.append(exc)
    synchronize_device(offloader.device)
    offloader.flush_profile_events(blocking=True)
    if errors:
        raise RuntimeError("Block-swap transfer failed before CPU parking") from errors[0]


def _release_execution_storage(offloader):
    offloader._mixed_representation_slots.clear()
    offloader._swap_gpu_slab_cache.clear()
    offloader._swap_gpu_scale_slab_cache.clear()
    offloader._slot_assignments.clear()
    offloader._slab_slot_by_block.clear()
    offloader._int8_resident_materialized.clear()


def _park(offloader, blocks):
    offloader._mixed_representation_slots.release_bindings(offloader, blocks)
    for index, block in enumerate(blocks):
        masters = offloader._cpu_weight_masters[index]
        for name, module in offloader._get_block_module_map(index, block).items():
            if not _can_swap_frozen_weight_to_cpu(module):
                continue
            master = masters.get(name)
            if isinstance(master, Params4bitBlockSwapCpuMaster):
                # parked=True intentionally retains GPU quant_state for normal
                # swaps. Full parking must also retire that metadata residency.
                _bind_params4bit_master(module, master, torch.device("cpu"))
            elif master is not None:
                module.weight.data = _parked_cpu_master_tensor(master)
            else:
                # Sparse masters omit the fixed middle. Active tensor carriers
                # there still need explicit placement because Module.to skips them.
                module.weight.data = module.weight.data.to("cpu")
        block.to("cpu")
    synchronize_device(offloader.device)
    _release_execution_storage(offloader)


def _restore(offloader, blocks, state):
    offloader.blocks_to_swap = state["blocks_to_swap"]
    offloader.forward_only = state["forward_only"]
    resident = len(blocks) - offloader.blocks_to_swap
    for index, block in enumerate(blocks):
        masters = offloader._cpu_weight_masters[index]
        dtypes = offloader._cpu_weight_master_dtypes[index]
        for name, master in masters.items():
            module = offloader._get_block_module_map(index, block)[name]
            if index < resident:
                if isinstance(master, Params4bitBlockSwapCpuMaster):
                    _bind_params4bit_master(module, master, offloader.device)
                else:
                    module.weight.data = _restore_cpu_master_tensor(
                        master, device=offloader.device, dtype=dtypes[name],
                    )
        # Resident sparse carriers have no master, and stay protected in .to().
        if index < resident:
            for name, module in offloader._get_block_module_map(index, block).items():
                if name not in masters and _can_swap_frozen_weight_to_cpu(module):
                    module.weight.data = module.weight.data.to(offloader.device)
            offloader._int8_resident_materialized.add(index)
        if not offloader.blocks_to_swap:
            block.to(offloader.device)
    if offloader.blocks_to_swap:
        # Marks payloads active, moves buffers/adapters, prepares mixed slots
        # and validates swap plans using the existing canonical layout contract.
        offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
    synchronize_device(offloader.device)


def _check_blocks(blocks, state):
    if tuple(id(block) for block in blocks) != state["blocks"]:
        raise ValueError("CPU park/restore must use the same block collection")


def park_blocks_on_cpu(offloader, blocks) -> bool:
    """Park once, returning False on repetition; preserve the entry mode/count.

    Parking failures attempt canonical recovery and propagate. A failed recovery
    leaves a retryable parked state and must not be treated as a decode failure.
    """
    state = getattr(offloader, "_cpu_park_state", None)
    if state is not None:
        _check_blocks(blocks, state)
        return False
    _drain(offloader)
    offloader._ensure_cpu_weight_masters(blocks)
    state = {"blocks": tuple(id(block) for block in blocks),
             "blocks_to_swap": offloader.blocks_to_swap,
             "forward_only": offloader.forward_only, "restoring": False}
    offloader._cpu_park_state = state
    try:
        _park(offloader, blocks)
    except Exception as exc:
        try:
            restore_parked_blocks(offloader, blocks)
        except Exception as recovery:
            raise RuntimeError("CPU parking and canonical recovery both failed") from recovery
        raise exc
    return True


def restore_parked_blocks(offloader, blocks) -> bool:
    """Restore once; failed restores re-park, propagate, and allow explicit retry."""
    state = getattr(offloader, "_cpu_park_state", None)
    if state is None:
        return False
    _check_blocks(blocks, state)
    state["restoring"] = True
    try:
        _restore(offloader, blocks, state)
    except Exception as exc:
        try:
            _drain(offloader)
            _park(offloader, blocks)
        except Exception as recovery:
            exc.add_note(f"Re-parking after restore failure also failed: {recovery!r}")
        raise RuntimeError("Block-swap CPU park restoration failed") from exc
    finally:
        state["restoring"] = False
    offloader._cpu_park_state = None
    return True
