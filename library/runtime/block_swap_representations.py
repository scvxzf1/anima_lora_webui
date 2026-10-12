"""Stable byte storage for slots shared by dense and packed NF4 weights.

bnb saves a transposed packed tensor outside ``save_for_backward``. Its storage
must be restored in place before backward, including non-reentrant checkpoint.
"""
from __future__ import annotations

import math
import torch

from .block_swap_masters import (
    Params4bitBlockSwapCpuMaster, _bind_params4bit_master,
    _parked_cpu_master_tensor,
)
from .block_swap_sparse_masters import participating_blocks, uses_sparse_masters


def logical_shape(master, module):
    if isinstance(master, Params4bitBlockSwapCpuMaster):
        packed = master.params4bit.data
        state = master.params4bit.quant_state
        shape = tuple(getattr(state, "shape", ()) or ())
        if (len(shape) != 2 or any(not isinstance(x, int) or x <= 0 for x in shape)
                or getattr(state, "quant_type", None) != "nf4"
                or packed.dtype != torch.uint8 or not packed.is_contiguous()
                or packed.numel() * 2 != math.prod(shape)
                or getattr(state, "absmax", None) is None
                or getattr(state, "code", None) is None):
            raise ValueError("Malformed NF4 block-swap representation")
    elif isinstance(master, torch.Tensor):
        shape = tuple(master.shape)
    else:
        raise ValueError("Mixed NF4 slots require native dense CPU masters")
    if shape != (getattr(module, "out_features", None), getattr(module, "in_features", None)):
        raise ValueError("Block-swap logical shape does not match Linear dimensions")
    return shape


def compatible_swap_shapes(source, target, source_module, target_module):
    """Validate logical matrices before accepting packed/dense physical shapes."""
    if not (isinstance(source, Params4bitBlockSwapCpuMaster)
            or isinstance(target, Params4bitBlockSwapCpuMaster)):
        return source_module.weight.shape == target_module.weight.shape
    if logical_shape(source, source_module) != logical_shape(target, target_module):
        raise ValueError("Block-swap NF4/dense logical shape mismatch")
    return True


class MixedRepresentationSlots:
    def __init__(self):
        self.buffers = {}
        self.owners = {}
        self.members = set()

    def clear(self):
        self.buffers.clear()
        self.owners.clear()
        self.members.clear()

    def release_bindings(self, offloader, blocks):
        # A sampling wrap can leave an ordinary resident weight on a mixed
        # buffer. Detach every owner before rebinding/reducing the slot pool.
        for index, name in self.owners:
            module = offloader._get_block_module_map(index, blocks[index])[name]
            module.weight.data = _parked_cpu_master_tensor(offloader._cpu_weight_masters[index][name])
        self.owners = {}

    def prepare(self, offloader, blocks):
        """Allocate maximum capacities before any forward creates saved aliases."""
        previous_members = frozenset(self.members)
        if not offloader.blocks_to_swap:
            self.clear()
            return bool(previous_members)
        resident = len(blocks) - offloader.blocks_to_swap
        masters = offloader._cpu_weight_masters
        dtypes = offloader._cpu_weight_master_dtypes
        active = (participating_blocks(len(blocks), offloader.blocks_to_swap)
                  if uses_sparse_masters(offloader) else set(range(len(blocks))))
        if offloader.forward_only:
            # Reuse the training pool instead of allocating one dense-capacity
            # buffer for every inference slot. bnb's completed backward ctx can
            # still own a packed view while its loss object remains in scope;
            # dropping/reallocating the pool would transiently retain both.
            names = {name for _, name in self.buffers}
            self.members = {(index, name) for index in active for name in names
                            if name in masters[index]}
            for (slot, name), buffer in self.buffers.items():
                module = offloader._get_block_module_map(slot, blocks[slot])[name]
                self.bind(module, masters[slot][name], dtypes[slot][name], buffer, offloader.device)
                self.owners[slot, name] = buffer
            return self.members != previous_members
        specs = {}
        members = set()
        for slot in range(resident):
            indices = (range(len(blocks)) if offloader.forward_only
                       else range(slot, len(blocks), resident))
            by_name = {}
            for index in indices:
                if index not in active:
                    continue
                for name, master in masters[index].items():
                    by_name.setdefault(name, []).append((index, master))
            for name, entries in by_name.items():
                kinds = {isinstance(master, Params4bitBlockSwapCpuMaster) for _, master in entries}
                if len(kinds) < 2:
                    continue
                shapes, capacity = set(), 0
                for index, master in entries:
                    module = offloader._get_block_module_map(index, blocks[index])[name]
                    shapes.add(logical_shape(master, module))
                    physical = _parked_cpu_master_tensor(master)
                    dtype = dtypes[index][name]
                    capacity = max(capacity, physical.numel() * torch.empty((), dtype=dtype).element_size())
                    members.add((index, name))
                if len(shapes) != 1:
                    raise ValueError("Mixed NF4 slot has incompatible logical shapes")
                specs[slot, name] = capacity
        self.buffers = {key: self.buffers[key] if key in self.buffers
                        and self.buffers[key].numel() == size else
                        torch.empty(size, dtype=torch.uint8, device=offloader.device)
                        for key, size in specs.items()}
        self.owners = {}
        self.members = members
        for (slot, name), buffer in self.buffers.items():
            module = offloader._get_block_module_map(slot, blocks[slot])[name]
            self.bind(module, masters[slot][name], dtypes[slot][name], buffer, offloader.device)
            self.owners[slot, name] = buffer
        return self.members != previous_members

    @staticmethod
    def bind(module, master, dtype, buffer, device):
        physical = _parked_cpu_master_tensor(master)
        size = physical.numel() * torch.empty((), dtype=dtype).element_size()
        view = buffer[:size].view(dtype).view(physical.shape)
        if isinstance(master, Params4bitBlockSwapCpuMaster):
            _bind_params4bit_master(module, master, device, storage=view)
        else:
            view.copy_(master, non_blocking=True)
            module.weight.data = view

    def swap(self, offloader, source_index, target_index, source_module, target_module,
             entry, stream=None):
        name, source_master, target_master, _, target_dtype = entry
        buffer = self.owners.pop((source_index, name), None)
        if buffer is None:
            raise RuntimeError("Mixed NF4 storage must be prepared before forward")
        if stream is not None:
            buffer.record_stream(stream)
        source_module.weight.data = _parked_cpu_master_tensor(source_master)
        self.bind(target_module, target_master, target_dtype, buffer, offloader.device)
        self.owners[target_index, name] = buffer


def swap_representation_jobs(offloader, source_index, target_index, source_modules,
                             target_modules, entries, stream=None):
    slots = offloader._mixed_representation_slots
    remaining = []
    for entry in entries:
        name = entry[0]
        source_nf4 = isinstance(entry[1], Params4bitBlockSwapCpuMaster)
        target_nf4 = isinstance(entry[2], Params4bitBlockSwapCpuMaster)
        if offloader.forward_only and (source_index, name) in slots.owners:
            slots.swap(offloader, source_index, target_index, source_modules[name],
                       target_modules[name], entry, stream)
        elif offloader.forward_only and source_nf4 != target_nf4:
            # Inference has no saved backward aliases. Retire the outgoing
            # payload before allocating the incoming representation, without
            # retaining a dense-capacity slot for a packed NF4 matrix.
            source_modules[name].weight.data = _parked_cpu_master_tensor(entry[1])
            if target_nf4:
                _bind_params4bit_master(target_modules[name], entry[2], offloader.device)
            else:
                target_modules[name].weight.data = entry[2].to(
                    device=offloader.device, dtype=entry[4], non_blocking=True,
                )
        elif offloader.forward_only and not source_nf4 and (source_index, name) in slots.members:
            # Plan membership follows the potential pool owners, while its
            # actual ownership rotates during the inference wraparound.
            storage = source_modules[name].weight.data
            source_modules[name].weight.data = _parked_cpu_master_tensor(entry[1])
            storage.copy_(entry[2], non_blocking=True)
            target_modules[name].weight.data = storage
        elif not offloader.forward_only and (source_index, name) in slots.members:
            slots.swap(offloader, source_index, target_index, source_modules[name],
                       target_modules[name], entry, stream)
        else:
            remaining.append(entry)
    return tuple(remaining)
