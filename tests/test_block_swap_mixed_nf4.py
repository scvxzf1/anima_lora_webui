"""Saved NF4 aliases survive dense/packed storage rotation."""
import copy

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from library.anima.mixed_nf4.runtime import restore_linear
from library.runtime.block_swap_masters import Params4bitBlockSwapCpuMaster
from library.runtime.block_swap_representations import logical_shape
from library.runtime.offloading import ModelOffloader


def linear(quantized):
    weight = torch.randn(64, 64, dtype=torch.bfloat16) * .02
    if quantized:
        from bitsandbytes.functional import quantize_4bit
        packed, state = quantize_4bit(weight, compress_statistics=True, quant_type="nf4")
        return restore_linear(packed, state.as_dict(packed=True), in_features=64, out_features=64)
    module = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
    module.weight = nn.Parameter(weight, requires_grad=False)
    return module


class Block(nn.Module):
    def __init__(self, quantized):
        super().__init__()
        self.projection = linear(quantized)
        self.output_projection = linear(True)
        self.adapter = nn.Linear(64, 64, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            self.adapter.weight.mul_(.01)

    def forward(self, x):
        return x + (self.output_projection(self.projection(x)) + self.adapter(x)) * .1


def cleanup(offloader):
    for index in list(offloader.futures):
        offloader._wait_blocks_move(index, phase="test_cleanup")
    offloader.thread_pool.shutdown(wait=True)


@pytest.mark.parametrize("reverse", [False, True])
def test_cpu_bidirectional_storage_and_identity(reverse):
    torch.manual_seed(1)
    blocks = nn.ModuleList([Block(not reverse), Block(reverse)])
    offloader = ModelOffloader(blocks, 1, torch.device("cpu"))
    try:
        offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
        parameters = [b.projection.weight for b in blocks]
        states = [getattr(p, "quant_state", None) for p in parameters]
        originals = [p.detach().clone() for p in parameters]
        backing = next(iter(offloader._mixed_representation_slots.buffers.values()))
        alias = blocks[0].projection.weight.t()
        for source, target in [(0, 1), (1, 0)] * 3:
            offloader.swap_weight_devices(source, blocks[source], target, blocks[target])
            assert blocks[target].projection.weight.untyped_storage().data_ptr() == backing.data_ptr()
            assert blocks[source].projection.weight.data_ptr() == offloader._cpu_weight_masters[source]["projection"].params4bit.data.data_ptr() if isinstance(offloader._cpu_weight_masters[source]["projection"], Params4bitBlockSwapCpuMaster) else blocks[source].projection.weight.data_ptr() == offloader._cpu_weight_masters[source]["projection"].data_ptr()
            for index, block in enumerate(blocks):
                assert block.projection.weight is parameters[index]
                assert getattr(block.projection.weight, "quant_state", None) is states[index]
            torch.testing.assert_close(blocks[target].projection.weight, originals[target], rtol=0, atol=0)
        torch.testing.assert_close(alias, originals[0].t(), rtol=0, atol=0)
        assert len(offloader._mixed_representation_slots.buffers) == 1
    finally:
        cleanup(offloader)


@pytest.mark.parametrize("mutation", ["missing_shape", "packed_size", "logical_shape", "quant_type"])
def test_malformed_representation_fails_closed(mutation):
    module = linear(True)
    from library.runtime.block_swap_masters import _capture_cpu_master
    master, _ = _capture_cpu_master(module.weight, pin_memory=False, transfer_dtype="bf16")
    if mutation == "missing_shape":
        master.params4bit.quant_state.shape = None
    elif mutation == "packed_size":
        master.params4bit.data = master.params4bit.data[:-1]
    elif mutation == "logical_shape":
        master.params4bit.quant_state.shape = (32, 128)
    else:
        master.params4bit.quant_state.quant_type = "fp4"
    with pytest.raises(ValueError):
        logical_shape(master, module)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA NF4 storage aliases")
@pytest.mark.parametrize("restore_mode", ["foreach", "slab"])
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("count", [12, 20])
def test_tiny_cuda_training_sample_training(restore_mode, checkpointed, count):
    torch.manual_seed(3)
    blocks = nn.ModuleList([Block(index not in range(14, 21)) for index in range(28)])
    baseline = copy.deepcopy(blocks).cuda()
    offloader = ModelOffloader(blocks, count, torch.device("cuda"), restore_mode=restore_mode)
    x = torch.randn(2, 64, device="cuda", dtype=torch.bfloat16)

    def run(layers, swapping=False, backward=True):
        for block in layers:
            block.adapter.zero_grad(set_to_none=True)
        input_tensor = x.clone().requires_grad_(backward)
        hidden = input_tensor
        for index, block in enumerate(layers):
            if swapping:
                offloader.wait_for_block(index)
            hidden = checkpoint(block, hidden, use_reentrant=False) if checkpointed and backward else block(hidden)
            if swapping:
                offloader.submit_move_blocks(layers, index)
        if backward:
            hidden.float().square().mean().backward()
        return hidden.detach(), input_tensor.grad, [b.adapter.weight.grad for b in layers]

    try:
        offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
        params = [b.projection.weight for b in blocks]
        states = [getattr(p, "quant_state", None) for p in params]
        expected = run(baseline)
        for _ in range(3):
            actual = run(blocks, True)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for index, block in enumerate(blocks):
                assert block.projection.weight is params[index]
                assert getattr(params[index], "quant_state", None) is states[index]
        offloader.set_forward_only(True)
        training_buffers = dict(offloader._mixed_representation_slots.buffers)
        offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
        assert all(buffer is training_buffers[key] for key, buffer
                   in offloader._mixed_representation_slots.buffers.items())
        with torch.no_grad():
            torch.testing.assert_close(run(blocks, True, False)[0], expected[0], rtol=0, atol=0)
        offloader.set_forward_only(False)
        offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
        torch.testing.assert_close(run(blocks, True), expected, rtol=0, atol=0)
        assert len(offloader._mixed_representation_slots.buffers) <= len(blocks) - count
        for masters in offloader._cpu_weight_masters:
            for master in masters.values():
                tensor = master.params4bit.data if isinstance(master, Params4bitBlockSwapCpuMaster) else master
                assert tensor.device.type == "cpu"
                assert all(tensor.untyped_storage().data_ptr() != buffer.untyped_storage().data_ptr()
                           for buffer in offloader._mixed_representation_slots.buffers.values())
    finally:
        cleanup(offloader)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("master_scope", ["all", "participating"])
def test_reconfigure_mixed_topology_without_extra_prepare(device, master_scope):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(19)
    blocks = nn.ModuleList([Block(index % 2 == 0) for index in range(6)])
    baseline = copy.deepcopy(blocks).to(device)
    offloader = ModelOffloader(blocks, 4, torch.device(device))
    offloader.master_scope = master_scope
    x = torch.randn(2, 64, device=device, dtype=torch.bfloat16)

    def run(layers, swapped):
        for block in layers:
            block.adapter.zero_grad(set_to_none=True)
        inputs = x.clone().requires_grad_()
        hidden = inputs
        for index, block in enumerate(layers):
            if swapped:
                offloader.wait_for_block(index)
            hidden = checkpoint(block, hidden, use_reentrant=False)
            if swapped:
                offloader.submit_move_blocks(layers, index)
        hidden.float().square().mean().backward()
        return hidden.detach(), inputs.grad, [b.adapter.weight.grad.clone() for b in layers]

    try:
        offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
        parameters = [p for block in blocks for p in block.parameters()]
        for count in (4, 3, 0, 2, 4):
            for block in blocks:
                block.adapter.zero_grad(set_to_none=True)
            offloader.reconfigure(blocks, count)
            torch.testing.assert_close(run(blocks, True), run(baseline, False), rtol=0, atol=0)
            assert all(p is original for p, original in zip(blocks.parameters(), parameters))
    finally:
        cleanup(offloader)


def test_dense_prepare_retains_cached_plan_identity():
    blocks = nn.ModuleList([nn.Sequential(nn.Linear(64, 64, bias=False).requires_grad_(False))
                            for _ in range(6)])
    offloader = ModelOffloader(blocks, 2, torch.device("cpu"))
    try:
        offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
        previous = dict(offloader._swap_plan_cache)
        offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
        assert previous and all(offloader._swap_plan_cache[key] is plan for key, plan in previous.items())
    finally:
        cleanup(offloader)


def test_mixed_slot_allocation_oom_restores_previous_topology(monkeypatch):
    blocks = nn.ModuleList([Block(index % 2 == 0) for index in range(6)])
    offloader = ModelOffloader(blocks, 3, torch.device("cpu"))
    slots = offloader._mixed_representation_slots
    prepare = slots.prepare

    def injected(offloader, blocks):
        if offloader.blocks_to_swap == 2:
            raise torch.cuda.OutOfMemoryError("injected mixed allocation")
        return prepare(offloader, blocks)

    try:
        offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
        monkeypatch.setattr(slots, "prepare", injected)
        with pytest.raises(torch.cuda.OutOfMemoryError, match="mixed allocation"):
            offloader.reconfigure(blocks, 2)
        assert offloader.blocks_to_swap == 3
        assert len(slots.buffers) == 3
        offloader.swap_weight_devices(0, blocks[0], 3, blocks[3])
        offloader.swap_weight_devices(3, blocks[3], 0, blocks[0])
    finally:
        cleanup(offloader)


def test_plan_validation_failure_restores_count_and_backward_hooks():
    blocks = nn.ModuleList([nn.Sequential(linear(True)) for _ in range(6)])
    from bitsandbytes.functional import quantize_4bit
    packed, state = quantize_4bit(torch.randn(128, 128, dtype=torch.bfloat16),
                                  compress_statistics=True, quant_type="nf4")
    blocks[3][0] = restore_linear(packed, state.as_dict(packed=True),
                                  in_features=128, out_features=128)
    offloader = ModelOffloader(blocks, 2, torch.device("cpu"))
    try:
        offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
        previous_hooks = list(offloader.remove_handles)
        previous_registrations = [dict(block._backward_hooks) for block in blocks]
        with pytest.raises(ValueError, match="logical shape mismatch"):
            offloader.reconfigure(blocks, 3)
        assert offloader.blocks_to_swap == 2
        assert offloader.remove_handles == previous_hooks
        assert [dict(block._backward_hooks) for block in blocks] == previous_registrations
        offloader.swap_weight_devices(0, blocks[0], 4, blocks[4])
    finally:
        cleanup(offloader)


def test_forward_only_membership_change_reuses_existing_training_pool():
    blocks = nn.ModuleList([Block(index % 3 == 0) for index in range(6)])
    offloader = ModelOffloader(blocks, 2, torch.device("cpu"))
    try:
        offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
        slots = offloader._mixed_representation_slots
        assert slots.members
        slots.release_bindings(offloader, blocks)
        offloader.forward_only = True
        assert slots.prepare(offloader, blocks)
        assert slots.members and len(slots.buffers) == 1
    finally:
        cleanup(offloader)
