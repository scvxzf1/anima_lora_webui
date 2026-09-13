import copy
import gc
import weakref

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from library.runtime.offloading import ModelOffloader
from library.runtime.block_swap_sparse_masters import participating_blocks
from tests.test_dynamic_block_swap_runtime import Block


@pytest.mark.parametrize("count,active", [(0, set()), (1, {0, 5}), (2, {0, 1, 4, 5}), (4, set(range(6)))])
def test_participating_blocks_follow_training_exchange_pairs(count, active):
    assert participating_blocks(6, count) == active


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sparse_initialization_and_bidirectional_training(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(17)
    blocks = nn.ModuleList([Block() for _ in range(6)]).to(device)
    reference = copy.deepcopy(blocks)
    identities = list(blocks.parameters())
    offloader = ModelOffloader(blocks, 1, torch.device(device))
    offloader.master_scope = "participating"
    offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
    assert {i for i, m in enumerate(offloader._cpu_weight_masters) if m} == {0, 5}
    initial_bytes = offloader._frozen_weight_master_bytes
    try:
        for count in [1, 4, 0, 2, 1]:
            offloader.reconfigure(blocks, count)
            assert {i for i, m in enumerate(offloader._cpu_weight_masters) if m} == participating_blocks(6, count)
            assert offloader._frozen_weight_master_bytes == initial_bytes * min(count, 3)
            assert {i for i, slab in enumerate(offloader._cpu_weight_master_slabs) if slab is not None} == participating_blocks(6, count)
            x = torch.randn(2, 8, device=device, requires_grad=True)
            actual, expected = x.clone(), x.clone()
            for index, (block, other) in enumerate(zip(blocks, reference)):
                offloader.wait_for_block(index)
                actual = checkpoint(block, actual, use_reentrant=False)
                offloader.submit_move_blocks(blocks, index)
                expected = checkpoint(other, expected, use_reentrant=False)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.square().mean().backward()
            expected.square().mean().backward()
            for block, other in zip(blocks, reference):
                torch.testing.assert_close(block.adapter.weight.grad, other.adapter.weight.grad, rtol=0, atol=0)
                block.adapter.weight.grad = other.adapter.weight.grad = None
            assert all(a is b for a, b in zip(identities, blocks.parameters()))
        with pytest.raises(ValueError, match="training only"):
            offloader.set_forward_only(True)
    finally:
        offloader.set_forward_only(False)
        offloader.thread_pool.shutdown(wait=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA rollback")
def test_sparse_migration_oom_restores_residency_and_releases_new_masters(monkeypatch):
    from library.runtime import block_swap_reconfigure as runtime

    blocks = nn.ModuleList([Block() for _ in range(6)]).cuda()
    reference = copy.deepcopy(blocks)
    offloader = ModelOffloader(blocks, 1, torch.device("cuda"))
    offloader.master_scope = "participating"
    offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
    original_place = runtime._place_frozen_weights
    hooks = list(offloader.remove_handles)
    initial_bytes = offloader._frozen_weight_master_bytes
    retired = []

    def partial_placement(offloader, blocks, count):
        if count == 2:
            retired.extend(weakref.ref(offloader._cpu_weight_masters[i]["base"]) for i in (1, 4))
            # Fail after parking one newly swapped block, before migration completes.
            blocks[4].base.weight.data = offloader._cpu_weight_masters[4]["base"]
            raise torch.cuda.OutOfMemoryError("injected partial placement")
        original_place(offloader, blocks, count)

    monkeypatch.setattr(runtime, "_place_frozen_weights", partial_placement)
    try:
        with pytest.raises(torch.cuda.OutOfMemoryError, match="partial placement"):
            offloader.reconfigure(blocks, 2)
        gc.collect()
        assert all(ref() is None for ref in retired)
        assert offloader.blocks_to_swap == 1
        assert offloader.remove_handles == hooks
        assert offloader._frozen_weight_master_bytes == initial_bytes
        assert {i for i, m in enumerate(offloader._cpu_weight_masters) if m} == {0, 5}
        assert not offloader._swap_plan_cache and not offloader._swap_slab_plan_cache
        assert [b.base.weight.device.type for b in blocks] == ["cuda"] * 5 + ["cpu"]
        x = torch.randn(2, 8, device="cuda", requires_grad=True)
        actual, expected = x.clone(), x.clone()
        for index, (block, other) in enumerate(zip(blocks, reference)):
            offloader.wait_for_block(index)
            actual = checkpoint(block, actual, use_reentrant=False)
            offloader.submit_move_blocks(blocks, index)
            expected = checkpoint(other, expected, use_reentrant=False)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().mean().backward()
        expected.square().mean().backward()
        for block, other in zip(blocks, reference):
            torch.testing.assert_close(block.adapter.weight.grad, other.adapter.weight.grad, rtol=0, atol=0)
    finally:
        offloader.set_forward_only(False)
        offloader.thread_pool.shutdown(wait=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA NF4")
@pytest.mark.parametrize("compiled", [False, True])
def test_nf4_sparse_masters_preserve_compiled_parameters_and_gradients(compiled):
    import bitsandbytes as bnb
    from types import SimpleNamespace
    from library.models.krea2_raw.dynamic_compile import refresh_resident_compile

    class NF4Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = bnb.nn.Linear4bit(32, 32, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4")
            self.adapter = nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)

        def _forward(self, x):
            return (self.base(x) + self.adapter(x)).tanh()

        def forward(self, x):
            return self._forward(x)

    blocks = nn.ModuleList([NF4Block() for _ in range(6)]).cuda()
    for block in blocks:
        block.base.requires_grad_(False)
    reference = copy.deepcopy(blocks)
    identities = list(blocks.parameters())
    model, ref_model = SimpleNamespace(blocks=blocks), SimpleNamespace(blocks=reference)
    if compiled:
        model._krea_compile_options = ref_model._krea_compile_options = {"backend": "inductor"}
    offloader = ModelOffloader(blocks, 1, torch.device("cuda"))
    offloader.master_scope = "participating"
    offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
    try:
        for count in [1, 4, 0, 2, 1]:
            offloader.reconfigure(blocks, count)
            model.blocks_to_swap = ref_model.blocks_to_swap = count
            refresh_resident_compile(model)
            refresh_resident_compile(ref_model)
            assert {i for i, m in enumerate(offloader._cpu_weight_masters) if m} == participating_blocks(6, count)
            for masters in offloader._cpu_weight_masters:
                for master in masters.values():
                    assert master.params4bit.device.type == "cpu"
                    assert master.params4bit.quant_state.absmax.device.type == "cpu"
            x = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
            actual, expected = x.clone(), x.clone()
            for index, (block, other) in enumerate(zip(blocks, reference)):
                offloader.wait_for_block(index)
                actual = checkpoint(block, actual, use_reentrant=False)
                offloader.submit_move_blocks(blocks, index)
                expected = checkpoint(other, expected, use_reentrant=False)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            actual.float().square().mean().backward()
            expected.float().square().mean().backward()
            for block, other in zip(blocks, reference):
                torch.testing.assert_close(block.adapter.weight.grad, other.adapter.weight.grad, rtol=0, atol=0)
                block.adapter.weight.grad = other.adapter.weight.grad = None
            assert all(a is b for a, b in zip(identities, blocks.parameters()))
    finally:
        offloader.set_forward_only(False)
        offloader.thread_pool.shutdown(wait=True)
