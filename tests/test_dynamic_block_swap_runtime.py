import copy

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from library.runtime.offloading import ModelOffloader
from library.models.krea2_raw.dynamic_compile import refresh_resident_compile


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(8, 8, bias=False).requires_grad_(False)
        self.adapter = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return x + 0.1 * (self.base(x) + self.adapter(x)).tanh()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reconfiguration_preserves_updates_masters_and_optimizer(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.manual_seed(4)
    blocks = nn.ModuleList([Block() for _ in range(6)]).to(device)
    reference = copy.deepcopy(blocks)
    params = [p for p in blocks.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.001)
    reference_optimizer = torch.optim.AdamW(
        [p for p in reference.parameters() if p.requires_grad], lr=0.001
    )
    offloader = ModelOffloader(blocks, 4, torch.device(device))
    offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
    masters = offloader._cpu_weight_masters
    try:
        for count in [4, 2, 0, 3, 1, 4]:
            offloader.reconfigure(blocks, count)
            assert offloader._cpu_weight_masters is masters
            assert all(
                p is q
                for p, q in zip(
                    params, [p for p in blocks.parameters() if p.requires_grad]
                )
            )
            x = torch.randn(2, 8, device=device, requires_grad=True)
            expected = x.clone()
            actual = x.clone()
            for index, (block, other) in enumerate(zip(blocks, reference)):
                offloader.wait_for_block(index)
                actual = checkpoint(block, actual, use_reentrant=False)
                offloader.submit_move_blocks(blocks, index)
                expected = checkpoint(other, expected, use_reentrant=False)
            torch.testing.assert_close(actual, expected)
            actual.square().mean().backward()
            expected.square().mean().backward()
            optimizer.step()
            reference_optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            for block, other in zip(blocks, reference):
                torch.testing.assert_close(block.adapter.weight, other.adapter.weight)
        assert all(int(optimizer.state[p]["step"]) == 6 for p in params)
    finally:
        offloader.set_forward_only(False)
        offloader.thread_pool.shutdown(wait=True)


def test_reconfigure_rejects_mid_accumulation():
    blocks = nn.ModuleList([Block() for _ in range(4)])
    offloader = ModelOffloader(blocks, 2, torch.device("cpu"))
    blocks[0].adapter.weight.grad = torch.ones_like(blocks[0].adapter.weight)
    try:
        with pytest.raises(RuntimeError, match="cleared"):
            offloader.reconfigure(blocks, 1)
    finally:
        offloader.thread_pool.shutdown(wait=True)


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_migration_oom_rolls_back_or_fails_closed(monkeypatch, rollback_fails):
    from library.runtime import block_swap_reconfigure as runtime

    blocks = nn.ModuleList([Block() for _ in range(4)])
    offloader = ModelOffloader(blocks, 2, torch.device("cpu"))
    offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
    hooks = list(offloader.remove_handles)
    calls = []

    def injected(offloader, blocks, count):
        calls.append(count)
        if count == 1 or rollback_fails:
            raise torch.cuda.OutOfMemoryError("injected migration")

    monkeypatch.setattr(runtime, "_place_frozen_weights", injected)
    try:
        error = RuntimeError if rollback_fails else torch.cuda.OutOfMemoryError
        with pytest.raises(error):
            offloader.reconfigure(blocks, 1)
        assert calls == [1, 2]
        assert offloader.blocks_to_swap == 2 and offloader.remove_handles == hooks
    finally:
        offloader.thread_pool.shutdown(wait=True)


def test_resident_compile_caches_callable_and_demotes_tail(monkeypatch):
    from tests.test_krea2_compile_blocks import _tiny_dit

    calls = []

    def compile_fn(fn, **kwargs):
        def wrapped(x):
            return fn(x)

        calls.append(wrapped)
        return wrapped

    monkeypatch.setattr(torch, "compile", compile_fn)
    model = _tiny_dit(blocks_to_swap=2)
    model.compile_blocks(backend="eager")
    initial = model.blocks[0]._forward
    model.blocks_to_swap = 0
    refresh_resident_compile(model)
    assert len(calls) == 4
    model.blocks_to_swap = 2
    refresh_resident_compile(model)
    assert model.blocks[2]._forward == model.blocks[2]._krea_compile_base_forward
    model.blocks_to_swap = 0
    refresh_resident_compile(model)
    assert len(calls) == 4
    assert model.blocks[0]._forward is initial


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NF4 CUDA")
def test_nf4_unindexed_cuda_reuses_storage_and_quant_state():
    import bitsandbytes as bnb
    from library.runtime.block_swap_masters import (
        _bind_params4bit_master,
        _capture_cpu_master,
    )

    module = bnb.nn.Linear4bit(
        32, 32, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4"
    ).cuda()
    master, _ = _capture_cpu_master(
        module.weight, pin_memory=True, transfer_dtype="bf16"
    )
    storage = module.weight.data
    quant_state = module.weight.quant_state
    _bind_params4bit_master(module, master, torch.device("cuda"), storage=storage)
    assert module.weight.data_ptr() == storage.data_ptr()
    assert module.weight.quant_state is quant_state
    assert master.params4bit.quant_state.absmax.device.type == "cpu"
    parameter = module.weight
    _bind_params4bit_master(module, master, torch.device("cpu"), parked=True)
    assert module.weight is parameter and module.weight.device.type == "cpu"
    assert module.weight.quant_state is quant_state
    assert module.weight.quant_state.absmax.device.type == "cuda"
    _bind_params4bit_master(module, master, torch.device("cpu"))
    assert module.weight is parameter
    assert module.weight.quant_state.absmax.device.type == "cpu"
    assert module.weight.quant_state is not master.params4bit.quant_state
    assert module.quant_state is module.weight.quant_state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NF4 CUDA")
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("device", ["cuda", "cuda:0"])
def test_nf4_reconfigure_preserves_cpu_masters_and_forward(compiled, device):
    import bitsandbytes as bnb

    class NF4Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = bnb.nn.Linear4bit(
                32, 32, bias=False, compute_dtype=torch.bfloat16, quant_type="nf4"
            )
            self.adapter = nn.Linear(32, 32, bias=False, dtype=torch.bfloat16)

        def _forward(self, x):
            return (self.base(x) + self.adapter(x)).tanh()

        def forward(self, x):
            return self._forward(x)

    blocks = nn.ModuleList([NF4Block() for _ in range(6)]).cuda()
    for block in blocks:
        block.base.requires_grad_(False)
    reference = copy.deepcopy(blocks)
    offloader = ModelOffloader(blocks, 4, torch.device(device))
    offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
    base_parameters = [block.base.weight for block in blocks]
    from types import SimpleNamespace

    model = SimpleNamespace(blocks=blocks)
    reference_model = SimpleNamespace(blocks=reference)
    compilations = []
    if compiled:
        from torch._dynamo.backends.registry import lookup_backend

        def backend(graph, inputs):
            compilations.append(graph)
            return lookup_backend("inductor")(graph, inputs)

        model._krea_compile_options = {"backend": backend}
        reference_model._krea_compile_options = {"backend": backend}
    try:
        for count in [4, 0, 3, 1, 4, 0]:
            offloader.reconfigure(blocks, count)
            model.blocks_to_swap = count
            reference_model.blocks_to_swap = count
            refresh_resident_compile(model)
            refresh_resident_compile(reference_model)
            compiled_before = len(compilations)
            for masters in offloader._cpu_weight_masters:
                assert masters["base"].params4bit.device.type == "cpu"
                assert (
                    masters["base"].params4bit.quant_state.absmax.device.type == "cpu"
                )
            actual = torch.randn(
                2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            expected = actual.clone()
            for index, (block, other) in enumerate(zip(blocks, reference)):
                offloader.wait_for_block(index)
                actual = checkpoint(block, actual, use_reentrant=False)
                offloader.submit_move_blocks(blocks, index)
                expected = checkpoint(other, expected, use_reentrant=False)
            tolerance = {"rtol": 0, "atol": 0}
            torch.testing.assert_close(actual, expected, **tolerance)
            actual.float().square().mean().backward()
            expected.float().square().mean().backward()
            for block, other in zip(blocks, reference):
                torch.testing.assert_close(
                    block.adapter.weight.grad, other.adapter.weight.grad, **tolerance
                )
                block.adapter.weight.grad = None
                other.adapter.weight.grad = None
            assert all(
                block.base.weight is weight
                for block, weight in zip(blocks, base_parameters)
            )
            if compiled and count in (3, 1):
                assert len(compilations) == compiled_before
    finally:
        offloader.set_forward_only(False)
        offloader.thread_pool.shutdown(wait=True)
