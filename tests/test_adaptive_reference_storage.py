import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from bench.adaptive_runtime.reference_storage import (
    Fp32ExecutionGuard, promote_placed_reference, validate_reference_storage,
)
from library.runtime.offloading import ModelOffloader


def reference_args(**changes):
    return SimpleNamespace(**{**dict(reference_bf16_storage=True, precision="fp32-reference",
                                    swap=4, disposable_probe=True, capture_training=True,
                                    resume=None, checkpoint_every_step=False, capture_linear=[]),
                              **changes})


@pytest.mark.parametrize("changes", [{"precision": "bf16"}, {"swap": 0},
                                     {"disposable_probe": False}, {"capture_training": False},
                                     {"resume": "step.pt"}, {"checkpoint_every_step": True},
                                     {"capture_linear": ["layer"]}])
def test_reference_storage_rejects_unsupported_modes(changes):
    with pytest.raises(ValueError, match="swapped disposable"):
        validate_reference_storage(reference_args(**changes))


def test_reference_storage_opt_in_and_guard():
    validate_reference_storage(reference_args())
    validate_reference_storage(SimpleNamespace())
    model = nn.Linear(2, 2)
    guard = Fp32ExecutionGuard(model, {})
    with pytest.raises(RuntimeError, match="execution tensor"):
        model(torch.ones(1, 2))
    guard.close()
    model(torch.ones(1, 2))
    assert not guard.handles


@pytest.mark.parametrize("mode,pending", [("slab", {}), ("foreach", {0: object()})])
def test_reference_storage_rejects_nonidle_or_slab(mode, pending):
    model = SimpleNamespace(offloader=SimpleNamespace(
        restore_mode=mode, futures=pending, _cpu_weight_masters=[{}],
        _swap_gpu_slab_cache={}), blocks=[object()])
    with pytest.raises(ValueError, match="idle foreach"):
        promote_placed_reference(model, {})


class ReferenceBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 8).to(torch.bfloat16).requires_grad_(False)
        self.offset = nn.Parameter(torch.ones(8, dtype=torch.bfloat16), requires_grad=False)
        self.register_buffer("gain", torch.full((8,), 0.125, dtype=torch.bfloat16))
        self.adapter = nn.Linear(8, 8, bias=False)

    def _forward(self, x):
        return x + (self.projection(x) + self.adapter(x) + self.offset) * self.gain

    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA reference storage")
def test_exact_bf16_storage_fp32_execution_with_checkpoint_wraparound():
    torch.manual_seed(20260921)
    model = nn.Module()
    model.blocks = nn.ModuleList([ReferenceBlock() for _ in range(6)])
    baseline = copy.deepcopy(model).cuda().float()
    offloader = ModelOffloader(model.blocks, blocks_to_swap=4, device=torch.device("cuda"),
                              restore_mode="foreach")
    model.offloader = offloader
    guard = None
    x = torch.randn(2, 8, device="cuda")

    def run(target, swapping=False):
        target.zero_grad(set_to_none=True)
        value = x.clone().requires_grad_(True)
        hidden = value
        for index, block in enumerate(target.blocks):
            if swapping:
                offloader.wait_for_block(index)
            assert all(p.dtype == torch.float32 for p in block.parameters())
            assert all(b.dtype == torch.float32 for b in block.buffers())
            hidden = block(hidden)
            if swapping:
                offloader.submit_move_blocks(target.blocks, index)
        hidden.square().mean().backward()
        return (hidden.detach(), value.grad.clone(),
                [p.grad.clone() for p in target.parameters() if p.requires_grad])

    try:
        offloader.prepare_block_devices_before_forward(model.blocks, free_cache=False)
        snapshots = [{name: value.clone() for name, value in table.items()}
                     for table in offloader._cpu_weight_masters]
        report = {}
        guard = promote_placed_reference(model, report)
        expected = run(baseline)
        for _ in range(3):
            torch.testing.assert_close(run(model, True), expected, rtol=0, atol=0)
        for table, snapshot in zip(offloader._cpu_weight_masters, snapshots, strict=True):
            for name, value in table.items():
                assert value.dtype == torch.bfloat16 and value.device.type == "cpu"
                assert torch.equal(value, snapshot[name])
        # More calls than one block/projection/adapter forward per step proves
        # the guard also ran inside non-reentrant checkpoint recomputation.
        assert report["reference_storage"]["module_calls_checked"] > 3 * 6 * 3
        assert report["reference_storage"]["cpu_master_bytes"] > 0
    finally:
        if guard is not None:
            guard.close()
        for index in list(offloader.futures):
            offloader._wait_blocks_move(index, phase="test_cleanup")
        offloader.thread_pool.shutdown(wait=True)
