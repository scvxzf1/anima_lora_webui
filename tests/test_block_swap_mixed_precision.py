"""CUDA regression for heterogeneous frozen block execution dtypes."""

import copy

import pytest
import torch
from torch import nn

from library.runtime.offloading import ModelOffloader
from library.training.adaptive_runtime.islands import install_precision_islands


class Block(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.projection = nn.Linear(8, 8, bias=False).requires_grad_(False)
        install_precision_islands(self, {"projection": dtype})

    def forward(self, x):
        return x + self.projection(x) * 0.1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA block swap")
@pytest.mark.parametrize("restore_mode", ["foreach", "slab"])
@pytest.mark.parametrize("backward", [False, True])
def test_mixed_block_dtype_wraparound(restore_mode, backward, monkeypatch):
    torch.manual_seed(20260921)
    blocks = nn.ModuleList([Block(dtype) for dtype in ("fp32", "fp16", "fp16", "fp32", "fp16", "fp32")])
    baseline = copy.deepcopy(blocks).cuda()
    expected_dtypes = [block.projection.weight.dtype for block in blocks]
    offloader = ModelOffloader(blocks, blocks_to_swap=4, device=torch.device("cuda"),
                              supports_backward=backward, restore_mode=restore_mode)
    original_restore_slab = offloader._get_cached_restore_slab

    def checked_restore_slab(*args, **kwargs):
        stream = offloader._get_copy_stream_for_slot(kwargs.get("slot_id"))
        assert torch.cuda.current_stream(offloader.device) == stream
        return original_restore_slab(*args, **kwargs)

    monkeypatch.setattr(offloader, "_get_cached_restore_slab", checked_restore_slab)
    x = torch.randn(2, 8, device="cuda")

    def run(layers, swapping=False):
        input_tensor = x.clone().requires_grad_(backward)
        hidden = input_tensor
        for index, block in enumerate(layers):
            if swapping:
                offloader.wait_for_block(index)
            assert block.projection.weight.dtype == expected_dtypes[index]
            hidden = block(hidden)
            if swapping:
                offloader.submit_move_blocks(blocks, index)
        if backward:
            hidden.square().mean().backward()
        return hidden.detach(), input_tensor.grad

    try:
        offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
        for index in range(len(blocks) - 4, len(blocks)):
            master = offloader._cpu_weight_masters[index]["projection"]
            assert blocks[index].projection.weight.data_ptr() == master.data_ptr()
        expected = run(baseline)
        for _ in range(3):
            torch.testing.assert_close(run(blocks, True), expected, rtol=0, atol=0)
    finally:
        for index in list(offloader.futures):
            offloader._wait_blocks_move(index, phase="test_cleanup")
        offloader.thread_pool.shutdown(wait=True)
