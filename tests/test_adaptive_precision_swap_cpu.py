"""CPU master/restore contracts for mixed islands; not CUDA transfer certification."""

from collections import Counter
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from library.runtime.block_swap_masters import _capture_cpu_master, _restore_cpu_master_tensor
from library.runtime.offloading import ModelOffloader
from library.training.adaptive_runtime.training_config import configuration_errors
from library.training.adaptive_runtime.training_precision import (
    install_training_precision, preserve_precision_cast, realized_precision_manifest_id,
)
from networks.lora_modules.lora import LoRAModule


@pytest.fixture(autouse=True)
def forbid_cuda_initialization(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU swap contracts must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    with torch.random.fork_rng(devices=[]):
        yield


class Block(torch.nn.Module):
    def __init__(self, index):
        super().__init__()
        for branch, name in enumerate(("projection", "condition")):
            layer = torch.nn.Linear(2, 2, bias=False, device="cpu", dtype=torch.float32)
            with torch.no_grad():
                layer.weight.copy_(torch.tensor([
                    [0.25 + (index + 1) * 2**-12 + 2**-20, -0.125],
                    [0.0625, 0.25 + branch * 2**-8],
                ]))
            setattr(self, name, layer)

    def forward(self, value):
        return value + (self.projection(value) + self.condition(value)) * 0.125


def make_model(layout):
    blocks = torch.nn.ModuleList([Block(index) for index in range(6)])
    patterns = ["*.condition"] if layout == "within" else ["0.*", "3.*", "5.*"]
    install_training_precision(blocks, SimpleNamespace(
        adaptive_fp32_modules=patterns, adaptive_loss_scale=1024.0, model_family="anima",
    ))
    adapters = torch.nn.ModuleList()
    for index, block in enumerate(blocks):
        for name in ("projection", "condition"):
            adapter = LoRAModule(f"swap_{index}_{name}", getattr(block, name), lora_dim=1, alpha=1)
            adapter.apply_to()
            adapter.train()
            with torch.no_grad():
                adapter.lora_down.weight.copy_(torch.tensor([[0.125, -0.125]]))
                adapter.lora_up.weight.copy_(torch.tensor([[0.0625], [-0.0625]]))
            adapters.append(adapter)
    return blocks, adapters


@contextmanager
def cpu_offloader(blocks):
    offloader = ModelOffloader(blocks, 4, torch.device("cpu"), supports_backward=True,
                              transfer_dtype="bf16", restore_mode="foreach")
    try:
        yield offloader
    finally:
        try:
            offloader.set_forward_only(False)
        finally:
            offloader.thread_pool.shutdown(wait=True)
            for handle in offloader.remove_handles:
                handle.remove()


def run_backward(blocks, adapters, value, *, offloader=None):
    adapters.zero_grad(set_to_none=True)
    source = value.detach().clone().requires_grad_(True)
    hidden = source
    for index, block in enumerate(blocks):
        if offloader is not None:
            offloader.wait_for_block(index)
        hidden = checkpoint(block, hidden, use_reentrant=False)
        if offloader is not None:
            offloader.submit_move_blocks(blocks, index)
    hidden.square().mean().backward()
    if offloader is not None:
        offloader.set_forward_only(False)
        assert not offloader.futures
    gradients = tuple(p.grad.detach().clone() for p in adapters.parameters())
    assert all(g.dtype == torch.float32 and torch.isfinite(g).all() for g in gradients)
    return hidden.detach(), source.grad, gradients


def snapshot(model):
    return {name: (p, p.detach().clone()) for name, p in model.named_parameters()}


def assert_unchanged(model, expected):
    current = dict(model.named_parameters())
    assert current.keys() == expected.keys()
    for name, (parameter, value) in expected.items():
        assert current[name] is parameter
        torch.testing.assert_close(parameter, value, rtol=0, atol=0)


def assert_masters(offloader, blocks, expected, layout):
    expected_bytes = 0
    for index, block in enumerate(blocks):
        masters = offloader._cpu_weight_masters[index]
        assert set(masters) == {"projection", "condition"}
        for name, master in masters.items():
            _, value = expected[f"{index}.{name}.weight"]
            layer = getattr(block, name)
            assert master.device.type == "cpu"
            assert master.dtype == layer._adaptive_dtype == layer.weight.dtype == value.dtype
            assert offloader._cpu_weight_master_dtypes[index][name] == value.dtype
            torch.testing.assert_close(master, value, rtol=0, atol=0)
            expected_bytes += value.numel() * value.element_size()
        slab = offloader._cpu_weight_master_slabs[index]
        if layout == "within":
            assert slab is None
        else:
            assert slab is not None and slab.dtype == block.projection.weight.dtype
    assert offloader._frozen_weight_master_bytes == expected_bytes == 144


@pytest.mark.parametrize("dtype,delta", [(torch.float16, 2**-10), (torch.float32, 2**-20)])
def test_native_bf16_transfer_policy_preserves_actual_dtype_and_values(dtype, delta):
    weight = torch.tensor([[1 + delta, -0.5], [0.25, 2]], dtype=dtype, device="cpu")
    original = weight.clone()
    # A real BF16 conversion would lose information in both fixtures.
    assert not torch.equal(weight.to(torch.bfloat16).to(dtype), original)
    master, stats = _capture_cpu_master(weight, pin_memory=False, transfer_dtype="bf16")
    assert master.dtype == dtype and master.device.type == "cpu"
    assert stats["stored_bytes"] == stats["source_bytes"] == weight.numel() * weight.element_size()
    for _ in range(3):
        restored = _restore_cpu_master_tensor(master, device=torch.device("cpu"), dtype=dtype)
        torch.testing.assert_close(restored, original, rtol=0, atol=0)
        torch.testing.assert_close(master, original, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["within", "between"])
def test_cpu_swap_wraparound_preserves_mixed_islands_and_lora_gradients(layout, monkeypatch):
    blocks, adapters = make_model(layout)
    reference, reference_adapters = make_model(layout)
    base_weights, adapter_weights = snapshot(blocks), snapshot(adapters)
    keys = list(blocks.state_dict())
    names = blocks._adaptive_precision_manifest_names
    manifest_id = blocks._adaptive_precision_manifest_id
    swaps = []
    with cpu_offloader(blocks) as offloader:
        original_swap = offloader._swap_weight_devices_cached_no_cuda

        def traced_swap(source, source_block, target, target_block, plan):
            result = original_swap(source, source_block, target, target_block, plan)
            swaps.append((source, target))
            return result

        monkeypatch.setattr(offloader, "_swap_weight_devices_cached_no_cuda", traced_swap)
        offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
        assert_masters(offloader, blocks, base_weights, layout)
        for turn in range(3):
            value = torch.tensor([[0.5, -0.25], [-0.75, 0.125]], device="cpu") / 2**turn
            expected = run_backward(reference, reference_adapters, value)
            actual = run_backward(blocks, adapters, value, offloader=offloader)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert_unchanged(blocks, base_weights)
            assert_unchanged(adapters, adapter_weights)
            assert_masters(offloader, blocks, base_weights, layout)
            assert preserve_precision_cast(blocks, torch.float32) is None
            assert realized_precision_manifest_id(blocks, names) == manifest_id
        pairs = [(index, index + 2) for index in range(4)]
        assert Counter(swaps) == Counter({pair: 3 for pair in pairs + [p[::-1] for p in pairs]})
        offloader.restore_blocks_to_device(blocks, torch.device("cpu"))
        assert_unchanged(blocks, base_weights)
        assert_unchanged(adapters, adapter_weights)
        assert_masters(offloader, blocks, base_weights, layout)
        assert list(blocks.state_dict()) == keys
        assert realized_precision_manifest_id(blocks, names) == manifest_id


@pytest.mark.parametrize("transfer_dtype", ["fp8_e4m3", "int8"])
def test_adaptive_training_still_rejects_lossy_transfer_policies(transfer_dtype):
    errors = configuration_errors({"adaptive_precision": "fp16_fp32",
                                   "block_swap_transfer_dtype": transfer_dtype})
    assert "FP16/FP32 training requires block_swap_transfer_dtype='bf16'" in errors
