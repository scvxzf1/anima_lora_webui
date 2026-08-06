from __future__ import annotations

import argparse
import json
import time

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from library.runtime.device import is_weight_swap_excluded, should_move_weight_to_device
from library.runtime.convrot.free_base import free_linear_weight_storage
from library.runtime.offloading import (
    Int8BlockSwapCpuMaster,
    ModelOffloader,
    normalize_block_swap_int8_scope,
    normalize_block_swap_int8_restore_mode,
    normalize_block_swap_restore_mode,
    normalize_block_swap_transfer_dtype,
    swap_weight_devices_no_cuda,
)
from library.runtime.block_swap_masters import (
    _can_swap_frozen_weight_to_cpu,
    _ensure_weight_on_device,
)
from library.runtime.block_swap_payload import (
    BlockSwapManagedTensor,
    block_swap_payload_residency,
)


def _make_training_offloader(blocks, blocks_to_swap, monkeypatch=None, depth=None):
    if depth is not None:
        monkeypatch.setenv("ANIMA_BLOCK_SWAP_PREFETCH_DEPTH", str(depth))
    return ModelOffloader(
        blocks,
        blocks_to_swap=blocks_to_swap,
        device=torch.device("cpu"),
        supports_backward=True,
    )


def test_submit_move_blocks_prefetch_depth_training_mode(monkeypatch) -> None:
    # Training mode (forward_only=False) prefetches ``depth`` blocks ahead.
    blocks = nn.ModuleList([_TinyBlock() for _ in range(6)])
    offloader = _make_training_offloader(blocks, 2, monkeypatch, depth=2)
    offloader.prepare_block_devices_before_forward(blocks, free_cache=False)

    offloader.submit_move_blocks(blocks, 0)

    # depth=2 → futures for both block_idx_to_cuda = 4 (step0) and 5 (step1)
    assert set(offloader.futures.keys()) == {4, 5}


def test_submit_move_blocks_prefetch_depth_default_one_in_forward_only(monkeypatch) -> None:
    # Forward-only (inference) ignores depth and keeps the exact lead of 1, so
    # the rotating slot storage is never overwritten before its block runs.
    blocks = nn.ModuleList([_TinyBlock() for _ in range(3)])
    offloader = _make_training_offloader(blocks, 1, monkeypatch, depth=3)
    offloader.set_forward_only(True)
    offloader.prepare_block_devices_before_forward(blocks, free_cache=False)

    offloader.submit_move_blocks(blocks, 0)

    assert set(offloader.futures.keys()) == {2}


def test_submit_move_blocks_prefetch_depth_env_override(monkeypatch) -> None:
    blocks = nn.ModuleList([_TinyBlock() for _ in range(6)])
    offloader = _make_training_offloader(blocks, 3, monkeypatch, depth=1)
    offloader.prepare_block_devices_before_forward(blocks, free_cache=False)

    offloader.submit_move_blocks(blocks, 0)

    # depth=1 → only the single next block
    assert set(offloader.futures.keys()) == {3}



class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(2, 2, bias=False)
        self.adapter = nn.Linear(2, 2, bias=False)
        self.base.weight.requires_grad_(False)
        self.adapter.weight.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.adapter(x)


class _Int8CandidateBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.layer1 = nn.Linear(8, 16, bias=False)
        self.mlp.layer2 = nn.Linear(16, 8, bias=False)
        self.self_attn = nn.Module()
        self.self_attn.qkv_proj = nn.Linear(8, 24, bias=False)
        self.self_attn.output_proj = nn.Linear(8, 8, bias=False)
        self.adaln_up_mlp = nn.Linear(8, 8, bias=False)
        self.adapter = nn.Linear(8, 8, bias=False)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.requires_grad_(False)
        self.adapter.weight.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(self.mlp.layer1(x))
        return self.mlp.layer2(hidden) + self.adapter(x)


class _ConvRotPayloadBlock(nn.Module):
    def __init__(self, value: int, *, device: torch.device) -> None:
        super().__init__()
        self.base = nn.Linear(4, 4, bias=False, device=device)
        self.base.weight.requires_grad_(False)
        payload = torch.full((4, 4), value, dtype=torch.int8, device=device)
        scale = torch.full((4,), float(value), dtype=torch.float32, device=device)
        self.base.add_module(
            "_convrot_quantized_weight", BlockSwapManagedTensor(payload)
        )
        self.base.add_module("_convrot_scale", BlockSwapManagedTensor(scale))
        free_linear_weight_storage(self.base)


def test_cpu_block_swap_policy_skips_trainable_weights() -> None:
    block = _TinyBlock()

    assert should_move_weight_to_device(
        block.base, torch.device("cpu"), include_trainable=False
    )
    assert not should_move_weight_to_device(
        block.adapter, torch.device("cpu"), include_trainable=False
    )
    assert should_move_weight_to_device(
        block.adapter, torch.device("cpu"), include_trainable=True
    )


def test_block_swap_skips_convrot_free_base_meta_weights() -> None:
    """Scheme A: free-base meta Linears must not enter masters or weighs_to_device."""
    block = _TinyBlock()
    free_linear_weight_storage(block.base)

    assert is_weight_swap_excluded(block.base)
    assert not is_weight_swap_excluded(block.adapter)
    assert not should_move_weight_to_device(
        block.base, torch.device("cpu"), include_trainable=False
    )
    assert not _can_swap_frozen_weight_to_cpu(block.base)
    # ensure is a no-op on meta (must not raise).
    _ensure_weight_on_device(block.base, torch.device("cpu"))
    assert block.base.weight.device.type == "meta"


def test_prepare_block_devices_skips_free_base_masters() -> None:
    """First prepare after free-base must not try to copy meta into CPU masters."""
    blocks = nn.ModuleList([_TinyBlock(), _TinyBlock(), _TinyBlock()])
    # Free base on every block (simulates scope covering those linears).
    for b in blocks:
        free_linear_weight_storage(b.base)

    offloader = ModelOffloader(
        blocks, blocks_to_swap=1, device=torch.device("cpu"), supports_backward=False
    )
    offloader.prepare_block_devices_before_forward(blocks, free_cache=False)

    # No masters for freed bases; adapters may still be non-swappable (trainable).
    assert offloader._cpu_weight_masters is not None
    for block_masters in offloader._cpu_weight_masters:
        assert "base" not in block_masters
    for b in blocks:
        assert b.base.weight.device.type == "meta"


def test_convrot_payload_uses_existing_block_swap_master_protocol() -> None:
    blocks = nn.ModuleList(
        [_ConvRotPayloadBlock(i + 1, device=torch.device("cpu")) for i in range(3)]
    )
    offloader = ModelOffloader(
        blocks, blocks_to_swap=1, device=torch.device("cpu"), supports_backward=False
    )
    try:
        offloader.prepare_block_devices_before_forward(blocks, free_cache=False)

        assert offloader._cpu_weight_masters is not None
        for masters in offloader._cpu_weight_masters:
            assert "base._convrot_quantized_weight" in masters
            assert "base._convrot_scale" in masters
            assert "base" not in masters
        assert all("_convrot" not in key for key in blocks.state_dict())
        residency = block_swap_payload_residency(blocks)
        assert residency["total_tensors"] == 6
        assert residency["tensors_by_device"] == {"cpu": 6}
    finally:
        offloader.thread_pool.shutdown(wait=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA residency swap")
def test_convrot_payload_moves_with_block_swap() -> None:
    device = torch.device("cuda")
    blocks = nn.ModuleList(
        [_ConvRotPayloadBlock(i + 1, device=device) for i in range(3)]
    )
    expected_target = blocks[2].base._convrot_quantized_weight.weight.cpu().clone()
    offloader = ModelOffloader(
        blocks, blocks_to_swap=1, device=device, supports_backward=False
    )
    try:
        offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
        assert blocks[0].base._convrot_quantized_weight.weight.device.type == "cuda"
        assert blocks[2].base._convrot_quantized_weight.weight.device.type == "cpu"

        offloader.swap_weight_devices(0, blocks[0], 2, blocks[2])
        torch.cuda.synchronize(device)

        assert blocks[0].base._convrot_quantized_weight.weight.device.type == "cpu"
        target = blocks[2].base._convrot_quantized_weight.weight
        assert target.device.type == "cuda"
        assert torch.equal(target.cpu(), expected_target)
        residency = block_swap_payload_residency(blocks)
        assert residency["tensors_by_device"] == {"cpu": 2, "cuda:0": 4}
    finally:
        offloader.thread_pool.shutdown(wait=False)


def test_prepare_block_devices_keeps_trainable_weights_on_main_device() -> None:
    blocks = nn.ModuleList([_TinyBlock(), _TinyBlock(), _TinyBlock()])
    offloader = ModelOffloader(
        blocks, blocks_to_swap=1, device=torch.device("cpu"), supports_backward=False
    )

    offloader.prepare_block_devices_before_forward(blocks, free_cache=False)

    tail = blocks[-1]
    assert tail.base.weight.device.type == "cpu"
    assert tail.adapter.weight.device.type == "cpu"
    assert tail.adapter.weight.requires_grad


def test_swap_weight_devices_no_cuda_does_not_swap_trainable_weights() -> None:
    block_to_cpu = _TinyBlock()
    block_to_device = _TinyBlock()
    source_trainable = block_to_cpu.adapter.weight.detach().clone()
    target_trainable = block_to_device.adapter.weight.detach().clone()

    swap_weight_devices_no_cuda(torch.device("cpu"), block_to_cpu, block_to_device)

    assert torch.equal(block_to_cpu.adapter.weight, source_trainable)
    assert torch.equal(block_to_device.adapter.weight, target_trainable)


def test_block_swap_pause_and_resume_state_machine() -> None:
    blocks = nn.ModuleList([_TinyBlock(), _TinyBlock(), _TinyBlock()])
    offloader = ModelOffloader(
        blocks, blocks_to_swap=1, device=torch.device("cpu"), supports_backward=True
    )

    assert not offloader.forward_only
    offloader.set_forward_only(True)
    assert offloader.forward_only
    offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
    offloader.set_forward_only(False)
    assert not offloader.forward_only


def test_block_swap_profile_jsonl_records_forward_wait(tmp_path) -> None:
    blocks = nn.ModuleList([_TinyBlock(), _TinyBlock(), _TinyBlock()])
    profile_path = tmp_path / "block_swap_profile.jsonl"
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=1,
        device=torch.device("cpu"),
        supports_backward=False,
        profile_jsonl=str(profile_path),
    )

    offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
    offloader.submit_move_blocks(blocks, 0)
    offloader.wait_for_block(2)

    events = [json.loads(line) for line in profile_path.read_text().splitlines()]
    wait_events = [event for event in events if event["ev"] == "block_swap"]
    assert len(wait_events) == 1
    event = wait_events[0]
    assert event["ev"] == "block_swap"
    assert event["phase"] == "forward_wait"
    assert event["submit_phase"] == "forward_prefetch"
    assert event["block_idx"] == 2
    assert event["block_idx_to_cpu"] == 0
    assert event["transfer_dtype"] == "bf16"
    assert event["wait_ms"] >= 0
    assert event["h2d_ms"] >= 0
    assert event["d2h_ms"] >= 0
    assert event["transfer_ms"] >= 0
    assert event["enqueue_ms"] >= 0
    assert event["submit_trigger_block_idx"] == 0
    assert event["wait_trigger_block_idx"] == 2
    assert event["prefetch_lead_blocks"] == 2
    assert event["slot_id"] == 0
    assert event["slot_count"] == 2
    assert event["slot_current_age_ms"] >= 0
    assert event["prefetch_runway_ms"] >= 0
    assert event["enqueue_to_wait_ms"] >= 0
    assert event["estimated_ready_slack_ms"] >= 0
    assert event["queued_at"] <= event["ready_at"]
    assert event["enqueued_at"] <= event["ready_at"]


def test_block_swap_profile_records_slot_reuse_age(tmp_path) -> None:
    blocks = nn.ModuleList([_TinyBlock() for _ in range(4)])
    profile_path = tmp_path / "block_swap_profile.jsonl"
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=2,
        device=torch.device("cpu"),
        supports_backward=False,
        profile_jsonl=str(profile_path),
    )

    offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
    offloader.submit_move_blocks(blocks, 0)
    offloader.wait_for_block(2)
    offloader.submit_move_blocks(blocks, 2)
    offloader.wait_for_block(0)

    events = [
        event
        for event in (json.loads(line) for line in profile_path.read_text().splitlines())
        if event["ev"] == "block_swap"
    ]
    assert len(events) == 2
    first, second = events
    assert first["block_idx"] == 2
    assert first["slot_id"] == 0
    assert first["slot_previous_block_idx"] is None
    assert second["block_idx"] == 0
    assert second["slot_id"] == 0
    assert second["slot_previous_block_idx"] == 2
    assert second["slot_previous_phase"] == "forward_prefetch"
    assert second["slot_reuse_age_ms"] >= first["slot_current_age_ms"]


def test_block_swap_profile_records_h2d_only_cpu_masters(tmp_path) -> None:
    blocks = nn.ModuleList([_TinyBlock(), _TinyBlock(), _TinyBlock()])
    profile_path = tmp_path / "block_swap_profile.jsonl"
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=1,
        device=torch.device("cpu"),
        supports_backward=False,
        profile_jsonl=str(profile_path),
    )

    offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
    offloader.submit_move_blocks(blocks, 0)
    offloader.wait_for_block(2)

    events = [json.loads(line) for line in profile_path.read_text().splitlines()]
    config = [event for event in events if event["ev"] == "block_swap_config"][0]
    waits = [event for event in events if event["ev"] == "block_swap"]
    assert config["h2d_only"] is True
    assert config["transfer_dtype"] == "bf16"
    assert config["frozen_weight_master_bytes"] > 0
    assert config["bf16_master_bytes"] == config["frozen_weight_master_bytes"]
    assert config["fp8_master_bytes"] == 0
    assert len(config["frozen_weight_bytes_by_block"]) == 3
    assert waits[0]["d2h_ms"] == 0
    assert waits[0]["h2d_ms"] >= 0


def test_block_swap_transfer_dtype_aliases_and_rejects_invalid() -> None:
    assert normalize_block_swap_transfer_dtype(None) == "bf16"
    assert normalize_block_swap_transfer_dtype("bfloat16") == "bf16"
    assert normalize_block_swap_transfer_dtype("float8_e4m3fn") == "fp8_e4m3"
    assert normalize_block_swap_transfer_dtype("int8") == "int8"
    assert normalize_block_swap_transfer_dtype("int8_linear") == "int8"
    assert normalize_block_swap_transfer_dtype("i8") == "int8"
    with pytest.raises(ValueError):
        normalize_block_swap_transfer_dtype("int4")


def test_block_swap_restore_mode_aliases_and_rejects_invalid() -> None:
    assert normalize_block_swap_restore_mode(None) == "foreach"
    assert normalize_block_swap_restore_mode("default") == "foreach"
    assert normalize_block_swap_restore_mode("loop") == "foreach"
    assert normalize_block_swap_restore_mode("slab") == "slab"
    with pytest.raises(ValueError):
        normalize_block_swap_restore_mode("ring")


def test_block_swap_int8_restore_mode_aliases_and_rejects_invalid() -> None:
    assert normalize_block_swap_int8_restore_mode(None) == "copy"
    assert normalize_block_swap_int8_restore_mode("default") == "copy"
    assert normalize_block_swap_int8_restore_mode("reuse") == "reuse_storage"
    assert normalize_block_swap_int8_restore_mode("into") == "reuse_storage"
    assert normalize_block_swap_int8_restore_mode("inplace") == "reuse_storage"
    assert normalize_block_swap_int8_restore_mode("direct") == "direct_bind"
    assert normalize_block_swap_int8_restore_mode("bind") == "direct_bind"
    assert normalize_block_swap_int8_restore_mode("direct_bind") == "direct_bind"
    with pytest.raises(ValueError):
        normalize_block_swap_int8_restore_mode("slab")


def test_block_swap_int8_scope_normalizes_and_rejects_invalid() -> None:
    assert normalize_block_swap_int8_scope(None) == "all"
    assert normalize_block_swap_int8_scope("") == "all"
    assert normalize_block_swap_int8_scope(" MLP , cross_q , mlp ") == "mlp,cross_q"
    assert normalize_block_swap_int8_scope("all,mlp") == "all"
    with pytest.raises(ValueError):
        normalize_block_swap_int8_scope("mlp,adaln")


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="torch build does not expose float8_e4m3fn",
)
def test_fp8_block_swap_cpu_master_restores_execution_dtype(tmp_path) -> None:
    blocks = nn.ModuleList([_TinyBlock(), _TinyBlock(), _TinyBlock()])
    profile_path = tmp_path / "block_swap_profile.jsonl"
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=1,
        device=torch.device("cpu"),
        supports_backward=False,
        profile_jsonl=str(profile_path),
        transfer_dtype="fp8_e4m3",
    )

    offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
    assert offloader._cpu_weight_masters is not None
    assert offloader._cpu_weight_masters[0]["base"].dtype == torch.float8_e4m3fn
    assert "adapter" not in offloader._cpu_weight_masters[0]

    offloader.submit_move_blocks(blocks, 0)
    offloader.wait_for_block(2)

    assert blocks[0].base.weight.dtype == torch.float8_e4m3fn
    assert blocks[2].base.weight.dtype == torch.float32
    assert blocks[2].adapter.weight.dtype == torch.float32
    assert blocks[2].adapter.weight.requires_grad

    events = [json.loads(line) for line in profile_path.read_text().splitlines()]
    config = [event for event in events if event["ev"] == "block_swap_config"][0]
    assert config["transfer_dtype"] == "fp8_e4m3"
    assert config["fp8_master_bytes"] > 0
    assert config["fp8_master_bytes"] < config["bf16_master_bytes"]
    assert len(config["fp8_mean_abs_error_by_block"]) == 3
    assert len(config["fp8_relative_l2_by_block"]) == 3


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="torch build does not expose float8_e4m3fn",
)
def test_fp8_block_swap_restore_all_blocks_to_device_uses_execution_dtype() -> None:
    blocks = nn.ModuleList([_TinyBlock(), _TinyBlock(), _TinyBlock()])
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=1,
        device=torch.device("cpu"),
        supports_backward=False,
        transfer_dtype="fp8_e4m3",
    )

    offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
    offloader.submit_move_blocks(blocks, 0)
    offloader.wait_for_block(2)
    assert blocks[0].base.weight.dtype == torch.float8_e4m3fn

    offloader.restore_blocks_to_device(blocks, torch.device("cpu"))

    assert all(block.base.weight.dtype == torch.float32 for block in blocks)
    assert all(block.adapter.weight.dtype == torch.float32 for block in blocks)
    assert all(block.adapter.weight.requires_grad for block in blocks)


def test_int8_block_swap_cpu_master_quantizes_only_candidate_frozen_linears(
    tmp_path,
) -> None:
    torch.manual_seed(0)
    blocks = nn.ModuleList([_Int8CandidateBlock() for _ in range(3)])
    target_mlp_weight = blocks[2].mlp.layer1.weight.detach().clone()
    target_adaln_weight = blocks[2].adaln_up_mlp.weight.detach().clone()
    profile_path = tmp_path / "block_swap_profile.jsonl"
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=1,
        device=torch.device("cpu"),
        supports_backward=False,
        profile_jsonl=str(profile_path),
        transfer_dtype="int8",
    )

    offloader.prepare_block_devices_before_forward(blocks, free_cache=False)

    assert offloader._cpu_weight_masters is not None
    masters = offloader._cpu_weight_masters[0]
    assert isinstance(masters["mlp.layer1"], Int8BlockSwapCpuMaster)
    assert isinstance(masters["mlp.layer2"], Int8BlockSwapCpuMaster)
    assert isinstance(masters["self_attn.qkv_proj"], Int8BlockSwapCpuMaster)
    assert isinstance(masters["self_attn.output_proj"], Int8BlockSwapCpuMaster)
    assert isinstance(masters["adaln_up_mlp"], torch.Tensor)
    assert "adapter" not in masters
    assert offloader._cpu_weight_master_slabs is not None
    assert offloader._cpu_weight_master_slabs[0] is None

    offloader.submit_move_blocks(blocks, 0)
    offloader.wait_for_block(2)

    assert blocks[0].mlp.layer1.weight.dtype == torch.int8
    assert blocks[0].adaln_up_mlp.weight.dtype == torch.float32
    assert blocks[0].adapter.weight.dtype == torch.float32
    assert blocks[0].adapter.weight.requires_grad
    assert blocks[2].mlp.layer1.weight.dtype == torch.float32
    assert blocks[2].adapter.weight.requires_grad
    mlp_rel = (blocks[2].mlp.layer1.weight - target_mlp_weight).norm() / target_mlp_weight.norm()
    adaln_rel = (
        (blocks[2].adaln_up_mlp.weight - target_adaln_weight).norm()
        / target_adaln_weight.norm()
    )
    assert mlp_rel.item() < 0.03
    assert adaln_rel.item() == pytest.approx(0.0)

    events = [json.loads(line) for line in profile_path.read_text().splitlines()]
    config = [event for event in events if event["ev"] == "block_swap_config"][0]
    assert config["transfer_dtype"] == "int8"
    assert config["int8_master_bytes"] > 0
    assert config["int8_master_bytes"] < config["bf16_master_bytes"]
    assert config["int8_quantized_tensors"] == 12
    assert len(config["int8_relative_l2_by_block"]) == 3
    assert max(config["int8_relative_l2_by_block"]) < 0.03
    assert config["fp8_master_bytes"] == 0


def test_int8_block_swap_cpu_master_respects_scope(tmp_path) -> None:
    blocks = nn.ModuleList([_Int8CandidateBlock() for _ in range(3)])
    profile_path = tmp_path / "block_swap_profile.jsonl"
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=1,
        device=torch.device("cpu"),
        supports_backward=False,
        profile_jsonl=str(profile_path),
        transfer_dtype="int8",
        int8_scope="mlp",
    )

    offloader.prepare_block_devices_before_forward(blocks, free_cache=False)

    assert offloader._cpu_weight_masters is not None
    masters = offloader._cpu_weight_masters[0]
    assert isinstance(masters["mlp.layer1"], Int8BlockSwapCpuMaster)
    assert isinstance(masters["mlp.layer2"], Int8BlockSwapCpuMaster)
    assert isinstance(masters["self_attn.qkv_proj"], torch.Tensor)
    assert isinstance(masters["self_attn.output_proj"], torch.Tensor)
    assert isinstance(masters["adaln_up_mlp"], torch.Tensor)
    assert offloader._int8_quantized_tensors == 6

    events = [json.loads(line) for line in profile_path.read_text().splitlines()]
    config = [event for event in events if event["ev"] == "block_swap_config"][0]
    assert config["transfer_dtype"] == "int8"
    assert config["int8_scope"] == "mlp"
    assert config["int8_quantized_tensors"] == 6
    assert config["int8_master_bytes"] > 0
    assert config["int8_master_bytes"] < config["bf16_master_bytes"]


def test_int8_block_swap_scope_can_come_from_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ANIMA_BLOCK_SWAP_INT8_SCOPE", "self_attn_out")
    blocks = nn.ModuleList([_Int8CandidateBlock() for _ in range(3)])
    profile_path = tmp_path / "block_swap_profile.jsonl"
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=1,
        device=torch.device("cpu"),
        supports_backward=False,
        profile_jsonl=str(profile_path),
        transfer_dtype="int8",
    )

    offloader.prepare_block_devices_before_forward(blocks, free_cache=False)

    assert offloader._cpu_weight_masters is not None
    masters = offloader._cpu_weight_masters[0]
    assert isinstance(masters["mlp.layer1"], torch.Tensor)
    assert isinstance(masters["mlp.layer2"], torch.Tensor)
    assert isinstance(masters["self_attn.qkv_proj"], torch.Tensor)
    assert isinstance(masters["self_attn.output_proj"], Int8BlockSwapCpuMaster)
    assert offloader._int8_quantized_tensors == 3

    events = [json.loads(line) for line in profile_path.read_text().splitlines()]
    config = [event for event in events if event["ev"] == "block_swap_config"][0]
    assert config["int8_scope"] == "self_attn_out"
    assert config["int8_quantized_tensors"] == 3


def test_int8_block_swap_restore_all_blocks_to_execution_dtype() -> None:
    blocks = nn.ModuleList([_Int8CandidateBlock() for _ in range(3)])
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=1,
        device=torch.device("cpu"),
        supports_backward=False,
        transfer_dtype="int8",
    )

    offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
    offloader.submit_move_blocks(blocks, 0)
    offloader.wait_for_block(2)
    assert blocks[0].mlp.layer1.weight.dtype == torch.int8

    offloader.restore_blocks_to_device(blocks, torch.device("cpu"))

    for block in blocks:
        assert block.mlp.layer1.weight.dtype == torch.float32
        assert block.mlp.layer2.weight.dtype == torch.float32
        assert block.self_attn.qkv_proj.weight.dtype == torch.float32
        assert block.adaln_up_mlp.weight.dtype == torch.float32
        assert block.adapter.weight.dtype == torch.float32
        assert block.adapter.weight.requires_grad


def test_block_swap_backward_next_use_wait_is_profiled(tmp_path) -> None:
    blocks = nn.ModuleList([_TinyBlock(), _TinyBlock(), _TinyBlock()])
    profile_path = tmp_path / "block_swap_profile.jsonl"
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=1,
        device=torch.device("cpu"),
        supports_backward=True,
        profile_jsonl=str(profile_path),
    )

    offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
    x = torch.ones(1, 2, requires_grad=True)
    y = x
    for block_idx, block in enumerate(blocks):
        offloader.wait_for_block(block_idx)
        y = block(y)
        offloader.submit_move_blocks(blocks, block_idx)
    y.sum().backward()

    events = [json.loads(line) for line in profile_path.read_text().splitlines()]
    backward_waits = [
        event
        for event in events
        if event["ev"] == "block_swap" and event["phase"] == "backward_wait"
    ]
    assert backward_waits
    assert backward_waits[0]["submit_phase"] == "backward_prefetch"
    assert backward_waits[0]["block_idx"] == 0


def test_cuda_block_swap_wait_uses_stream_event_without_host_sync(monkeypatch) -> None:
    class _FakeEvent:
        def __init__(self) -> None:
            self.synchronized = False

        def synchronize(self) -> None:
            self.synchronized = True
            raise AssertionError("production wait path must not host-synchronize")

    class _FakeStream:
        def __init__(self) -> None:
            self.waited_events = []

        def wait_event(self, event) -> None:
            self.waited_events.append(event)

    class _DoneFuture:
        def __init__(self, event: _FakeEvent) -> None:
            self.event = event

        def result(self):
            timings = {
                "d2h_ms": 0.0,
                "h2d_ms": 0.0,
                "event_wait_ms": 0.0,
                "_h2d_end_event": self.event,
            }
            return None, 0, timings, time.time()

    event = _FakeEvent()
    stream = _FakeStream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: stream)

    blocks = nn.ModuleList([_TinyBlock(), _TinyBlock(), _TinyBlock()])
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=1,
        device=torch.device("cuda"),
        supports_backward=False,
        profile_jsonl=None,
    )
    try:
        offloader.futures[0] = (
            _DoneFuture(event),
            {"phase": "forward_prefetch", "block_idx": 0, "queued_at": time.time()},
        )

        offloader._wait_blocks_move(0, phase="forward_wait")

        assert stream.waited_events == [event]
        assert event.synchronized is False
        assert offloader.futures == {}
    finally:
        offloader._stop_profile_poller()
        offloader.thread_pool.shutdown(wait=False)


def test_cuda_block_swap_profile_flush_avoids_inline_event_synchronize(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("ANIMA_BLOCK_SWAP_PROFILE_GPU_WAIT", "1")

    class _FakeEvent:
        def __init__(self, *, t_ms: float, ready: bool) -> None:
            self.t_ms = t_ms
            self.ready = ready
            self.synchronize_calls = 0
            self.recorded_streams = []

        def query(self) -> bool:
            return self.ready

        def synchronize(self) -> None:
            self.synchronize_calls += 1
            self.ready = True

        def record(self, stream) -> None:
            self.recorded_streams.append(stream)

        def elapsed_time(self, other) -> float:
            return float(other.t_ms - self.t_ms)

    class _FakeStream:
        def __init__(self) -> None:
            self.waited_events = []

        def wait_event(self, event) -> None:
            self.waited_events.append(event)

    class _DoneFuture:
        def __init__(self, ready_event, start_event, end_event) -> None:
            self.ready_event = ready_event
            self.start_event = start_event
            self.end_event = end_event

        def result(self):
            timings = {
                "d2h_ms": 0.0,
                "h2d_ms": 0.0,
                "event_wait_ms": 0.0,
                "enqueue_ms": 2.0,
                "_ready_event": self.ready_event,
                "_h2d_start_event": self.start_event,
                "_h2d_end_event": self.end_event,
                "_event_timing": True,
            }
            return None, 0, timings, time.time()

    stream = _FakeStream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: stream)

    profile_path = tmp_path / "block_swap_profile.jsonl"
    blocks = nn.ModuleList([_TinyBlock(), _TinyBlock(), _TinyBlock()])
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=1,
        device=torch.device("cuda"),
        supports_backward=False,
        profile_jsonl=str(profile_path),
    )
    try:
        ready_event = _FakeEvent(t_ms=0.0, ready=False)
        start_event = _FakeEvent(t_ms=1.5, ready=False)
        end_event = _FakeEvent(t_ms=14.0, ready=False)
        wait_start_event = _FakeEvent(t_ms=20.0, ready=False)
        wait_end_event = _FakeEvent(t_ms=24.0, ready=False)
        acquired_events = [wait_start_event, wait_end_event]
        monkeypatch.setattr(
            offloader,
            "_acquire_cuda_event",
            lambda *, enable_timing: acquired_events.pop(0),
        )
        monkeypatch.setattr(offloader, "_ensure_profile_poller", lambda: None)
        offloader.futures[0] = (
            _DoneFuture(ready_event, start_event, end_event),
            {"phase": "forward_prefetch", "block_idx": 0, "queued_at": time.time()},
        )

        offloader._wait_blocks_move(0, phase="forward_wait")

        assert stream.waited_events == [end_event]
        assert end_event.synchronize_calls == 0
        assert profile_path.exists() is False or profile_path.read_text() == ""

        ready_event.ready = True
        start_event.ready = True
        end_event.ready = True
        wait_start_event.ready = True
        wait_end_event.ready = True
        time.sleep(0.02)
        offloader.flush_profile_events(blocking=False)

        events = [json.loads(line) for line in profile_path.read_text().splitlines()]
        assert len(events) == 1
        event = events[0]
        assert event["phase"] == "forward_wait"
        assert event["submit_phase"] == "forward_prefetch"
        assert event["h2d_ms"] == pytest.approx(12.5)
        assert event["event_wait_ms"] == pytest.approx(1.5)
        assert event["gpu_wait_ms"] == pytest.approx(4.0)
        assert event["wait_ms"] == pytest.approx(
            event["host_wait_ms"] + event["gpu_wait_ms"]
        )
        assert event["host_queue_ms"] >= 0
        assert event["enqueue_to_ready_ms"] >= 0
        assert event["wait_requested_at"] <= event["ready_at"]
        assert event["waited_at"] > event["ready_at"]
        assert end_event.synchronize_calls == 0
    finally:
        offloader.thread_pool.shutdown(wait=False)


def test_block_swap_plan_is_cached(monkeypatch) -> None:
    blocks = nn.ModuleList([_TinyBlock(), _TinyBlock(), _TinyBlock()])
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=1,
        device=torch.device("cpu"),
        supports_backward=False,
    )
    build_calls: list[tuple[int, int]] = []
    original = offloader._build_swap_plan

    def wrapped_build_swap_plan(*args, **kwargs):
        build_calls.append((args[0], args[2]))
        return original(*args, **kwargs)

    monkeypatch.setattr(offloader, "_build_swap_plan", wrapped_build_swap_plan)
    try:
        offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
        offloader.swap_weight_devices(0, blocks[0], 2, blocks[2])
        offloader.swap_weight_devices(0, blocks[0], 2, blocks[2])

        assert build_calls == [(0, 2)]
    finally:
        offloader.thread_pool.shutdown(wait=False)


def test_cuda_block_swap_profile_gpu_wait_timing_is_opt_in(
    monkeypatch, tmp_path
) -> None:
    class _FakeEvent:
        def __init__(self, *, t_ms: float, ready: bool) -> None:
            self.t_ms = t_ms
            self.ready = ready
            self.recorded_streams = []

        def query(self) -> bool:
            return self.ready

        def record(self, stream) -> None:
            self.recorded_streams.append(stream)

        def elapsed_time(self, other) -> float:
            return float(other.t_ms - self.t_ms)

    class _FakeStream:
        def __init__(self) -> None:
            self.waited_events = []

        def wait_event(self, event) -> None:
            self.waited_events.append(event)

    class _DoneFuture:
        def __init__(self, ready_event, start_event, end_event) -> None:
            self.ready_event = ready_event
            self.start_event = start_event
            self.end_event = end_event

        def result(self):
            timings = {
                "d2h_ms": 0.0,
                "h2d_ms": 0.0,
                "event_wait_ms": 0.0,
                "enqueue_ms": 2.0,
                "_ready_event": self.ready_event,
                "_h2d_start_event": self.start_event,
                "_h2d_end_event": self.end_event,
                "_event_timing": True,
            }
            return None, 0, timings, time.time()

    stream = _FakeStream()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: stream)

    profile_path = tmp_path / "block_swap_profile.jsonl"
    blocks = nn.ModuleList([_TinyBlock(), _TinyBlock(), _TinyBlock()])
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=1,
        device=torch.device("cuda"),
        supports_backward=False,
        profile_jsonl=str(profile_path),
    )
    try:
        monkeypatch.setattr(offloader, "_ensure_profile_poller", lambda: None)

        def fail_acquire(*, enable_timing):
            raise AssertionError("GPU wait timing events should be opt-in")

        monkeypatch.setattr(offloader, "_acquire_cuda_event", fail_acquire)
        ready_event = _FakeEvent(t_ms=0.0, ready=True)
        start_event = _FakeEvent(t_ms=1.5, ready=True)
        end_event = _FakeEvent(t_ms=14.0, ready=True)
        offloader.futures[0] = (
            _DoneFuture(ready_event, start_event, end_event),
            {"phase": "forward_prefetch", "block_idx": 0, "queued_at": time.time()},
        )

        offloader._wait_blocks_move(0, phase="forward_wait")
        offloader.flush_profile_events(blocking=False)

        events = [json.loads(line) for line in profile_path.read_text().splitlines()]
        assert len(events) == 1
        assert stream.waited_events == [end_event]
        assert events[0]["gpu_wait_ms"] == 0.0
        assert events[0]["h2d_ms"] == pytest.approx(12.5)
    finally:
        offloader._stop_profile_poller()
        offloader.thread_pool.shutdown(wait=False)


def test_block_swap_slab_plan_tracks_offsets_and_total_numel() -> None:
    class _FakeWeight:
        def __init__(self, data):
            self.data = data
            self.shape = tuple(data.shape)

    class _FakeModule:
        def __init__(self, data) -> None:
            self.weight = _FakeWeight(data)

    class _FakeBlock:
        def __init__(self) -> None:
            self._module0 = _FakeModule(torch.empty(2, 3))
            self._module1 = _FakeModule(torch.empty(4, 1))

        def named_modules(self):
            return [
                ("", self),
                ("base0", self._module0),
                ("base1", self._module1),
            ]

    blocks = nn.ModuleList([_TinyBlock(), _TinyBlock(), _TinyBlock()])
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=1,
        device=torch.device("cpu"),
        supports_backward=False,
    )
    swap_plan = (
        (
            ("base0", torch.randn(2, 3), torch.randn(2, 3), torch.float32, torch.float32),
            ("base1", torch.randn(4, 1), torch.randn(4, 1), torch.float32, torch.float32),
        ),
        (),
    )
    try:
        slab_plan = offloader._get_swap_slab_plan(0, 2, swap_plan)
        slab_entries, slab_numel = slab_plan

        assert slab_numel == 10
        assert slab_entries == (
            ("base0", (0, 6, (2, 3), torch.float32)),
            ("base1", (6, 4, (4, 1), torch.float32)),
        )
        assert offloader._get_swap_slab_plan(0, 2, swap_plan) is slab_plan
    finally:
        offloader.thread_pool.shutdown(wait=False)


def test_block_swap_copy_stream_is_reused(monkeypatch) -> None:
    class _FakeTensorView:
        def __init__(self) -> None:
            self.copied_from = []
            self.recorded_streams = []

        def record_stream(self, stream) -> None:
            self.recorded_streams.append(stream)

        def copy_(self, source, non_blocking: bool = False):
            self.copied_from.append((source, non_blocking))
            return self

    class _FakeWeight:
        def __init__(self, data):
            self.data = data
            self.shape = (2, 2)

    class _FakeModule:
        def __init__(self, weight) -> None:
            self.weight = _FakeWeight(weight)

    class _FakeBlock:
        def __init__(self, weight) -> None:
            self._module = _FakeModule(weight)

        def named_modules(self):
            return [("", self), ("base", self._module)]

    class _FakeEvent:
        def __init__(self, enable_timing: bool = False) -> None:
            self.enable_timing = enable_timing
            self.recorded_on = []

        def record(self, stream) -> None:
            self.recorded_on.append(stream)

    class _FakeStream:
        def __init__(self, device=None) -> None:
            self.device = device
            self.waited_events = []

        def wait_event(self, event) -> None:
            self.waited_events.append(event)

    class _FakeStreamContext:
        def __init__(self, stream) -> None:
            self.stream = stream

        def __enter__(self):
            return self.stream

        def __exit__(self, exc_type, exc, tb):
            return False

    stream_creations: list[_FakeStream] = []

    def fake_stream(device=None):
        stream = _FakeStream(device=device)
        stream_creations.append(stream)
        return stream

    monkeypatch.setattr(torch.cuda, "Stream", fake_stream)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: _FakeStreamContext(stream))

    blocks = nn.ModuleList([_TinyBlock(), _TinyBlock(), _TinyBlock()])
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=1,
        device=torch.device("cuda"),
        supports_backward=False,
    )
    offloader._cpu_weight_masters = [{"base": torch.randn(2, 2)} for _ in range(3)]
    offloader._cpu_weight_master_dtypes = [{"base": torch.float32} for _ in range(3)]
    offloader._block_module_maps = [None for _ in range(3)]
    source_views = [_FakeTensorView(), _FakeTensorView()]
    block_to_cpu_a = _FakeBlock(source_views[0])
    block_to_cpu_b = _FakeBlock(source_views[1])
    block_to_cuda = _FakeBlock(_FakeTensorView())
    try:
        offloader.swap_weight_devices(0, block_to_cpu_a, 2, block_to_cuda)
        offloader.swap_weight_devices(1, block_to_cpu_b, 2, block_to_cuda)

        assert len(stream_creations) == 1
        assert source_views[0].recorded_streams == [stream_creations[0]]
        assert source_views[1].recorded_streams == [stream_creations[0]]
    finally:
        offloader.thread_pool.shutdown(wait=False)


def test_block_swap_event_pool_reuses_events() -> None:
    class _FakeEvent:
        def __init__(self, enable_timing: bool = False) -> None:
            self.enable_timing = enable_timing

    blocks = nn.ModuleList([_TinyBlock(), _TinyBlock(), _TinyBlock()])
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=1,
        device=torch.device("cuda"),
        supports_backward=False,
    )
    try:
        marker = _FakeEvent(enable_timing=False)
        timing = _FakeEvent(enable_timing=True)

        offloader._release_cuda_event(marker, enable_timing=False)
        offloader._release_cuda_event(timing, enable_timing=True)

        assert offloader._acquire_cuda_event(enable_timing=False) is marker
        assert offloader._acquire_cuda_event(enable_timing=True) is timing
    finally:
        offloader.thread_pool.shutdown(wait=False)


def test_block_swap_cached_cuda_prefers_foreach_copy(monkeypatch) -> None:
    class _FakeWeight:
        def __init__(self, data):
            self.data = data
            self.shape = (2, 2)

    class _FakeModule:
        def __init__(self, weight) -> None:
            self.weight = _FakeWeight(weight)

    class _FakeBlock:
        def __init__(self, weight0, weight1) -> None:
            self._module0 = _FakeModule(weight0)
            self._module1 = _FakeModule(weight1)

        def named_modules(self):
            return [
                ("", self),
                ("base0", self._module0),
                ("base1", self._module1),
            ]

    class _FakeEvent:
        def __init__(self, enable_timing: bool = False) -> None:
            self.enable_timing = enable_timing

        def record(self, stream) -> None:
            pass

    class _FakeStream:
        def __init__(self, device=None) -> None:
            self.device = device
            self.waited_events = []

        def wait_event(self, event) -> None:
            self.waited_events.append(event)

    class _FakeStreamContext:
        def __init__(self, stream) -> None:
            self.stream = stream

        def __enter__(self):
            return self.stream

        def __exit__(self, exc_type, exc, tb):
            return False

    foreach_calls = []

    def fake_foreach_copy(dsts, srcs, non_blocking=True):
        foreach_calls.append((list(dsts), list(srcs), non_blocking))
        return dsts

    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: _FakeStream(device=device))
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: _FakeStreamContext(stream))
    monkeypatch.setattr(torch, "_foreach_copy_", fake_foreach_copy)
    record_calls = []

    def fake_record_stream(tensor, stream) -> None:
        record_calls.append((tensor, stream))

    monkeypatch.setattr(torch.Tensor, "record_stream", fake_record_stream)

    blocks = nn.ModuleList([_TinyBlock(), _TinyBlock(), _TinyBlock()])
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=1,
        device=torch.device("cuda"),
        supports_backward=False,
    )
    offloader._cpu_weight_masters = [
        {"base0": torch.randn(2, 2), "base1": torch.randn(2, 2)}
        for _ in range(3)
    ]
    offloader._cpu_weight_master_dtypes = [
        {"base0": torch.float32, "base1": torch.float32}
        for _ in range(3)
    ]
    block_to_cpu = _FakeBlock(
        torch.empty(2, 2, device="cuda", dtype=torch.float32),
        torch.empty(2, 2, device="cuda", dtype=torch.float32),
    )
    block_to_cuda = _FakeBlock(
        torch.empty(2, 2, device="cuda", dtype=torch.float32),
        torch.empty(2, 2, device="cuda", dtype=torch.float32),
    )
    try:
        offloader.swap_weight_devices(0, block_to_cpu, 2, block_to_cuda)

        assert len(foreach_calls) == 1
        dsts, srcs, non_blocking = foreach_calls[0]
        assert len(dsts) == 2
        assert len(srcs) == 2
        assert non_blocking is True
        assert len(record_calls) == 2
    finally:
        offloader.thread_pool.shutdown(wait=False)


def test_block_swap_cached_cuda_falls_back_when_foreach_copy_is_incompatible(
    monkeypatch,
) -> None:
    class _FakeWeight:
        def __init__(self, data):
            self.data = data
            self.shape = (2, 2)

    class _FakeModule:
        def __init__(self, weight) -> None:
            self.weight = _FakeWeight(weight)

    class _FakeBlock:
        def __init__(self, weight0, weight1) -> None:
            self._module0 = _FakeModule(weight0)
            self._module1 = _FakeModule(weight1)

        def named_modules(self):
            return [
                ("", self),
                ("base0", self._module0),
                ("base1", self._module1),
            ]

    class _FakeEvent:
        def __init__(self, enable_timing: bool = False) -> None:
            self.enable_timing = enable_timing

        def record(self, stream) -> None:
            pass

    class _FakeStream:
        def __init__(self, device=None) -> None:
            self.device = device

        def wait_event(self, event) -> None:
            pass

    class _FakeStreamContext:
        def __init__(self, stream) -> None:
            self.stream = stream

        def __enter__(self):
            return self.stream

        def __exit__(self, exc_type, exc, tb):
            return False

    foreach_calls = []

    def fake_foreach_copy(dsts, srcs, non_blocking=True):
        foreach_calls.append((list(dsts), list(srcs), non_blocking))
        raise RuntimeError("force fallback")

    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: _FakeStream(device=device))
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: _FakeStreamContext(stream))
    monkeypatch.setattr(torch, "_foreach_copy_", fake_foreach_copy)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: None)

    blocks = nn.ModuleList([_TinyBlock(), _TinyBlock(), _TinyBlock()])
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=1,
        device=torch.device("cuda"),
        supports_backward=False,
    )
    offloader._cpu_weight_masters = [
        {"base0": torch.randn(2, 2, dtype=torch.float32), "base1": torch.randn(2, 2, dtype=torch.float32)}
        for _ in range(3)
    ]
    offloader._cpu_weight_master_dtypes = [
        {"base0": torch.float32, "base1": torch.float32}
        for _ in range(3)
    ]
    copy_calls = []
    original_copy = torch.Tensor.copy_

    def wrapped_copy(tensor, source, non_blocking: bool = False):
        copy_calls.append((tensor, source, non_blocking))
        return original_copy(tensor, source, non_blocking=non_blocking)

    monkeypatch.setattr(torch.Tensor, "copy_", wrapped_copy)
    block_to_cpu = _FakeBlock(
        torch.empty(2, 2, device="cuda", dtype=torch.float32),
        torch.empty(2, 2, device="cuda", dtype=torch.float32),
    )
    block_to_cuda = _FakeBlock(
        torch.empty(2, 2, device="cuda", dtype=torch.float32),
        torch.empty(2, 2, device="cuda", dtype=torch.float32),
    )
    try:
        offloader.swap_weight_devices(0, block_to_cpu, 2, block_to_cuda)

        assert len(foreach_calls) == 1
        assert len(copy_calls) == 2
        _, source0, non_blocking0 = copy_calls[0]
        _, source1, non_blocking1 = copy_calls[1]
        assert source0 is offloader._cpu_weight_masters[2]["base0"]
        assert source1 is offloader._cpu_weight_masters[2]["base1"]
        assert non_blocking0 is True
        assert non_blocking1 is True
    finally:
        offloader.thread_pool.shutdown(wait=False)


def test_block_swap_cached_cuda_int8_master_uses_dequant_restore(monkeypatch) -> None:
    import library.runtime.offloading as offloading_module

    class _FakeTensorView:
        def __init__(self) -> None:
            self.shape = (2, 2)
            self.recorded_streams = []
            self.copy_calls = []

        def record_stream(self, stream) -> None:
            self.recorded_streams.append(stream)

        def copy_(self, source, non_blocking: bool = False):
            self.copy_calls.append((source, non_blocking))
            return self

    class _FakeWeight:
        def __init__(self, data):
            self.data = data
            self.shape = (2, 2)

    class _FakeModule:
        def __init__(self, weight) -> None:
            self.weight = _FakeWeight(weight)

    class _FakeBlock:
        def __init__(self, weight) -> None:
            self._module = _FakeModule(weight)

        def named_modules(self):
            return [("", self), ("mlp.layer1", self._module)]

    class _FakeEvent:
        def __init__(self, enable_timing: bool = False) -> None:
            self.enable_timing = enable_timing

        def record(self, stream) -> None:
            pass

    class _FakeStream:
        def __init__(self, device=None) -> None:
            self.device = device

        def wait_event(self, event) -> None:
            pass

    class _FakeStreamContext:
        def __init__(self, stream) -> None:
            self.stream = stream

        def __enter__(self):
            return self.stream

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: _FakeStream(device=device))
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: _FakeStreamContext(stream))
    monkeypatch.setattr(
        torch,
        "_foreach_copy_",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("int8 master must not use foreach copy")
        ),
    )
    restored_marker = object()
    restore_calls = []

    def fake_restore_cpu_master_tensor(master, *, device, dtype, non_blocking=True):
        restore_calls.append((master, device, dtype, non_blocking))
        return restored_marker

    monkeypatch.setattr(
        offloading_module,
        "_restore_cpu_master_tensor",
        fake_restore_cpu_master_tensor,
    )

    int8_master = Int8BlockSwapCpuMaster(
        quantized=torch.ones(2, 2, dtype=torch.int8),
        scale=torch.ones(2, dtype=torch.float32),
        shape=(2, 2),
    )
    source_view = _FakeTensorView()
    target_view = _FakeTensorView()
    blocks = nn.ModuleList([_TinyBlock(), _TinyBlock(), _TinyBlock()])
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=1,
        device=torch.device("cuda"),
        supports_backward=False,
        restore_mode="slab",
    )
    offloader._cpu_weight_masters = [
        {"mlp.layer1": int8_master},
        {"mlp.layer1": int8_master},
        {"mlp.layer1": int8_master},
    ]
    offloader._cpu_weight_master_dtypes = [
        {"mlp.layer1": torch.float32},
        {"mlp.layer1": torch.float32},
        {"mlp.layer1": torch.float32},
    ]
    offloader._cpu_weight_master_slabs = [None, None, None]
    offloader._cpu_weight_master_slab_plans = [None, None, None]
    block_to_cpu = _FakeBlock(source_view)
    block_to_cuda = _FakeBlock(target_view)
    try:
        offloader.swap_weight_devices(0, block_to_cpu, 2, block_to_cuda)

        assert restore_calls == [
            (int8_master, torch.device("cuda"), torch.float32, True)
        ]
        assert source_view.copy_calls == [(restored_marker, True)]
        assert block_to_cpu._module.weight.data is int8_master.quantized
        assert block_to_cuda._module.weight.data is source_view
    finally:
        offloader.thread_pool.shutdown(wait=False)


def test_block_swap_cached_cuda_int8_direct_bind_skips_extra_copy(monkeypatch) -> None:
    import library.runtime.offloading as offloading_module

    class _FakeTensorView:
        def __init__(self) -> None:
            self.shape = (2, 2)
            self.recorded_streams = []
            self.copy_calls = []

        def record_stream(self, stream) -> None:
            self.recorded_streams.append(stream)

        def copy_(self, source, non_blocking: bool = False):
            self.copy_calls.append((source, non_blocking))
            return self

    class _FakeWeight:
        def __init__(self, data):
            self.data = data
            self.shape = (2, 2)

    class _FakeModule:
        def __init__(self, weight) -> None:
            self.weight = _FakeWeight(weight)

    class _FakeBlock:
        def __init__(self, weight) -> None:
            self._module = _FakeModule(weight)

        def named_modules(self):
            return [("", self), ("mlp.layer1", self._module)]

    class _FakeEvent:
        def __init__(self, enable_timing: bool = False) -> None:
            self.enable_timing = enable_timing

        def record(self, stream) -> None:
            pass

    class _FakeStream:
        def __init__(self, device=None) -> None:
            self.device = device

        def wait_event(self, event) -> None:
            pass

    class _FakeStreamContext:
        def __init__(self, stream) -> None:
            self.stream = stream

        def __enter__(self):
            return self.stream

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: _FakeStream(device=device))
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: _FakeStreamContext(stream))
    monkeypatch.setattr(
        torch,
        "_foreach_copy_",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("int8 master must not use foreach copy")
        ),
    )
    restored_marker = object()
    restore_calls = []

    def fake_restore_cpu_master_tensor(master, *, device, dtype, non_blocking=True):
        restore_calls.append((master, device, dtype, non_blocking))
        return restored_marker

    monkeypatch.setattr(
        offloading_module,
        "_restore_cpu_master_tensor",
        fake_restore_cpu_master_tensor,
    )

    int8_master = Int8BlockSwapCpuMaster(
        quantized=torch.ones(2, 2, dtype=torch.int8),
        scale=torch.ones(2, dtype=torch.float32),
        shape=(2, 2),
    )
    source_view = _FakeTensorView()
    target_view = _FakeTensorView()
    blocks = nn.ModuleList([_TinyBlock(), _TinyBlock(), _TinyBlock()])
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=1,
        device=torch.device("cuda"),
        supports_backward=False,
        restore_mode="slab",
        int8_restore_mode="direct_bind",
    )
    offloader._cpu_weight_masters = [
        {"mlp.layer1": int8_master},
        {"mlp.layer1": int8_master},
        {"mlp.layer1": int8_master},
    ]
    offloader._cpu_weight_master_dtypes = [
        {"mlp.layer1": torch.float32},
        {"mlp.layer1": torch.float32},
        {"mlp.layer1": torch.float32},
    ]
    offloader._cpu_weight_master_slabs = [None, None, None]
    offloader._cpu_weight_master_slab_plans = [None, None, None]
    block_to_cpu = _FakeBlock(source_view)
    block_to_cuda = _FakeBlock(target_view)
    try:
        offloader.swap_weight_devices(0, block_to_cpu, 2, block_to_cuda)

        assert restore_calls == [
            (int8_master, torch.device("cuda"), torch.float32, True)
        ]
        assert source_view.copy_calls == []
        assert block_to_cpu._module.weight.data is int8_master.quantized
        assert block_to_cuda._module.weight.data is restored_marker
    finally:
        offloader.thread_pool.shutdown(wait=False)


def test_block_swap_cached_cuda_int8_reuse_storage_restores_into_old_view(
    monkeypatch,
) -> None:
    import library.runtime.offloading as offloading_module

    class _FakeTensorView:
        def __init__(self) -> None:
            self.shape = (2, 2)
            self.recorded_streams = []
            self.copy_calls = []

        def record_stream(self, stream) -> None:
            self.recorded_streams.append(stream)

        def copy_(self, source, non_blocking: bool = False):
            self.copy_calls.append((source, non_blocking))
            return self

    class _FakeWeight:
        def __init__(self, data):
            self.data = data
            self.shape = (2, 2)

    class _FakeModule:
        def __init__(self, weight) -> None:
            self.weight = _FakeWeight(weight)

    class _FakeBlock:
        def __init__(self, weight) -> None:
            self._module = _FakeModule(weight)

        def named_modules(self):
            return [("", self), ("mlp.layer1", self._module)]

    class _FakeEvent:
        def __init__(self, enable_timing: bool = False) -> None:
            self.enable_timing = enable_timing

        def record(self, stream) -> None:
            pass

    class _FakeStream:
        def __init__(self, device=None) -> None:
            self.device = device

        def wait_event(self, event) -> None:
            pass

    class _FakeStreamContext:
        def __init__(self, stream) -> None:
            self.stream = stream

        def __enter__(self):
            return self.stream

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: _FakeStream(device=device))
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: _FakeStreamContext(stream))
    monkeypatch.setattr(
        torch,
        "_foreach_copy_",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("int8 master must not use foreach copy")
        ),
    )
    restore_into_calls = []

    def fake_restore_int8_cpu_master_into_tensor(
        master,
        dst,
        *,
        device,
        dtype,
        non_blocking=True,
        chunk_rows=0,
    ):
        restore_into_calls.append((master, dst, device, dtype, non_blocking, chunk_rows))
        return dst

    monkeypatch.setattr(
        offloading_module,
        "_restore_int8_cpu_master_into_tensor",
        fake_restore_int8_cpu_master_into_tensor,
    )

    int8_master = Int8BlockSwapCpuMaster(
        quantized=torch.ones(2, 2, dtype=torch.int8),
        scale=torch.ones(2, dtype=torch.float32),
        shape=(2, 2),
    )
    source_view = _FakeTensorView()
    target_view = _FakeTensorView()
    blocks = nn.ModuleList([_TinyBlock(), _TinyBlock(), _TinyBlock()])
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=1,
        device=torch.device("cuda"),
        supports_backward=False,
        restore_mode="slab",
        int8_restore_mode="reuse_storage",
        int8_restore_chunk_rows=2,
    )
    offloader._cpu_weight_masters = [
        {"mlp.layer1": int8_master},
        {"mlp.layer1": int8_master},
        {"mlp.layer1": int8_master},
    ]
    offloader._cpu_weight_master_dtypes = [
        {"mlp.layer1": torch.float32},
        {"mlp.layer1": torch.float32},
        {"mlp.layer1": torch.float32},
    ]
    offloader._cpu_weight_master_slabs = [None, None, None]
    offloader._cpu_weight_master_slab_plans = [None, None, None]
    block_to_cpu = _FakeBlock(source_view)
    block_to_cuda = _FakeBlock(target_view)
    try:
        offloader.swap_weight_devices(0, block_to_cpu, 2, block_to_cuda)

        assert restore_into_calls == [
            (int8_master, source_view, torch.device("cuda"), torch.float32, True, 2)
        ]
        assert source_view.copy_calls == []
        assert block_to_cpu._module.weight.data is int8_master.quantized
        assert block_to_cuda._module.weight.data is source_view
    finally:
        offloader.thread_pool.shutdown(wait=False)


def test_block_swap_cached_cuda_slab_restore_uses_single_gpu_slab(monkeypatch) -> None:
    class _FakeWeight:
        def __init__(self, data):
            self.data = data
            self.shape = tuple(data.shape)

    class _FakeModule:
        def __init__(self, weight) -> None:
            self.weight = _FakeWeight(weight)

    class _FakeBlock:
        def __init__(self, weight0, weight1) -> None:
            self._module0 = _FakeModule(weight0)
            self._module1 = _FakeModule(weight1)

        def named_modules(self):
            return [
                ("", self),
                ("base0", self._module0),
                ("base1", self._module1),
            ]

    class _FakeEvent:
        def __init__(self, enable_timing: bool = False) -> None:
            self.enable_timing = enable_timing

        def record(self, stream) -> None:
            pass

    class _FakeStream:
        def __init__(self, device=None) -> None:
            self.device = device

        def wait_event(self, event) -> None:
            pass

    class _FakeStreamContext:
        def __init__(self, stream) -> None:
            self.stream = stream

        def __enter__(self):
            return self.stream

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: _FakeStream(device=device))
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: _FakeStreamContext(stream))

    foreach_calls = []

    def fake_foreach_copy(dsts, srcs, non_blocking=True):
        foreach_calls.append((list(dsts), list(srcs), non_blocking))
        return dsts

    monkeypatch.setattr(torch, "_foreach_copy_", fake_foreach_copy)

    copy_calls = []
    original_copy = torch.Tensor.copy_

    def wrapped_copy(tensor, source, non_blocking: bool = False):
        copy_calls.append((tensor, source, non_blocking))
        return original_copy(tensor, source, non_blocking=non_blocking)

    monkeypatch.setattr(torch.Tensor, "copy_", wrapped_copy)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: None)

    blocks = nn.ModuleList([_TinyBlock(), _TinyBlock(), _TinyBlock()])
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=1,
        device=torch.device("cuda"),
        supports_backward=False,
        restore_mode="slab",
    )
    offloader._cpu_weight_masters = [
        {
            "base0": torch.randn(2, 2).pin_memory(),
            "base1": torch.randn(2, 2).pin_memory(),
        }
        for _ in range(3)
    ]
    offloader._cpu_weight_master_dtypes = [
        {"base0": torch.float32, "base1": torch.float32}
        for _ in range(3)
    ]
    monkeypatch.setattr(torch.Tensor, "copy_", original_copy)
    offloader._cpu_weight_master_slabs = []
    offloader._cpu_weight_master_slab_plans = []
    for masters in offloader._cpu_weight_masters:
        slab_views, slab, slab_plan = offloader._pack_cpu_master_block(masters, pin_memory=True)
        offloader._cpu_weight_masters[offloader._cpu_weight_master_slabs.__len__()] = slab_views
        offloader._cpu_weight_master_slabs.append(slab)
        offloader._cpu_weight_master_slab_plans.append(slab_plan)
    monkeypatch.setattr(torch.Tensor, "copy_", wrapped_copy)

    block_to_cpu = _FakeBlock(
        torch.empty(2, 2, device="cuda", dtype=torch.float32),
        torch.empty(2, 2, device="cuda", dtype=torch.float32),
    )
    block_to_cuda = _FakeBlock(
        torch.empty(2, 2, device="cuda", dtype=torch.float32),
        torch.empty(2, 2, device="cuda", dtype=torch.float32),
    )
    try:
        offloader.swap_weight_devices(0, block_to_cpu, 2, block_to_cuda)

        assert len(foreach_calls) == 0
        assert len(copy_calls) == 1
        copied_tensor, copied_source, non_blocking = copy_calls[0]
        assert copied_tensor.ndim == 1
        assert copied_source.ndim == 1
        assert copied_tensor.numel() == 8
        assert copied_source.numel() == 8
        assert non_blocking is True
        assert (
            block_to_cuda._module0.weight.data.untyped_storage().data_ptr()
            == block_to_cuda._module1.weight.data.untyped_storage().data_ptr()
        )
    finally:
        offloader.thread_pool.shutdown(wait=False)


def test_block_swap_slab_gpu_cache_reuses_physical_slot() -> None:
    blocks = nn.ModuleList([_TinyBlock() for _ in range(6)])
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=4,
        device=torch.device("cpu"),
        supports_backward=False,
        restore_mode="slab",
    )
    swap_plan = (
        (("base", torch.randn(2, 2), torch.randn(2, 2), torch.float32, torch.float32),),
        (),
    )
    try:
        offloader._cpu_weight_master_slabs = [None for _ in range(6)]
        offloader._cpu_weight_master_slab_plans = [None for _ in range(6)]
        for block_idx in range(6):
            masters = {"base": torch.randn(2, 2)}
            slab_views, slab, slab_plan = offloader._pack_cpu_master_block(masters, pin_memory=False)
            offloader._cpu_weight_master_slabs[block_idx] = slab
            offloader._cpu_weight_master_slab_plans[block_idx] = slab_plan
        bundle_a = offloader._get_cached_restore_slab(0, 2, swap_plan)
        bundle_b = offloader._get_cached_restore_slab(2, 4, swap_plan)
        assert bundle_a is not None
        assert bundle_b is not None
        _, gpu_slab_a, _ = bundle_a
        _, gpu_slab_b, _ = bundle_b
        assert gpu_slab_a is gpu_slab_b
        assert len(offloader._swap_gpu_slab_cache) == 1
    finally:
        offloader.thread_pool.shutdown(wait=False)


def test_block_swap_backward_hooks_work_with_standard_checkpointing(tmp_path) -> None:
    blocks = nn.ModuleList([_TinyBlock() for _ in range(6)])
    profile_path = tmp_path / "block_swap_profile.jsonl"
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=2,
        device=torch.device("cpu"),
        supports_backward=True,
        profile_jsonl=str(profile_path),
    )

    offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
    x = torch.ones(1, 2, requires_grad=True)
    y = x
    for block_idx, block in enumerate(blocks):
        offloader.wait_for_block(block_idx)
        y = torch_checkpoint(block, y, use_reentrant=False)
        offloader.submit_move_blocks(blocks, block_idx)
    y.sum().backward()

    assert x.grad is not None
    events = [json.loads(line) for line in profile_path.read_text().splitlines()]
    assert any(
        event["ev"] == "block_swap"
        and event["phase"] == "backward_wait"
        and event["submit_phase"] == "backward_prefetch"
        for event in events
    )


def test_backward_hooks_prefetch_from_completed_tail_block() -> None:
    blocks = nn.ModuleList([_TinyBlock() for _ in range(6)])
    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=2,
        device=torch.device("cpu"),
        supports_backward=True,
    )
    submitted: list[tuple[int, int, str]] = []
    waited: list[tuple[int, str]] = []

    def fake_submit(blocks_arg, block_idx_to_cpu, block_idx_to_cuda, *, phase):
        submitted.append((block_idx_to_cpu, block_idx_to_cuda, phase))

    def fake_wait(block_idx, *, phase=""):
        waited.append((block_idx, phase))

    offloader._submit_move_blocks = fake_submit  # type: ignore[method-assign]
    offloader._wait_blocks_move = fake_wait  # type: ignore[method-assign]

    hook_5 = offloader.create_backward_hook(blocks, 5)
    hook_4 = offloader.create_backward_hook(blocks, 4)
    hook_2 = offloader.create_backward_hook(blocks, 2)
    assert hook_5 is not None
    assert hook_4 is not None
    assert hook_2 is not None

    hook_5(blocks[5], (), ())
    hook_4(blocks[4], (), ())
    hook_2(blocks[2], (), ())

    assert submitted == [
        (5, 1, "backward_prefetch"),
        (4, 0, "backward_prefetch"),
    ]
    assert waited == [(1, "backward_wait")]


class _FakeCheckpointBlock:
    def __init__(self) -> None:
        self.disable_gradient_checkpointing()

    def enable_gradient_checkpointing(
        self, cpu_offload: bool = False, unsloth_offload: bool = False
    ) -> None:
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload
        self.unsloth_offload_checkpointing = unsloth_offload
        self.adapter_aware_checkpointing = False
        self.mlp_checkpointing = False
        self.mlp_layer1_checkpointing = False

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.unsloth_offload_checkpointing = False
        self.adapter_aware_checkpointing = False
        self.mlp_checkpointing = False
        self.mlp_layer1_checkpointing = False

    def enable_mlp_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.unsloth_offload_checkpointing = False
        self.adapter_aware_checkpointing = False
        self.mlp_checkpointing = True
        self.mlp_layer1_checkpointing = False

    def enable_mlp_layer1_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.unsloth_offload_checkpointing = False
        self.adapter_aware_checkpointing = False
        self.mlp_checkpointing = False
        self.mlp_layer1_checkpointing = True

    def enable_adapter_aware_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.unsloth_offload_checkpointing = False
        self.adapter_aware_checkpointing = True
        self.mlp_checkpointing = False
        self.mlp_layer1_checkpointing = False


def test_selective_checkpoint_modes_set_block_flags() -> None:
    from library.anima.models import Anima

    model = object.__new__(Anima)
    model.blocks = [_FakeCheckpointBlock() for _ in range(4)]

    Anima.enable_selective_checkpointing(model, "every_other")
    assert model.selective_checkpoint == "every_other"
    assert [block.gradient_checkpointing for block in model.blocks] == [
        True,
        False,
        True,
        False,
    ]
    assert not any(block.mlp_checkpointing for block in model.blocks)
    assert not any(block.cpu_offload_checkpointing for block in model.blocks)
    assert not any(block.unsloth_offload_checkpointing for block in model.blocks)
    assert not any(block.adapter_aware_checkpointing for block in model.blocks)

    Anima.enable_selective_checkpointing(model, "adapter_aware")
    assert model.selective_checkpoint == "adapter_aware"
    assert not any(block.gradient_checkpointing for block in model.blocks)
    assert all(block.adapter_aware_checkpointing for block in model.blocks)
    assert not any(block.mlp_checkpointing for block in model.blocks)
    assert not any(block.mlp_layer1_checkpointing for block in model.blocks)

    Anima.enable_selective_checkpointing(model, "mlp_only")
    assert model.selective_checkpoint == "mlp_only"
    assert not any(block.gradient_checkpointing for block in model.blocks)
    assert all(block.mlp_checkpointing for block in model.blocks)
    assert not any(block.mlp_layer1_checkpointing for block in model.blocks)

    Anima.enable_selective_checkpointing(model, "mlp_layer1_only")
    assert model.selective_checkpoint == "mlp_layer1_only"
    assert not any(block.gradient_checkpointing for block in model.blocks)
    assert not any(block.mlp_checkpointing for block in model.blocks)
    assert all(block.mlp_layer1_checkpointing for block in model.blocks)

    Anima.enable_selective_checkpointing(
        model,
        "peak_blocks_mlp_layer1",
        blocks="1-2",
    )
    assert model.selective_checkpoint == "peak_blocks_mlp_layer1"
    assert [block.mlp_layer1_checkpointing for block in model.blocks] == [
        False,
        True,
        True,
        False,
    ]
    assert not any(block.mlp_checkpointing for block in model.blocks)

    Anima.enable_selective_checkpointing(model, "peak_blocks_mlp", blocks="2,3")
    assert model.selective_checkpoint == "peak_blocks_mlp"
    assert [block.mlp_checkpointing for block in model.blocks] == [
        False,
        False,
        True,
        True,
    ]
    assert not any(block.mlp_layer1_checkpointing for block in model.blocks)

    Anima.enable_selective_checkpointing(
        model,
        "peak_blocks_adapter_aware",
        blocks="0,3",
    )
    assert model.selective_checkpoint == "peak_blocks_adapter_aware"
    assert [block.adapter_aware_checkpointing for block in model.blocks] == [
        True,
        False,
        False,
        True,
    ]
    assert not any(block.gradient_checkpointing for block in model.blocks)

    Anima.enable_selective_checkpointing(
        model,
        "peak_blocks_adapter_aware",
        blocks="",
    )
    assert [block.adapter_aware_checkpointing for block in model.blocks] == [
        False,
        True,
        True,
        True,
    ]

    Anima.enable_selective_checkpointing(model, "off")
    assert model.selective_checkpoint == "off"
    assert not any(block.gradient_checkpointing for block in model.blocks)
    assert not any(block.mlp_checkpointing for block in model.blocks)
    assert not any(block.mlp_layer1_checkpointing for block in model.blocks)
    assert not any(block.adapter_aware_checkpointing for block in model.blocks)


def test_selective_checkpoint_rejects_invalid_peak_block_index() -> None:
    from library.anima.models import Anima

    model = object.__new__(Anima)
    model.blocks = [_FakeCheckpointBlock() for _ in range(4)]

    with pytest.raises(ValueError, match="invalid block index"):
        Anima.enable_selective_checkpointing(
            model,
            "peak_blocks_adapter_aware",
            blocks="4",
        )


def test_peak_probe_attachment_sets_block_indices() -> None:
    from library.anima.models import Anima

    model = object.__new__(Anima)
    model.blocks = [_FakeCheckpointBlock() for _ in range(3)]
    probe = object()

    Anima.enable_peak_probe(model, probe)

    assert model._peak_probe is probe
    assert [block._peak_probe for block in model.blocks] == [probe, probe, probe]
    assert [block._block_idx for block in model.blocks] == [0, 1, 2]


def test_adapter_aware_checkpoint_policy_saves_only_small_trainable_ops() -> None:
    import torch
    from torch.utils.checkpoint import CheckpointPolicy

    from library.anima.models import _adapter_aware_checkpoint_policy

    class _Ctx:
        is_recompute = True
        op_output = None

    policy = _adapter_aware_checkpoint_policy(max_save_numel=128)
    small_x = torch.randn(4, 32, requires_grad=True)
    small_w = torch.randn(32, 8, requires_grad=True)
    large_w = torch.randn(32, 1024, requires_grad=True)
    frozen_x = torch.randn(4, 32)
    frozen_w = torch.randn(32, 8)

    assert (
        policy(_Ctx(), torch.ops.aten.mm.default, small_x, small_w)
        is CheckpointPolicy.MUST_SAVE
    )
    assert (
        policy(_Ctx(), torch.ops.aten.mm.default, small_x, large_w)
        is CheckpointPolicy.PREFER_RECOMPUTE
    )
    assert (
        policy(_Ctx(), torch.ops.aten.mm.default, frozen_x, frozen_w)
        is CheckpointPolicy.PREFER_RECOMPUTE
    )

    gate = torch.randn(2, 4, requires_grad=True)
    routed_weight = torch.randn(4, 8, 3, requires_grad=True)
    for equation in (
        "be,eor->bor",
        "be,eod->bod",
        "bc,cor->bor",
        "bf,for->bor",
    ):
        assert (
            policy(_Ctx(), torch.ops.aten.einsum.default, equation, [gate, routed_weight])
            is CheckpointPolicy.MUST_SAVE
        )


def test_adapter_aware_checkpoint_matches_eager_block_forward_backward() -> None:
    import copy

    import torch

    from library.anima.models import Block
    from networks.attention_dispatch import AttentionParams
    from networks.lora_modules.lora import LoRAModule

    def attach_loras(block: Block) -> None:
        for name, linear in [
            ("self_qkv", block.self_attn.qkv_proj),
            ("cross_q", block.cross_attn.q_proj),
            ("mlp1", block.mlp.layer1),
        ]:
            module = LoRAModule(name, linear, lora_dim=2, alpha=2)
            module.apply_to()
            block.add_module(f"_test_lora_{name}", module)

    torch.manual_seed(0)
    eager = Block(x_dim=8, context_dim=8, num_heads=2, mlp_ratio=2.0)
    attach_loras(eager)
    eager.train()
    checkpointed = copy.deepcopy(eager)
    checkpointed.enable_adapter_aware_checkpointing()

    x = torch.randn(2, 1, 3, 1, 8, requires_grad=True)
    emb = torch.randn(2, 1, 8, requires_grad=True)
    context = torch.randn(2, 5, 8, requires_grad=True)
    x_ckpt = x.detach().clone().requires_grad_(True)
    emb_ckpt = emb.detach().clone().requires_grad_(True)
    context_ckpt = context.detach().clone().requires_grad_(True)
    attn_params = AttentionParams.create_attention_params("torch")

    eager_out = eager(x, emb, context, attn_params)
    ckpt_out = checkpointed(x_ckpt, emb_ckpt, context_ckpt, attn_params)
    torch.testing.assert_close(ckpt_out, eager_out)

    eager_out.square().mean().backward()
    ckpt_out.square().mean().backward()

    torch.testing.assert_close(x_ckpt.grad, x.grad)
    torch.testing.assert_close(emb_ckpt.grad, emb.grad)
    torch.testing.assert_close(context_ckpt.grad, context.grad)
    for (name, eager_param), (ckpt_name, ckpt_param) in zip(
        eager.named_parameters(), checkpointed.named_parameters()
    ):
        assert ckpt_name == name
        if eager_param.grad is None and ckpt_param.grad is None:
            continue
        torch.testing.assert_close(ckpt_param.grad, eager_param.grad)


def test_adapter_aware_checkpoint_invokes_selective_policy(monkeypatch) -> None:
    import torch
    from torch.utils.checkpoint import CheckpointPolicy

    import library.anima.models as anima_models
    from library.anima.models import Block
    from networks.attention_dispatch import AttentionParams
    from networks.lora_modules.lora import LoRAModule

    seen: list[tuple[str, CheckpointPolicy, bool]] = []

    def recording_context(max_save_numel: int = 16):
        base_policy = anima_models._adapter_aware_checkpoint_policy(max_save_numel)

        def policy(ctx, op, *args, **kwargs):
            result = base_policy(ctx, op, *args, **kwargs)
            seen.append((str(op), result, bool(getattr(ctx, "is_recompute", False))))
            return result

        return anima_models.create_selective_checkpoint_contexts(policy)

    monkeypatch.setattr(
        anima_models,
        "_adapter_aware_checkpoint_context",
        recording_context,
    )

    torch.manual_seed(0)
    block = Block(x_dim=8, context_dim=8, num_heads=2, mlp_ratio=2.0)
    module = LoRAModule("mlp1", block.mlp.layer1, lora_dim=2, alpha=2)
    module.apply_to()
    block.add_module("_test_lora_mlp1", module)
    block.train()
    block.enable_adapter_aware_checkpointing(max_save_numel=16)

    x = torch.randn(2, 1, 3, 1, 8, requires_grad=True)
    emb = torch.randn(2, 1, 8, requires_grad=True)
    context = torch.randn(2, 5, 8, requires_grad=True)
    attn_params = AttentionParams.create_attention_params("torch")

    out = block(x, emb, context, attn_params)
    out.square().mean().backward()

    assert any(result is CheckpointPolicy.MUST_SAVE for _, result, _ in seen)
    assert any(result is CheckpointPolicy.PREFER_RECOMPUTE for _, result, _ in seen)
    assert any(is_recompute for _, _, is_recompute in seen)


@pytest.mark.parametrize(
    ("checkpoint_mode", "enable_fn"),
    [
        ("eager", None),
        ("adapter_aware", lambda block: block.enable_adapter_aware_checkpointing()),
        ("grad_ckpt", lambda block: block.enable_gradient_checkpointing()),
        (
            "cpu_offload_ckpt",
            lambda block: block.enable_gradient_checkpointing(cpu_offload=True),
        ),
        (
            "unsloth_ckpt",
            lambda block: block.enable_gradient_checkpointing(unsloth_offload=True),
        ),
    ],
)
def test_block_forward_preserves_use_fp32_across_wrappers(
    monkeypatch,
    checkpoint_mode: str,
    enable_fn,
) -> None:
    import library.anima.models as anima_models
    from library.anima.models import Block
    from networks.attention_dispatch import AttentionParams

    seen: list[bool] = []

    def fake_forward(
        self,
        x_B_T_H_W_D,
        emb_B_T_D,
        crossattn_emb,
        attn_params,
        rope_cos_sin=None,
        adaln_lora_B_T_3D=None,
        use_fp32: bool = False,
    ):
        seen.append(use_fp32)
        return x_B_T_H_W_D

    monkeypatch.setattr(Block, "_forward", fake_forward)
    monkeypatch.setattr(
        anima_models,
        "torch_checkpoint",
        lambda func, *args, **kwargs: func(*args),
    )
    monkeypatch.setattr(
        anima_models,
        "unsloth_checkpoint",
        lambda func, *args, **kwargs: func(*args),
    )

    block = Block(x_dim=8, context_dim=8, num_heads=2, mlp_ratio=2.0)
    block.train()
    if enable_fn is not None:
        enable_fn(block)

    x = torch.randn(2, 1, 3, 1, 8, requires_grad=True)
    emb = torch.randn(2, 1, 8, requires_grad=True)
    context = torch.randn(2, 5, 8, requires_grad=True)
    attn_params = AttentionParams.create_attention_params("torch")

    out = block(x, emb, context, attn_params, use_fp32=True)

    assert seen
    assert all(flag is True for flag in seen)
    assert out is x


def test_block_swap_compile_mode_is_downgraded_for_cudagraph_modes(caplog) -> None:
    import train

    args = _args_for_assert_extra(
        cache_text_encoder_outputs_to_disk=False,
        cache_text_encoder_outputs=False,
        cache_llm_adapter_outputs=False,
        network_train_unet_only=True,
        cpu_offload_checkpointing=False,
        unsloth_offload_checkpointing=False,
        gradient_checkpointing=False,
        blocks_to_swap=8,
        torch_compile=True,
        dynamo_backend="inductor",
        compile_inductor_mode="reduce-overhead",
    )

    train.AnimaTrainer().assert_extra_args(args, _CacheableDataset(), None)

    assert args.torch_compile is True
    assert args.compile_inductor_mode is None
    assert any("CUDAGraph" in rec.getMessage() for rec in caplog.records)


def test_block_swap_max_autotune_uses_no_cudagraph_compile_mode(caplog) -> None:
    import train

    args = _args_for_assert_extra(
        blocks_to_swap=8,
        torch_compile=True,
        dynamo_backend="inductor",
        compile_inductor_mode="max-autotune",
    )

    train.AnimaTrainer().assert_extra_args(args, _CacheableDataset(), None)

    assert args.compile_inductor_mode == "max-autotune-no-cudagraphs"
    assert any("CUDAGraph" in rec.getMessage() for rec in caplog.records)


def test_lokr_full_checkpoint_keeps_torch_compile(caplog) -> None:
    import train

    args = _args_for_assert_extra(
        blocks_to_swap=8,
        gradient_checkpointing=True,
        cpu_offload_checkpointing=False,
        unsloth_offload_checkpointing=False,
        use_lokr=True,
        torch_compile=True,
        compile_dynamic_seq=True,
    )

    train.AnimaTrainer().assert_extra_args(args, _CacheableDataset(), None)

    assert args.torch_compile is True
    assert any(
        "LoKr" in rec.getMessage() and "Dynamo graph budget" in rec.getMessage()
        for rec in caplog.records
    )


def test_block_swap_rejects_soft_tokens_multi_forward_override() -> None:
    import train

    args = _args_for_assert_extra(
        blocks_to_swap=8,
        network_module="networks.methods.soft_tokens",
        network_args=[],
    )

    with pytest.raises(ValueError, match="soft_tokens"):
        train.AnimaTrainer().assert_extra_args(args, _CacheableDataset(), None)


def test_block_swap_rejects_cpu_activation_offload() -> None:
    import train

    args = _args_for_assert_extra(
        blocks_to_swap=8,
        cpu_offload_checkpointing=True,
    )

    with pytest.raises(ValueError, match="cpu_offload_checkpointing"):
        train.AnimaTrainer().assert_extra_args(args, _CacheableDataset(), None)


def test_block_swap_rejects_unsloth_activation_offload() -> None:
    import train

    args = _args_for_assert_extra(
        blocks_to_swap=8,
        gradient_checkpointing=True,
        unsloth_offload_checkpointing=True,
    )

    with pytest.raises(ValueError, match="unsloth_offload_checkpointing"):
        train.AnimaTrainer().assert_extra_args(args, _CacheableDataset(), None)


def test_block_swap_allows_standard_gradient_checkpointing() -> None:
    import train

    args = _args_for_assert_extra(
        blocks_to_swap=8,
        gradient_checkpointing=True,
        cpu_offload_checkpointing=False,
        unsloth_offload_checkpointing=False,
    )

    train.AnimaTrainer().assert_extra_args(args, _CacheableDataset(), None)


def test_block_swap_allows_selective_mlp_checkpointing() -> None:
    import train

    args = _args_for_assert_extra(
        blocks_to_swap=12,
        gradient_checkpointing=False,
        selective_checkpoint="mlp_only",
    )

    train.AnimaTrainer().assert_extra_args(args, _CacheableDataset(), None)


def test_training_args_reject_negative_blocks_to_swap() -> None:
    import train

    from library.training.cli_args import verify_training_args

    args = train.setup_parser().parse_args([])
    args.blocks_to_swap = -1

    with pytest.raises(ValueError, match="blocks_to_swap"):
        verify_training_args(args)


def test_selective_checkpoint_rejects_full_checkpointing() -> None:
    import train

    args = _args_for_assert_extra(
        selective_checkpoint="mlp_only",
        gradient_checkpointing=True,
    )

    with pytest.raises(ValueError, match="selective_checkpoint"):
        train.AnimaTrainer().assert_extra_args(args, _CacheableDataset(), None)


def _args_for_assert_extra(**overrides) -> argparse.Namespace:
    values = {
        "cache_text_encoder_outputs_to_disk": False,
        "cache_text_encoder_outputs": False,
        "cache_llm_adapter_outputs": False,
        "network_train_unet_only": True,
        "cpu_offload_checkpointing": False,
        "unsloth_offload_checkpointing": False,
        "gradient_checkpointing": False,
        "blocks_to_swap": 0,
        "torch_compile": False,
        "dynamo_backend": "inductor",
        "compile_inductor_mode": None,
        "network_module": "networks.lora_anima",
        "network_args": [],
        "functional_loss_weight": 0.0,
        "selective_checkpoint": "off",
        "block_swap_profile_jsonl": None,
        "block_swap_restore_mode": "foreach",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class _CacheableDataset:
    datasets: list[object] = []

    def is_text_encoder_output_cacheable(self, *, cache_supports_dropout: bool) -> bool:
        return True

    def verify_bucket_reso_steps(self, steps: int) -> None:
        assert steps == 16
