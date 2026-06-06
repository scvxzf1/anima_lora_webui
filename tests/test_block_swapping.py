from __future__ import annotations

import argparse
import json

import pytest
import torch
from torch import nn

from library.runtime.device import should_move_weight_to_device
from library.runtime.offloading import (
    ModelOffloader,
    normalize_block_swap_transfer_dtype,
    swap_weight_devices_no_cuda,
)


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = nn.Linear(2, 2, bias=False)
        self.adapter = nn.Linear(2, 2, bias=False)
        self.base.weight.requires_grad_(False)
        self.adapter.weight.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.adapter(x)


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
    assert event["queued_at"] <= event["ready_at"]
    assert event["enqueued_at"] <= event["ready_at"]


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
    with pytest.raises(ValueError):
        normalize_block_swap_transfer_dtype("int8")


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
        self.mlp_checkpointing = False
        self.mlp_layer1_checkpointing = False

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.unsloth_offload_checkpointing = False
        self.mlp_checkpointing = False
        self.mlp_layer1_checkpointing = False

    def enable_mlp_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.unsloth_offload_checkpointing = False
        self.mlp_checkpointing = True
        self.mlp_layer1_checkpointing = False

    def enable_mlp_layer1_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        self.unsloth_offload_checkpointing = False
        self.mlp_checkpointing = False
        self.mlp_layer1_checkpointing = True


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

    Anima.enable_selective_checkpointing(model, "off")
    assert model.selective_checkpoint == "off"
    assert not any(block.gradient_checkpointing for block in model.blocks)
    assert not any(block.mlp_checkpointing for block in model.blocks)
    assert not any(block.mlp_layer1_checkpointing for block in model.blocks)


def test_peak_probe_attachment_sets_block_indices() -> None:
    from library.anima.models import Anima

    model = object.__new__(Anima)
    model.blocks = [_FakeCheckpointBlock() for _ in range(3)]
    probe = object()

    Anima.enable_peak_probe(model, probe)

    assert model._peak_probe is probe
    assert [block._peak_probe for block in model.blocks] == [probe, probe, probe]
    assert [block._block_idx for block in model.blocks] == [0, 1, 2]


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
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class _CacheableDataset:
    datasets: list[object] = []

    def is_text_encoder_output_cacheable(self, *, cache_supports_dropout: bool) -> bool:
        return True

    def verify_bucket_reso_steps(self, steps: int) -> None:
        assert steps == 16
