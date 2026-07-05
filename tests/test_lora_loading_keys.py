from __future__ import annotations

import pytest
import torch

from networks.lora_anima.loading import (
    _refuse_split_chimera_keys,
    _refuse_split_hydra_keys,
    _stack_chimera_lora_ups,
    _stack_lora_ups,
)


def test_stack_lora_ups_stacks_sorted_experts_and_keeps_shared_down() -> None:
    prefix = "lora_unet_blocks_0_mlp"
    state_dict = {
        f"{prefix}.lora_down.weight": torch.randn(2, 8),
        f"{prefix}.lora_ups.2.weight": torch.full((4, 2), 2.0),
        f"{prefix}.lora_ups.0.weight": torch.full((4, 2), 0.0),
        f"{prefix}.lora_ups.1.weight": torch.full((4, 2), 1.0),
        f"{prefix}.alpha": torch.tensor(2.0),
    }

    out = _stack_lora_ups(state_dict)

    assert out is state_dict
    assert f"{prefix}.lora_down.weight" in out
    assert f"{prefix}.lora_down_weight" not in out
    assert f"{prefix}.lora_ups.0.weight" not in out
    assert torch.equal(out[f"{prefix}.lora_up_weight"][0], torch.full((4, 2), 0.0))
    assert torch.equal(out[f"{prefix}.lora_up_weight"][1], torch.full((4, 2), 1.0))
    assert torch.equal(out[f"{prefix}.lora_up_weight"][2], torch.full((4, 2), 2.0))


def test_stack_lora_ups_malformed_expert_index_raises_value_error() -> None:
    with pytest.raises(ValueError):
        _stack_lora_ups({"lora_unet_blocks_0_mlp.lora_ups.bad.weight": torch.randn(4, 2)})


def test_refuse_split_hydra_keys_refuses_qkv_and_keeps_plain_lora_leg() -> None:
    shared = "lora_unet_blocks_0_self_attn_"
    plain = "lora_unet_blocks_0_mlp"
    rank = 2
    experts = 3
    state_dict: dict[str, torch.Tensor] = {
        f"{plain}.lora_down.weight": torch.full((rank, 8), 50.0),
        f"{plain}.lora_up.weight": torch.full((4, rank), 60.0),
        f"{plain}.alpha": torch.tensor(float(rank)),
    }
    for idx, letter in enumerate(("q", "k", "v")):
        prefix = f"{shared}{letter}_proj"
        state_dict[f"{prefix}.lora_down.weight"] = torch.full((rank, 8), 10.0 + idx)
        state_dict[f"{prefix}.lora_up_weight"] = torch.full((experts, 4, rank), float(idx))
        state_dict[f"{prefix}.router.weight"] = torch.full((experts, rank), 30.0 + idx)
        state_dict[f"{prefix}.router.bias"] = torch.full((experts,), 40.0 + idx)
        state_dict[f"{prefix}.alpha"] = torch.tensor(float(rank))

    out = _refuse_split_hydra_keys(state_dict)
    fused = "lora_unet_blocks_0_self_attn_qkv_proj"

    assert out is state_dict
    assert torch.equal(out[f"{fused}.lora_down.weight"], torch.full((rank, 8), 10.0))
    assert out[f"{fused}.lora_up_weight"].shape == (experts, 12, rank)
    assert torch.equal(out[f"{fused}.router.weight"], torch.full((experts, rank), 30.0))
    assert "lora_unet_blocks_0_self_attn_q_proj.lora_up_weight" not in out
    assert torch.equal(out[f"{plain}.lora_up.weight"], torch.full((4, rank), 60.0))


def test_refuse_split_chimera_keys_refuses_dual_pool_qkv() -> None:
    shared = "lora_unet_blocks_0_self_attn_"
    rank = 2
    content_experts = 2
    freq_experts = 3
    state_dict: dict[str, torch.Tensor] = {}
    for idx, letter in enumerate(("q", "k", "v")):
        prefix = f"{shared}{letter}_proj"
        state_dict[f"{prefix}.lora_down_c.weight"] = torch.full((rank, 8), 10.0 + idx)
        state_dict[f"{prefix}.lora_down_f.weight"] = torch.full((rank, 8), 20.0 + idx)
        for expert in range(content_experts):
            state_dict[f"{prefix}.lora_ups_c.{expert}.weight"] = torch.full(
                (4, rank), float(100 + idx * 10 + expert)
            )
        for expert in range(freq_experts):
            state_dict[f"{prefix}.lora_ups_f.{expert}.weight"] = torch.full(
                (4, rank), float(200 + idx * 10 + expert)
            )
        state_dict[f"{prefix}.router.weight"] = torch.full((content_experts, rank), 30.0 + idx)
        state_dict[f"{prefix}.router.bias"] = torch.full((content_experts,), 40.0 + idx)
        state_dict[f"{prefix}.alpha"] = torch.tensor(float(rank))

    out = _stack_chimera_lora_ups(state_dict)
    out = _refuse_split_chimera_keys(out)
    fused = "lora_unet_blocks_0_self_attn_qkv_proj"

    assert out[f"{fused}.lora_up_c_weight"].shape == (content_experts, 12, rank)
    assert out[f"{fused}.lora_up_f_weight"].shape == (freq_experts, 12, rank)
    assert torch.equal(out[f"{fused}.lora_down_c.weight"], torch.full((rank, 8), 10.0))
    assert torch.equal(out[f"{fused}.lora_down_f.weight"], torch.full((rank, 8), 20.0))
    assert torch.equal(out[f"{fused}.router.weight"], torch.full((content_experts, rank), 30.0))
    assert "lora_unet_blocks_0_self_attn_q_proj.lora_up_c_weight" not in out
    assert "lora_unet_blocks_0_self_attn_q_proj.lora_ups_c.0.weight" not in out
