from __future__ import annotations

import pytest
import torch

from networks.lora_anima.loading import (
    _refuse_split_chimera_keys,
    _refuse_split_hydra_keys,
    _refuse_split_stacked_experts_keys,
    _refuse_unfused_attn_lora_keys,
    _rename_dora_scale_for_load,
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


def test_stack_lora_ups_stacks_sorted_independent_a_downs() -> None:
    prefix = "lora_unet_blocks_0_mlp"
    state_dict = {
        f"{prefix}.lora_ups.1.weight": torch.full((4, 2), 11.0),
        f"{prefix}.lora_ups.0.weight": torch.full((4, 2), 10.0),
        f"{prefix}.lora_downs.1.weight": torch.full((2, 8), 21.0),
        f"{prefix}.lora_downs.0.weight": torch.full((2, 8), 20.0),
        f"{prefix}.alpha": torch.tensor(2.0),
    }

    out = _stack_lora_ups(state_dict)

    assert out is state_dict
    assert torch.equal(out[f"{prefix}.lora_up_weight"][0], torch.full((4, 2), 10.0))
    assert torch.equal(out[f"{prefix}.lora_up_weight"][1], torch.full((4, 2), 11.0))
    assert torch.equal(out[f"{prefix}.lora_down_weight"][0], torch.full((2, 8), 20.0))
    assert torch.equal(out[f"{prefix}.lora_down_weight"][1], torch.full((2, 8), 21.0))
    assert f"{prefix}.lora_ups.0.weight" not in out
    assert f"{prefix}.lora_downs.0.weight" not in out


def test_stack_lora_ups_malformed_expert_index_raises_value_error() -> None:
    with pytest.raises(ValueError):
        _stack_lora_ups({"lora_unet_blocks_0_mlp.lora_ups.bad.weight": torch.randn(4, 2)})


def test_stack_lora_ups_non_contiguous_expert_indices_raise_value_error() -> None:
    with pytest.raises(ValueError, match="must be contiguous from 0"):
        _stack_lora_ups(
            {
                "lora_unet_blocks_0_mlp.lora_ups.0.weight": torch.randn(4, 2),
                "lora_unet_blocks_0_mlp.lora_ups.2.weight": torch.randn(4, 2),
            }
        )


def test_stack_lora_downs_non_contiguous_expert_indices_raise_value_error() -> None:
    with pytest.raises(ValueError, match="lora_downs expert indices"):
        _stack_lora_ups(
            {
                "lora_unet_blocks_0_mlp.lora_downs.0.weight": torch.randn(2, 8),
                "lora_unet_blocks_0_mlp.lora_downs.2.weight": torch.randn(2, 8),
            }
        )


def test_stack_chimera_lora_ups_non_contiguous_pool_indices_raise_value_error() -> None:
    with pytest.raises(ValueError, match="lora_ups_f expert indices"):
        _stack_chimera_lora_ups(
            {
                "lora_unet_blocks_0_mlp.lora_ups_f.0.weight": torch.randn(4, 2),
                "lora_unet_blocks_0_mlp.lora_ups_f.2.weight": torch.randn(4, 2),
            }
        )


def test_stack_chimera_lora_ups_stacks_sorted_dual_pools_independently() -> None:
    prefix = "lora_unet_blocks_0_mlp"
    state_dict = {
        f"{prefix}.lora_ups_c.1.weight": torch.full((4, 2), 11.0),
        f"{prefix}.lora_ups_c.0.weight": torch.full((4, 2), 10.0),
        f"{prefix}.lora_ups_f.2.weight": torch.full((4, 2), 22.0),
        f"{prefix}.lora_ups_f.0.weight": torch.full((4, 2), 20.0),
        f"{prefix}.lora_ups_f.1.weight": torch.full((4, 2), 21.0),
    }

    out = _stack_chimera_lora_ups(state_dict)

    assert out is state_dict
    assert torch.equal(out[f"{prefix}.lora_up_c_weight"][0], torch.full((4, 2), 10.0))
    assert torch.equal(out[f"{prefix}.lora_up_c_weight"][1], torch.full((4, 2), 11.0))
    assert torch.equal(out[f"{prefix}.lora_up_f_weight"][0], torch.full((4, 2), 20.0))
    assert torch.equal(out[f"{prefix}.lora_up_f_weight"][1], torch.full((4, 2), 21.0))
    assert torch.equal(out[f"{prefix}.lora_up_f_weight"][2], torch.full((4, 2), 22.0))
    assert f"{prefix}.lora_ups_c.0.weight" not in out
    assert f"{prefix}.lora_ups_f.0.weight" not in out


def test_rename_dora_scale_for_load_maps_export_scale_keys_to_magnitude() -> None:
    state_dict = {
        "lora_unet_blocks_0_mlp.dora_scale": torch.full((4,), 1.0),
        "lora_unet_blocks_1_mlp.dora_magnitude": torch.full((4,), 2.0),
        "lora_unet_blocks_2_mlp.lora_up.weight": torch.full((4, 2), 3.0),
    }

    out = _rename_dora_scale_for_load(state_dict)

    assert out is state_dict
    assert torch.equal(out["lora_unet_blocks_0_mlp.magnitude"], torch.full((4,), 1.0))
    assert torch.equal(out["lora_unet_blocks_1_mlp.magnitude"], torch.full((4,), 2.0))
    assert "lora_unet_blocks_0_mlp.dora_scale" not in out
    assert "lora_unet_blocks_1_mlp.dora_magnitude" not in out
    assert "lora_unet_blocks_2_mlp.lora_up.weight" in out


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


def test_refuse_split_hydra_keys_missing_component_leaves_keys_untouched() -> None:
    shared = "lora_unet_blocks_0_self_attn_"
    rank = 2
    experts = 3
    state_dict: dict[str, torch.Tensor] = {}
    for letter in ("q", "v"):
        prefix = f"{shared}{letter}_proj"
        state_dict[f"{prefix}.lora_down.weight"] = torch.full((rank, 8), 10.0)
        state_dict[f"{prefix}.lora_up_weight"] = torch.full((experts, 4, rank), 20.0)
        state_dict[f"{prefix}.alpha"] = torch.tensor(float(rank))

    out = _refuse_split_hydra_keys(state_dict)

    assert out is state_dict
    assert "lora_unet_blocks_0_self_attn_qkv_proj.lora_up_weight" not in out
    assert "lora_unet_blocks_0_self_attn_q_proj.lora_up_weight" in out
    assert "lora_unet_blocks_0_self_attn_v_proj.lora_up_weight" in out


def test_refuse_split_hydra_keys_inconsistent_up_shape_leaves_keys_untouched() -> None:
    shared = "lora_unet_blocks_0_self_attn_"
    rank = 2
    experts = 3
    state_dict: dict[str, torch.Tensor] = {}
    for idx, letter in enumerate(("q", "k", "v")):
        prefix = f"{shared}{letter}_proj"
        up_rank = 3 if letter == "k" else rank
        state_dict[f"{prefix}.lora_down.weight"] = torch.full((rank, 8), 10.0)
        state_dict[f"{prefix}.lora_up_weight"] = torch.full((experts, 4, up_rank), float(idx))
        state_dict[f"{prefix}.alpha"] = torch.tensor(float(rank))

    out = _refuse_split_hydra_keys(state_dict)

    assert out is state_dict
    assert "lora_unet_blocks_0_self_attn_qkv_proj.lora_up_weight" not in out
    assert "lora_unet_blocks_0_self_attn_k_proj.lora_up_weight" in out


def test_refuse_split_hydra_keys_rehomes_sigma_mlp_and_inv_scale() -> None:
    shared = "lora_unet_blocks_0_self_attn_"
    rank = 2
    experts = 3
    state_dict: dict[str, torch.Tensor] = {}
    for idx, letter in enumerate(("q", "k", "v")):
        prefix = f"{shared}{letter}_proj"
        state_dict[f"{prefix}.lora_down.weight"] = torch.full((rank, 8), 10.0 + idx)
        state_dict[f"{prefix}.lora_up_weight"] = torch.full((experts, 4, rank), float(idx))
        state_dict[f"{prefix}.router.weight"] = torch.full((experts, rank), 30.0 + idx)
        state_dict[f"{prefix}.router.bias"] = torch.full((experts,), 40.0 + idx)
        state_dict[f"{prefix}.inv_scale"] = torch.full((8,), 50.0 + idx)
        state_dict[f"{prefix}.sigma_mlp.0.weight"] = torch.full((rank, rank), 60.0 + idx)
        state_dict[f"{prefix}.sigma_mlp.0.bias"] = torch.full((rank,), 70.0 + idx)
        state_dict[f"{prefix}.alpha"] = torch.tensor(float(rank))

    out = _refuse_split_hydra_keys(state_dict)
    fused = "lora_unet_blocks_0_self_attn_qkv_proj"

    assert out is state_dict
    assert torch.equal(out[f"{fused}.inv_scale"], torch.full((8,), 50.0))
    assert torch.equal(out[f"{fused}.sigma_mlp.0.weight"], torch.full((rank, rank), 60.0))
    assert torch.equal(out[f"{fused}.sigma_mlp.0.bias"], torch.full((rank,), 70.0))
    assert f"{shared}q_proj.inv_scale" not in out
    assert f"{shared}q_proj.sigma_mlp.0.weight" not in out
    assert f"{shared}k_proj.sigma_mlp.0.bias" not in out


def test_refuse_unfused_attn_lora_keys_prefused_roundtrip_keeps_rank() -> None:
    shared = "lora_unet_blocks_0_self_attn_"
    rank = 2
    in_dim = 8
    out_dim = 4
    down = torch.arange(rank * in_dim, dtype=torch.float32).reshape(rank, in_dim)
    state_dict: dict[str, torch.Tensor] = {}
    for idx, letter in enumerate(("q", "k", "v")):
        prefix = f"{shared}{letter}_proj"
        state_dict[f"{prefix}.lora_down.weight"] = down.clone()
        state_dict[f"{prefix}.lora_up.weight"] = torch.full((out_dim, rank), float(idx + 1))
        state_dict[f"{prefix}.alpha"] = torch.tensor(float(rank))

    out = _refuse_unfused_attn_lora_keys(state_dict)
    fused = "lora_unet_blocks_0_self_attn_qkv_proj"

    assert out is state_dict
    assert torch.equal(out[f"{fused}.lora_down.weight"], down)
    assert out[f"{fused}.lora_up.weight"].shape == (out_dim * 3, rank)
    assert out[f"{fused}.alpha"].item() == pytest.approx(float(rank))
    assert torch.equal(out[f"{fused}.lora_up.weight"][:out_dim], torch.full((out_dim, rank), 1.0))
    assert f"{shared}q_proj.lora_down.weight" not in out
    assert f"{shared}k_proj.lora_up.weight" not in out


def test_refuse_unfused_attn_lora_keys_missing_component_leaves_keys_untouched() -> None:
    shared = "lora_unet_blocks_0_self_attn_"
    rank = 2
    out_dim = 4
    state_dict: dict[str, torch.Tensor] = {}
    for letter in ("q", "v"):
        prefix = f"{shared}{letter}_proj"
        state_dict[f"{prefix}.lora_down.weight"] = torch.full((rank, 8), 10.0)
        state_dict[f"{prefix}.lora_up.weight"] = torch.full((out_dim, rank), 20.0)
        state_dict[f"{prefix}.alpha"] = torch.tensor(float(rank))

    out = _refuse_unfused_attn_lora_keys(state_dict)

    assert out is state_dict
    assert "lora_unet_blocks_0_self_attn_qkv_proj.lora_up.weight" not in out
    assert f"{shared}q_proj.lora_up.weight" in out
    assert f"{shared}v_proj.lora_down.weight" in out


def test_refuse_unfused_attn_lora_keys_inconsistent_up_shape_leaves_keys_untouched() -> None:
    shared = "lora_unet_blocks_0_self_attn_"
    rank = 2
    state_dict: dict[str, torch.Tensor] = {}
    for letter in ("q", "k", "v"):
        prefix = f"{shared}{letter}_proj"
        out_dim = 5 if letter == "k" else 4
        state_dict[f"{prefix}.lora_down.weight"] = torch.full((rank, 8), 10.0)
        state_dict[f"{prefix}.lora_up.weight"] = torch.full((out_dim, rank), 20.0)
        state_dict[f"{prefix}.alpha"] = torch.tensor(float(rank))

    out = _refuse_unfused_attn_lora_keys(state_dict)

    assert out is state_dict
    assert "lora_unet_blocks_0_self_attn_qkv_proj.lora_up.weight" not in out
    assert f"{shared}k_proj.lora_up.weight" in out


def test_refuse_unfused_attn_lora_keys_inconsistent_down_shape_leaves_keys_untouched() -> None:
    shared = "lora_unet_blocks_0_self_attn_"
    rank = 2
    state_dict: dict[str, torch.Tensor] = {}
    for letter in ("q", "k", "v"):
        prefix = f"{shared}{letter}_proj"
        in_dim = 9 if letter == "v" else 8
        state_dict[f"{prefix}.lora_down.weight"] = torch.full((rank, in_dim), 10.0)
        state_dict[f"{prefix}.lora_up.weight"] = torch.full((4, rank), 20.0)
        state_dict[f"{prefix}.alpha"] = torch.tensor(float(rank))

    out = _refuse_unfused_attn_lora_keys(state_dict)

    assert out is state_dict
    assert "lora_unet_blocks_0_self_attn_qkv_proj.lora_down.weight" not in out
    assert f"{shared}v_proj.lora_down.weight" in out


def test_refuse_unfused_attn_lora_keys_fuses_split_dora_magnitudes() -> None:
    shared = "lora_unet_blocks_0_self_attn_"
    rank = 2
    in_dim = 8
    out_dim = 4
    dora_leaf_by_letter = {
        "q": "dora_scale",
        "k": "dora_magnitude",
        "v": "magnitude",
    }
    state_dict: dict[str, torch.Tensor] = {}
    for idx, letter in enumerate(("q", "k", "v")):
        prefix = f"{shared}{letter}_proj"
        state_dict[f"{prefix}.lora_down.weight"] = torch.full((rank, in_dim), float(idx + 1))
        state_dict[f"{prefix}.lora_up.weight"] = torch.full((out_dim, rank), float(idx + 10))
        state_dict[f"{prefix}.alpha"] = torch.tensor(float(rank))
        state_dict[f"{prefix}.{dora_leaf_by_letter[letter]}"] = torch.full(
            (out_dim,),
            float(idx + 20),
        )

    out = _refuse_unfused_attn_lora_keys(state_dict)
    fused = "lora_unet_blocks_0_self_attn_qkv_proj"

    assert out[f"{fused}.lora_down.weight"].shape == (rank * 3, in_dim)
    assert out[f"{fused}.lora_up.weight"].shape == (out_dim * 3, rank * 3)
    assert torch.equal(
        out[f"{fused}.magnitude"],
        torch.cat([torch.full((out_dim,), float(idx + 20)) for idx in range(3)]),
    )
    assert f"{shared}q_proj.dora_scale" not in out
    assert f"{shared}k_proj.dora_magnitude" not in out
    assert f"{shared}v_proj.magnitude" not in out


def test_refuse_unfused_attn_lora_keys_partial_scaling_metadata_is_dropped() -> None:
    shared = "lora_unet_blocks_0_self_attn_"
    rank = 2
    in_dim = 8
    out_dim = 4
    state_dict: dict[str, torch.Tensor] = {}
    for idx, letter in enumerate(("q", "k", "v")):
        prefix = f"{shared}{letter}_proj"
        state_dict[f"{prefix}.lora_down.weight"] = torch.full(
            (rank, in_dim), float(idx + 1)
        )
        state_dict[f"{prefix}.lora_up.weight"] = torch.full(
            (out_dim, rank), float(idx + 10)
        )
        state_dict[f"{prefix}.alpha"] = torch.tensor(float(rank))
    state_dict[f"{shared}q_proj.inv_scale"] = torch.full((in_dim,), 20.0)
    state_dict[f"{shared}k_proj.magnitude"] = torch.full((out_dim,), 30.0)

    out = _refuse_unfused_attn_lora_keys(state_dict)
    fused = "lora_unet_blocks_0_self_attn_qkv_proj"

    assert f"{fused}.inv_scale" not in out
    assert f"{fused}.magnitude" not in out
    assert f"{shared}q_proj.inv_scale" not in out
    assert f"{shared}k_proj.magnitude" not in out
    assert out[f"{fused}.lora_down.weight"].shape == (rank * 3, in_dim)
    assert out[f"{fused}.lora_up.weight"].shape == (out_dim * 3, rank * 3)


def test_refuse_unfused_attn_lora_keys_split_components_block_diag_and_alpha_scales() -> None:
    shared = "lora_unet_blocks_0_self_attn_"
    rank = 2
    in_dim = 5
    out_dim = 3
    alphas = {"q": 2.0, "k": 4.0, "v": 1.0}
    up_values = {"q": 10.0, "k": 20.0, "v": 30.0}
    state_dict: dict[str, torch.Tensor] = {}
    for idx, letter in enumerate(("q", "k", "v")):
        prefix = f"{shared}{letter}_proj"
        down = torch.full((rank, in_dim), float(idx + 1))
        down[0, 0] = float(idx + 10)
        state_dict[f"{prefix}.lora_down.weight"] = down
        state_dict[f"{prefix}.lora_up.weight"] = torch.full((out_dim, rank), up_values[letter])
        state_dict[f"{prefix}.alpha"] = torch.tensor(alphas[letter])

    out = _refuse_unfused_attn_lora_keys(state_dict)
    fused = "lora_unet_blocks_0_self_attn_qkv_proj"
    fused_up = out[f"{fused}.lora_up.weight"]
    fused_down = out[f"{fused}.lora_down.weight"]

    assert fused_down.shape == (rank * 3, in_dim)
    assert fused_down[0, 0].item() == pytest.approx(10.0)
    assert fused_down[rank, 0].item() == pytest.approx(11.0)
    assert fused_down[rank * 2, 0].item() == pytest.approx(12.0)
    assert fused_up.shape == (out_dim * 3, rank * 3)
    assert out[f"{fused}.alpha"].item() == pytest.approx(float(rank * 3))
    for idx, letter in enumerate(("q", "k", "v")):
        row = slice(idx * out_dim, (idx + 1) * out_dim)
        col = slice(idx * rank, (idx + 1) * rank)
        assert torch.equal(
            fused_up[row, col],
            torch.full((out_dim, rank), up_values[letter] * (alphas[letter] / rank)),
        )
    assert torch.count_nonzero(fused_up[:out_dim, rank:]).item() == 0
    assert torch.count_nonzero(fused_up[out_dim : 2 * out_dim, :rank]).item() == 0
    assert torch.count_nonzero(fused_up[2 * out_dim :, : 2 * rank]).item() == 0


def test_refuse_unfused_attn_lora_keys_missing_alpha_uses_unit_scale() -> None:
    shared = "lora_unet_blocks_0_self_attn_"
    rank = 2
    out_dim = 3
    state_dict: dict[str, torch.Tensor] = {}
    for idx, letter in enumerate(("q", "k", "v")):
        prefix = f"{shared}{letter}_proj"
        state_dict[f"{prefix}.lora_down.weight"] = torch.full((rank, 5), float(idx + 1))
        state_dict[f"{prefix}.lora_up.weight"] = torch.full((out_dim, rank), float(idx + 10))

    out = _refuse_unfused_attn_lora_keys(state_dict)
    fused = "lora_unet_blocks_0_self_attn_qkv_proj"
    fused_up = out[f"{fused}.lora_up.weight"]

    assert out[f"{fused}.alpha"].item() == pytest.approx(float(rank * 3))
    for idx, value in enumerate((10.0, 11.0, 12.0)):
        row = slice(idx * out_dim, (idx + 1) * out_dim)
        col = slice(idx * rank, (idx + 1) * rank)
        assert torch.equal(fused_up[row, col], torch.full((out_dim, rank), value))


def test_refuse_unfused_attn_lora_keys_mixed_missing_alpha_uses_unit_scale() -> None:
    shared = "lora_unet_blocks_0_self_attn_"
    rank = 2
    out_dim = 3
    state_dict: dict[str, torch.Tensor] = {}
    for idx, letter in enumerate(("q", "k", "v")):
        prefix = f"{shared}{letter}_proj"
        state_dict[f"{prefix}.lora_down.weight"] = torch.full((rank, 5), float(idx + 1))
        state_dict[f"{prefix}.lora_up.weight"] = torch.full((out_dim, rank), 10.0)
    state_dict[f"{shared}q_proj.alpha"] = torch.tensor(4.0)
    state_dict[f"{shared}v_proj.alpha"] = torch.tensor(1.0)

    out = _refuse_unfused_attn_lora_keys(state_dict)
    fused = "lora_unet_blocks_0_self_attn_qkv_proj"
    fused_up = out[f"{fused}.lora_up.weight"]

    expected_scales = (2.0, 1.0, 0.5)
    for idx, scale in enumerate(expected_scales):
        row = slice(idx * out_dim, (idx + 1) * out_dim)
        col = slice(idx * rank, (idx + 1) * rank)
        assert torch.equal(fused_up[row, col], torch.full((out_dim, rank), 10.0 * scale))


def test_refuse_split_stacked_experts_keys_refuses_qkv() -> None:
    shared = "lora_unet_blocks_0_self_attn_"
    rank = 2
    experts = 3
    state_dict: dict[str, torch.Tensor] = {}
    for idx, letter in enumerate(("q", "k", "v")):
        prefix = f"{shared}{letter}_proj"
        state_dict[f"{prefix}.lora_down_weight"] = torch.full(
            (experts, rank, 8), 10.0 + idx
        )
        state_dict[f"{prefix}.lora_up_weight"] = torch.full(
            (experts, 4, rank), float(idx)
        )
        state_dict[f"{prefix}.alpha"] = torch.tensor(float(rank))

    out = _refuse_split_stacked_experts_keys(state_dict)
    fused = "lora_unet_blocks_0_self_attn_qkv_proj"

    assert out is state_dict
    assert torch.equal(out[f"{fused}.lora_down_weight"], torch.full((experts, rank, 8), 10.0))
    assert out[f"{fused}.lora_up_weight"].shape == (experts, 12, rank)
    assert "lora_unet_blocks_0_self_attn_q_proj.lora_up_weight" not in out
    assert "lora_unet_blocks_0_self_attn_q_proj.lora_down_weight" not in out


def test_refuse_split_stacked_experts_keys_missing_component_leaves_keys_untouched() -> None:
    shared = "lora_unet_blocks_0_self_attn_"
    rank = 2
    experts = 3
    state_dict: dict[str, torch.Tensor] = {}
    for letter in ("q", "v"):
        prefix = f"{shared}{letter}_proj"
        state_dict[f"{prefix}.lora_down_weight"] = torch.full((experts, rank, 8), 10.0)
        state_dict[f"{prefix}.lora_up_weight"] = torch.full((experts, 4, rank), 20.0)
        state_dict[f"{prefix}.alpha"] = torch.tensor(float(rank))

    out = _refuse_split_stacked_experts_keys(state_dict)

    assert out is state_dict
    assert "lora_unet_blocks_0_self_attn_qkv_proj.lora_up_weight" not in out
    assert "lora_unet_blocks_0_self_attn_q_proj.lora_up_weight" in out
    assert "lora_unet_blocks_0_self_attn_v_proj.lora_down_weight" in out


def test_refuse_split_stacked_experts_keys_inconsistent_down_shape_leaves_keys_untouched() -> None:
    shared = "lora_unet_blocks_0_self_attn_"
    rank = 2
    experts = 3
    state_dict: dict[str, torch.Tensor] = {}
    for letter in ("q", "k", "v"):
        prefix = f"{shared}{letter}_proj"
        down_rank = 3 if letter == "k" else rank
        state_dict[f"{prefix}.lora_down_weight"] = torch.full((experts, down_rank, 8), 10.0)
        state_dict[f"{prefix}.lora_up_weight"] = torch.full((experts, 4, rank), 20.0)
        state_dict[f"{prefix}.alpha"] = torch.tensor(float(rank))

    out = _refuse_split_stacked_experts_keys(state_dict)

    assert out is state_dict
    assert "lora_unet_blocks_0_self_attn_qkv_proj.lora_down_weight" not in out
    assert "lora_unet_blocks_0_self_attn_k_proj.lora_down_weight" in out


def test_refuse_split_stacked_experts_keys_inconsistent_up_shape_leaves_keys_untouched() -> None:
    shared = "lora_unet_blocks_0_self_attn_"
    rank = 2
    experts = 3
    state_dict: dict[str, torch.Tensor] = {}
    for letter in ("q", "k", "v"):
        prefix = f"{shared}{letter}_proj"
        up_rank = 3 if letter == "k" else rank
        state_dict[f"{prefix}.lora_down_weight"] = torch.full(
            (experts, rank, 8), 10.0
        )
        state_dict[f"{prefix}.lora_up_weight"] = torch.full(
            (experts, 4, up_rank), 20.0
        )
        state_dict[f"{prefix}.alpha"] = torch.tensor(float(rank))

    out = _refuse_split_stacked_experts_keys(state_dict)

    assert out is state_dict
    assert "lora_unet_blocks_0_self_attn_qkv_proj.lora_up_weight" not in out
    assert "lora_unet_blocks_0_self_attn_k_proj.lora_up_weight" in out


def test_stack_lora_ups_then_refuse_split_stacked_experts_keys_refuses_qkv() -> None:
    shared = "lora_unet_blocks_0_self_attn_"
    rank = 2
    experts = 3
    out_dim = 4
    state_dict: dict[str, torch.Tensor] = {}
    for comp_idx, letter in enumerate(("q", "k", "v")):
        prefix = f"{shared}{letter}_proj"
        for expert in range(experts):
            state_dict[f"{prefix}.lora_ups.{expert}.weight"] = torch.full(
                (out_dim, rank),
                float(100 + comp_idx * 10 + expert),
            )
            state_dict[f"{prefix}.lora_downs.{expert}.weight"] = torch.full(
                (rank, 8),
                float(200 + comp_idx * 10 + expert),
            )
        state_dict[f"{prefix}.alpha"] = torch.tensor(float(rank))

    out = _stack_lora_ups(state_dict)
    out = _refuse_split_stacked_experts_keys(out)
    fused = "lora_unet_blocks_0_self_attn_qkv_proj"

    assert out[f"{fused}.lora_up_weight"].shape == (experts, out_dim * 3, rank)
    assert out[f"{fused}.lora_down_weight"].shape == (experts, rank, 8)
    assert torch.equal(
        out[f"{fused}.lora_up_weight"][1, out_dim : out_dim * 2],
        torch.full((out_dim, rank), 111.0),
    )
    assert torch.equal(
        out[f"{fused}.lora_down_weight"][2],
        torch.full((rank, 8), 202.0),
    )
    assert f"{shared}q_proj.lora_ups.0.weight" not in out
    assert f"{shared}q_proj.lora_up_weight" not in out
    assert f"{shared}v_proj.lora_down_weight" not in out


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


def test_refuse_split_chimera_keys_missing_pool_leaves_keys_untouched() -> None:
    shared = "lora_unet_blocks_0_self_attn_"
    rank = 2
    content_experts = 2
    freq_experts = 3
    state_dict: dict[str, torch.Tensor] = {}
    for letter in ("q", "k", "v"):
        prefix = f"{shared}{letter}_proj"
        state_dict[f"{prefix}.lora_down_c.weight"] = torch.full((rank, 8), 10.0)
        state_dict[f"{prefix}.lora_down_f.weight"] = torch.full((rank, 8), 20.0)
        for expert in range(content_experts):
            state_dict[f"{prefix}.lora_ups_c.{expert}.weight"] = torch.full(
                (4, rank), 30.0
            )
        if letter != "k":
            for expert in range(freq_experts):
                state_dict[f"{prefix}.lora_ups_f.{expert}.weight"] = torch.full(
                    (4, rank), 40.0
                )

    out = _stack_chimera_lora_ups(state_dict)
    out = _refuse_split_chimera_keys(out)

    assert "lora_unet_blocks_0_self_attn_qkv_proj.lora_up_c_weight" not in out
    assert "lora_unet_blocks_0_self_attn_q_proj.lora_up_c_weight" in out
    assert "lora_unet_blocks_0_self_attn_k_proj.lora_up_c_weight" in out
    assert "lora_unet_blocks_0_self_attn_k_proj.lora_down_f.weight" in out
