from __future__ import annotations

import torch

from library.inference.selective_lora import (
    apply_anima_selective_lora,
    classify_anima_lora_key,
    enabled_blocks_from_anima_selective_strengths,
    normalize_anima_selective_blocks,
    normalize_anima_selective_block_strengths,
)


def test_classify_anima_lora_key_routes_known_block_types() -> None:
    assert classify_anima_lora_key("lora_unet_blocks_7_self_attn_q_proj.lora_down.weight") == "block_7"
    assert classify_anima_lora_key("lora_unet_llm_adapter_blocks_3_mlp_fc1.lora_down.weight") == "llm_adapter_3"
    assert classify_anima_lora_key("lora_unet_llm_adapter_embed_proj.lora_down.weight") == "llm_adapter_io"
    assert classify_anima_lora_key("lora_unet_final_layer_linear.lora_down.weight") == "final_layer"
    assert classify_anima_lora_key("lora_unet_t_embedder_mlp_0.lora_down.weight") == "t_embedder"
    assert classify_anima_lora_key("lora_unet_x_embedder_proj.weight") == "x_embedder"
    assert classify_anima_lora_key("lora_te_text_model_encoder.layers.0.self_attn.q_proj.weight") == "other_weights"


def test_apply_anima_selective_lora_filters_and_scales_selected_blocks() -> None:
    weights = {
        "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight": torch.ones(1),
        "lora_unet_blocks_22_self_attn_q_proj.lora_down.weight": torch.full((1,), 2.0),
        "lora_unet_llm_adapter_blocks_2_mlp_fc1.lora_down.weight": torch.full((1,), 3.0),
        "lora_unet_final_layer_linear.lora_down.weight": torch.full((1,), 4.0),
    }

    filtered = apply_anima_selective_lora(
        weights,
        ["block_22", "final_layer"],
        strength=0.5,
        preset="custom",
    )

    assert set(filtered.keys()) == {
        "lora_unet_blocks_22_self_attn_q_proj.lora_down.weight",
        "lora_unet_final_layer_linear.lora_down.weight",
    }
    assert torch.equal(
        filtered["lora_unet_blocks_22_self_attn_q_proj.lora_down.weight"],
        torch.full((1,), 1.0),
    )
    assert torch.equal(
        filtered["lora_unet_final_layer_linear.lora_down.weight"],
        torch.full((1,), 2.0),
    )


def test_apply_anima_selective_lora_uses_per_block_strength_map() -> None:
    weights = {
        "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight": torch.full((1,), 4.0),
        "lora_unet_blocks_1_self_attn_q_proj.lora_down.weight": torch.full((1,), 6.0),
        "lora_unet_final_layer_linear.lora_down.weight": torch.full((1,), 8.0),
    }

    filtered = apply_anima_selective_lora(
        weights,
        None,
        preset="custom",
        block_strengths={
            "block_0": 0.25,
            "block_1": 0.0,
            "final_layer": 1.5,
        },
    )

    assert set(filtered.keys()) == {
        "lora_unet_blocks_0_self_attn_q_proj.lora_down.weight",
        "lora_unet_final_layer_linear.lora_down.weight",
    }
    assert torch.equal(
        filtered["lora_unet_blocks_0_self_attn_q_proj.lora_down.weight"],
        torch.full((1,), 1.0),
    )
    assert torch.equal(
        filtered["lora_unet_final_layer_linear.lora_down.weight"],
        torch.full((1,), 12.0),
    )


def test_normalize_anima_selective_blocks_uses_preset_only_when_missing() -> None:
    assert normalize_anima_selective_blocks(None, preset="all_off") == []
    assert normalize_anima_selective_blocks([], preset="default") == []
    assert normalize_anima_selective_blocks(["block_1", "block_0", "unknown"], preset="default") == [
        "block_0",
        "block_1",
    ]


def test_normalize_anima_selective_block_strengths_rounds_to_step() -> None:
    normalized = normalize_anima_selective_block_strengths(
        {
            "block_0": 0.63,
            "block_1": -1,
            "final_layer": 9,
        },
        preset="default",
    )

    assert normalized["block_0"] == 0.65
    assert normalized["block_1"] == 0.0
    assert normalized["final_layer"] == 2.0


def test_enabled_blocks_from_anima_selective_strengths_keeps_order() -> None:
    assert enabled_blocks_from_anima_selective_strengths(
        {"block_3": 0.5, "block_1": 0.25, "final_layer": 1.0},
        preset="all_off",
    ) == ["block_1", "block_3", "final_layer"]
