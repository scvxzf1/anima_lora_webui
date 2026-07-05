from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from networks import lora_save


def _read_safetensors(path: Path) -> tuple[set[str], dict[str, str]]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return set(handle.keys()), dict(handle.metadata() or {})


def test_save_network_weights_hydra_writes_moe_sibling_with_metadata(tmp_path: Path) -> None:
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    rank = 2
    experts = 3
    state_dict = {
        f"{prefix}.lora_down.weight": torch.randn(rank, 8),
        f"{prefix}.lora_up_weight": torch.randn(experts, 24, rank),
        f"{prefix}.router.weight": torch.randn(experts, rank),
        f"{prefix}.router.bias": torch.randn(experts),
        f"{prefix}.alpha": torch.tensor(float(rank)),
    }
    out = tmp_path / "hydra.safetensors"

    lora_save.save_network_weights(
        state_dict,
        file=str(out),
        dtype=torch.float32,
        metadata={"ss_network_spec": "hydra", "marker": "kept"},
        save_variant="hydra_moe",
    )

    moe_out = tmp_path / "hydra_moe.safetensors"
    assert not out.exists()
    assert moe_out.exists()
    keys, metadata = _read_safetensors(moe_out)

    assert metadata["ss_network_spec"] == "hydra"
    assert metadata["marker"] == "kept"
    assert "lora_unet_blocks_0_self_attn_q_proj.lora_ups.0.weight" in keys
    assert "lora_unet_blocks_0_self_attn_k_proj.lora_ups.1.weight" in keys
    assert "lora_unet_blocks_0_self_attn_v_proj.router.weight" in keys
    assert f"{prefix}.lora_up_weight" not in keys


def test_save_network_weights_hydra_fallback_writes_moe_sibling(tmp_path: Path) -> None:
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    rank = 2
    experts = 3
    state_dict = {
        f"{prefix}.lora_down.weight": torch.randn(rank, 8),
        f"{prefix}.lora_up_weight": torch.randn(experts, 24, rank),
        f"{prefix}.router.weight": torch.randn(experts, rank),
        f"{prefix}.router.bias": torch.randn(experts),
        f"{prefix}.alpha": torch.tensor(float(rank)),
    }
    out = tmp_path / "fallback.safetensors"

    lora_save.save_network_weights(
        state_dict,
        file=str(out),
        dtype=torch.float32,
        metadata={"marker": "fallback"},
        save_variant="",
    )

    moe_out = tmp_path / "fallback_moe.safetensors"
    assert not out.exists()
    assert moe_out.exists()
    keys, metadata = _read_safetensors(moe_out)

    assert metadata["marker"] == "fallback"
    assert "lora_unet_blocks_0_self_attn_q_proj.lora_ups.0.weight" in keys
    assert f"{prefix}.lora_up_weight" not in keys


def test_save_network_weights_hydra_fallback_empty_metadata_writes_typed_moe(
    tmp_path: Path,
) -> None:
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    rank = 2
    experts = 3
    state_dict = {
        f"{prefix}.lora_down.weight": torch.randn(rank, 8),
        f"{prefix}.lora_up_weight": torch.arange(
            experts * 24 * rank,
            dtype=torch.float32,
        ).reshape(experts, 24, rank),
        f"{prefix}.router.weight": torch.randn(experts, rank),
        f"{prefix}.router.bias": torch.randn(experts),
        f"{prefix}.alpha": torch.tensor(float(rank)),
    }
    out = tmp_path / "fallback_empty.safetensors"

    lora_save.save_network_weights(
        state_dict,
        file=str(out),
        dtype=torch.float16,
        metadata={},
        save_variant="",
    )

    moe_out = tmp_path / "fallback_empty_moe.safetensors"
    assert not out.exists()
    assert moe_out.exists()
    with safe_open(str(moe_out), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        q_expert_0 = handle.get_tensor(
            "lora_unet_blocks_0_self_attn_q_proj.lora_ups.0.weight"
        )
        k_expert_1 = handle.get_tensor(
            "lora_unet_blocks_0_self_attn_k_proj.lora_ups.1.weight"
        )
        v_expert_2 = handle.get_tensor(
            "lora_unet_blocks_0_self_attn_v_proj.lora_ups.2.weight"
        )

    assert "ss_network_spec" not in metadata
    assert q_expert_0.dtype == torch.float16
    assert q_expert_0.shape == (8, rank)
    assert k_expert_1.shape == (8, rank)
    assert v_expert_2.shape == (8, rank)
    assert q_expert_0[0, 0].item() == 0.0
    assert k_expert_1[0, 0].item() == float((1 * 24 + 8) * rank)
    assert v_expert_2[0, 0].item() == float((2 * 24 + 16) * rank)


def test_save_network_weights_standard_preserves_metadata_and_adds_hashes(
    tmp_path: Path,
) -> None:
    prefix = "lora_unet_blocks_0_mlp"
    rank = 2
    state_dict = {
        f"{prefix}.lora_down.weight": torch.randn(rank, 8),
        f"{prefix}.lora_up.weight": torch.randn(4, rank),
        f"{prefix}.alpha": torch.tensor(float(rank)),
    }
    out = tmp_path / "standard.safetensors"

    lora_save.save_network_weights(
        state_dict,
        file=str(out),
        dtype=torch.float32,
        metadata={"marker": "standard"},
        save_variant="",
    )

    keys, metadata = _read_safetensors(out)

    assert metadata["marker"] == "standard"
    assert metadata["sshs_model_hash"]
    assert metadata["sshs_legacy_hash"]
    assert f"{prefix}.lora_down.weight" in keys
    assert f"{prefix}.lora_up.weight" in keys


def test_save_network_weights_standard_empty_metadata_still_adds_hashes(
    tmp_path: Path,
) -> None:
    prefix = "lora_unet_blocks_0_mlp"
    rank = 2
    state_dict = {
        f"{prefix}.lora_down.weight": torch.randn(rank, 8),
        f"{prefix}.lora_up.weight": torch.randn(4, rank),
        f"{prefix}.alpha": torch.tensor(float(rank)),
    }
    out = tmp_path / "standard_empty_metadata.safetensors"

    lora_save.save_network_weights(
        state_dict,
        file=str(out),
        dtype=torch.float32,
        metadata={},
        save_variant="",
    )

    _keys, metadata = _read_safetensors(out)

    assert metadata["sshs_model_hash"]
    assert metadata["sshs_legacy_hash"]
    assert "ss_network_spec" not in metadata


def test_save_network_weights_standard_torch_save_branch_writes_loadable_pt(
    tmp_path: Path,
) -> None:
    prefix = "lora_unet_blocks_0_mlp"
    rank = 2
    state_dict = {
        f"{prefix}.lora_down.weight": torch.randn(rank, 8),
        f"{prefix}.lora_up.weight": torch.randn(4, rank),
        f"{prefix}.alpha": torch.tensor(float(rank)),
    }
    out = tmp_path / "standard.pt"

    lora_save.save_network_weights(
        state_dict,
        file=str(out),
        dtype=torch.float16,
        metadata={"marker": "ignored-for-pt"},
        save_variant="",
    )

    loaded = torch.load(out, map_location="cpu", weights_only=True)

    assert set(loaded) == set(state_dict)
    assert loaded[f"{prefix}.lora_down.weight"].dtype == torch.float16
    assert loaded[f"{prefix}.lora_up.weight"].dtype == torch.float16
    assert loaded[f"{prefix}.alpha"].dtype == torch.float16


def test_save_network_weights_stacked_experts_writes_moe_sibling(tmp_path: Path) -> None:
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    rank = 2
    experts = 3
    state_dict = {
        f"{prefix}.lora_down_weight": torch.randn(experts, rank, 8),
        f"{prefix}.lora_up_weight": torch.randn(experts, 24, rank),
        f"{prefix}.alpha": torch.tensor(float(rank)),
        "global_router.net.0.weight": torch.randn(4, 2),
    }
    out = tmp_path / "stacked.safetensors"

    lora_save.save_network_weights(
        state_dict,
        file=str(out),
        dtype=torch.float32,
        metadata={"ss_network_spec": "stacked_experts_global_fei"},
        save_variant="stacked_experts_global_fei",
    )

    moe_out = tmp_path / "stacked_moe.safetensors"
    assert not out.exists()
    assert moe_out.exists()
    keys, metadata = _read_safetensors(moe_out)

    assert metadata["ss_network_spec"] == "stacked_experts_global_fei"
    assert "lora_unet_blocks_0_self_attn_q_proj.lora_downs.0.weight" in keys
    assert "lora_unet_blocks_0_self_attn_k_proj.lora_ups.1.weight" in keys
    assert "global_router.net.0.weight" in keys
    assert f"{prefix}.lora_down_weight" not in keys


def test_save_network_weights_stacked_experts_writes_typed_expert_tensors(
    tmp_path: Path,
) -> None:
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    rank = 2
    experts = 3
    state_dict = {
        f"{prefix}.lora_down_weight": torch.arange(
            experts * rank * 8,
            dtype=torch.float32,
        ).reshape(experts, rank, 8),
        f"{prefix}.lora_up_weight": torch.arange(
            experts * 24 * rank,
            dtype=torch.float32,
        ).reshape(experts, 24, rank),
        f"{prefix}.alpha": torch.tensor(float(rank)),
        "global_router.net.0.weight": torch.randn(4, 2),
    }
    out = tmp_path / "stacked_typed.safetensors"

    lora_save.save_network_weights(
        state_dict,
        file=str(out),
        dtype=torch.float16,
        metadata={"ss_network_spec": "stacked_experts_global_fei"},
        save_variant="stacked_experts_global_fei",
    )

    moe_out = tmp_path / "stacked_typed_moe.safetensors"
    with safe_open(str(moe_out), framework="pt", device="cpu") as handle:
        q_down_2 = handle.get_tensor(
            "lora_unet_blocks_0_self_attn_q_proj.lora_downs.2.weight"
        )
        k_up_1 = handle.get_tensor(
            "lora_unet_blocks_0_self_attn_k_proj.lora_ups.1.weight"
        )
        v_up_0 = handle.get_tensor(
            "lora_unet_blocks_0_self_attn_v_proj.lora_ups.0.weight"
        )

    assert q_down_2.dtype == torch.float16
    assert q_down_2.shape == (rank, 8)
    assert k_up_1.shape == (8, rank)
    assert v_up_0.shape == (8, rank)
    assert q_down_2[0, 0].item() == float(2 * rank * 8)
    assert k_up_1[0, 0].item() == float((1 * 24 + 8) * rank)
    assert v_up_0[0, 0].item() == float(16 * rank)


def test_save_network_weights_chimera_writes_chimera_sibling(tmp_path: Path) -> None:
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    rank = 2
    content_experts = 2
    freq_experts = 3
    state_dict = {
        f"{prefix}.lora_down_c.weight": torch.randn(rank, 8),
        f"{prefix}.lora_up_c_weight": torch.randn(content_experts, 24, rank),
        f"{prefix}.lora_down_f.weight": torch.randn(rank, 8),
        f"{prefix}.lora_up_f_weight": torch.randn(freq_experts, 24, rank),
        f"{prefix}.router.weight": torch.randn(content_experts, rank),
        f"{prefix}.router.bias": torch.randn(content_experts),
        f"{prefix}.alpha": torch.tensor(float(rank)),
        "freq_router.net.0.weight": torch.randn(4, 2),
    }
    out = tmp_path / "chimera.safetensors"

    lora_save.save_network_weights(
        state_dict,
        file=str(out),
        dtype=torch.float32,
        metadata={"ss_use_chimera_hydra": "true", "marker": "kept"},
        save_variant="chimera_hydra_moe",
    )

    chimera_out = tmp_path / "chimera_chimera.safetensors"
    assert not out.exists()
    assert chimera_out.exists()
    keys, metadata = _read_safetensors(chimera_out)

    assert metadata["ss_use_chimera_hydra"] == "true"
    assert metadata["marker"] == "kept"
    assert "lora_unet_blocks_0_self_attn_q_proj.lora_ups_c.0.weight" in keys
    assert "lora_unet_blocks_0_self_attn_k_proj.lora_ups_f.1.weight" in keys
    assert "lora_unet_blocks_0_self_attn_v_proj.router.weight" in keys
    assert "freq_router.net.0.weight" in keys
    assert f"{prefix}.lora_up_c_weight" not in keys


def test_save_network_weights_chimera_writes_typed_pool_tensors(
    tmp_path: Path,
) -> None:
    prefix = "lora_unet_blocks_0_self_attn_qkv_proj"
    rank = 2
    content_experts = 2
    freq_experts = 3
    state_dict = {
        f"{prefix}.lora_down_c.weight": torch.arange(
            rank * 8,
            dtype=torch.float32,
        ).reshape(rank, 8),
        f"{prefix}.lora_up_c_weight": torch.arange(
            content_experts * 24 * rank,
            dtype=torch.float32,
        ).reshape(content_experts, 24, rank),
        f"{prefix}.lora_down_f.weight": torch.arange(
            rank * 8,
            dtype=torch.float32,
        ).reshape(rank, 8)
        + 1000,
        f"{prefix}.lora_up_f_weight": torch.arange(
            freq_experts * 24 * rank,
            dtype=torch.float32,
        ).reshape(freq_experts, 24, rank)
        + 2000,
        f"{prefix}.router.weight": torch.randn(content_experts, rank),
        f"{prefix}.router.bias": torch.randn(content_experts),
        f"{prefix}.alpha": torch.tensor(float(rank)),
        "freq_router.net.0.weight": torch.randn(4, 2),
    }
    out = tmp_path / "chimera_typed.safetensors"

    lora_save.save_network_weights(
        state_dict,
        file=str(out),
        dtype=torch.float16,
        metadata={"ss_use_chimera_hydra": "true"},
        save_variant="chimera_hydra_moe",
    )

    chimera_out = tmp_path / "chimera_typed_chimera.safetensors"
    with safe_open(str(chimera_out), framework="pt", device="cpu") as handle:
        q_down_c = handle.get_tensor(
            "lora_unet_blocks_0_self_attn_q_proj.lora_down_c.weight"
        )
        k_up_c_1 = handle.get_tensor(
            "lora_unet_blocks_0_self_attn_k_proj.lora_ups_c.1.weight"
        )
        v_up_f_2 = handle.get_tensor(
            "lora_unet_blocks_0_self_attn_v_proj.lora_ups_f.2.weight"
        )

    assert q_down_c.dtype == torch.float16
    assert q_down_c.shape == (rank, 8)
    assert k_up_c_1.shape == (8, rank)
    assert v_up_f_2.shape == (8, rank)
    assert q_down_c[0, 0].item() == 0.0
    assert k_up_c_1[0, 0].item() == float((1 * 24 + 8) * rank)
    assert v_up_f_2[0, 0].item() == float(2000 + (2 * 24 + 16) * rank)
