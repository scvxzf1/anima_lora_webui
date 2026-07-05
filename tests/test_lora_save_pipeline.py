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
