from __future__ import annotations

import pytest
import torch
from safetensors import safe_open

from networks import NETWORK_REGISTRY, resolve_network_spec
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.factory import create_network, create_network_from_weights
from networks.lora_anima.persistence import stamp_lora_save_metadata
from networks.plugins.lokr.module import LoKrModule
from web.services.config.preflight import _validate_lokr_config


class Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8, bias=False)


class _TinyUnet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList([Block()])


def test_preflight_rejects_legacy_lokr_dim_without_opt_in():
    errors = _validate_lokr_config(
        {
            "use_lokr": True,
            "network_dim": 114514,
            "network_alpha": 32,
            "lokr_full_factor": False,
        }
    )
    assert len(errors) == 1
    assert "lokr_full_factor=true" in errors[0]


def test_preflight_accepts_recommended_full_factor_config():
    errors = _validate_lokr_config(
        {
            "use_lokr": True,
            "network_dim": 32,
            "network_alpha": 32,
            "lokr_full_factor": True,
            "lokr_decompose_w2": False,
        }
    )
    assert errors == []


def test_preflight_allows_legacy_dim_with_explicit_flag():
    errors = _validate_lokr_config(
        {
            "use_lokr": True,
            "network_dim": 114514,
            "network_alpha": 32,
            "lokr_allow_legacy_dim": True,
        }
    )
    assert errors == []


def test_resolve_accepts_full_factor_mode():
    spec = resolve_network_spec(
        {
            "use_lokr": "true",
            "lokr_full_factor": "true",
            "network_dim": 32,
            "network_alpha": 32,
        }
    )
    assert spec.name == "lokr"


def test_stamp_lokr_full_factor_parses_string_false():
    cfg = LoRANetworkCfg.from_kwargs(
        {"use_lokr": "true", "lokr_full_factor": "false"},
        network_dim=32,
        network_alpha=32,
        neuron_dropout=None,
        module_class=LoKrModule,
    )
    metadata = {}
    stamp_lora_save_metadata(metadata, cfg, NETWORK_REGISTRY["lokr"], network=None)
    assert metadata["ss_lokr_full_factor"] == "false"
    assert metadata["ss_network_dim"] == "32"
    assert metadata["ss_network_alpha"] == "32"


def test_default_lokr_already_keeps_full_w2_without_full_factor_flag():
    """anima_lora default path is already full-w2 when decompose is off."""
    base = torch.nn.Linear(2048, 2048, bias=False)
    m = LoKrModule(
        "test",
        base,
        lora_dim=32,
        alpha=32,
        factor=8,
        lokr_full_factor=False,
        lokr_decompose_w2=False,
    )
    assert m._use_decomposed_w2 is False
    assert m.lokr_w1.shape == (8, 8)
    assert m.lokr_w2.shape == (256, 256)
    assert m.scale == 1.0


def test_full_factor_round_trip_stamps_and_restores_layout(tmp_path):
    unet = _TinyUnet()
    net = create_network(
        multiplier=1.0,
        network_dim=32,
        network_alpha=32,
        vae=None,
        text_encoders=[],
        unet=unet,
        use_lokr="true",
        lokr_factor="4",
        lokr_full_factor="true",
        lokr_decompose_w2="false",
    )
    assert len(net.unet_loras) == 1
    source = net.unet_loras[0]
    assert source.lokr_full_factor is True
    assert source._use_decomposed_w2 is False
    net.apply_to([], unet, False, True)
    with torch.no_grad():
        source.lokr_w1.normal_(0, 0.1)
        source.lokr_w2.normal_(0, 0.1)
    expected = source.get_weight().clone()

    out = tmp_path / "lokr-full.safetensors"
    net.save_weights(str(out), torch.float32, metadata={})
    with safe_open(str(out), framework="pt") as f:
        meta = f.metadata() or {}
        keys = list(f.keys())
    assert meta["ss_lokr_full_factor"] == "true"
    assert meta["ss_network_dim"] == "32"
    assert meta["ss_network_alpha"] == "32"
    assert any(k.endswith(".lokr_w2") for k in keys)
    assert not any(k.endswith((".lokr_w2_a", ".lokr_w2_b")) for k in keys)

    restored, weights_sd = create_network_from_weights(
        multiplier=1.0,
        file=str(out),
        ae=None,
        text_encoders=[],
        unet=_TinyUnet(),
    )
    assert restored.cfg.plugin_args.get("lokr_full_factor") is True
    assert restored.unet_loras[0]._use_decomposed_w2 is False
    restored.apply_to([], _TinyUnet(), False, True)
    info = restored.load_state_dict(weights_sd, strict=False)
    assert info.missing_keys == []
    assert info.unexpected_keys == []
    assert torch.allclose(restored.unet_loras[0].get_weight(), expected, atol=1e-5, rtol=1e-5)


def test_naturally_full_layout_round_trip_stamps_actual_layout(tmp_path):
    """Even without the opt-in flag, a full-w2 layout should stamp true."""
    unet = _TinyUnet()
    net = create_network(
        multiplier=1.0,
        network_dim=32,
        network_alpha=16,
        vae=None,
        text_encoders=[],
        unet=unet,
        use_lokr="true",
        lokr_factor="4",
        lokr_full_factor="false",
        lokr_decompose_w2="false",
    )
    source = net.unet_loras[0]
    assert source.lokr_full_factor is False
    assert source._use_decomposed_w2 is False
    net.apply_to([], unet, False, True)
    with torch.no_grad():
        source.lokr_w2.normal_(0, 0.1)
    expected = source.get_weight().clone()

    out = tmp_path / "naturally-full.safetensors"
    net.save_weights(str(out), torch.float32, metadata={})
    with safe_open(str(out), framework="pt") as f:
        meta = f.metadata() or {}
    assert meta["ss_lokr_full_factor"] == "true"

    restored, weights_sd = create_network_from_weights(
        multiplier=1.0,
        file=str(out),
        ae=None,
        text_encoders=[],
        unet=_TinyUnet(),
    )
    assert restored.cfg.plugin_args.get("lokr_full_factor") is True
    restored.apply_to([], _TinyUnet(), False, True)
    info = restored.load_state_dict(weights_sd, strict=False)
    assert info.missing_keys == []
    assert torch.allclose(restored.unet_loras[0].get_weight(), expected, atol=1e-5, rtol=1e-5)


def test_full_factor_checkpoint_stamp_restores_runtime_layout():
    lora_name = "lora_unet_blocks_0_q_proj"
    native_sd = {
        f"{lora_name}.lokr_w1": torch.randn(2, 2),
        f"{lora_name}.lokr_w2": torch.randn(4, 4),
        f"{lora_name}.alpha": torch.tensor(32.0),
    }
    network, _ = create_network_from_weights(
        multiplier=1.0,
        file="",
        weights_sd=native_sd,
        metadata={
            "ss_network_spec": "lokr",
            "ss_network_dim": "32",
            "ss_lokr_full_factor": "true",
        },
        ae=None,
        text_encoders=[],
        unet=_TinyUnet(),
    )
    assert network.cfg.plugin_args.get("lokr_full_factor") is True
    assert network.unet_loras[0]._use_decomposed_w2 is False


def test_full_factor_checkpoint_stamp_rejects_decomposed_factor_keys():
    lora_name = "lora_unet_blocks_0_q_proj"
    native_sd = {
        f"{lora_name}.lokr_w1": torch.randn(2, 2),
        f"{lora_name}.lokr_w2_a": torch.randn(4, 2),
        f"{lora_name}.lokr_w2_b": torch.randn(2, 4),
        f"{lora_name}.alpha": torch.tensor(2.0),
    }
    with pytest.raises(RuntimeError, match="contains decomposed factor keys"):
        create_network_from_weights(
            multiplier=1.0,
            file="",
            weights_sd=native_sd,
            metadata={
                "ss_network_spec": "lokr",
                "ss_network_dim": "2",
                "ss_lokr_full_factor": "true",
            },
            ae=None,
            text_encoders=[],
            unet=_TinyUnet(),
        )


def test_legacy_unstamped_full_w2_checkpoint_is_inferred():
    lora_name = "lora_unet_blocks_0_q_proj"
    native_sd = {
        f"{lora_name}.lokr_w1": torch.randn(2, 2),
        f"{lora_name}.lokr_w2": torch.randn(4, 4),
        f"{lora_name}.alpha": torch.tensor(32.0),
    }
    network, _ = create_network_from_weights(
        multiplier=1.0,
        file="",
        weights_sd=native_sd,
        metadata={"ss_network_spec": "lokr", "ss_network_dim": "32"},
        ae=None,
        text_encoders=[],
        unet=_TinyUnet(),
    )
    assert network.cfg.plugin_args.get("lokr_full_factor") is True
    assert network.unet_loras[0]._use_decomposed_w2 is False
