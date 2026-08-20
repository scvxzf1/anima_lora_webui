from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from library.anima import checkpoint, weights
from library.anima.checkpoint import AnimaCheckpointLayout
from library.anima.compat import (
    adapter_identity_metadata,
    require_training_compatibility,
    validate_adapter_metadata,
)
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.persistence import stamp_lora_save_metadata
from networks.registry import NETWORK_REGISTRY


def _layout(blocks: int) -> AnimaCheckpointLayout:
    return AnimaCheckpointLayout(
        arch=f"anima-2048-{blocks}",
        variant="anima-2.9b-preview-v1" if blocks == 40 else "anima-28-block",
        num_blocks=blocks,
        model_channels=2048,
        num_heads=16,
        key_prefix="net.",
    )


def _header(blocks: int, *, prefix: str = "net.") -> dict[str, dict[str, object]]:
    header: dict[str, dict[str, object]] = {}
    for index in range(blocks):
        root = f"{prefix}blocks.{index}"
        header[f"{root}.self_attn.q_proj.weight"] = {"shape": [2048, 2048]}
        header[f"{root}.self_attn.q_norm.weight"] = {"shape": [128]}
    return header


class _FakeSafeOpen:
    header: dict[str, dict[str, object]] = {}

    def __init__(self, _path: str):
        self.header = type(self).header

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def keys(self):
        return list(self.header)


@pytest.mark.parametrize("blocks", [28, 40])
def test_inspect_anima_checkpoint_detects_supported_depths(
    tmp_path, monkeypatch, blocks
) -> None:
    path = tmp_path / "model.safetensors"
    path.touch()
    _FakeSafeOpen.header = _header(blocks)
    monkeypatch.setattr(checkpoint, "MemoryEfficientSafeOpen", _FakeSafeOpen)

    layout = checkpoint.inspect_anima_checkpoint(path)

    assert layout.num_blocks == blocks
    assert layout.model_channels == 2048
    assert layout.num_heads == 16


def test_inspect_anima_checkpoint_rejects_unknown_depth(tmp_path, monkeypatch) -> None:
    path = tmp_path / "model.safetensors"
    path.touch()
    _FakeSafeOpen.header = _header(29)
    monkeypatch.setattr(checkpoint, "MemoryEfficientSafeOpen", _FakeSafeOpen)

    with pytest.raises(ValueError, match="expected exactly 0..27 or 0..39"):
        checkpoint.inspect_anima_checkpoint(path)


def test_load_anima_model_constructs_detected_block_count(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeAnima:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def to(self, _dtype):
            return self

        def load_state_dict(self, _state, **_kwargs):
            return [], []

    monkeypatch.setattr(weights.anima_models, "Anima", FakeAnima)
    monkeypatch.setattr(weights, "init_empty_weights", nullcontext)
    monkeypatch.setattr(weights, "load_safetensors_with_lora", lambda **_kwargs: {})

    model = weights.load_anima_model(
        "cpu",
        "unused.safetensors",
        "torch",
        "cpu",
        torch.bfloat16,
        checkpoint_layout=_layout(40),
    )

    assert captured["num_blocks"] == 40
    assert model._anima_checkpoint_layout.num_blocks == 40


def _plain_args(**overrides):
    values = {
        "network_module": "networks.lora_anima",
        "network_args": [],
        "network_train_unet_only": True,
        "base_compute": "bf16",
        "vr_loss_weight": 0.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_40_block_training_allows_plain_lora_and_rejects_hydra() -> None:
    assert require_training_compatibility(_plain_args(), _layout(40)) == "plain_lora"
    with pytest.raises(ValueError, match="Hydra/MoE routing"):
        require_training_compatibility(
            _plain_args(network_args=["use_moe_style=shared_A"]), _layout(40)
        )


def test_40_block_adapter_metadata_round_trip_and_cross_depth_refusal() -> None:
    layout = _layout(40)
    base_hash = "a" * 64
    network = SimpleNamespace(
        _anima_checkpoint_layout=layout,
        _anima_base_sha256=base_hash,
    )
    metadata: dict[str, str] = {}

    stamp_lora_save_metadata(
        metadata,
        LoRANetworkCfg(model_family="anima"),
        NETWORK_REGISTRY["lora"],
        network=network,
    )

    assert metadata == adapter_identity_metadata(layout, base_hash)
    validate_adapter_metadata(metadata, network)
    with pytest.raises(ValueError, match="incompatible"):
        validate_adapter_metadata(
            metadata,
            SimpleNamespace(
                _anima_checkpoint_layout=_layout(28),
                _anima_base_sha256=base_hash,
            ),
        )


def test_28_block_adapter_metadata_remains_legacy_unstamped() -> None:
    metadata: dict[str, str] = {}

    stamp_lora_save_metadata(
        metadata,
        LoRANetworkCfg(model_family="anima"),
        NETWORK_REGISTRY["lora"],
        network=SimpleNamespace(_anima_checkpoint_layout=_layout(28)),
    )

    assert not set(metadata).intersection(
        {"ss_anima_arch", "ss_anima_num_blocks", "ss_anima_model_channels"}
    )


def test_40_block_model_rejects_unstamped_adapter() -> None:
    with pytest.raises(ValueError, match="requires adapter architecture metadata"):
        validate_adapter_metadata(
            {}, SimpleNamespace(_anima_checkpoint_layout=_layout(40))
        )
