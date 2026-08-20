from __future__ import annotations

from types import SimpleNamespace

import pytest

from library.env import normalize_model_family, resolve_model_family
from library.training.bootstrap import TrainingBootstrap
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.persistence import stamp_lora_save_metadata
from networks.registry import NETWORK_REGISTRY


def test_normalize_model_family_accepts_only_canonical_values() -> None:
    assert normalize_model_family(" AnImA ") == "anima"
    assert normalize_model_family("KREA2_RAW") == "krea2_raw"
    assert normalize_model_family("", allow_empty=True) == ""
    with pytest.raises(ValueError, match="unknown"):
        normalize_model_family("unknown")


def test_resolve_model_family_rejects_unknown_args_and_env(monkeypatch) -> None:
    monkeypatch.delenv("ANIMA_MODEL_FAMILY", raising=False)
    with pytest.raises(ValueError, match="args.model_family"):
        resolve_model_family(SimpleNamespace(model_family="typo"))

    monkeypatch.setenv("ANIMA_MODEL_FAMILY", "typo")
    with pytest.raises(ValueError, match="ANIMA_MODEL_FAMILY"):
        resolve_model_family(SimpleNamespace(model_family=None))


def test_bootstrap_allows_matching_network_family(monkeypatch) -> None:
    monkeypatch.delenv("ANIMA_MODEL_FAMILY", raising=False)
    args = SimpleNamespace(
        model_family="krea2_raw",
        network_args=["model_family=krea2_raw"],
    )

    net_kwargs = TrainingBootstrap.build_net_kwargs(args)

    assert net_kwargs["model_family"] == "krea2_raw"
    assert net_kwargs["unet_target_replace_modules"] == '["SingleStreamBlock"]'


def test_bootstrap_rejects_conflicting_network_family(monkeypatch) -> None:
    monkeypatch.delenv("ANIMA_MODEL_FAMILY", raising=False)
    args = SimpleNamespace(
        model_family="anima",
        network_args=["model_family=krea2_raw"],
    )

    with pytest.raises(ValueError, match="conflicts with the runtime family"):
        TrainingBootstrap.build_net_kwargs(args)


def test_lora_config_and_persistence_reject_unknown_family() -> None:
    with pytest.raises(ValueError, match="LoRA network model_family"):
        LoRANetworkCfg.from_kwargs(
            {"model_family": "unknown"},
            network_dim=4,
            network_alpha=4,
            neuron_dropout=None,
            module_class=NETWORK_REGISTRY["lora"].module_class,
        )

    metadata: dict[str, str] = {}
    with pytest.raises(ValueError, match="LoRA checkpoint model_family"):
        stamp_lora_save_metadata(
            metadata,
            LoRANetworkCfg(model_family="unknown"),
            NETWORK_REGISTRY["lora"],
        )
