"""Metadata-flow regression tests for ``create_network_from_weights``.

``load_file()`` discards safetensors ``__metadata__``, so a caller that
pre-loads tensors and passes ``weights_sd=`` used to silently drop the
three-axis routing stamps (ss_use_moe_style / ss_route_per_layer /
ss_router_source) and trip the "missing three-axis stamps" raise in
``LoRANetworkCfg.from_weights`` — blaming the checkpoint for a call-site fault.

These tests pin the de-footgun: metadata reaches the cfg via the explicit
``metadata=`` channel, via ``file=`` even when ``weights_sd=`` is also given,
via the plain ``file=`` path, and that the bare ``weights_sd=`` case raises an
error that names the real cause.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest
from safetensors.torch import save_file

from networks.lora_anima.factory import create_network_from_weights


# Class name must be "Block" to match LoRANetwork.ANIMA_TARGET_REPLACE_MODULE.
class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)


class _TinyDiT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block = _Block()


_LORA = "lora_unet_block_proj"
_RANK = 4
_NUM_EXPERTS = 3
_MOE_META = {
    "ss_use_moe_style": "shared_A",
    "ss_route_per_layer": "True",
    "ss_router_source": "input",
}
_INDEPENDENT_A_META = {
    "ss_use_moe_style": "independent_A",
    "ss_route_per_layer": "False",
    "ss_router_source": "fei",
    "ss_fei_feature_dim": "2",
}


def _moe_state_dict() -> dict[str, torch.Tensor]:
    """Synthetic Hydra-moe (shared_A, per-layer input router) state dict.

    Only the key shapes are sniffed by the factory — the tensors are never
    loaded here (``create_network_from_weights`` returns before any
    ``load_state_dict``), so random values are fine.
    """
    return {
        f"{_LORA}.lora_down.weight": torch.randn(_RANK, 8),
        f"{_LORA}.lora_up_weight": torch.randn(_NUM_EXPERTS, 8, _RANK),
        f"{_LORA}.router.weight": torch.randn(_NUM_EXPERTS, _RANK),
        f"{_LORA}.alpha": torch.tensor(float(_RANK)),
    }


def _stacked_experts_state_dict() -> dict[str, torch.Tensor]:
    """Synthetic StackedExperts / FeRA-form state dict with independent-A keys."""
    return {
        f"{_LORA}.lora_down_weight": torch.randn(_NUM_EXPERTS, _RANK, 8),
        f"{_LORA}.lora_up_weight": torch.randn(_NUM_EXPERTS, 8, _RANK),
        f"{_LORA}.alpha": torch.tensor(float(_RANK)),
        "global_router.net.0.weight": torch.randn(4, 2),
    }


def _split_hydra_sigma_mlp_state_dict() -> dict[str, torch.Tensor]:
    """Synthetic split q/k/v Hydra dict carrying old sigma_mlp keys."""
    shared = "lora_unet_blocks_0_self_attn_"
    state_dict: dict[str, torch.Tensor] = {}
    for letter in ("q", "k", "v"):
        prefix = f"{shared}{letter}_proj"
        state_dict[f"{prefix}.lora_down.weight"] = torch.randn(_RANK, 8)
        state_dict[f"{prefix}.lora_up_weight"] = torch.randn(_NUM_EXPERTS, 4, _RANK)
        state_dict[f"{prefix}.router.weight"] = torch.randn(_NUM_EXPERTS, _RANK)
        state_dict[f"{prefix}.router.bias"] = torch.randn(_NUM_EXPERTS)
        state_dict[f"{prefix}.sigma_mlp.0.weight"] = torch.randn(_RANK, _RANK)
        state_dict[f"{prefix}.alpha"] = torch.tensor(float(_RANK))
    return state_dict


def _build(**kwargs):
    network, _sd = create_network_from_weights(
        multiplier=1.0,
        ae=None,
        text_encoders=[],
        unet=_TinyDiT(),
        for_inference=True,
        **kwargs,
    )
    return network


def _assert_axes(network) -> None:
    assert network.cfg.use_moe_style == "shared_A"
    assert network.cfg.route_per_layer is True
    assert network.cfg.router_source == "input"


def test_metadata_kwarg_lands_three_axes():
    """Explicit ``metadata=`` carries the stamps even with a pre-loaded sd."""
    net = _build(file=None, weights_sd=_moe_state_dict(), metadata=dict(_MOE_META))
    _assert_axes(net)


def test_file_path_reads_metadata(tmp_path):
    """Regression: the plain ``file=`` path still reads the stamps."""
    path = tmp_path / "moe.safetensors"
    save_file(_moe_state_dict(), str(path), metadata=_MOE_META)
    net = _build(file=str(path), weights_sd=None)
    _assert_axes(net)


def test_file_recovers_metadata_when_weights_supplied(tmp_path):
    """``file=`` + ``weights_sd=`` together must still recover the stamps.

    Previously the read was gated on ``weights_sd is None`` so a caller that
    supplied both lost the metadata anyway.
    """
    path = tmp_path / "moe.safetensors"
    save_file(_moe_state_dict(), str(path), metadata=_MOE_META)
    net = _build(file=str(path), weights_sd=_moe_state_dict())
    _assert_axes(net)


def test_explicit_metadata_overrides_file_metadata(tmp_path):
    """Explicit ``metadata=`` wins when the safetensors file has stale stamps."""
    path = tmp_path / "moe_stale.safetensors"
    stale_meta = {
        "ss_use_moe_style": "shared_A",
        "ss_route_per_layer": "False",
        "ss_router_source": "fei",
        "ss_fei_feature_dim": "2",
    }
    save_file(_moe_state_dict(), str(path), metadata=stale_meta)

    net = _build(
        file=str(path),
        weights_sd=_moe_state_dict(),
        metadata=dict(_MOE_META),
    )

    _assert_axes(net)


def test_independent_a_metadata_kwarg_lands_three_axes_and_expert_count():
    """Independent-A metadata must preserve the checkpoint expert count."""
    net = _build(
        file=None,
        weights_sd=_stacked_experts_state_dict(),
        metadata=dict(_INDEPENDENT_A_META),
    )

    assert net.cfg.use_moe_style == "independent_A"
    assert net.cfg.route_per_layer is False
    assert net.cfg.router_source == "fei"
    assert net.cfg.fei_feature_dim == 2
    assert net.cfg.num_experts == _NUM_EXPERTS
    assert net.global_router is not None
    assert net.global_router.num_experts == _NUM_EXPERTS


def test_bare_weights_sd_raises_actionable_error():
    """No metadata, no file → loud error naming load_file / metadata=."""
    import pytest

    with pytest.raises(RuntimeError) as exc:
        _build(file=None, weights_sd=_moe_state_dict())
    msg = str(exc.value)
    assert "three-axis" in msg
    assert "load_file" in msg
    assert "metadata=" in msg


def test_bare_stacked_experts_weights_sd_raises_actionable_error():
    """Independent-A MoE keys also need safetensors metadata."""
    import pytest

    with pytest.raises(RuntimeError) as exc:
        _build(file=None, weights_sd=_stacked_experts_state_dict())
    msg = str(exc.value)
    assert "three-axis" in msg
    assert "load_file" in msg
    assert "metadata=" in msg


def test_split_hydra_sigma_mlp_keys_are_rejected_after_refuse():
    """Split legacy sigma_mlp keys must not silently survive load refusion."""
    import pytest

    with pytest.raises(RuntimeError, match="legacy σ-router") as exc:
        _build(
            file=None,
            weights_sd=_split_hydra_sigma_mlp_state_dict(),
            metadata=dict(_MOE_META),
        )
    assert "sigma_mlp" in str(exc.value)


def test_old_global_hydra_router_keys_are_rejected() -> None:
    """Old global Hydra router checkpoints must fail before cfg inference."""
    import pytest

    state_dict = _moe_state_dict()
    state_dict["_hydra_router.net.0.weight"] = torch.randn(4, _RANK)

    with pytest.raises(RuntimeError, match="old global HydraLoRA router") as exc:
        _build(file=None, weights_sd=state_dict, metadata=dict(_MOE_META))
    assert "_hydra_router" in str(exc.value)


# ── Model family stamp (Krea-2-Raw migration, stage 6) ──

def _plain_lora_state_dict() -> dict[str, torch.Tensor]:
    """Synthetic plain-LoRA state dict for the ``_Block.proj`` Linear."""
    return {
        f"{_LORA}.lora_down.weight": torch.randn(_RANK, 8),
        f"{_LORA}.lora_up.weight": torch.randn(8, _RANK),
        f"{_LORA}.alpha": torch.tensor(float(_RANK)),
    }


def test_ss_model_family_krea2_read_into_cfg() -> None:
    """A Krea-2 checkpoint's ss_model_family stamp lands on the cfg."""
    meta = {"ss_model_family": "krea2_raw"}
    net = _build(file=None, weights_sd=_plain_lora_state_dict(), metadata=meta)
    assert net.cfg.model_family == "krea2_raw"


def test_ss_model_family_absent_defaults_to_anima() -> None:
    """Anima / old unstamped checkpoints load as anima (absence = anima)."""
    net = _build(file=None, weights_sd=_plain_lora_state_dict(), metadata={})
    assert net.cfg.model_family == "anima"


def test_ss_model_family_unknown_is_rejected() -> None:
    """Only an absent stamp may use the legacy Anima default."""
    meta = {"ss_model_family": "unknown_family"}
    with pytest.raises(ValueError, match="checkpoint ss_model_family"):
        _build(file=None, weights_sd=_plain_lora_state_dict(), metadata=meta)


def test_ss_model_family_round_trip_through_stamp(tmp_path) -> None:
    """stamp_lora_save_metadata writes ss_model_family only for non-anima."""
    from networks.lora_anima.config import LoRANetworkCfg
    from networks.lora_anima.persistence import stamp_lora_save_metadata
    from networks.registry import NETWORK_REGISTRY

    # Build a Krea-2 cfg (family set + SingleStreamBlock target container).
    krea2_cfg = LoRANetworkCfg(
        unet_target_replace_modules=["SingleStreamBlock"],
        model_family="krea2_raw",
    )
    krea2_meta: dict[str, str] = {}
    stamp_lora_save_metadata(
        krea2_meta, krea2_cfg, NETWORK_REGISTRY["lora"], network=None
    )
    assert krea2_meta.get("ss_model_family") == "krea2_raw"
    assert krea2_meta.get("ss_unet_target_replace_modules") == '["SingleStreamBlock"]'

    # Anima cfg: family field left default → stamp must OMIT ss_model_family
    # so anima checkpoints stay byte-identical (key absent, not "anima").
    anima_cfg = LoRANetworkCfg()
    anima_meta: dict[str, str] = {}
    stamp_lora_save_metadata(
        anima_meta, anima_cfg, NETWORK_REGISTRY["lora"], network=None
    )
    assert "ss_model_family" not in anima_meta
    assert "ss_unet_target_replace_modules" not in anima_meta
