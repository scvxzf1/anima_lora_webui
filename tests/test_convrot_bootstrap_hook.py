"""Bootstrap thin-hook tests for ConvRot base_compute."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from library.training.bootstrap import TrainingBootstrap


class _FakeLoRAModule(nn.Module):
    def __init__(self, name: str, linear: nn.Linear) -> None:
        super().__init__()
        self.lora_name = f"lora_unet_{name.replace('.', '_')}"
        self.original_name = name
        self.org_module_ref = [linear]
        self.org_forward = linear.forward
        self.lora_down = nn.Linear(linear.in_features, 2, bias=False)
        self.lora_up = nn.Linear(2, linear.out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.org_forward(x) + self.lora_up(self.lora_down(x))


class _FakeNetwork(nn.Module):
    def __init__(self, loras: list[nn.Module]) -> None:
        super().__init__()
        self.unet_loras = nn.ModuleList(loras)


def _args(**kwargs):
    defaults = dict(
        base_compute="bf16",
        convrot_group_size=16,
        convrot_scope="mlp",
        convrot_weight_source="online_from_bf16",
        convrot_prequant_path=None,
        block_swap_transfer_dtype="bf16",
        network_args=None,
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_maybe_apply_convrot_base_noop_for_bf16() -> None:
    linear = nn.Linear(32, 32, bias=False)
    linear.weight.requires_grad_(False)
    network = _FakeNetwork([_FakeLoRAModule("blocks.0.mlp.layer1", linear)])
    applied = TrainingBootstrap.maybe_apply_convrot_base(_args(), network)
    assert applied is False
    assert not hasattr(network.unet_loras[0], "_convrot_quantized_weight")


def test_maybe_apply_convrot_base_patches_w8a16() -> None:
    linear = nn.Linear(32, 32, bias=False)
    linear.weight.requires_grad_(False)
    network = _FakeNetwork([_FakeLoRAModule("blocks.0.mlp.layer1", linear)])
    args = _args(base_compute="w8a16_convrot")
    applied = TrainingBootstrap.maybe_apply_convrot_base(args, network)
    assert applied is True
    assert hasattr(network.unet_loras[0], "_convrot_quantized_weight")
    assert args.base_compute == "w8a16_convrot"


def test_maybe_apply_convrot_base_mutex_with_block_swap_int8() -> None:
    linear = nn.Linear(32, 32, bias=False)
    linear.weight.requires_grad_(False)
    network = _FakeNetwork([_FakeLoRAModule("blocks.0.mlp.layer1", linear)])
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainingBootstrap.maybe_apply_convrot_base(
            _args(
                base_compute="w8a16_convrot",
                block_swap_transfer_dtype="int8",
            ),
            network,
        )


def test_maybe_apply_convrot_base_rejects_dora_network_args() -> None:
    linear = nn.Linear(32, 32, bias=False)
    linear.weight.requires_grad_(False)
    network = _FakeNetwork([_FakeLoRAModule("blocks.0.mlp.layer1", linear)])
    with pytest.raises(ValueError, match="DoRA"):
        TrainingBootstrap.maybe_apply_convrot_base(
            _args(
                base_compute="w8a16_convrot",
                network_args=["dora_wd=true"],
            ),
            network,
        )


def test_maybe_apply_convrot_base_freezes_unet_before_quant() -> None:
    """Mirrors real bootstrap: base Linear still requires_grad until late freeze.

    Regression for hot-test failure where patched=0 because ConvRot ran before
    ``unet.requires_grad_(False)`` in prepare_models_for_accelerator.
    """
    linear = nn.Linear(32, 32, bias=False)
    assert linear.weight.requires_grad is True
    unet = nn.Module()
    unet.linear = linear
    network = _FakeNetwork([_FakeLoRAModule("blocks.0.mlp.layer1", linear)])
    args = _args(base_compute="w8a16_convrot")
    applied = TrainingBootstrap.maybe_apply_convrot_base(args, network, unet=unet)
    assert applied is True
    assert linear.weight.requires_grad is False
    assert hasattr(network.unet_loras[0], "_convrot_quantized_weight")


def test_maybe_apply_convrot_base_freezes_org_refs_when_unet_missing() -> None:
    linear = nn.Linear(32, 32, bias=False)
    assert linear.weight.requires_grad is True
    network = _FakeNetwork([_FakeLoRAModule("blocks.0.mlp.layer1", linear)])
    applied = TrainingBootstrap.maybe_apply_convrot_base(
        _args(base_compute="w8a16_convrot"),
        network,
        unet=None,
    )
    assert applied is True
    assert linear.weight.requires_grad is False
    assert hasattr(network.unet_loras[0], "_convrot_quantized_weight")
