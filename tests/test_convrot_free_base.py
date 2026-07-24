"""Tests for freeing dual-resident bf16 base weights after ConvRot quant."""

from __future__ import annotations

import torch
from torch import nn

from library.runtime.convrot.apply import apply_convrot_to_lora_network
from library.runtime.convrot.free_base import free_linear_weight_storage, is_base_weight_freed


class _FakeLoRAModule(nn.Module):
    def __init__(self, name: str, linear: nn.Linear) -> None:
        super().__init__()
        self.lora_name = f"lora_unet_{name.replace('.', '_')}"
        self.original_name = name
        self.org_module_ref = [linear]
        self.org_forward = linear.forward
        self.lora_down = nn.Linear(linear.in_features, 4, bias=False)
        self.lora_up = nn.Linear(4, linear.out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.org_forward(x) + self.lora_up(self.lora_down(x))


class _FakeLoRANetwork(nn.Module):
    def __init__(self, loras: list[nn.Module]) -> None:
        super().__init__()
        self.unet_loras = nn.ModuleList(loras)


def test_free_linear_weight_storage_uses_meta_and_is_idempotent() -> None:
    linear = nn.Linear(32, 16, bias=False)
    linear.weight.requires_grad_(False)
    nbytes = free_linear_weight_storage(linear)
    assert nbytes == 32 * 16 * 4  # fp32 default
    assert is_base_weight_freed(linear)
    assert linear.weight.device.type == "meta"
    assert free_linear_weight_storage(linear) == 0


def test_apply_frees_base_weight_by_default() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(32, 32, bias=False)
    linear.weight.requires_grad_(False)
    # move to cuda if available to assert storage release intent
    device = torch.device("cpu")
    linear = linear.to(device)
    network = _FakeLoRANetwork([_FakeLoRAModule("blocks.0.mlp.layer1", linear)])
    x = torch.randn(2, 32, device=device)
    y_before = linear(x)

    result = apply_convrot_to_lora_network(
        network, mode="w8a16", scope="mlp", group_size=16
    )
    assert result.patched_count == 1
    assert result.freed_modules == 1
    assert result.freed_bytes > 0
    assert is_base_weight_freed(linear)
    assert linear.weight.device.type == "meta"

    # org_forward still works via int8 buffers on the LoRA module
    y_after = network.unet_loras[0].org_forward(x)
    rel = (y_after - y_before).norm() / y_before.norm().clamp_min(1e-8)
    assert rel.item() < 0.05


def test_apply_can_keep_base_weights() -> None:
    linear = nn.Linear(32, 32, bias=False)
    linear.weight.requires_grad_(False)
    network = _FakeLoRANetwork([_FakeLoRAModule("blocks.0.mlp.layer1", linear)])
    result = apply_convrot_to_lora_network(
        network,
        mode="w8a16",
        scope="mlp",
        group_size=16,
        free_base_weights=False,
    )
    assert result.patched_count == 1
    assert result.freed_modules == 0
    assert not is_base_weight_freed(linear)
    assert linear.weight.device.type != "meta"


def test_dry_run_does_not_free_base_weights() -> None:
    linear = nn.Linear(32, 32, bias=False)
    linear.weight.requires_grad_(False)
    network = _FakeLoRANetwork([_FakeLoRAModule("blocks.0.mlp.layer1", linear)])
    result = apply_convrot_to_lora_network(
        network, mode="w8a16", scope="mlp", group_size=16, dry_run=True
    )
    assert result.patched_count == 1
    assert result.freed_modules == 0
    assert not is_base_weight_freed(linear)
