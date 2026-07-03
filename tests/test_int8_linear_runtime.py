from __future__ import annotations

import torch
from torch import nn

from library.runtime.int8_linear import (
    Int8FrozenLinear,
    classify_frozen_linear_module,
    replace_frozen_base_linears_with_int8,
)


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.layer1 = nn.Linear(8, 16, bias=False)
        self.mlp.layer2 = nn.Linear(16, 8, bias=False)
        self.self_attn = nn.Module()
        self.self_attn.qkv_proj = nn.Linear(8, 24, bias=False)
        self.self_attn.output_proj = nn.Linear(8, 8, bias=False)
        self.adaln_up_mlp = nn.Linear(2, 8, bias=False)
        self.trainable_adapter = nn.Linear(8, 8, bias=False)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.requires_grad_(False)
        self.trainable_adapter.weight.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.mlp.layer1(x)
        hidden = torch.nn.functional.gelu(hidden)
        return self.mlp.layer2(hidden)


class _TinyAnima(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_TinyBlock()])
        self.final_layer = nn.Linear(8, 8, bias=False)
        self.final_layer.weight.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks[0](x)


def test_classify_frozen_linear_module_scopes() -> None:
    assert classify_frozen_linear_module("blocks.0.mlp.layer1") == (0, "mlp")
    assert classify_frozen_linear_module("net.blocks.0.mlp.layer1") == (0, "mlp")
    assert classify_frozen_linear_module("blocks.0.mlp.layer2") == (0, "mlp")
    assert classify_frozen_linear_module("blocks.0.self_attn.qkv_proj") is None
    assert classify_frozen_linear_module("blocks.0.self_attn.qkv_proj", scope="attention") == (
        0,
        "attention",
    )
    assert classify_frozen_linear_module("final_layer") is None
    assert classify_frozen_linear_module("blocks.0.adaln_up_mlp", scope="all") is None


def test_int8_frozen_linear_matches_float_linear_closely() -> None:
    torch.manual_seed(0)
    linear = nn.Linear(5, 7, bias=False)
    linear.weight.requires_grad_(False)
    wrapped = Int8FrozenLinear.from_linear(linear)
    x = torch.randn(3, 5)

    expected = linear(x)
    actual = wrapped(x)

    assert wrapped.quantized_weight.dtype is torch.int8
    assert wrapped.scale.shape == (7,)
    assert not list(wrapped.parameters())
    assert actual.shape == expected.shape
    rel = (actual - expected).norm() / expected.norm()
    assert rel.item() < 0.02


def test_replace_frozen_base_linears_with_int8_respects_scope_and_trainable_weights() -> None:
    torch.manual_seed(1)
    model = _TinyAnima()
    x = torch.randn(2, 8)
    expected = model(x)

    replacements = replace_frozen_base_linears_with_int8(model, scope="mlp")
    actual = model(x)

    assert [item.name for item in replacements] == [
        "blocks.0.mlp.layer1",
        "blocks.0.mlp.layer2",
    ]
    assert isinstance(model.blocks[0].mlp.layer1, Int8FrozenLinear)
    assert isinstance(model.blocks[0].mlp.layer2, Int8FrozenLinear)
    assert isinstance(model.blocks[0].self_attn.qkv_proj, nn.Linear)
    assert isinstance(model.blocks[0].adaln_up_mlp, nn.Linear)
    assert isinstance(model.blocks[0].trainable_adapter, nn.Linear)
    assert isinstance(model.final_layer, nn.Linear)
    rel = (actual - expected).norm() / expected.norm()
    assert rel.item() < 0.03


def test_replace_frozen_base_linears_can_dry_run_attention_without_mutating() -> None:
    model = _TinyAnima()

    replacements = replace_frozen_base_linears_with_int8(model, scope="all", dry_run=True)

    assert [item.name for item in replacements] == [
        "blocks.0.mlp.layer1",
        "blocks.0.mlp.layer2",
        "blocks.0.self_attn.qkv_proj",
        "blocks.0.self_attn.output_proj",
    ]
    assert all(item.payload_bytes < item.bf16_bytes for item in replacements)
    assert isinstance(model.blocks[0].mlp.layer1, nn.Linear)
    assert isinstance(model.blocks[0].self_attn.qkv_proj, nn.Linear)
