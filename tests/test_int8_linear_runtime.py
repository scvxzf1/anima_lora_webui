from __future__ import annotations

import torch
from torch import nn

from library.runtime.int8_linear import (
    Int8LoRABaseForwardPatch,
    Int8FrozenLinear,
    classify_frozen_linear_module,
    patch_lora_frozen_base_forwards_with_int8,
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


class _FakeLoRAModule(nn.Module):
    def __init__(self, name: str, linear: nn.Linear) -> None:
        super().__init__()
        self.lora_name = f"lora_unet_{name.replace('.', '_')}"
        self.original_name = name
        self.org_module_ref = [linear]
        self.org_forward = linear.forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.org_forward(x)


class _FakeLoRANetwork(nn.Module):
    def __init__(self, loras: list[_FakeLoRAModule]) -> None:
        super().__init__()
        self.unet_loras = nn.ModuleList(loras)


def test_classify_frozen_linear_module_scopes() -> None:
    assert classify_frozen_linear_module("blocks.0.mlp.layer1") == (0, "mlp")
    assert classify_frozen_linear_module("net.blocks.0.mlp.layer1") == (0, "mlp")
    assert classify_frozen_linear_module("blocks.0.mlp.layer2") == (0, "mlp")
    assert classify_frozen_linear_module("blocks.0.self_attn.qkv_proj") is None
    assert classify_frozen_linear_module("blocks.0.self_attn.qkv_proj", scope="attention") == (
        0,
        "attention",
    )
    assert classify_frozen_linear_module("blocks.0.self_attn.qkv_proj", scope="self_attn_qkv") == (
        0,
        "attention",
    )
    assert classify_frozen_linear_module("blocks.0.self_attn.output_proj", scope="self_attn_qkv") is None
    assert classify_frozen_linear_module("blocks.0.self_attn.output_proj", scope="self_attn_out") == (
        0,
        "attention",
    )
    assert classify_frozen_linear_module("blocks.0.cross_attn.kv_proj", scope="cross_attn_kv") == (
        0,
        "attention",
    )
    assert classify_frozen_linear_module("blocks.0.cross_attn.q_proj", scope="cross_attn_kv") is None
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


def test_replace_frozen_base_linears_accepts_projection_subsets() -> None:
    model = _TinyAnima()

    replacements = replace_frozen_base_linears_with_int8(
        model,
        scope="mlp,self_attn_out",
        dry_run=True,
    )

    assert [item.name for item in replacements] == [
        "blocks.0.mlp.layer1",
        "blocks.0.mlp.layer2",
        "blocks.0.self_attn.output_proj",
    ]


def test_patch_lora_frozen_base_forwards_with_int8_respects_scope() -> None:
    torch.manual_seed(2)
    mlp = nn.Linear(8, 16, bias=False)
    attn = nn.Linear(8, 8, bias=False)
    trainable = nn.Linear(8, 16, bias=False)
    mlp.weight.requires_grad_(False)
    attn.weight.requires_grad_(False)
    trainable.weight.requires_grad_(True)
    network = _FakeLoRANetwork(
        [
            _FakeLoRAModule("blocks.0.mlp.layer1", mlp),
            _FakeLoRAModule("blocks.0.self_attn.output_proj", attn),
            _FakeLoRAModule("blocks.0.mlp.layer2", trainable),
        ]
    )
    x = torch.randn(3, 8)
    expected = network.unet_loras[0](x)

    patches = patch_lora_frozen_base_forwards_with_int8(network, scope="mlp")
    actual = network.unet_loras[0](x)

    assert patches == [
        Int8LoRABaseForwardPatch(
            lora_name="lora_unet_blocks_0_mlp_layer1",
            name="blocks.0.mlp.layer1",
            family="mlp",
            block_idx=0,
            shape=(16, 8),
            bf16_bytes=256,
            payload_bytes=192,
        )
    ]
    assert hasattr(network.unet_loras[0], "_int8_base_quantized_weight")
    assert not hasattr(network.unet_loras[1], "_int8_base_quantized_weight")
    assert not hasattr(network.unet_loras[2], "_int8_base_quantized_weight")
    rel = (actual - expected).norm() / expected.norm()
    assert rel.item() < 0.03
