"""CLI-to-bootstrap contracts for experimental Qwen projection packing."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21Attention,
    QwenImage21SwiGLUFeedForward,
    QwenImage21TransformerBlock,
)
from library.models.qwen_image_2_1.fused_mlp import FrozenQwenImage21SwiGLU
from library.models.qwen_image_2_1.packed_qkv import FrozenPackedQKV
from library.models.qwen_image_2_1.training_blocks import install_training_block_projections
from library.training.compat_matrix import check_training_compat
from library.training.cli_args import add_sd_models_arguments
from library.config import schema as config_schema
from library.config.io import _flatten_toml, _render_merged_toml
from networks.lora_modules.lora import LoRAModule


def _args(**overrides):
    values = dict(model_family="qwen_image_2_1", qwen_fused_projections="off",
                  mixed_precision="bf16", base_compute="bf16",
                  block_swap_transfer_dtype="bf16", gradient_checkpointing=True,
                  selective_checkpoint="off", network_module="networks.lora_anima")
    values.update(overrides)
    return SimpleNamespace(**values)


def _fixture(block_count=2, *, native_blocks=False):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = QwenImage21Attention(dim=8, heads=2, dim_head=4)
            self.img_mlp = QwenImage21SwiGLUFeedForward(8, 24)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = nn.ModuleList(
                QwenImage21TransformerBlock(8, 2, 4, mlp_ratio=2)
                if native_blocks else Block() for _ in range(block_count)
            )
            self.gradient_checkpointing = True
            self._gradient_checkpointing_func = lambda fn, *args, **kwargs: fn(*args, **kwargs)

    model = Model().to(torch.bfloat16)
    loras = []
    for block in model.transformer_blocks:
        for name in ("to_q", "to_k", "to_v"):
            lora = LoRAModule(name, getattr(block.attn, name), lora_dim=2, alpha=2).to(torch.bfloat16)
            lora.apply_to()
            loras.append(lora)
    return model, SimpleNamespace(unet_loras=loras, text_encoder_loras=[])


@pytest.mark.parametrize("mode,expected", [("off", (0, 0)), ("mlp", (2, 0)),
                                             ("qkv", (0, 2)), ("all", (2, 2))])
def test_real_diffusers_blocks_with_applied_loras(mode, expected):
    model, network = _fixture()
    result = install_training_block_projections(_args(qwen_fused_projections=mode), model, network)
    assert result == expected
    for block in model.transformer_blocks:
        assert isinstance(block.img_mlp, FrozenQwenImage21SwiGLU) == (mode in {"mlp", "all"})
        assert isinstance(getattr(block.attn, "packed_qkv", None), FrozenPackedQKV) == (mode in {"qkv", "all"})
        if mode in {"mlp", "all"}:
            assert not any(p.requires_grad for p in block.img_mlp.parameters())
    if mode in {"qkv", "all"}:
        x = torch.randn(1, 4, 8, dtype=torch.bfloat16, requires_grad=True)
        model.transformer_blocks[0].attn.packed_qkv(x)[0].float().sum().backward()
        assert x.grad is not None
        assert network.unet_loras[0].lora_down.weight.grad is not None


@pytest.mark.parametrize("override", [
    {"model_family": "anima"}, {"mixed_precision": "fp16"},
    {"base_compute": "nf4"}, {"block_swap_transfer_dtype": "int8"},
    {"gradient_checkpointing": False}, {"selective_checkpoint": "every_other"},
    {"cpu_offload_checkpointing": True},
])
def test_rejects_unsupported_modes_before_mutating_model(override):
    model, network = _fixture()
    with pytest.raises(ValueError):
        install_training_block_projections(_args(qwen_fused_projections="all", **override), model, network)
    assert all(not hasattr(block.attn, "packed_qkv") for block in model.transformer_blocks)
    assert all(isinstance(block.img_mlp, QwenImage21SwiGLUFeedForward) for block in model.transformer_blocks)


def test_all_preflights_qkv_before_fusing_any_mlp():
    model, network = _fixture()
    network.unet_loras.pop()
    with pytest.raises(TypeError, match="three plain"):
        install_training_block_projections(_args(qwen_fused_projections="all"), model, network)
    assert all(isinstance(block.img_mlp, QwenImage21SwiGLUFeedForward) for block in model.transformer_blocks)


@pytest.mark.parametrize("failure,expected", [
    ("count", "within 1..block_count"),
    ("checkpointing", "requires gradient checkpointing"),
    ("block_type", "requires native Qwen Image 2.1 blocks"),
    ("checkpoint_func", "requires a checkpoint function"),
])
def test_saved_projection_preflight_preserves_modules_and_lora_refs(failure, expected):
    model, network = _fixture(native_blocks=failure != "block_type")
    args = _args(qwen_fused_projections="all", qwen_saved_projection_blocks=1,
                 qwen_projection_budget_mib=1)
    if failure == "count":
        args.qwen_saved_projection_blocks = 3
    elif failure == "checkpointing":
        model.gradient_checkpointing = False
    elif failure == "checkpoint_func":
        del model._gradient_checkpointing_func

    modules = [
        (block.img_mlp, block.attn, block.attn.to_q, block.attn.to_k, block.attn.to_v)
        for block in model.transformer_blocks
    ]
    lora_refs = [(lora.org_module_ref[0], lora.org_forward) for lora in network.unet_loras]
    with pytest.raises((ValueError, TypeError), match=expected):
        install_training_block_projections(args, model, network)
    for block, original in zip(model.transformer_blocks, modules):
        assert all(current is before for current, before in zip(
            (block.img_mlp, block.attn, block.attn.to_q, block.attn.to_k, block.attn.to_v), original,
        ))
        assert not hasattr(block.attn, "packed_qkv")
    assert all(lora.org_module_ref[0] is module and lora.org_forward is forward
               for lora, (module, forward) in zip(network.unet_loras, lora_refs))


def test_rejects_non_attention_lora_and_non_bf16_weights():
    model, network = _fixture()
    extra = LoRAModule("mlp", model.transformer_blocks[0].img_mlp.out, lora_dim=2, alpha=2).to(torch.bfloat16)
    extra.apply_to()
    network.unet_loras.append(extra)
    with pytest.raises((ValueError, RuntimeError), match="attention-only|adapter"):
        install_training_block_projections(_args(qwen_fused_projections="all"), model, network)
    model, network = _fixture()
    model.transformer_blocks[1].img_mlp.float()
    with pytest.raises(ValueError, match="BF16 MLP"):
        install_training_block_projections(_args(qwen_fused_projections="all"), model, network)
    assert all(isinstance(block.img_mlp, QwenImage21SwiGLUFeedForward) for block in model.transformer_blocks)


def test_compat_matrix_rejects_family_transfer_and_invalid_mode():
    assert any(issue.code == "qwen_fused_projections_family" for issue in check_training_compat({
        "model_family": "anima", "qwen_fused_projections": "all"}).errors)
    assert any(issue.code == "qwen_fused_projections_block_swap_transfer_dtype" for issue in check_training_compat({
        "model_family": "qwen_image_2_1", "qwen_fused_projections": "all",
        "block_swap_transfer_dtype": "int8", "gradient_checkpointing": True}).errors)
    assert any(issue.code == "invalid_qwen_fused_projections" for issue in check_training_compat({
        "model_family": "qwen_image_2_1", "qwen_fused_projections": "invalid"}).errors)


def test_bootstrap_installation_order_source_contract():
    source = (Path(__file__).resolve().parents[1] / "library/training/bootstrap.py").read_text()
    body = source[source.index("network.apply_to(text_encoder, unet"):]
    assert body.index("network.load_weights(args.network_weights)") < body.index("install_training_block_projections(args, unet, network)")
    assert body.index("install_training_block_projections(args, unet, network)") < body.index("compile_qwen_image_2_1_blocks(")


def test_cli_toml_and_snapshot_keep_experiment_opt_in():
    parser = argparse.ArgumentParser()
    add_sd_models_arguments(parser)
    config_schema.populate_schema(parser)
    assert parser.parse_args([]).qwen_fused_projections == "off"
    assert parser.parse_args(["--qwen_fused_projections", "all"]).qwen_fused_projections == "all"
    values = _flatten_toml({"qwen_fused_projections": "mlp"}, strict=True)
    assert values["qwen_fused_projections"] == "mlp"
    args = parser.parse_args([])
    args.qwen_fused_projections = values["qwen_fused_projections"]
    snapshot = _render_merged_toml(args, parser, {"qwen_fused_projections": "method.toml"})
    assert 'qwen_fused_projections = "mlp"' in snapshot
