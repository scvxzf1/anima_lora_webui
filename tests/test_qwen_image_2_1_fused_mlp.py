"""CPU contracts for opt-in Qwen Image 2.1 frozen MLP packing."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21SwiGLUFeedForward,
)
from library.models.qwen_image_2_1.fused_mlp import (
    FrozenQwenImage21SwiGLU,
    fuse_qwen_image_2_1_mlp,
    install_qwen_image_2_1_fused_mlps,
)
from library.models.qwen_image_2_1.lora_targets import qwen_image_2_1_target_kwargs
from library.runtime.offloading import ModelOffloader


def _mlp(dtype: torch.dtype = torch.float32) -> QwenImage21SwiGLUFeedForward:
    return QwenImage21SwiGLUFeedForward(8, 24).to(dtype=dtype)


def _model(dtype: torch.dtype = torch.float32, count: int = 3) -> nn.Module:
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.img_mlp = _mlp(dtype)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = nn.ModuleList(Block() for _ in range(count))

    return Model()


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_forward_and_input_gradient_match_frozen_base(dtype: torch.dtype) -> None:
    original = _mlp(dtype).eval().requires_grad_(False)
    fused = fuse_qwen_image_2_1_mlp(original)
    assert not fused.training
    assert fused.out is original.out
    assert fused.gate_up.weight.device == original.gate_layer.weight.device
    assert fused.gate_up.weight.dtype == dtype
    assert not any(p.requires_grad for p in fused.parameters())
    assert set(fused.state_dict()) == {"gate_up.weight", "out.weight"}
    assert torch.equal(fused.gate_up.weight[:24], original.gate_layer.weight)
    assert torch.equal(fused.gate_up.weight[24:], original.proj.weight)

    x = torch.randn(2, 5, 8, dtype=dtype)
    x_original = x.clone().requires_grad_()
    x_fused = x.clone().requires_grad_()
    y_original = original(x_original)
    y_fused = fused(x_fused)
    torch.testing.assert_close(y_fused, y_original, rtol=1e-13 if dtype == torch.float64 else 2e-6, atol=1e-13 if dtype == torch.float64 else 2e-6)
    grad = torch.randn_like(y_original)
    y_original.backward(grad)
    y_fused.backward(grad)
    torch.testing.assert_close(x_fused.grad, x_original.grad, rtol=1e-13 if dtype == torch.float64 else 2e-6, atol=1e-13 if dtype == torch.float64 else 2e-6)


def test_checkpoint_recomputation_and_state_dict_roundtrip() -> None:
    original = _mlp().requires_grad_(False)
    fused = fuse_qwen_image_2_1_mlp(original)
    x = torch.randn(2, 3, 8, requires_grad=True)
    reference = x.detach().clone().requires_grad_()
    checkpoint(fused, x, use_reentrant=False).square().sum().backward()
    checkpoint(original, reference, use_reentrant=False).square().sum().backward()
    torch.testing.assert_close(x.grad, reference.grad)

    state = copy.deepcopy(fused.state_dict())
    loaded = fuse_qwen_image_2_1_mlp(_mlp().requires_grad_(False))
    loaded.load_state_dict(state, strict=True)
    torch.testing.assert_close(loaded(x.detach()), fused(x.detach()))


def test_explicit_freeze_and_reject_patched_mlp_adapter() -> None:
    source = _mlp()
    with pytest.raises(RuntimeError, match="pass freeze=True"):
        fuse_qwen_image_2_1_mlp(source)
    fused = fuse_qwen_image_2_1_mlp(source, freeze=True)
    assert not any(parameter.requires_grad for parameter in fused.parameters())

    source = _mlp().requires_grad_(False)
    source.gate_layer.forward = lambda x: nn.functional.linear(x, source.gate_layer.weight)
    with pytest.raises(RuntimeError, match="attached MLP adapter"):
        fuse_qwen_image_2_1_mlp(source)

    source = _mlp().requires_grad_(False)
    source.activation_fn.register_forward_hook(lambda _module, _input, output: output)
    with pytest.raises(RuntimeError, match="activation hooks"):
        fuse_qwen_image_2_1_mlp(source)


def test_rejects_incompatible_shape_and_preserves_child_training_flags() -> None:
    source = _mlp().requires_grad_(False)
    source.proj = nn.Linear(7, 24, bias=False)
    with pytest.raises(ValueError, match="shapes differ"):
        fuse_qwen_image_2_1_mlp(source)

    source = _mlp().requires_grad_(False)
    source.gate_layer.eval()
    source.out.eval()
    fused = fuse_qwen_image_2_1_mlp(source)
    assert fused.training == source.training
    assert fused.gate_up.training == source.gate_layer.training
    assert fused.out.training == source.out.training


def test_model_install_preflight_and_attention_only_target_unchanged() -> None:
    model = _model()
    with pytest.raises(RuntimeError, match="pass freeze=True"):
        install_qwen_image_2_1_fused_mlps(model)
    assert all(isinstance(block.img_mlp, QwenImage21SwiGLUFeedForward) for block in model.transformer_blocks)

    original_out = [block.img_mlp.out for block in model.transformer_blocks]
    assert install_qwen_image_2_1_fused_mlps(model, freeze=True) == 3
    for block, out in zip(model.transformer_blocks, original_out):
        assert isinstance(block.img_mlp, FrozenQwenImage21SwiGLU)
        assert block.img_mlp.out is out
        assert all("gate_layer" not in key and ".proj." not in key for key in block.state_dict())
    with pytest.raises(TypeError, match="unmodified"):
        install_qwen_image_2_1_fused_mlps(model, freeze=True)
    assert qwen_image_2_1_target_kwargs()["exclude_patterns"] == [
        r".*\.img_mlp\..*", r".*\.img_mod\..*"
    ]


def test_model_preflight_is_all_or_none() -> None:
    model = _model()
    model.transformer_blocks[-1].img_mlp.proj = nn.Linear(7, 24, bias=False)
    with pytest.raises(ValueError, match="shapes differ"):
        install_qwen_image_2_1_fused_mlps(model, freeze=True)
    assert all(isinstance(block.img_mlp, QwenImage21SwiGLUFeedForward) for block in model.transformer_blocks)


@pytest.mark.parametrize("late", ["masters", "compiled"])
def test_install_refuses_late_swap_or_compile(late: str) -> None:
    model = _model(torch.bfloat16)
    if late == "compiled":
        model._qwen_image_2_1_blocks_compiled = True
    else:
        model.offloader = ModelOffloader(
            model.transformer_blocks, 1, torch.device("cpu"),
            supports_backward=False, transfer_dtype="bf16", restore_mode="slab",
        )
        model.offloader._ensure_cpu_weight_masters(model.transformer_blocks)
    with pytest.raises(RuntimeError, match="before block"):
        install_qwen_image_2_1_fused_mlps(model, freeze=True)


def test_fused_linear_is_discovered_in_bf16_cpu_slab() -> None:
    model = _model(torch.bfloat16)
    offloader = ModelOffloader(
        model.transformer_blocks, 1, torch.device("cpu"),
        supports_backward=False, transfer_dtype="bf16", restore_mode="slab",
    )
    model.offloader = offloader
    install_qwen_image_2_1_fused_mlps(model, freeze=True)
    offloader._ensure_cpu_weight_masters(model.transformer_blocks)
    masters = offloader._cpu_weight_masters
    plans = offloader._cpu_weight_master_slab_plans
    assert masters is not None and plans is not None
    assert all("img_mlp.gate_up" in block for block in masters)
    assert all("img_mlp.out" in block for block in masters)
    assert all("img_mlp.gate_up" in plan for plan in plans)
    assert all("img_mlp.gate_layer" not in block and "img_mlp.proj" not in block for block in masters)
    assert all(master.dtype == torch.bfloat16 for block in masters for master in block.values())
