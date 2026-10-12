"""Regression coverage for Qwen Image 2.1 all-linear NF4 checkpoints."""

import pytest
import torch
from diffusers import QwenImage21Transformer2DModel

from library.quantization import (
    BF16AutocastLinear4bit,
    inspect_mixed_nf4,
    load_mixed_nf4_into,
    save_mixed_nf4,
)


def _tiny_qwen(*, device="cpu"):
    return QwenImage21Transformer2DModel(
        num_layers=2,
        num_attention_heads=4,
        attention_head_dim=16,
        context_in_dim=64,
        axes_dims_rope=(4, 6, 6),
        mlp_ratio=2,
    ).to(device=device, dtype=torch.bfloat16)


def test_qwen_all_linear_nf4_roundtrip(tmp_path):
    torch.manual_seed(7)
    source = _tiny_qwen()
    before = {key: value.detach().clone() for key, value in source.state_dict().items()}
    path = tmp_path / "qwen-all-linear.safetensors"

    result = save_mixed_nf4(
        source,
        path,
        family="qwen_image_2_1",
        policy="all-linear-nf4-v1",
        source_sha256="a" * 64,
    )
    manifest = inspect_mixed_nf4(path, family="qwen_image_2_1")
    linear_paths = sorted(
        name for name, module in source.named_modules() if type(module) is torch.nn.Linear
    )
    assert result["quantized_weights"] == len(linear_paths)
    assert manifest["policy"] == "all-linear-nf4-v1"
    assert manifest["modules"] == linear_paths

    with torch.device("meta"):
        restored = _tiny_qwen(device="meta")
    load_mixed_nf4_into(
        restored,
        path,
        family="qwen_image_2_1",
        dtype=torch.bfloat16,
        device="cpu",
    )
    assert all(not value.is_meta for value in restored.state_dict().values())
    assert sum(isinstance(module, BF16AutocastLinear4bit) for module in restored.modules()) == len(
        linear_paths
    )
    for key, value in source.state_dict().items():
        assert torch.equal(value, before[key])
        if key.removesuffix(".weight") not in manifest["modules"]:
            torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)

    x = torch.randn(2, 64, dtype=torch.bfloat16, requires_grad=True)
    output = restored.get_submodule(manifest["modules"][0])(x)
    output.float().square().sum().backward()
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(x.grad).all()


def test_qwen_all_linear_rejects_unaligned_layer(tmp_path):
    model = _tiny_qwen()
    model.proj_out = torch.nn.Linear(64, 65, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="proj_out"):
        save_mixed_nf4(
            model,
            tmp_path / "qwen-all-linear-unaligned.safetensors",
            family="qwen_image_2_1",
            policy="all-linear-nf4-v1",
            source_sha256="b" * 64,
        )
