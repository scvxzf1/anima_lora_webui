"""fp16 residual-stream overflow guards for Block / FinalLayer / Anima."""

from __future__ import annotations

import torch

from library.anima.models import Anima, Block, FinalLayer

_FP16_MAX = torch.finfo(torch.float16).max


def _tiny_anima(num_blocks: int = 3) -> Anima:
    """Minimal real Anima DiT constructible on CPU (matches test_native_flatten)."""
    return Anima(
        max_img_h=256,
        max_img_w=256,
        max_frames=4,
        in_channels=16,
        out_channels=16,
        patch_spatial=2,
        patch_temporal=1,
        concat_padding_mask=False,
        model_channels=64,
        num_blocks=num_blocks,
        num_heads=4,
        mlp_ratio=2.0,
        crossattn_emb_channels=64,
        use_adaln_lora=True,
        adaln_lora_dim=16,
        use_llm_adapter=False,
        attn_mode="torch",
    )


def test_residual_add_unit_overflow_guard():
    block = Block(x_dim=64, context_dim=64, num_heads=4)
    a = torch.full((2, 4), 0.9 * _FP16_MAX, dtype=torch.float16)
    b = torch.full((2, 4), 0.9 * _FP16_MAX, dtype=torch.float16)

    naive = (a + b).to(torch.float16)
    assert torch.isinf(naive).any()

    block.fp32_residual = False
    assert torch.equal(block._residual_add(a, b), naive)

    block.fp32_residual = True
    guarded = block._residual_add(a, b)
    assert guarded.dtype == torch.float32
    assert torch.isfinite(guarded).all()


def test_gated_residual_add_overflow_guard():
    block = Block(x_dim=64, context_dim=64, num_heads=4)
    residual = torch.full((2, 4), 1000.0, dtype=torch.float16)
    gate = torch.full((2, 4), 8.0, dtype=torch.float16)
    branch = torch.full((2, 4), 0.9 * _FP16_MAX, dtype=torch.float16)

    block.fp32_residual = False
    naive = residual + gate * branch
    assert torch.isinf(naive).any()

    block.fp32_residual = True
    guarded = block._gated_residual_add(residual, gate, branch)
    assert guarded.dtype == torch.float32
    assert torch.isfinite(guarded).all()


def test_enable_fp32_residual_propagates_to_all_modules():
    anima = _tiny_anima(num_blocks=3)
    assert all(not b.fp32_residual for b in anima.blocks)
    assert anima.final_layer.fp32_residual is False

    anima.enable_fp32_residual()

    assert all(b.fp32_residual for b in anima.blocks)
    assert anima.final_layer.fp32_residual is True
