"""fp16 residual-stream overflow guards for Block / FinalLayer / Anima."""

from __future__ import annotations

import torch

from library.anima.models import Anima, Block, FinalLayer, RMSNorm

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


def test_final_layer_fp32_residual_projection_stays_finite():
    """FinalLayer fp32_residual path stays finite when naive fp16 projection overflows."""
    layer = FinalLayer(
        hidden_size=4,
        spatial_patch_size=1,
        temporal_patch_size=1,
        out_channels=2,
        use_adaln_lora=False,
    ).half()
    with torch.no_grad():
        layer.adaln_modulation[1].weight.zero_()
        # Second half of adaln outputs is scale; amplify post-LN features into fp16 danger zone.
        layer.adaln_modulation[1].weight[4:, :].fill_(50.0)
        # Non-uniform projection so mean-zero LN features cannot cancel.
        layer.linear.weight.zero_()
        layer.linear.weight[:, 0] = 400.0

    # Asymmetric token so LayerNorm keeps a large positive first channel.
    x = torch.tensor([[[[[100.0, -1.0, -1.0, -1.0]]]]], dtype=torch.float16)
    emb = torch.ones(1, 1, 4, dtype=torch.float16)

    layer.fp32_residual = False
    naive = layer(x, emb)
    assert naive.dtype == torch.float16
    assert torch.isinf(naive).any()

    layer.fp32_residual = True
    guarded = layer(x, emb)
    assert guarded.dtype == torch.float32
    assert torch.isfinite(guarded).all()
    # Same overflow case projected in fp32 must remain finite and non-trivial.
    assert guarded.abs().max() > 0


def test_enable_fp32_residual_propagates_to_all_modules():
    anima = _tiny_anima(num_blocks=3)
    assert all(not b.fp32_residual for b in anima.blocks)
    assert anima.final_layer.fp32_residual is False

    anima.enable_fp32_residual()

    assert all(b.fp32_residual for b in anima.blocks)
    assert anima.final_layer.fp32_residual is True


def test_rmsnorm_preserves_half_dtype() -> None:
    norm = RMSNorm(16)
    for dtype in (torch.bfloat16, torch.float16):
        inputs = torch.randn(2, 4, 16, dtype=dtype)
        output = norm(inputs)
        assert output.dtype == dtype
        assert output.shape == inputs.shape
        assert torch.isfinite(output.float()).all()


def test_rmsnorm_matches_legacy_fp32_affine_up_to_half_rounding() -> None:
    torch.manual_seed(0)
    norm = RMSNorm(32)
    inputs = torch.randn(3, 8, 32, dtype=torch.bfloat16)

    current = norm(inputs)
    legacy = (norm._norm(inputs.float()) * norm.weight).to(inputs.dtype)
    relative_error = (
        (current.float() - legacy.float()).norm()
        / legacy.float().norm().clamp_min(1e-8)
    )

    assert relative_error.item() < 1e-3
