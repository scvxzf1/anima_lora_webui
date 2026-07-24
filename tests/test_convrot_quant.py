"""Unit tests for ConvRot rotated-domain quant helpers."""

from __future__ import annotations

import torch

from library.runtime.convrot.quant import (
    dequantize_weight,
    fake_quantize_activation_int8,
    quantize_weight_per_output_channel,
    rotate_and_quantize_weight,
)
from library.runtime.convrot.rht import group_rht_weight


def test_quantize_weight_per_output_channel_shapes_and_roundtrip() -> None:
    torch.manual_seed(0)
    w = torch.randn(11, 32)
    q, scale = quantize_weight_per_output_channel(w)
    assert q.dtype is torch.int8
    assert q.shape == w.shape
    assert scale.shape == (11,)
    restored = dequantize_weight(q, scale, dtype=w.dtype)
    rel = (restored - w).norm() / w.norm().clamp_min(1e-8)
    assert rel.item() < 0.05


def test_rotate_and_quantize_weight_uses_rotated_domain() -> None:
    torch.manual_seed(1)
    w = torch.randn(5, 64)
    group = 16
    q, scale = rotate_and_quantize_weight(w, group)
    rotated = group_rht_weight(w.to(torch.float32), group)
    q_ref, scale_ref = quantize_weight_per_output_channel(rotated)
    assert torch.equal(q, q_ref)
    assert torch.allclose(scale, scale_ref)


def test_fake_quantize_activation_int8_ste_passes_grad() -> None:
    torch.manual_seed(2)
    x = torch.randn(4, 32, requires_grad=True)
    y = fake_quantize_activation_int8(x)
    loss = y.pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    # STE: grad should be non-zero almost everywhere for random input.
    assert x.grad.abs().sum().item() > 0


def test_fake_quantize_stays_in_float_scale() -> None:
    torch.manual_seed(3)
    x = torch.randn(2, 16) * 3
    y = fake_quantize_activation_int8(x)
    # Fake path should remain near original magnitude, not int8 code range.
    assert y.abs().max().item() > 1.0
    assert (y - x).abs().max().item() < 0.5
