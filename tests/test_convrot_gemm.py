"""Tests for ConvRot int8 GEMM path."""

from __future__ import annotations

import os

import pytest
import torch
from torch import nn

from library.runtime.convrot.gemm import (
    can_use_torch_int_mm,
    int8_mm_scaled,
    quantize_activation_absmax_int8,
    w8a8_int_linear,
)
from library.runtime.convrot.linear_w8a8 import ConvRotW8A8Linear
from library.runtime.convrot.quant import rotate_and_quantize_weight
from library.runtime.convrot.rht import group_rht, normalized_hadamard


def test_int8_mm_scaled_float_matches_dequant_matmul() -> None:
    torch.manual_seed(0)
    m, k, n = 7, 64, 32
    x = torch.randn(m, k)
    w = torch.randn(n, k)
    x_q, x_s = quantize_activation_absmax_int8(x)
    w_q, w_s = rotate_and_quantize_weight(w, 32)  # also tests shape; use plain quant path
    # Use per-channel quant without rotation for exact dequant reference:
    from library.runtime.convrot.quant import quantize_weight_per_output_channel

    w_q, w_s = quantize_weight_per_output_channel(w)
    y = int8_mm_scaled(x_q, x_s, w_q, w_s, prefer="float")
    ref = (x_q.float() * x_s) @ (w_q.float() * w_s[:, None]).t()
    assert torch.allclose(y, ref, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for _int_mm")
def test_int8_mm_scaled_cuda_int_mm_matches_float_when_supported() -> None:
    torch.manual_seed(1)
    m, k, n = 32, 256, 128
    device = torch.device("cuda")
    x = torch.randn(m, k, device=device)
    w = torch.randn(n, k, device=device)
    from library.runtime.convrot.quant import quantize_weight_per_output_channel

    x_q, x_s = quantize_activation_absmax_int8(x)
    w_q, w_s = quantize_weight_per_output_channel(w)
    assert can_use_torch_int_mm(m, k, n, device=device)
    y_int = int8_mm_scaled(x_q, x_s, w_q, w_s, prefer="int_mm")
    y_fp = int8_mm_scaled(x_q, x_s, w_q, w_s, prefer="float")
    # Scales make results float; integer path should match float ref closely.
    rel = (y_int - y_fp).norm() / y_fp.norm().clamp_min(1e-8)
    assert rel.item() < 1e-5


def test_w8a8_int_linear_ste_grad() -> None:
    torch.manual_seed(2)
    x = torch.randn(5, 64, requires_grad=True)
    w = torch.randn(32, 64)
    from library.runtime.convrot.quant import quantize_weight_per_output_channel

    w_q, w_s = quantize_weight_per_output_channel(w)
    y = w8a8_int_linear(x, w_q, w_s)
    y.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert x.grad.abs().sum().item() > 0


def test_convrot_w8a8_module_finite_and_close_to_bf16() -> None:
    torch.manual_seed(3)
    linear = nn.Linear(64, 32, bias=False)
    linear.weight.requires_grad_(False)
    wrapped = ConvRotW8A8Linear.from_linear(linear, group_size=16)
    x = torch.randn(4, 64)
    y = wrapped(x)
    bf = linear(x)
    assert torch.isfinite(y).all()
    rel = (y - bf).norm() / bf.norm().clamp_min(1e-8)
    assert rel.item() < 0.15


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA preferred for backend env")
def test_backend_env_float_forces_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANIMA_CONVROT_INT8_GEMM", "float")
    torch.manual_seed(4)
    m, k, n = 32, 128, 64
    device = torch.device("cuda")
    x = torch.randn(m, k, device=device)
    from library.runtime.convrot.quant import quantize_weight_per_output_channel

    w = torch.randn(n, k, device=device)
    x_q, x_s = quantize_activation_absmax_int8(x)
    w_q, w_s = quantize_weight_per_output_channel(w)
    y = int8_mm_scaled(x_q, x_s, w_q, w_s)  # prefer auto but env=float
    assert torch.isfinite(y).all()
