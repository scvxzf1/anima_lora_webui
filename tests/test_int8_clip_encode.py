"""Contracts for the per-row L2-optimal INT8 clip encoder."""

from __future__ import annotations

import pytest
import torch

from library.runtime.int8_encode import (
    dequantize_nf8,
    quantize_weight_per_channel,
    quantize_weight_per_channel_clipped,
    quantize_weight_per_channel_nf8,
)
from library.runtime.int8_linear import quantize_weight_per_channel as baseline


def _rel_error(weight, codes, scale):
    restored = codes.float() * scale.float()[:, None]
    target = weight.detach().float()
    return float(((restored - target) ** 2).sum() / (target ** 2).sum())


def test_clip_gain_stays_small_even_with_outliers():
    """P2/3A is closed by measurement: the uniform int8 error sits at the noise
    floor, so a per-row clip search buys at most a couple of percent."""
    torch.manual_seed(0)
    rows = torch.randn(64, 512)
    rows[3, 17] = 14.0
    rows[9, 200] = 9.5
    base_codes, base_scale = baseline(rows)
    clip_codes, clip_scale = quantize_weight_per_channel_clipped(rows)
    base_error = _rel_error(rows, base_codes, base_scale)
    clip_error = _rel_error(rows, clip_codes, clip_scale)
    assert clip_error <= base_error, (base_error, clip_error)
    assert clip_error >= base_error * 0.95, (base_error, clip_error)


def test_clip_never_regresses_without_outliers():
    torch.manual_seed(1)
    rows = torch.randn(48, 256)
    base_codes, base_scale = baseline(rows)
    clip_codes, clip_scale = quantize_weight_per_channel_clipped(rows)
    assert _rel_error(rows, clip_codes, clip_scale) <= _rel_error(rows, base_codes, base_scale) + 1e-12


def test_clip_payload_is_byte_identical():
    rows = torch.randn(32, 128)
    rows[7, 4] = 11.0
    base_codes, base_scale = baseline(rows)
    clip_codes, clip_scale = quantize_weight_per_channel_clipped(rows)
    assert clip_codes.dtype is torch.int8 and base_codes.dtype is torch.int8
    assert clip_codes.shape == base_codes.shape
    assert clip_scale.dtype is torch.float32 and clip_scale.shape == base_scale.shape
    assert clip_codes.numel() * clip_codes.element_size() == base_codes.numel() * base_codes.element_size()
    assert clip_scale.numel() * clip_scale.element_size() == base_scale.numel() * base_scale.element_size()


def test_entry_point_defaults_to_baseline():
    torch.manual_seed(2)
    rows = torch.randn(16, 64)
    off_codes, off_scale = quantize_weight_per_channel(rows)
    base_codes, base_scale = baseline(rows)
    assert torch.equal(off_codes, base_codes) and torch.equal(off_scale, base_scale)
    on_codes, on_scale = quantize_weight_per_channel(rows, clip=True)
    clip_codes, clip_scale = quantize_weight_per_channel_clipped(rows)
    assert torch.equal(on_codes, clip_codes) and torch.equal(on_scale, clip_scale)


def test_rejects_non_2d_weights():
    with pytest.raises(ValueError):
        quantize_weight_per_channel_clipped(torch.randn(4))

def _peaked_rows(rows: int = 32, cols: int = 256, seed: int = 3) -> torch.Tensor:
    """Stand-in for real DiT rows: peaked near zero with a sparse heavy tail."""
    torch.manual_seed(seed)
    base = torch.randn(rows, cols) * 0.25
    tail = torch.randn(rows, cols).abs() * 1.5
    mask = torch.rand(rows, cols) < 0.03
    return torch.where(mask, tail.sign() * tail, base)


def test_nf8_beats_uniform_on_peaked_rows():
    """Real DiT rows are peaked with a sparse tail: measured +18~64% MSE there."""
    rows = _peaked_rows()
    base_error = _rel_error(rows, *baseline(rows))
    codes, scale = quantize_weight_per_channel_nf8(rows)
    restored = dequantize_nf8(codes, scale)  # codes are indices, not values
    nf8_error = float(((restored - rows) ** 2).sum() / (rows ** 2).sum())
    assert nf8_error < base_error * 0.9, (base_error, nf8_error)
    assert torch.isfinite(restored).all()


def test_nf8_regresses_on_gaussian_rows_so_it_must_stay_opt_in():
    """A fixed NormalFloat codebook is not a universal replacement.

    Measured on Gaussian-like rows it is ~2.4x worse than uniform int8, so the
    runtime must choose per row/tensor instead of switching unconditionally.
    """
    torch.manual_seed(5)
    rows = torch.randn(32, 256) * 0.35
    base_error = _rel_error(rows, *baseline(rows))
    codes, scale = quantize_weight_per_channel_nf8(rows)
    restored = dequantize_nf8(codes, scale)
    nf8_error = float(((restored - rows) ** 2).sum() / (rows ** 2).sum())
    assert nf8_error > base_error * 1.5, (base_error, nf8_error)


def test_nf8_payload_is_byte_identical_to_int8():
    rows = _peaked_rows()
    base_codes, base_scale = baseline(rows)
    codes, scale = quantize_weight_per_channel_nf8(rows)
    assert codes.dtype is torch.uint8 and codes.shape == base_codes.shape
    assert scale.dtype is torch.float32 and scale.shape == base_scale.shape
    assert codes.numel() * codes.element_size() == base_codes.numel() * base_codes.element_size()


def test_nf8_reproduces_row_extremes():
    rows = _peaked_rows(rows=4)
    codes, scale = quantize_weight_per_channel_nf8(rows)
    restored = dequantize_nf8(codes, scale)
    for index in range(rows.shape[0]):
        amax = float(rows[index].abs().max())
        assert float(restored[index].abs().max()) <= amax * 1.02
