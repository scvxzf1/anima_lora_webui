"""Per-row INT8 encoders for block-swap CPU masters.

The baseline encoder in int8_linear.py uses scale = amax / 127 per output channel.
A single 3.5-5 sigma outlier in a row inflates that scale by 2-3x and leaves the
other 99% of the row an effective ~5-bit resolution. This module adds an opt-in
encoder that searches a small candidate grid per row and keeps the scale with the
lowest reconstruction L2 error, so the payload stays exactly 1 byte per element
plus one FP32 scale per row (no extra bytes on the wire).

The search costs one extra pass per candidate at master-construction time only;
training steps are unaffected. The default encoder stays untouched.
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch

from library.runtime.int8_linear import INT8_MAX, SCALE_EPS

# Fractions of row amax, plus empirical percentiles, form the candidate grid.
CLIP_FRACTIONS: tuple = (0.85, 0.90, 0.925, 0.95, 0.975, 1.0)
CLIP_PERCENTILES: tuple = (0.9995,)


def _candidate_scales(
    rows: torch.Tensor,
    amax: torch.Tensor,
    fractions: Iterable,
    percentiles: Iterable,
) -> list:
    scales = [(amax * float(fraction)).clamp_min(SCALE_EPS) for fraction in fractions]
    absolute = rows.abs()
    for percentile in percentiles:
        try:
            cut = torch.quantile(absolute, float(percentile), dim=1)
        except (RuntimeError, ValueError):
            continue
        scales.append(cut.clamp_min(SCALE_EPS))
    return [scale / INT8_MAX for scale in scales]


def quantize_weight_per_channel_clipped(
    weight: torch.Tensor,
    *,
    fractions: Optional[Iterable] = None,
    percentiles: Optional[Iterable] = None,
):
    """Per-row L2-optimal INT8 encoding with a clipped scale search.

    Returns (int8 codes, FP32 scale) with the same shapes and byte count as
    quantize_weight_per_channel, so it is a drop-in replacement for the int8
    block-swap master.
    """
    if weight.dim() != 2:
        raise ValueError("expected a 2D Linear weight")
    rows = weight.detach().to(torch.float32)
    amax = rows.abs().amax(dim=1)
    candidates = _candidate_scales(
        rows,
        amax,
        CLIP_FRACTIONS if fractions is None else fractions,
        CLIP_PERCENTILES if percentiles is None else percentiles,
    )
    if not candidates:
        raise ValueError("at least one clip candidate is required")
    best_error = None
    best_codes = None
    best_scale = None
    for scale in candidates:
        codes = (rows / scale[:, None]).round().clamp(-INT8_MAX, INT8_MAX)
        error = ((codes * scale[:, None] - rows) ** 2).sum(dim=1)
        if best_error is None:
            best_error, best_codes, best_scale = error, codes, scale
            continue
        better = error < best_error
        best_codes = torch.where(better[:, None], codes, best_codes)
        best_scale = torch.where(better, scale, best_scale)
        best_error = torch.where(better, error, best_error)
    return best_codes.to(torch.int8), best_scale


def quantize_weight_per_channel(weight: torch.Tensor, *, clip: bool = False):
    """Entry point for block-swap masters; clip=True opts into the search."""
    if clip:
        return quantize_weight_per_channel_clipped(weight)
    from library.runtime.int8_linear import quantize_weight_per_channel as baseline

    return baseline(weight)


# --- NF8 (fixed NormalFloat-style) encoding ---------------------------------
# Measurements in bench/block_swap_int8/encode_error.py show:
#   * uniform int8 with a per-row clip search gains at most ~2% MSE (closed),
#   * a fixed equiprobable NormalFloat codebook gains 18-64% MSE on real Anima
#     rows at an identical byte count (1 byte/element + 1 FP32 scale per row).
# This is therefore the encoder worth wiring into the int8 block-swap master.

NF8_LEVELS = 256


def nf8_codebook(levels: int = NF8_LEVELS) -> torch.Tensor:
    """Fixed equiprobable standard-normal codebook (the NormalFloat grid)."""
    from statistics import NormalDist

    dist = NormalDist()
    return torch.tensor(
        [dist.inv_cdf((index + 0.5) / levels) for index in range(levels)],
        dtype=torch.float32,
    )


def quantize_weight_per_channel_nf8(
    weight: torch.Tensor, *, codebook=None
):
    """NF8 encode: uint8 codes + per-row FP32 scale, byte-identical to int8."""
    if weight.dim() != 2:
        raise ValueError("expected a 2D Linear weight")
    book = nf8_codebook() if codebook is None else codebook.float()
    rows = weight.detach().to(torch.float32)
    span = book.abs().max().clamp_min(SCALE_EPS)
    amax = rows.abs().amax(dim=1).clamp_min(SCALE_EPS)
    scale = amax / span
    midpoints = (book[:-1] + book[1:]) / 2.0
    codes = torch.bucketize(rows / scale[:, None], midpoints).to(torch.uint8)
    return codes, scale


def dequantize_nf8(codes: torch.Tensor, scale: torch.Tensor, *, codebook=None) -> torch.Tensor:
    book = nf8_codebook() if codebook is None else codebook.float()
    return book[codes.long()] * scale.float()[:, None]
