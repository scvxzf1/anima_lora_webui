"""Weight / activation int8 quant helpers for ConvRot (rotated domain)."""

from __future__ import annotations

import torch

from library.runtime.convrot.rht import (
    assert_group_divides,
    group_rht_weight,
    hadamard_kind,
    normalized_hadamard,
)

INT8_MAX = 127.0
SCALE_EPS = 1e-12


def quantize_weight_per_output_channel(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel absmax int8 quant on a 2D weight ``[out, in]``."""
    if weight.dim() != 2:
        raise ValueError(f"expected a 2D Linear weight, got shape={tuple(weight.shape)}")
    rows = weight.detach().to(torch.float32)
    amax = rows.abs().amax(dim=1)
    scale = (amax / INT8_MAX).clamp_min(SCALE_EPS)
    quantized = (rows / scale[:, None]).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return quantized, scale


def dequantize_weight(
    quantized: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if quantized.dim() != 2:
        raise ValueError("quantized weight must be 2D")
    if scale.shape != (quantized.shape[0],):
        raise ValueError("scale must have one value per output channel")
    weight = quantized.to(torch.float32) * scale.to(torch.float32)[:, None]
    return weight.to(dtype) if dtype is not None else weight


def rotate_and_quantize_weight(
    weight: torch.Tensor,
    group_size: int,
    *,
    hadamard: torch.Tensor | None = None,
    kind: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RHT on in-features then per-output-channel int8 quant.

    Returns ``(W_q int8 [out, in], scale float32 [out])``.
    ``kind`` defaults to ``ANIMA_CONVROT_HADAMARD`` (sylvester|regular).
    """
    kind_resolved = kind or hadamard_kind()
    assert_group_divides(int(weight.shape[-1]), group_size, kind=kind_resolved)  # type: ignore[arg-type]
    if hadamard is None:
        hadamard = normalized_hadamard(
            group_size,
            device=weight.device,
            dtype=torch.float32,
            kind=kind_resolved,  # type: ignore[arg-type]
        )
    rotated = group_rht_weight(
        weight.detach().to(torch.float32),
        group_size,
        hadamard=hadamard,
        kind=kind_resolved,  # type: ignore[arg-type]
    )
    return quantize_weight_per_output_channel(rotated)


def dynamic_absmax_quantize_activation(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-token absmax int8 quant on last dim.

    Returns ``(x_q int8, scale float32)`` where scale broadcasts over the last
    feature dim: shape is ``x.shape[:-1] + (1,)``.
    """
    if x.shape[-1] == 0:
        empty_q = x.to(torch.int8)
        empty_s = torch.ones(
            *x.shape[:-1],
            1,
            device=x.device,
            dtype=torch.float32,
        )
        return empty_q, empty_s
    work = x.to(torch.float32)
    # STE-friendly path: quantize in forward, straight-through via identity on backward
    # for the activation tensor when training W8A8 fake path.
    amax = work.detach().abs().amax(dim=-1, keepdim=True).clamp_min(SCALE_EPS)
    scale = amax / INT8_MAX
    quantized = (work / scale).round().clamp(-INT8_MAX, INT8_MAX)
    dequant = quantized * scale
    # Fake-quant in original float scale; STE treats round as identity.
    fake = work + (dequant - work).detach()
    return fake, scale


def fake_quantize_activation_int8(x: torch.Tensor) -> torch.Tensor:
    """Return STE fake-quantized activation in the same dtype as ``x``."""
    fake, _scale = dynamic_absmax_quantize_activation(x)
    return fake.to(dtype=x.dtype)
