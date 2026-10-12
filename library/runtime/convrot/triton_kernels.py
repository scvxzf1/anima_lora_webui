"""Experimental, explicit CUDA-only ConvRot kernels; no production dispatch.

Importing this module neither imports Triton nor initializes CUDA. These are
forward-only primitives, not autograd/STE wrappers. No floating GEMM fallback
exists. Finite inputs are the numerical contract; ``check_finite=True`` offers
an explicit, synchronizing diagnostic check and rejects NaN/Inf.
"""

from __future__ import annotations

from functools import lru_cache
from math import prod
from typing import Literal

import torch

WeightLayout = Literal["kn", "nk"]
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MAX_QUANT_K = 65536


@lru_cache(maxsize=1)
def _load_kernels():
    try:
        from library.runtime.convrot import _triton_impl
    except ImportError as exc:
        raise RuntimeError(
            "ConvRot Triton kernels require an installed CUDA Triton runtime"
        ) from exc
    return _triton_impl


def _cuda_tensor(tensor: torch.Tensor, name: str) -> None:
    if tensor.device.type != "cuda":
        raise ValueError(
            f"{name} must be a CUDA tensor; CPU/float fallback is disabled"
        )


def _finite(tensor: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} contains NaN or Inf")


def quantize_rowwise(
    x: torch.Tensor,
    *,
    check_finite: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match gemm.quantize_activation_absmax_int8 for finite FP16/BF16/FP32.

    Native absmax is promoted to FP32 and clamped at 1e-12. Scale uses
    multiplication by float32(1/127), matching PyTorch's scalar division.
    FP32 x/scale uses correctly rounded division and ties-to-even round;
    codes are clamped to [-127,127]. Returns int8 ``x.shape`` and FP32
    ``x.shape[:-1] + (1,)``. Noncontiguous input incurs an explicit copy.
    Last-axis sizes 1..65536 are supported; empty leading axes are allowed.
    Nonfinite input without diagnostic checking is outside this contract.
    """
    if x.dtype not in _FLOAT_DTYPES:
        raise TypeError("x must have float16, bfloat16 or float32 dtype")
    if x.ndim < 1 or not 1 <= x.shape[-1] <= _MAX_QUANT_K:
        raise ValueError("x must have a last dimension K in 1..65536")
    _cuda_tensor(x, "x")
    kernels = _load_kernels()
    if check_finite:
        _finite(x, "x")
    work = x.detach().contiguous()
    k = x.shape[-1]
    m = prod(x.shape[:-1])
    q = torch.empty(x.shape, device=x.device, dtype=torch.int8)
    scale = torch.empty((*x.shape[:-1], 1), device=x.device, dtype=torch.float32)
    if m:
        kernels.quantize(work, q, scale, m, k)
    return q, scale


def _matmul_shape(x_q: torch.Tensor, w_q: torch.Tensor, weight_layout: str):
    if weight_layout not in ("kn", "nk"):
        raise ValueError("weight_layout must be 'kn' or 'nk'")
    if x_q.dtype != torch.int8 or w_q.dtype != torch.int8:
        raise TypeError("x_q and w_q must have torch.int8 dtype")
    if x_q.ndim < 1 or w_q.ndim != 2:
        raise ValueError("x_q must have at least one dimension and w_q must be 2D")
    k = x_q.shape[-1]
    k_w, n = w_q.shape if weight_layout == "kn" else reversed(w_q.shape)
    if k <= 0 or n <= 0 or k_w != k:
        raise ValueError(f"invalid K/N or K mismatch: x K={k}, weight K={k_w}, N={n}")
    _cuda_tensor(x_q, "x_q")
    _cuda_tensor(w_q, "w_q")
    if w_q.device != x_q.device:
        raise ValueError("x_q and w_q must be on the same CUDA device")
    return prod(x_q.shape[:-1]), k, n


def _scale(tensor: torch.Tensor, name: str, shape: tuple, device: torch.device) -> None:
    if tensor.dtype not in _FLOAT_DTYPES:
        raise TypeError(f"{name} must have float16, bfloat16 or float32 dtype")
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} shape must be {shape}, got {tuple(tensor.shape)}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on the same CUDA device as x_q")


def int8_matmul_scaled(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    w_q: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    weight_layout: WeightLayout = "kn",
    out_dtype: torch.dtype = torch.bfloat16,
    bias: torch.Tensor | None = None,
    check_finite: bool = False,
) -> torch.Tensor:
    """INT8 dot -> INT32 -> FP32 * sx -> FP32 * sw -> optional bias -> cast.

    x_q is [...,K], w_q is [K,N] (kn) or [N,K] (nk). x_scale must
    exactly match [...,1], w_scale [N]. Scales load/cast to FP32. Optional
    [N] bias is added in FP32 before the final cast (not after output cast).
    Copies noncontiguous activations/scales; weight strides are supported.
    All M/K/N tails are masked, including M=1/16/17/4032/4200. Fusion and
    floating-point reassociation are disabled. INT32 overflow has the same
    limitation as torch._int_mm; callers must choose K/payload accordingly.
    """
    m, k, n = _matmul_shape(x_q, w_q, weight_layout)
    if out_dtype not in _FLOAT_DTYPES:
        raise TypeError("out_dtype must be float16, bfloat16 or float32")
    _scale(x_scale, "x_scale", (*x_q.shape[:-1], 1), x_q.device)
    _scale(w_scale, "w_scale", (n,), x_q.device)
    if bias is not None:
        _scale(bias, "bias", (n,), x_q.device)
    kernels = _load_kernels()
    if check_finite:
        for name, tensor in (
            ("x_scale", x_scale),
            ("w_scale", w_scale),
            ("bias", bias),
        ):
            if tensor is not None:
                _finite(tensor, name)
    out = torch.empty((*x_q.shape[:-1], n), device=x_q.device, dtype=out_dtype)
    if m:
        kernels.matmul(
            x_q.contiguous(),
            x_scale.contiguous(),
            w_q,
            w_scale.contiguous(),
            out,
            m,
            k,
            n,
            weight_layout,
            bias.contiguous() if bias is not None else None,
            False,
        )
    return out


def debug_int8_matmul_accumulator(
    x_q: torch.Tensor,
    w_q: torch.Tensor,
    *,
    weight_layout: WeightLayout = "kn",
) -> torch.Tensor:
    """Expose exact INT32 dot from the same kernel, without scale/cast."""
    m, k, n = _matmul_shape(x_q, w_q, weight_layout)
    kernels = _load_kernels()
    out = torch.empty((*x_q.shape[:-1], n), device=x_q.device, dtype=torch.int32)
    if m:
        kernels.matmul(
            x_q.contiguous(), x_q, w_q, w_q, out, m, k, n, weight_layout, None, True
        )
    return out
