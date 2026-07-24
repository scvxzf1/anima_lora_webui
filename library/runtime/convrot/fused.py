"""Fused ConvRot forward paths: RHT + quant/GEMM in one autograd Function.

Fusion goals (training-safe, no Triton dependency):
1. Single ``autograd.Function`` for W8A16 / W8A8 so RHT → matmul (→ act quant)
   share one backward without Python-level intermediate graphs.
2. RHT backend selectable: dense (default, fastest on consumer GPUs) or FWHT
   via ``ANIMA_CONVROT_RHT=fwht``.
3. W8A16 weight path: dequant+``F.linear`` (default) or ``_weight_int8pack_mm``
   via ``ANIMA_CONVROT_W8A16_KERNEL=int8pack`` (often *slower* on 3080-class).
4. W8A8: RHT → absmax quant → ``int8_mm_scaled`` (true ``_int_mm`` or float).
"""

from __future__ import annotations

import os
from typing import Literal

import torch
from torch.nn import functional as F

from library.runtime.convrot.gemm import int8_mm_scaled, quantize_activation_absmax_int8
from library.runtime.convrot.quant import dequantize_weight
from library.runtime.convrot.rht import (
    assert_group_divides,
    group_fwht,
    group_rht,
    rht_backend,
)

_FUSED_ENV = "ANIMA_CONVROT_FUSED"
_W8A16_KERNEL_ENV = "ANIMA_CONVROT_W8A16_KERNEL"


def fused_enabled() -> bool:
    raw = str(os.environ.get(_FUSED_ENV, "1") or "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def w8a16_kernel_backend() -> Literal["auto", "int8pack", "dequant"]:
    raw = str(os.environ.get(_W8A16_KERNEL_ENV, "auto") or "auto").strip().lower()
    if raw in {"int8pack", "pack", "weight_int8pack"}:
        return "int8pack"
    if raw in {"dequant", "float", "linear", "f.linear"}:
        return "dequant"
    return "auto"


def _rotate_acts(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """RHT on activations using configured backend (dense default)."""
    with torch.profiler.record_function("convrot::rht"):
        if rht_backend() == "fwht":
            return group_fwht(x, group_size)
        return group_rht(x, group_size)


def can_use_weight_int8pack_mm(
    *,
    device: torch.device,
    x_dtype: torch.dtype,
    m: int,
    k: int,
    n: int,
) -> bool:
    if device.type != "cuda":
        return False
    if not hasattr(torch, "_weight_int8pack_mm"):
        return False
    if x_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        return False
    if m <= 0 or k < 16 or n < 16:
        return False
    return True


def _w8a16_linear_core(
    x_rot: torch.Tensor,
    w_q: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    backend = w8a16_kernel_backend()
    *leading, k = x_rot.shape
    flat = x_rot.reshape(-1, k)
    m = flat.shape[0]
    n = int(w_q.shape[0])
    # Default/auto: dequant — measured much faster than int8pack on RTX 3080.
    use_pack = backend == "int8pack" or (
        backend == "auto"
        and False  # keep int8pack opt-in only until profiled faster on target GPU
        and can_use_weight_int8pack_mm(
            device=flat.device,
            x_dtype=flat.dtype,
            m=m,
            k=k,
            n=n,
        )
    )
    # Prefer the activation dtype for dequant linear so Ampere can stay on
    # bf16/fp16 Tensor Cores. Forcing fp32 here was the main W8A16 tax on 3080.
    compute_dtype = flat.dtype if flat.is_floating_point() else torch.float32
    if compute_dtype == torch.float32 and flat.device.type == "cuda":
        # Host often promotes to fp32; fall back to bf16 TC when available.
        compute_dtype = torch.bfloat16

    if use_pack:
        try:
            x_in = flat.to(dtype=compute_dtype)
            with torch.profiler.record_function("convrot::gemm_int8pack"):
                y = torch._weight_int8pack_mm(
                    x_in.contiguous(),
                    w_q.contiguous(),
                    w_scale.to(device=flat.device, dtype=torch.float32).contiguous(),
                )
            return y.to(torch.float32).reshape(*leading, n)
        except RuntimeError:
            if backend == "int8pack":
                raise
    with torch.profiler.record_function("convrot::dequant"):
        weight = dequantize_weight(w_q, w_scale, dtype=compute_dtype).to(
            device=flat.device, dtype=compute_dtype
        )
    with torch.profiler.record_function("convrot::gemm_dequant_linear"):
        y = F.linear(flat.to(dtype=compute_dtype), weight, None)
    return y.to(torch.float32).reshape(*leading, n)


class _FusedW8A16Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_q, w_scale, group_size: int):
        # RHT in float32 (cheap vs GEMM on measured 3080 profile); GEMM in act TC dtype.
        x_rot = _rotate_acts(x.to(torch.float32), int(group_size))
        compute_dtype = (
            x.dtype
            if x.is_floating_point() and x.dtype in (torch.float16, torch.bfloat16)
            else torch.bfloat16
            if x.device.type == "cuda"
            else torch.float32
        )
        y = _w8a16_linear_core(x_rot.to(dtype=compute_dtype), w_q, w_scale)
        ctx.group_size = int(group_size)
        ctx.save_for_backward(w_q, w_scale)
        ctx.x_dtype = x.dtype
        ctx.compute_dtype = compute_dtype
        return y

    @staticmethod
    def backward(ctx, grad_y):
        w_q, w_scale = ctx.saved_tensors
        dtype = getattr(ctx, "compute_dtype", torch.float32)
        w_hat = dequantize_weight(w_q, w_scale, dtype=dtype).to(
            device=grad_y.device, dtype=dtype
        )
        grad_rot = grad_y.to(dtype=dtype) @ w_hat
        grad_x = _rotate_acts(grad_rot.to(torch.float32), ctx.group_size)
        if ctx.x_dtype != grad_x.dtype:
            grad_x = grad_x.to(dtype=ctx.x_dtype)
        return grad_x, None, None, None


class _FusedW8A8Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_q, w_scale, group_size: int):
        x_f = x.to(torch.float32)
        x_rot = _rotate_acts(x_f, int(group_size))
        with torch.profiler.record_function("convrot::act_quant"):
            x_q, x_scale = quantize_activation_absmax_int8(x_rot)
        with torch.profiler.record_function("convrot::gemm_int8"):
            y = int8_mm_scaled(x_q, x_scale, w_q, w_scale)
        ctx.group_size = int(group_size)
        ctx.save_for_backward(w_q, w_scale)
        ctx.x_dtype = x.dtype
        return y

    @staticmethod
    def backward(ctx, grad_y):
        w_q, w_scale = ctx.saved_tensors
        w_hat = dequantize_weight(w_q, w_scale, dtype=torch.float32).to(
            device=grad_y.device
        )
        grad_rot = grad_y.to(torch.float32) @ w_hat
        grad_x = _rotate_acts(grad_rot, ctx.group_size)
        if ctx.x_dtype != grad_x.dtype:
            grad_x = grad_x.to(dtype=ctx.x_dtype)
        return grad_x, None, None, None


def fused_w8a16_forward(
    x: torch.Tensor,
    quantized_weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor:
    assert_group_divides(int(quantized_weight.shape[1]), group_size)
    y = _FusedW8A16Fn.apply(
        x,
        quantized_weight.to(device=x.device),
        scale.to(device=x.device, dtype=torch.float32),
        int(group_size),
    )
    return y.to(dtype=x.dtype) if x.is_floating_point() else y


def fused_w8a8_forward(
    x: torch.Tensor,
    quantized_weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    group_size: int,
) -> torch.Tensor:
    assert_group_divides(int(quantized_weight.shape[1]), group_size)
    y = _FusedW8A8Fn.apply(
        x,
        quantized_weight.to(device=x.device),
        scale.to(device=x.device, dtype=torch.float32),
        int(group_size),
    )
    return y.to(dtype=x.dtype) if x.is_floating_point() else y
