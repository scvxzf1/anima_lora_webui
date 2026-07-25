"""Int8 GEMM helpers for ConvRot W8A8 (true path + float fallback)."""

from __future__ import annotations

import os
from typing import Literal

import torch

# torch._int_mm requires M (rows of A) > 16 on current CUDA builds; some shapes
# also trip cublasLt NOT_SUPPORTED. Prefer true path when safe, else float.
_INT_MM_MIN_M = 17
_BACKEND_ENV = "ANIMA_CONVROT_INT8_GEMM"


def int8_gemm_backend() -> Literal["auto", "int_mm", "float"]:
    raw = str(os.environ.get(_BACKEND_ENV, "auto") or "auto").strip().lower()
    if raw in {"auto", "int_mm", "int8", "true", "1"}:
        if raw in {"int_mm", "int8", "true", "1"}:
            return "int_mm"
        return "auto"
    if raw in {"float", "fake", "fp", "0", "false", "off"}:
        return "float"
    return "auto"


def can_use_torch_int_mm(
    m: int,
    k: int,
    n: int,
    *,
    device: torch.device,
) -> bool:
    if device.type != "cuda":
        return False
    if not hasattr(torch, "_int_mm"):
        return False
    if m < _INT_MM_MIN_M:
        return False
    # Heuristic: cublasLt path is picky on tiny dims; require reasonable K/N.
    if k < 16 or n < 16:
        return False
    return True


def quantize_activation_absmax_int8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token absmax quant → real int8 + scale (no STE).

    ``scale`` shape is ``x.shape[:-1] + (1,)`` such that
    ``x ≈ x_q.float() * scale``.
    """
    from library.runtime.convrot.quant import INT8_MAX, SCALE_EPS

    work = x.detach().to(torch.float32)
    amax = work.abs().amax(dim=-1, keepdim=True).clamp_min(SCALE_EPS)
    scale = amax / INT8_MAX
    q = (work / scale).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
    return q, scale


def int8_mm_scaled(
    x_q: torch.Tensor,
    x_scale: torch.Tensor,
    w_q: torch.Tensor,
    w_scale: torch.Tensor,
    *,
    prefer: Literal["auto", "int_mm", "float"] | None = None,
) -> torch.Tensor:
    """Compute ``(x_q @ w_q.T) * x_scale * w_scale`` in float32.

    Parameters
    ----------
    x_q:
        int8 activations ``[..., K]``
    x_scale:
        float scales broadcastable to ``[..., 1]``
    w_q:
        int8 weights ``[N, K]`` (out, in)
    w_scale:
        float per-out-channel scales ``[N]``
    """
    if x_q.dtype is not torch.int8 or w_q.dtype is not torch.int8:
        raise TypeError("x_q and w_q must be torch.int8")
    if w_q.dim() != 2:
        raise ValueError("w_q must be 2D [out, in]")
    *leading, k = x_q.shape
    n, k_w = w_q.shape
    if k != k_w:
        raise ValueError(f"K mismatch: x has {k}, w has {k_w}")

    flat = x_q.reshape(-1, k)
    m = flat.shape[0]
    backend = prefer or int8_gemm_backend()
    use_int = backend == "int_mm" or (
        backend == "auto"
        and can_use_torch_int_mm(m, k, n, device=flat.device)
    )

    x_scale_flat = x_scale.reshape(-1, 1).to(device=flat.device, dtype=torch.float32)
    w_scale_row = w_scale.reshape(1, -1).to(device=flat.device, dtype=torch.float32)

    if use_int:
        try:
            # _int_mm(A[M,K], B[K,N]) -> int32[M,N]
            # Prefer a view when w_q is already contiguous in the needed layout.
            b = w_q.t().contiguous()
            acc = torch._int_mm(flat.contiguous(), b)
            # In-place scale chain: fewer temporaries than two out-of-place muls.
            y = acc.to(torch.float32)
            y.mul_(x_scale_flat)
            y.mul_(w_scale_row)
            return y.reshape(*leading, n)
        except RuntimeError:
            if backend == "int_mm":
                raise
            # fall through to float reference

    # Float reference / fallback: exact for int8×int8 when accumulated in fp32.
    y = flat.to(torch.float32) @ w_q.to(torch.float32).t()
    y.mul_(x_scale_flat)
    y.mul_(w_scale_row)
    return y.reshape(*leading, n)


class _W8A8IntLinearFn(torch.autograd.Function):
    """Forward: true/fake int8 GEMM; backward: STE via dequant weight @ grad."""

    @staticmethod
    def forward(
        ctx,
        x_rot: torch.Tensor,
        w_q: torch.Tensor,
        w_scale: torch.Tensor,
    ) -> torch.Tensor:
        x_q, x_scale = quantize_activation_absmax_int8(x_rot)
        y = int8_mm_scaled(x_q, x_scale, w_q, w_scale)
        # STE: save float rotated input and dequant-ready weight scales.
        ctx.save_for_backward(x_rot, w_q, w_scale)
        return y

    @staticmethod
    def backward(ctx, grad_y: torch.Tensor):
        x_rot, w_q, w_scale = ctx.saved_tensors
        # Base weights frozen — only grad_x. STE ignores act quant.
        from library.runtime.convrot.quant import dequantize_weight

        w_hat = dequantize_weight(w_q, w_scale, dtype=torch.float32)
        if w_hat.device != grad_y.device:
            w_hat = w_hat.to(device=grad_y.device)
        grad = grad_y.to(torch.float32)
        # y = x @ W^T  => grad_x = grad_y @ W
        grad_x = grad @ w_hat
        if x_rot.dtype != grad_x.dtype:
            grad_x = grad_x.to(dtype=x_rot.dtype)
        return grad_x, None, None


def w8a8_int_linear(x_rot: torch.Tensor, w_q: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    """Autograd-aware W8A8 linear in rotated space."""
    return _W8A8IntLinearFn.apply(x_rot, w_q, w_scale)
