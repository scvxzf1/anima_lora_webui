"""Lazily imported Triton implementation for experimental ConvRot primitives.

Original implementation; no external kernel/framework code is vendored.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _quantize(X, Q, S, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    # Abs and max commute with lossless native->FP32 conversion for finite
    # FP16/BF16 values, so this retains native absmax semantics.
    values = tl.load(X + row * K + col, col < K, other=0).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(values), axis=0), 1.0e-12)
    # PyTorch's scalar Tensor / 127 uses multiplication by the FP32
    # reciprocal. Correctly rounded division differs by one ULP for some
    # rows, so match the actual production operation rather than real math.
    scale = _mul_rn(amax, 1.0 / 127.0)
    codes = libdevice.nearbyint(tl.div_rn(values, scale))
    codes = tl.minimum(tl.maximum(codes, -127.0), 127.0).to(tl.int8)
    tl.store(Q + row * K + col, codes, col < K)
    tl.store(S + row, scale)


@triton.jit
def _mul_rn(a, b):
    # Explicit rounding barriers forbid reassociation of the two scale
    # multiplies, independently of optimizer algebraic simplifications.
    return tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a, b],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _matmul(
    X,
    SX,
    W,
    SW,
    Y,
    BIAS,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    W_K: tl.constexpr,
    W_N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    RETURN_ACC: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.int32)
    for block in range(tl.cdiv(K, BK)):
        ks = block * BK + kk
        a = tl.load(
            X + rows[:, None] * K + ks[None, :],
            (rows[:, None] < M) & (ks[None, :] < K),
            other=0,
        )
        b = tl.load(
            W + ks[:, None] * W_K + cols[None, :] * W_N,
            (ks[:, None] < K) & (cols[None, :] < N),
            other=0,
        )
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
    if RETURN_ACC:
        result = acc
    else:
        sx = tl.load(SX + rows, rows < M, other=0).to(tl.float32)
        sw = tl.load(SW + cols, cols < N, other=0).to(tl.float32)
        result = _mul_rn(acc.to(tl.float32), sx[:, None])
        result = _mul_rn(result, sw[None, :])
        if HAS_BIAS:
            bias = tl.load(BIAS + cols, cols < N, other=0).to(tl.float32)
            result = result + bias[None, :]
    tl.store(
        Y + rows[:, None] * N + cols[None, :],
        result,
        (rows[:, None] < M) & (cols[None, :] < N),
    )


def quantize(x, q, scale, m, k):
    block = triton.next_power_of_2(k)
    warps = 4 if block <= 8192 else 8 if block <= 16384 else 16
    with torch.cuda.device(x.device):
        _quantize[(m,)](x, q, scale, k, block, num_warps=warps, enable_fp_fusion=False)


def matmul(x, sx, w, sw, out, m, k, n, layout, bias, return_acc):
    # Bounded static SM86 starting point, no runtime autotuning/large compile
    # matrix. Subsequent shape-specific tuning requires separate GPU evidence.
    bm, bn, bk = (16 if m <= 16 else 32), 64, 64
    w_k, w_n = w.stride() if layout == "kn" else reversed(w.stride())
    with torch.cuda.device(x.device):
        _matmul[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            x,
            sx,
            w,
            sw,
            out,
            bias if bias is not None else out,
            m,
            k,
            n,
            w_k,
            w_n,
            bm,
            bn,
            bk,
            return_acc,
            bias is not None,
            num_warps=4,
            num_stages=3,
            enable_fp_fusion=False,
        )
