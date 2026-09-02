"""Triton-only backward helpers for LoKr grouped-delta projection."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - import availability is environment-specific
    triton = None
    tl = None


if triton is not None:  # pragma: no branch - definition only when Triton imports

    @triton.jit
    def _lokr_grad_w1_partials_kernel(
        grad_ptr,
        x_ptr,
        w2_ptr,
        partials_ptr,
        rows,
        in_dim,
        out_dim,
        grad_row_stride,
        grad_factor_stride,
        x_row_stride,
        w2_row_stride,
        num_out_tiles,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        FACTOR: tl.constexpr,
        GROUP: tl.constexpr,
        USE_COMPENSATED_TF32: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_in = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < rows
        mask_n = offs_n < out_dim

        projected = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_iter in range(0, tl.cdiv(in_dim, BLOCK_K)):
            offs_k = k_iter * BLOCK_K + tl.arange(0, BLOCK_K)
            mask_k = offs_k < in_dim
            x_ptrs = (
                x_ptr
                + offs_m[:, None] * x_row_stride
                + pid_in * in_dim
                + offs_k[None, :]
            )
            w2_ptrs = w2_ptr + offs_n[:, None] * w2_row_stride + offs_k[None, :]
            x_block = tl.load(
                x_ptrs,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            w2_block = tl.load(
                w2_ptrs,
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            if USE_COMPENSATED_TF32:
                w2_bits = w2_block.to(tl.uint32, bitcast=True)
                tf32_mask = tl.full(w2_block.shape, 0xFFFFE000, tl.uint32)
                w2_high = (w2_bits & tf32_mask).to(tl.float32, bitcast=True)
                w2_residual = w2_block - w2_high
                projected += tl.dot(
                    x_block,
                    tl.trans(w2_high),
                    input_precision="tf32",
                )
                projected += tl.dot(
                    x_block,
                    tl.trans(w2_residual),
                    input_precision="tf32",
                )
            else:
                projected += tl.dot(
                    x_block,
                    tl.trans(w2_block),
                    input_precision="ieee",
                )

        tile_id = pid_m * num_out_tiles + pid_n
        for out_factor in range(GROUP):
            grad_ptrs = (
                grad_ptr
                + offs_m[:, None] * grad_row_stride
                + out_factor * grad_factor_stride
                + offs_n[None, :]
            )
            grad_block = tl.load(
                grad_ptrs,
                mask=mask_m[:, None] & mask_n[None, :],
                other=0.0,
            ).to(tl.float32)
            row_partials = tl.sum(projected * grad_block, axis=1)
            partial = tl.sum(row_partials, axis=0)
            partial_offset = (tile_id * FACTOR + pid_in) * GROUP + out_factor
            tl.store(partials_ptr + partial_offset, partial)


def lokr_grad_w1_triton_available() -> bool:
    return triton is not None and tl is not None


def reduce_lokr_grad_w1_triton(
    grad: torch.Tensor,
    x_chunk: torch.Tensor,
    w2: torch.Tensor,
    factor: int,
    in_dim: int,
    out_dim: int,
    partial_buffer: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one grouped ``grad_w1`` contribution and its reusable workspace."""

    if not lokr_grad_w1_triton_available():
        raise RuntimeError("Triton is unavailable for LoKr grad_w1 reduction")

    rows = int(grad.shape[0])
    out_count = int(grad.shape[1])
    block_m = 32 if rows >= 32 else 16
    block_n = 32 if out_dim >= 32 else 16
    block_k = 32 if in_dim >= 32 else 16
    num_row_tiles = triton.cdiv(rows, block_m)
    num_out_tiles = triton.cdiv(out_dim, block_n)
    partial_count = num_row_tiles * num_out_tiles * factor * out_count

    if partial_buffer is None or partial_buffer.numel() < partial_count:
        partial_buffer = torch.empty(
            partial_count,
            device=grad.device,
            dtype=torch.float32,
        )
    partials = partial_buffer[:partial_count].view(
        num_row_tiles * num_out_tiles,
        factor,
        out_count,
    )

    # Low-precision activations are exactly representable by TF32. Splitting
    # only the FP32 weight tile into TF32 high/residual terms recovers most FP32
    # accuracy with two Tensor Core dots instead of Triton's general tf32x3.
    use_compensated_tf32 = x_chunk.dtype != torch.float32
    num_warps = 8 if block_m == 32 and block_n == 32 else 4
    _lokr_grad_w1_partials_kernel[(num_row_tiles, num_out_tiles, factor)](
        grad,
        x_chunk,
        w2,
        partials,
        rows,
        in_dim,
        out_dim,
        grad.stride(0),
        grad.stride(1),
        x_chunk.stride(0),
        w2.stride(0),
        num_out_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        FACTOR=factor,
        GROUP=out_count,
        USE_COMPENSATED_TF32=use_compensated_tf32,
        num_warps=num_warps,
        num_stages=2,
    )
    return partials.sum(dim=0).transpose(0, 1), partial_buffer
