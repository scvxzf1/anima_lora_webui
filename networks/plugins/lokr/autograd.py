"""Memory-saving projection helpers for the LoKR plugin."""

from __future__ import annotations

import torch


_LOKR_PROJECT_CHUNK_BYTES = 4 * 1024 * 1024


def _projection_chunk_rows(in_dim: int, out_dim: int) -> int:
    bytes_per_row = max(1, int(in_dim) + int(out_dim)) * 4
    return max(1, int(_LOKR_PROJECT_CHUNK_BYTES) // bytes_per_row)


class LoKrProjectFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, factor, in_dim, out_dim):
        factor = int(factor)
        in_dim = int(in_dim)
        out_dim = int(out_dim)
        x_float = x.float()
        leading_shape = x_float.shape[:-1]
        x_view = x_float.reshape(-1, factor, in_dim)
        w1_float = w1.float()
        w2_float = w2.float()
        w2_t = w2_float.transpose(0, 1)
        mixed = x_float.new_empty((x_view.shape[0], factor, out_dim))
        for out_factor in range(factor):
            out_slice = mixed[:, out_factor, :]
            out_slice.zero_()
            for in_factor in range(factor):
                projected = x_view[:, in_factor, :].matmul(w2_t)
                out_slice.add_(projected * w1_float[out_factor, in_factor])
        ctx.save_for_backward(x, w1, w2)
        ctx.factor = factor
        ctx.in_dim = in_dim
        ctx.out_dim = out_dim
        ctx.leading_shape = leading_shape
        return mixed.reshape(*leading_shape, factor * out_dim)

    @staticmethod
    def backward(ctx, grad_out):
        x, w1, w2 = ctx.saved_tensors
        factor = ctx.factor
        in_dim = ctx.in_dim
        out_dim = ctx.out_dim

        x_float = x.float().reshape(-1, factor, in_dim)
        w1_float = w1.float()
        w2_float = w2.float()
        grad = grad_out.float().reshape(-1, factor, out_dim)

        # Recompute one factor slice at a time instead of saving the forward
        # intermediate or expanding a full Kronecker weight.
        grad_w1 = w1_float.new_empty(w1_float.shape)
        grad_w2 = w2_float.new_zeros(w2_float.shape)
        grad_x = x_float.new_empty(x_float.shape)
        for in_factor in range(factor):
            x_slice = x_float[:, in_factor, :]
            projected = x_slice.matmul(w2_float.transpose(0, 1))
            grad_w1[:, in_factor] = torch.einsum("nao,no->a", grad, projected)

            grad_projected = torch.einsum(
                "a,nao->no", w1_float[:, in_factor], grad
            )
            grad_w2.add_(grad_projected.transpose(0, 1).matmul(x_slice))
            grad_x[:, in_factor, :] = grad_projected.matmul(w2_float)
        grad_x = grad_x.reshape(*ctx.leading_shape, factor * in_dim)

        return (
            grad_x.to(x.dtype),
            grad_w1.to(w1.dtype),
            grad_w2.to(w2.dtype),
            None,
            None,
            None,
        )


class LoKrProjectFactorFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, out_factor, factor, in_dim, out_dim):
        out_factor = int(out_factor)
        factor = int(factor)
        in_dim = int(in_dim)
        out_dim = int(out_dim)
        leading_shape = x.shape[:-1]
        x_view = x.reshape(-1, factor, in_dim)
        w1_float = w1.float()
        w2_float = w2.float()
        w2_t = w2_float.transpose(0, 1)
        out = w2_float.new_empty((x_view.shape[0], out_dim))
        out.zero_()
        chunk_rows = _projection_chunk_rows(in_dim, out_dim)
        for row_start in range(0, x_view.shape[0], chunk_rows):
            row_end = min(row_start + chunk_rows, x_view.shape[0])
            out_chunk = out[row_start:row_end]
            for in_factor in range(factor):
                x_slice = x_view[row_start:row_end, in_factor, :].float()
                projected = x_slice.matmul(w2_t)
                out_chunk.add_(projected * w1_float[out_factor, in_factor])
        ctx.save_for_backward(x, w1, w2)
        ctx.out_factor = out_factor
        ctx.factor = factor
        ctx.in_dim = in_dim
        ctx.out_dim = out_dim
        ctx.leading_shape = leading_shape
        return out.reshape(*leading_shape, out_dim)

    @staticmethod
    def backward(ctx, grad_out):
        x, w1, w2 = ctx.saved_tensors
        out_factor = ctx.out_factor
        factor = ctx.factor
        in_dim = ctx.in_dim
        out_dim = ctx.out_dim

        x_view = x.reshape(-1, factor, in_dim)
        w1_float = w1.float()
        w2_float = w2.float()
        grad_view = grad_out.reshape(-1, out_dim)

        grad_w1 = w1_float.new_zeros(w1_float.shape)
        grad_w2 = w2_float.new_zeros(w2_float.shape)
        grad_x = torch.empty(
            (x_view.shape[0], factor, in_dim), dtype=x.dtype, device=x.device
        )
        w2_t = w2_float.transpose(0, 1)
        chunk_rows = _projection_chunk_rows(in_dim, out_dim)
        for row_start in range(0, x_view.shape[0], chunk_rows):
            row_end = min(row_start + chunk_rows, x_view.shape[0])
            grad = grad_view[row_start:row_end].float()
            for in_factor in range(factor):
                x_slice = x_view[row_start:row_end, in_factor, :].float()
                projected = x_slice.matmul(w2_t)
                grad_w1[out_factor, in_factor].add_((grad * projected).sum())

                coeff = w1_float[out_factor, in_factor]
                grad_projected = grad * coeff
                grad_w2.add_(grad_projected.transpose(0, 1).matmul(x_slice))
                grad_x[row_start:row_end, in_factor, :] = grad_projected.matmul(
                    w2_float
                ).to(x.dtype)
        grad_x = grad_x.reshape(*ctx.leading_shape, factor * in_dim)

        return (
            grad_x,
            grad_w1.to(w1.dtype),
            grad_w2.to(w2.dtype),
            None,
            None,
            None,
            None,
        )


class LoKrProjectFactorGroupFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, out_start, out_count, factor, in_dim, out_dim):
        out_start = int(out_start)
        out_count = int(out_count)
        factor = int(factor)
        in_dim = int(in_dim)
        out_dim = int(out_dim)
        out_count = max(1, min(out_count, factor - out_start))
        leading_shape = x.shape[:-1]
        x_view = x.reshape(-1, factor, in_dim)
        w1_float = w1.float()
        w2_float = w2.float()
        w2_t = w2_float.transpose(0, 1)
        out = w2_float.new_empty((x_view.shape[0], out_count, out_dim))
        out.zero_()
        chunk_rows = _projection_chunk_rows(in_dim, out_dim)
        for row_start in range(0, x_view.shape[0], chunk_rows):
            row_end = min(row_start + chunk_rows, x_view.shape[0])
            out_chunk = out[row_start:row_end]
            for in_factor in range(factor):
                x_slice = x_view[row_start:row_end, in_factor, :].float()
                projected = x_slice.matmul(w2_t)
                coeffs = w1_float[out_start : out_start + out_count, in_factor]
                out_chunk.add_(projected.unsqueeze(1) * coeffs.view(1, out_count, 1))
        ctx.save_for_backward(x, w1, w2)
        ctx.out_start = out_start
        ctx.out_count = out_count
        ctx.factor = factor
        ctx.in_dim = in_dim
        ctx.out_dim = out_dim
        ctx.leading_shape = leading_shape
        return out.reshape(*leading_shape, out_count * out_dim)

    @staticmethod
    def backward(ctx, grad_out):
        x, w1, w2 = ctx.saved_tensors
        out_start = ctx.out_start
        out_count = ctx.out_count
        factor = ctx.factor
        in_dim = ctx.in_dim
        out_dim = ctx.out_dim

        x_view = x.reshape(-1, factor, in_dim)
        w1_float = w1.float()
        w2_float = w2.float()
        grad_view = grad_out.reshape(-1, out_count, out_dim)

        grad_w1 = w1_float.new_zeros(w1_float.shape)
        grad_w2 = w2_float.new_zeros(w2_float.shape)
        grad_x = torch.empty(
            (x_view.shape[0], factor, in_dim), dtype=x.dtype, device=x.device
        )
        w2_t = w2_float.transpose(0, 1)
        chunk_rows = _projection_chunk_rows(in_dim, out_dim)
        out_slice = slice(out_start, out_start + out_count)
        for row_start in range(0, x_view.shape[0], chunk_rows):
            row_end = min(row_start + chunk_rows, x_view.shape[0])
            grad = grad_view[row_start:row_end].float()
            for in_factor in range(factor):
                x_slice = x_view[row_start:row_end, in_factor, :].float()
                projected = x_slice.matmul(w2_t)
                grad_w1[out_slice, in_factor].add_(
                    torch.einsum("ngo,no->g", grad, projected)
                )

                coeffs = w1_float[out_slice, in_factor]
                grad_projected = torch.einsum("g,ngo->no", coeffs, grad)
                grad_w2.add_(grad_projected.transpose(0, 1).matmul(x_slice))
                grad_x[row_start:row_end, in_factor, :] = grad_projected.matmul(
                    w2_float
                ).to(x.dtype)
        grad_x = grad_x.reshape(*ctx.leading_shape, factor * in_dim)

        return (
            grad_x,
            grad_w1.to(w1.dtype),
            grad_w2.to(w2.dtype),
            None,
            None,
            None,
            None,
            None,
        )


@torch.compiler.disable(recursive=True)
def lokr_project(x, w1, w2, factor, in_dim, out_dim):
    """Project through the LyCORIS-compatible Kronecker LoKR weight.

    ``torch.kron(w1, w2)`` materializes the full ``out_features x in_features``
    adapter weight. On Anima DiT's wide projections that temporary is large
    enough to erase the memory saved by block swap, so training uses the
    equivalent two-stage contraction instead:

    1. apply ``w2`` inside each Kronecker factor slice;
    2. mix factor slices with ``w1``.
    """

    return LoKrProjectFn.apply(x, w1, w2, factor, in_dim, out_dim)


@torch.compiler.disable(recursive=True)
def lokr_project_factor(x, w1, w2, out_factor, factor, in_dim, out_dim):
    """Project one output factor slice of the Kronecker LoKr weight."""

    return LoKrProjectFactorFn.apply(
        x, w1, w2, out_factor, factor, in_dim, out_dim
    )


@torch.compiler.disable(recursive=True)
def lokr_project_factor_group(x, w1, w2, out_start, out_count, factor, in_dim, out_dim):
    """Project a small group of output factor slices.

    This keeps the peak below a full LoKr output tensor but avoids recomputing
    the same ``x @ w2.T`` once for every single output factor.
    """

    return LoKrProjectFactorGroupFn.apply(
        x, w1, w2, out_start, out_count, factor, in_dim, out_dim
    )
