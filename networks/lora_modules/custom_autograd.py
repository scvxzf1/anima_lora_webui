# Memory-saving autograd for the LoRA down projection.
#
# `F.linear(x.float(), weight.float())` saves the fp32-cast input for backward
# (~32 MiB per 2048-wide Linear at 4096 tokens, ×N adapted modules). These
# Functions save the bf16 `x` and recompute the cast in backward. The unscaled
# Function is bitwise-identical to the legacy path; the scaled variant folds
# `inv_scale` into the weight at the matmul (avoiding a (B, L, in_dim) bf16
# intermediate), so it is equivalent up to fp32-vs-bf16 rounding order.
#
# Two Functions (scaled / unscaled) instead of one with an optional tensor:
# keeps the compile graph shape fixed.

from __future__ import annotations

import torch


class LoRADownProjectFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight):
        out = torch.nn.functional.linear(x.float(), weight.float())
        ctx.save_for_backward(x, weight)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        go = grad_out.float()
        w_f = weight.float()
        x_f = x.float()

        grad_x = go.matmul(w_f).to(x.dtype)
        grad_weight = go.reshape(-1, go.shape[-1]).transpose(0, 1).matmul(
            x_f.reshape(-1, x_f.shape[-1])
        )
        return grad_x, grad_weight.to(weight.dtype)


class ScaledLoRADownProjectFn(torch.autograd.Function):
    """Scaled variant: equivalent to ``F.linear(x * inv_scale, weight)`` in fp32.

    Identity used: ``F.linear(x * c, W) == F.linear(x, W * c)`` for a per-input
    feature scale ``c``. We fold ``inv_scale`` into ``weight`` at the matmul
    instead of materializing ``x_work = x * inv`` (a ``(B, L, in_dim)`` bf16
    tensor). Under ``compile_inductor_mode = "reduce-overhead"`` the
    intermediate would otherwise get pinned in the CUDA-Graph pool across all
    adapted Linears — ~16 MiB × N modules of avoidable activation memory.

    ``inv_scale`` is a calibration buffer (no gradient), stored fp32 for
    1/s_norm precision; cast to fp32 at the matmul boundary is free since the
    matmul is already fp32. Saved-for-backward stays bf16 ``x`` + bf16
    ``weight`` + fp32 ``inv_scale`` (in_features-sized, negligible).
    """

    @staticmethod
    def forward(ctx, x, weight, inv_scale):
        inv_f = inv_scale.float()
        w_scaled = weight.float() * inv_f.unsqueeze(0)
        out = torch.nn.functional.linear(x.float(), w_scaled)
        ctx.save_for_backward(x, weight, inv_scale)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, inv_scale = ctx.saved_tensors
        go = grad_out.float()
        inv_f = inv_scale.float()
        w_f = weight.float()
        x_f = x.float()

        # out = x_f @ (W_f * inv).T
        #   ⇒ grad_W_scaled = go^T @ x_f         (fp32, (r, in))
        #   ⇒ grad_W        = grad_W_scaled * inv  (chain rule: W_scaled = W * inv)
        grad_w_scaled = go.reshape(-1, go.shape[-1]).transpose(0, 1).matmul(
            x_f.reshape(-1, x_f.shape[-1])
        )
        grad_weight = grad_w_scaled * inv_f.unsqueeze(0)

        # grad_x = go @ W_scaled (fp32 throughout, cast at end); avoids
        # materializing a bf16 grad_x_work intermediate.
        w_scaled = w_f * inv_f.unsqueeze(0)
        grad_x = go.matmul(w_scaled).to(x.dtype)
        return grad_x, grad_weight.to(weight.dtype), None


def lora_down_project(x, weight, inv_scale):
    """Dispatch helper: picks the scaled or unscaled Function based on inv_scale."""
    if inv_scale is None:
        return LoRADownProjectFn.apply(x, weight)
    return ScaledLoRADownProjectFn.apply(x, weight, inv_scale)


class LoKrProjectFn(torch.autograd.Function):
    """Memory-saving LoKr projection.

    The naive path materializes ``kron(w1, w2)`` as a full Linear weight. During
    training that full matrix can be saved by autograd for every adapted module.
    This Function saves only the original input plus the two small Kronecker
    factors, then recomputes the intermediate projection in backward.
    """

    @staticmethod
    def forward(ctx, x, w1, w2, factor: int, in_dim: int, out_dim: int):
        x_f = x.float()
        w1_f = w1.float()
        w2_f = w2.float()
        x_view = x_f.reshape(*x_f.shape[:-1], factor, in_dim)
        tmp = torch.nn.functional.linear(x_view, w2_f)
        mixed = torch.matmul(tmp.movedim(-2, -1), w1_f.transpose(0, 1)).movedim(
            -1, -2
        )
        ctx.factor = factor
        ctx.in_dim = in_dim
        ctx.out_dim = out_dim
        ctx.x_shape = x.shape
        ctx.save_for_backward(x, w1, w2)
        return mixed.reshape(*x_f.shape[:-1], factor * out_dim)

    @staticmethod
    def backward(ctx, grad_out):
        x, w1, w2 = ctx.saved_tensors
        factor = ctx.factor
        in_dim = ctx.in_dim
        out_dim = ctx.out_dim

        x_f = x.float().reshape(-1, factor, in_dim)
        w1_f = w1.float()
        w2_f = w2.float()
        go = grad_out.float().reshape(-1, factor, out_dim)

        tmp = torch.nn.functional.linear(x_f, w2_f)
        grad_w1 = torch.einsum("nap,nbp->ab", go, tmp)

        grad_tmp = torch.matmul(go.movedim(-2, -1), w1_f).movedim(-1, -2)
        grad_w2 = torch.einsum("nbp,nbi->pi", grad_tmp, x_f)
        grad_x = torch.matmul(grad_tmp, w2_f).reshape(ctx.x_shape)

        return (
            grad_x.to(x.dtype),
            grad_w1.to(w1.dtype),
            grad_w2.to(w2.dtype),
            None,
            None,
            None,
        )


def lokr_project(x, w1, w2, factor: int, in_dim: int, out_dim: int):
    return LoKrProjectFn.apply(x, w1, w2, factor, in_dim, out_dim)
