"""Memory-conscious LoHa linear autograd.

Forward still builds the Hadamard-product weight (same math as PEFT/LyCORIS),
but only the four low-rank factors and the activation are saved for backward.
The full ``out × in`` weight is recomputed in backward instead of retained on
the autograd tape.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def make_hada_weight(
    w1_a: torch.Tensor,
    w1_b: torch.Tensor,
    w2_a: torch.Tensor,
    w2_b: torch.Tensor,
) -> torch.Tensor:
    """``(w1_a @ w1_b) ⊙ (w2_a @ w2_b)`` in fp32 matmul precision."""

    return (w1_a.float() @ w1_b.float()) * (w2_a.float() @ w2_b.float())


class HadaLinearFn(torch.autograd.Function):
    """``y = scale * F.linear(x, (w1a@w1b) ⊙ (w2a@w2b))``."""

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        w1_a: torch.Tensor,
        w1_b: torch.Tensor,
        w2_a: torch.Tensor,
        w2_b: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        weight = make_hada_weight(w1_a, w1_b, w2_a, w2_b)
        out = F.linear(x.float(), weight)
        # Keep factors + activation only; drop the materialized weight.
        ctx.save_for_backward(x, w1_a, w1_b, w2_a, w2_b)
        ctx.scale = float(scale)
        return (out * ctx.scale).to(dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, w1_a, w1_b, w2_a, w2_b = ctx.saved_tensors
        scale = ctx.scale
        go = grad_out.float() * scale
        x_f = x.float()

        weight = make_hada_weight(w1_a, w1_b, w2_a, w2_b)
        grad_x = go.matmul(weight).to(dtype=x.dtype)

        # dL/dW = go^T @ x  → shape (out, in)
        grad_w = go.reshape(-1, go.shape[-1]).transpose(0, 1).matmul(
            x_f.reshape(-1, x_f.shape[-1])
        )

        w1 = w1_a.float() @ w1_b.float()
        w2 = w2_a.float() @ w2_b.float()
        # d/dW1 (W1 ⊙ W2) = W2; d/dW2 (W1 ⊙ W2) = W1
        grad_w1 = grad_w * w2
        grad_w2 = grad_w * w1

        grad_w1_a = grad_w1.matmul(w1_b.float().transpose(0, 1)).to(dtype=w1_a.dtype)
        grad_w1_b = w1_a.float().transpose(0, 1).matmul(grad_w1).to(dtype=w1_b.dtype)
        grad_w2_a = grad_w2.matmul(w2_b.float().transpose(0, 1)).to(dtype=w2_a.dtype)
        grad_w2_b = w2_a.float().transpose(0, 1).matmul(grad_w2).to(dtype=w2_b.dtype)

        return grad_x, grad_w1_a, grad_w1_b, grad_w2_a, grad_w2_b, None


def loha_linear(
    x: torch.Tensor,
    w1_a: torch.Tensor,
    w1_b: torch.Tensor,
    w2_a: torch.Tensor,
    w2_b: torch.Tensor,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    """Public helper used by :class:`LoHaModule` forward."""

    return HadaLinearFn.apply(x, w1_a, w1_b, w2_a, w2_b, float(scale))
