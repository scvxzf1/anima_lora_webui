"""Compile-safe finite-value diagnostics for Anima attention and training."""

from __future__ import annotations

import torch
from torch import nn


def finite_checks_enabled(default: bool = False) -> bool:
    return bool(default)


def assert_finite_tensor(
    tensor: torch.Tensor,
    label: str,
    *,
    block: nn.Module | None = None,
) -> None:
    if torch.compiler.is_compiling():
        torch._assert_async(torch.isfinite(tensor).all(), label)
        return

    if block is not None:
        block_index = getattr(
            block,
            "_block_index",
            getattr(block, "_block_idx", "?"),
        )
        label = f"block={block_index}.{label}"

    detached = tensor.detach()
    finite = torch.isfinite(detached)
    if finite.all():
        return

    finite_values = detached[finite]
    if finite_values.numel() > 0:
        finite_range = (
            f" finite_min={float(finite_values.min().item()):.6g}"
            f" finite_max={float(finite_values.max().item()):.6g}"
        )
    else:
        finite_range = " no_finite_values"
    raise FloatingPointError(
        f"non-finite tensor at {label}: dtype={tensor.dtype} "
        f"shape={tuple(tensor.shape)}{finite_range}"
    )
