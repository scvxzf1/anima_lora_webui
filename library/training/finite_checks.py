"""Fail-fast checks for invalid training loss and adapter gradients."""

from __future__ import annotations

import torch

from library.training.v100_flash import env_flag


def debug_finite_enabled(args) -> bool:
    return bool(getattr(args, "debug_finite_checks", False)) or env_flag(
        "ANIMA_DEBUG_FINITE"
    )


def check_loss_finite(
    loss: torch.Tensor,
    *,
    mixed_precision: str | None = None,
) -> None:
    if torch.isfinite(loss.detach()).all():
        return
    hint = ""
    if mixed_precision == "fp16":
        hint = (
            " fp16 autocast is active; inspect batch/sigma, residual-range "
            "guards, and the fp16-safe FinalLayer projection path."
        )
    raise FloatingPointError(
        "non-finite training loss before backward; aborting to avoid logging "
        "NaN averages or saving an invalid checkpoint." + hint
    )


def check_trainable_grads_finite(network) -> None:
    bad: list[str] = []
    for name, parameter in network.named_parameters():
        grad = parameter.grad
        if grad is None:
            continue
        if not torch.isfinite(grad.detach()).all():
            bad.append(f"{name}: dtype={grad.dtype} shape={tuple(grad.shape)}")
            if len(bad) >= 8:
                break
    if bad:
        raise FloatingPointError(
            "non-finite trainable gradients after backward: " + "; ".join(bad)
        )
