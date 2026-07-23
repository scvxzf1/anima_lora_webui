"""Spatial mask reduction helpers for training losses.

The legacy path multiplies a per-pixel loss by the mask and then averages over
the full latent. ``foreground_mean`` keeps the foreground gradient scale
independent of mask coverage while preserving the legacy default.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def mask_image_for_loss(
    loss: torch.Tensor, batch: dict[str, Any]
) -> torch.Tensor | None:
    """Return a broadcastable ``[B, 1, H, W]`` mask, if the batch has one."""
    if "conditioning_images" in batch:
        mask = batch["conditioning_images"].to(dtype=loss.dtype)[:, 0].unsqueeze(1)
        return F.interpolate(mask / 2 + 0.5, size=loss.shape[2:], mode="area")
    alpha_masks = batch.get("alpha_masks")
    if alpha_masks is None:
        return None
    mask = alpha_masks.to(dtype=loss.dtype).unsqueeze(1)
    return F.interpolate(mask, size=loss.shape[2:], mode="area")


def apply_mask(loss: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
    """Apply a spatial mask without changing the tensor shape."""
    mask = mask_image_for_loss(loss, batch)
    return loss if mask is None else loss * mask


def reduce_masked_loss(
    loss: torch.Tensor,
    batch: dict[str, Any],
    *,
    normalize: str = "none",
    foreground_weight: float = 1.0,
) -> torch.Tensor:
    """Reduce a per-pixel loss to ``[B]`` with optional foreground normalization.

    ``none`` is the historical full-latent mean. ``foreground_mean`` averages
    only over the soft mask and falls back to the unmasked mean for an empty
    mask, preventing NaNs and making malformed masks observable by callers.
    """
    mask = mask_image_for_loss(loss, batch)
    full_mean = loss.mean(dim=tuple(range(1, loss.ndim)))
    if mask is None:
        return full_mean
    expanded_mask = mask.expand_as(loss)
    masked = loss * expanded_mask
    if normalize == "none":
        return masked.mean(dim=tuple(range(1, loss.ndim)))
    if normalize != "foreground_mean":
        raise ValueError(f"unsupported mask loss normalization: {normalize!r}")

    reduce_dims = tuple(range(1, loss.ndim))
    area = expanded_mask.sum(dim=reduce_dims)
    valid = area > 1e-6
    denom = area.clamp_min(1e-6)
    foreground = masked.sum(dim=reduce_dims) / denom
    result = torch.where(valid, foreground * float(foreground_weight), full_mean)
    return result
