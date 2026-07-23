"""Synchronous affine augmentation for cached latent training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def affine_probabilities(
    state: dict[str, Any], args: Any, timesteps: torch.Tensor
) -> torch.Tensor | None:
    if not bool(getattr(args, "adaptive_personalization_affine", False)):
        return None
    bins = state.get("bins")
    if not isinstance(bins, list) or not bins:
        return timesteps.new_zeros(timesteps.shape, dtype=torch.float32)
    minimum = max(1, int(getattr(args, "adaptive_personalization_min_bin_samples", 16)))
    maximum = min(
        1.0,
        max(
            0.0,
            float(
                getattr(args, "adaptive_personalization_affine_probability_max", 0.5)
            ),
        ),
    )
    values = [
        min(maximum, max(0.0, float(item.get("gamma", 0.0) or 0.0)))
        if int(item.get("count", 0) or 0) >= minimum
        else 0.0
        for item in bins
    ]
    table = timesteps.new_tensor(values, dtype=torch.float32)
    indices = torch.clamp(
        (timesteps.detach().float() * len(values)).long(), 0, len(values) - 1
    )
    return table[indices]


def _theta(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    args: Any,
    selected: torch.Tensor,
) -> torch.Tensor:
    theta = torch.zeros(batch_size, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    count = int(selected.sum().item())
    if count == 0:
        return theta
    indices = torch.nonzero(selected, as_tuple=False).flatten()
    max_angle = float(
        getattr(args, "adaptive_personalization_affine_rotation_deg", 10.0)
    )
    max_translate = float(
        getattr(args, "adaptive_personalization_affine_translation", 0.05)
    )
    max_scale = float(
        getattr(args, "adaptive_personalization_affine_scale_delta", 0.05)
    )
    angle = (torch.rand(count, device=device) * 2.0 - 1.0) * max_angle
    angle = angle * torch.pi / 180.0
    scale = 1.0 + (torch.rand(count, device=device) * 2.0 - 1.0) * max_scale
    cosine = angle.cos() * scale
    sine = angle.sin() * scale
    theta[indices, 0, 0] = cosine
    theta[indices, 0, 1] = -sine
    theta[indices, 1, 0] = sine
    theta[indices, 1, 1] = cosine
    theta[indices, 0, 2] = (
        torch.rand(count, device=device) * 2.0 - 1.0
    ) * max_translate
    theta[indices, 1, 2] = (
        torch.rand(count, device=device) * 2.0 - 1.0
    ) * max_translate
    return theta


def _warp(
    value: torch.Tensor, theta: torch.Tensor, *, mode: str = "bilinear"
) -> torch.Tensor:
    original_dtype = value.dtype
    if value.ndim == 3:
        value = value.unsqueeze(1)
        squeezed = True
    else:
        squeezed = False
    work = value.float()
    grid = F.affine_grid(theta.float(), work.shape, align_corners=False)
    output = F.grid_sample(
        work,
        grid,
        mode=mode,
        padding_mode="border",
        align_corners=False,
    )
    output = output.to(dtype=original_dtype)
    return output.squeeze(1) if squeezed else output


def apply_synchronous_affine(
    latents: torch.Tensor,
    noise: torch.Tensor,
    noisy_input: torch.Tensor,
    batch: dict[str, Any],
    probabilities: torch.Tensor,
    args: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Warp all spatial training inputs with one per-sample affine transform."""
    selected = torch.rand_like(probabilities) < probabilities.to(probabilities.device)
    theta = _theta(
        latents.shape[0],
        device=latents.device,
        dtype=latents.dtype,
        args=args,
        selected=selected,
    )
    if not bool(selected.any()):
        return latents, noise, noisy_input, 0.0
    latents = _warp(latents, theta)
    noise = _warp(noise, theta)
    noisy_input = _warp(noisy_input, theta)
    for key in ("alpha_masks", "conditioning_images"):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            transformed = _warp(value.to(device=latents.device), theta)
            if key == "alpha_masks":
                transformed = transformed.clamp(0.0, 1.0)
            batch[key] = transformed
    return latents, noise, noisy_input, float(selected.float().mean().item())
