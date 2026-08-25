"""Z-Image VAE encoding and latent normalization."""

from __future__ import annotations

import torch

from library.models.latent_space import Z_IMAGE_F8C16_P2


def normalize_z_image_latents(latents: torch.Tensor) -> torch.Tensor:
    """Apply the affine normalization expected by the Z-Image transformer."""
    return (latents - float(Z_IMAGE_F8C16_P2.shift_factor)) * float(
        Z_IMAGE_F8C16_P2.scaling_factor
    )


def encode_z_image_latents(vae, images: torch.Tensor) -> torch.Tensor:
    return normalize_z_image_latents(vae.encode(images).latent_dist.mode())
