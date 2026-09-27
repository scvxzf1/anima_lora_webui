"""RGBA VAE input and 64-channel latent normalization."""

from __future__ import annotations

import torch


def encode_qwen_image_2_1_latents(vae, images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] not in (3, 4):
        raise ValueError("Qwen Image 2.1 expects RGB/RGBA images [B,C,H,W]")
    if images.shape[-2] % 16 or images.shape[-1] % 16:
        raise ValueError("Qwen Image 2.1 image dimensions must be multiples of 16")
    if images.shape[1] == 3:
        images = torch.cat([images, torch.ones_like(images[:, :1])], dim=1)
    latents = vae.encode(images.unsqueeze(2)).latent_dist.mode()
    if latents.shape[1] != 64 or latents.shape[2] != 1:
        raise ValueError(f"Qwen Image 2.1 VAE returned invalid shape {tuple(latents.shape)}")
    mean = latents.new_tensor(vae.config.latents_mean).view(1, 64, 1, 1, 1)
    std = latents.new_tensor(vae.config.latents_std).view(1, 64, 1, 1, 1)
    return ((latents - mean) / std).squeeze(2)
