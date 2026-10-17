# SPDX-License-Identifier: Apache-2.0
# MOSS-Image-SFT, revision daf706d2e1c7a6369c17b6d1658671aee6db1ebe.
"""FLUX.2 native 32-channel latents ↔ normalized packed MOSS 128 channels."""
from __future__ import annotations
import torch


def pack_moss_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4 or latents.shape[1] != 32:
        raise ValueError("MOSS VAE native latents must be [B,32,H,W]")
    b, c, h, w = latents.shape
    if h % 2 or w % 2:
        raise ValueError("MOSS native latent height/width must be even")
    return latents.reshape(b, c, h // 2, 2, w // 2, 2).permute(
        0, 1, 3, 5, 2, 4
    ).reshape(b, 128, h // 2, w // 2)


def unpack_moss_latents(latents: torch.Tensor) -> torch.Tensor:
    if latents.ndim != 4 or latents.shape[1] != 128:
        raise ValueError("MOSS packed latents must be [B,128,H,W]")
    b, _, h, w = latents.shape
    return latents.reshape(b, 32, 2, 2, h, w).permute(
        0, 1, 4, 2, 5, 3
    ).reshape(b, 32, h * 2, w * 2)


def _bn_stats(vae, latents):
    mean = vae.bn.running_mean.to(latents).view(1, -1, 1, 1)
    std = (vae.bn.running_var + vae.config.batch_norm_eps).sqrt().to(latents).view(1, -1, 1, 1)
    if mean.shape[1] != 128:
        raise ValueError("MOSS requires FLUX.2 VAE BN with 128 packed channels")
    return mean, std


def encode_moss_latents(vae, images: torch.Tensor) -> torch.Tensor:
    """Encode posterior mode, pack 2×2, normalize using frozen BN statistics."""
    if images.ndim != 4 or images.shape[1] != 3 or any(s % 16 for s in images.shape[-2:]):
        raise ValueError("MOSS images must be [B,3,H,W] with H,W divisible by 16")
    packed = pack_moss_latents(vae.encode(images).latent_dist.mode())
    mean, std = _bn_stats(vae, packed)
    return (packed - mean) / std


def decode_moss_latents(vae, latents: torch.Tensor) -> torch.Tensor:
    """Invert BN normalization and packing before native FLUX.2 decoding."""
    mean, std = _bn_stats(vae, latents)
    native = unpack_moss_latents(latents * std + mean)
    return vae.decode(native, return_dict=False)[0]
