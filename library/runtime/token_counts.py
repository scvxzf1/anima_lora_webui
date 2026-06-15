"""Token-count helpers shared by Anima compile paths."""

from __future__ import annotations

from collections.abc import Iterable

ANIMA_VAE_SPATIAL_COMPRESSION = 8


def pixel_bucket_token_count(
    resolution: tuple[int, int],
    *,
    patch_spatial: int,
    vae_spatial_compression: int = ANIMA_VAE_SPATIAL_COMPRESSION,
) -> int:
    """Return DiT block token count for a pixel-space bucket resolution.

    Web/preprocess buckets are stored as image pixels. The DiT sees VAE latents,
    then patches them again, so the real block sequence length is:
    ``(width / vae_scale / patch) * (height / vae_scale / patch)``.
    """

    width, height = (int(resolution[0]), int(resolution[1]))
    stride = int(patch_spatial) * int(vae_spatial_compression)
    if stride <= 0:
        raise ValueError("patch_spatial 和 vae_spatial_compression 必须为正数")
    return max(1, width // stride) * max(1, height // stride)


def pixel_bucket_token_counts(
    resolutions: Iterable[tuple[int, int]],
    *,
    patch_spatial: int,
    vae_spatial_compression: int = ANIMA_VAE_SPATIAL_COMPRESSION,
) -> set[int]:
    return {
        pixel_bucket_token_count(
            resolution,
            patch_spatial=patch_spatial,
            vae_spatial_compression=vae_spatial_compression,
        )
        for resolution in resolutions
    }
