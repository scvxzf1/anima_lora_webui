"""DC-AE loading and training-space encode for the DC-Gen Anima forge.

The diffusers-format checkpoint ``mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers``
loads directly through ``AutoencoderDC.from_pretrained``. Its ``encode`` returns
*raw* latents (no scaling applied), so :func:`encode_images_to_latents` applies
the config ``scaling_factor`` before returning — matching what the DiT training
loop consumes, and matching Anima's old VAE whose ``encode_pixels_to_latents``
already returns normalized latents.
"""

from __future__ import annotations

from pathlib import Path

import torch
from diffusers import AutoencoderDC

from library.env import resolve_under_home
from library.models.latent_space import DCGEN_F32C32_P1

# Local mirror of the diffusers-format DC-AE used for the first Anima forge.
DEFAULT_DC_AE_DIR = Path("models/dc_ae/dc-ae-f32c32-sana-1.1-diffusers")


def load_dc_ae(
    path: str | Path = DEFAULT_DC_AE_DIR,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> AutoencoderDC:
    """Load the f32c32 DC-AE and verify it matches the target space spec."""
    path = Path(resolve_under_home(str(path)))
    ae = AutoencoderDC.from_pretrained(str(path), torch_dtype=dtype)
    ae = ae.to(device).eval()

    cfg = ae.config
    if int(cfg.latent_channels) != DCGEN_F32C32_P1.latent_channels:
        raise ValueError(
            f"DC-AE latent_channels={cfg.latent_channels}, expected "
            f"{DCGEN_F32C32_P1.latent_channels}"
        )
    # AutoencoderDC has no explicit compression field; derive from a 32px probe
    # only when it matters. The spec's compression is enforced by shape checks
    # in the cache probe instead of every load.
    return ae


@torch.no_grad()
def encode_images_to_latents(
    ae: AutoencoderDC,
    images: torch.Tensor,
    *,
    scaling_factor: float | None = None,
) -> torch.Tensor:
    """Encode ``(B, 3, H, W)`` images into training-space latents.

    Images are expected in the same [-1, 1] range as Anima's ``IMAGE_TRANSFORMS``
    output. The returned latent is ``(B, 32, H/32, W/32)`` multiplied by the
    DC-AE ``scaling_factor`` (raw VAE output if ``scaling_factor=0`` is passed).
    """
    if scaling_factor is None:
        scaling_factor = DCGEN_F32C32_P1.scaling_factor
    if images.dim() != 4:
        raise ValueError(f"images must be 4D (B,C,H,W), got {tuple(images.shape)}")
    raw = ae.encode(images).latent
    if scaling_factor:
        raw = raw * scaling_factor
    return raw
