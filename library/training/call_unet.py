"""Default UNet forward used by AnimaTrainer.call_unet."""

from __future__ import annotations


def call_unet(
    args,
    accelerator,
    unet,
    noisy_latents,
    timesteps,
    text_conds,
    batch,
    weight_dtype,
    **kwargs,
):
    noise_pred = unet(noisy_latents, timesteps, text_conds[0]).sample
    return noise_pred
