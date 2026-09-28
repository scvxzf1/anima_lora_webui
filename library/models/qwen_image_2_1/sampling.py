"""Official flow-match Euler scheduling for Qwen Image 2.1 previews."""

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler

from library.models.qwen_image_2_1.family import forward_for_loss


@torch.no_grad()
def generate_preview(dit, prompt, positive, negative, reference, *, device, dtype, seed, forward=forward_for_loss):
    width, height = int(prompt.get("width", 512)), int(prompt.get("height", 512))
    count = int(prompt.get("sample_steps", 28))
    scale = float(prompt.get("guidance_scale", prompt.get("scale", 1.0)))
    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn((1, 64, height // 16, width // 16), generator=generator, device=device, dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    tokens = latent.shape[-2] * latent.shape[-1]
    mu = 0.5 + (tokens - 256) * (1.15 - 0.5) / (4096 - 256)
    scheduler.set_timesteps(count, device=device, sigmas=np.linspace(1.0, 1 / count, count), mu=mu)

    def condition(values):
        return tuple(value.to(device=device, dtype=dtype if index == 0 else torch.bool) for index, value in enumerate(values))

    positive = condition(positive)
    negative = condition(negative) if scale > 1 and prompt.get("negative_prompt") is not None and negative is not None else None
    if reference is not None:
        if isinstance(reference, torch.Tensor):
            reference = reference.to(device=device, dtype=dtype)
        else:
            reference = tuple(value.to(device=device, dtype=dtype) for value in reference)

    def predict(values, sigma):
        prepare = getattr(dit, "prepare_block_swap_before_forward", None)
        if prepare:
            prepare()
        return forward(dit, latent, values[0], values[1], sigma,
                       reference_latents=reference,
                       image_slot_mask=values[2] if reference is not None else None)

    for timestep in scheduler.timesteps:
        sigma = (timestep / 1000).expand(1)
        prediction = predict(positive, sigma)
        if negative is not None:
            unconditional = predict(negative, sigma)
            prediction = unconditional + scale * (prediction - unconditional)
        latent = scheduler.step(prediction, timestep, latent, return_dict=False)[0].to(dtype)
    return latent
