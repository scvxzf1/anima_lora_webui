"""Replay cached 1024 input without trimming text attention sinks."""

from __future__ import annotations
from pathlib import Path
import torch
from .manifest import file_hash, object_hash
from .numerics import tensor_hash
from scripts.experiments.int8_linear_equivalence_probe import select_cached_batch_pair


def validate_tensors(latents: torch.Tensor, context: torch.Tensor) -> None:
    if latents.ndim != 3 or tuple(latents.shape) != (16, 128, 128):
        raise ValueError(
            f"expected 1024 Anima latent [16,128,128], got {tuple(latents.shape)}"
        )
    if context.ndim != 2 or context.shape[0] != 512:
        raise ValueError(
            f"expected max-padded text [512,D], got {tuple(context.shape)}"
        )
    if not torch.isfinite(latents).all() or not torch.isfinite(context).all():
        raise ValueError("cache contains NaN/Inf")


def load_input(
    cache_dir: Path,
    image_dir: Path,
    index: int,
    seed: int,
    timestep: float | None = None,
    device="cpu",
):
    from library.io.cache import load_cached_latents, load_cached_crossattn_emb

    pair = select_cached_batch_pair(cache_dir, index)
    images = sorted(
        p
        for p in image_dir.glob(pair.base_stem + ".*")
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    if len(images) != 1:
        raise ValueError(
            f"expected exactly one source image for {pair.base_stem}, got {len(images)}"
        )
    caption = image_dir / (pair.base_stem + ".txt")
    latents, resolution, orig_h, orig_w = load_cached_latents(str(pair.latent_path))
    context = load_cached_crossattn_emb(str(pair.text_path), variant=0)
    if context is None:
        raise ValueError("missing crossattn_emb")
    validate_tensors(latents, context)
    generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    noise = torch.randn((1, *latents.shape), generator=generator)
    sigma = (
        float(torch.rand((), generator=generator).clamp(1e-4, 1 - 1e-4))
        if timestep is None
        else timestep
    )
    if not 0 < sigma < 1:
        raise ValueError("timestep is actual flow sigma and must be in (0,1)")
    # Generate replay tensors on CPU, so different device/backend modes see identical bits.
    latent = latents.unsqueeze(0).to(torch.float32)
    noise = noise.to(torch.float32)
    sigma_tensor = torch.tensor([sigma], dtype=torch.bfloat16)
    sigma_work = torch.tensor([sigma], dtype=torch.float32).view(1, 1, 1, 1)
    x = ((1 - sigma_work) * latent + sigma_work * noise).to(torch.bfloat16).unsqueeze(2)
    target = (noise - latent).unsqueeze(2)
    tensors = {
        "x": x,
        "timesteps": sigma_tensor,
        "context": context.unsqueeze(0).to(torch.bfloat16),
        "padding_mask": torch.zeros(1, 1, 128, 128, dtype=torch.bfloat16),
        "target": target,
    }
    hashes = {
        str(p.resolve()): file_hash(p)
        for p in (pair.latent_path, pair.text_path, images[0], caption)
    }
    metadata = {
        "files": hashes,
        "cache_index": index,
        "seed": seed,
        "sigma_requested": sigma,
        "sigma_actual": float(sigma_tensor[0]),
        "resolution": resolution,
        "original_size": [orig_w, orig_h],
        "shapes": {k: list(v.shape) for k, v in tensors.items()},
        "text_padding": "full 512 positions retained; no sequence mask",
    }
    metadata["tensor_hashes"] = {k: tensor_hash(v) for k, v in tensors.items()}
    metadata["input_hash"] = object_hash(metadata)
    return {k: v.to(device) for k, v in tensors.items()}, metadata
