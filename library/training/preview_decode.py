"""Shared preview persistence for Anima/Krea-2 and Z-Image VAE contracts."""

import logging
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from library.runtime.device import clean_memory_on_device
from library.training.preview_decode_config import decode_policy
from library.training.preview_decode_events import decode_event_sink
from library.training.preview_decode_policy import PreviewDecodeRestoreError, decode_with_policy, model_device

logger = logging.getLogger(__name__)


def pixels_to_image(decoded):
    image = torch.clamp((decoded.float() + 1.0) / 2.0, 0.0, 1.0)[0]
    if image.ndim == 4:
        if image.shape[1] != 1:
            raise ValueError("Training image previews require a singleton temporal dimension")
        image = image[:, 0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("Training image previews require RGB pixels")
    array = (255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)).astype(np.uint8)
    return Image.fromarray(array)


def decode_anima_latent(vae, latent):
    latent = latent.to(device=model_device(vae), dtype=vae.dtype)
    return pixels_to_image(vae.decode_to_pixels(latent))


def decode_z_image_latent(vae, latent):
    if latent.ndim != 4 or latent.shape[1] != 16:
        raise ValueError("Z-Image preview decoder expects [B,16,H,W]")
    latent = latent.to(device=model_device(vae), dtype=vae.dtype)
    latent = latent / float(vae.config.scaling_factor) + float(vae.config.shift_factor)
    return pixels_to_image(vae.decode(latent, return_dict=False)[0])


def decode_pending_samples(accelerator, args, vae, *, decode_latent, wandb_enabled=False):
    root = Path(args.output_dir) / "sample"
    staged = root / "latents"
    files = sorted(staged.glob("*.pt"))
    if not files or vae is None:
        return
    tracker = None
    if wandb_enabled and any(item.name == "wandb" for item in getattr(accelerator, "trackers", [])):
        import wandb
        tracker = accelerator.get_tracker("wandb")
    policy = decode_policy(args)
    try:
        for path in files:
            try:
                record = torch.load(path, map_location="cpu", weights_only=True)
                image = decode_with_policy(vae, record["latents"], accelerator.device, policy, decode_latent,
                                           event_sink=decode_event_sink(args.output_dir, path))
                try:
                    image.save(root / f"{path.stem}.png")
                    if tracker is not None:
                        tracker.log({f"sample_{record.get('enum', 0)}":
                                     wandb.Image(image, caption=record.get("prompt", ""))}, commit=False)
                finally:
                    image.close()
                path.unlink()
            except PreviewDecodeRestoreError:
                raise
            except Exception:
                logger.exception("Failed to decode preview %s; keeping latent for retry", path)
            clean_memory_on_device(accelerator.device)
    finally:
        clean_memory_on_device(accelerator.device)
    try:
        staged.rmdir()
    except OSError:
        pass


def raise_decode_failure(accelerator, error, processes):
    """Coordinate fatal decoder/placement errors before any rank waits again."""
    from library.training.sample_preview_common import failed_on_any_process

    if failed_on_any_process(accelerator, error is not None, processes):
        if error is not None:
            raise error
        raise RuntimeError("Training preview decode failed on another process")
