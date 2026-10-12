"""Read-only standalone decoding of normalized Qwen Image 2.1 latent files."""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import torch
from safetensors import safe_open

from library.models.qwen_image_2_1 import inference_runner

logger = logging.getLogger(__name__)


def _validate_decode_args(args) -> None:
    task = str(getattr(args, "qwen_image_2_1_task", None) or "t2i").lower()
    if task == "edit" or any(getattr(args, key, None) for key in (
        "reference_image", "reference_images", "lora_weight", "from_file", "interactive",
    )):
        raise SystemExit(
            "Qwen Image 2.1 latent decode does not accept Edit/reference flags, "
            "--lora_weight, --from_file or --interactive."
        )
    inference_runner.validate_qwen_image_2_1_inference_args(args, mode="single")


def _read_latent(path: Path) -> tuple[torch.Tensor, dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Qwen Image 2.1 latent file does not exist: {path}")
    metadata = {}
    if path.suffix == ".safetensors":
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if "latent" not in handle.keys():
                raise ValueError(f"Qwen Image 2.1 latent file lacks 'latent': {path}")
            latent = handle.get_tensor("latent")
            metadata = handle.metadata() or {}
    else:
        latent = torch.load(path, map_location="cpu", weights_only=True)
    if (not isinstance(latent, torch.Tensor) or latent.ndim != 4
            or latent.shape[:2] != (1, 64) or any(size <= 0 for size in latent.shape[-2:])):
        shape = tuple(latent.shape) if isinstance(latent, torch.Tensor) else type(latent).__name__
        raise ValueError(f"Qwen Image 2.1 latent must be [1,64,H,W] with positive H/W: {path}: {shape}")
    if not latent.is_floating_point() or not torch.isfinite(latent).all():
        raise ValueError(f"Qwen Image 2.1 latent must be floating and finite: {path}")
    for key, expected in (("height", latent.shape[-2] * 16), ("width", latent.shape[-1] * 16)):
        if key in metadata and int(metadata[key]) != expected:
            raise ValueError(f"Qwen Image 2.1 latent metadata {key} must equal {expected}: {path}")
    return latent, metadata


def decode_qwen_latent_files(args, device: torch.device) -> None:
    """Preflight every input before loading only the family-specific output VAE."""
    _validate_decode_args(args)
    records = []
    for filename in args.latent_path:
        path = Path(filename)
        latent, metadata = _read_latent(path)
        output_args = copy.copy(args)
        output_args.image_size = [latent.shape[-2] * 16, latent.shape[-1] * 16]
        output_args.seed = int(metadata.get("seeds", args.seed if args.seed is not None else 0))
        for key in ("prompt", "negative_prompt"):
            if key in metadata:
                setattr(output_args, key, metadata[key])
        records.append((path, latent, output_args))
    for path, latent, output_args in records:
        logger.info("Loaded Qwen Image 2.1 latent from %s. Shape: %s", path, tuple(latent.shape))
        inference_runner.save_qwen_image_2_1_output(
            output_args, latent, device, original_base_name=Path(path.name).stem,
        )
