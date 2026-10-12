"""Decode Qwen preview latents and persist inspectable edit comparisons."""

import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps, PngImagePlugin
import torch

from library.runtime.device import clean_memory_on_device
from library.models.qwen_image_2_1.preview_decode_policy import (
    PreviewDecodeRestoreError, decode_with_policy,
)
from library.training.preview_decode_config import decode_policy
from library.training.preview_decode_events import decode_event_sink

logger = logging.getLogger(__name__)


def decode_latent(vae, latent):
    if latent.ndim != 4 or latent.shape[1] != 64:
        raise ValueError("Qwen preview decoder expects [B,64,H,W]")
    latent = latent.unsqueeze(2).to(device=vae.device, dtype=vae.dtype)
    mean = latent.new_tensor(vae.config.latents_mean).view(1, 64, 1, 1, 1)
    std = latent.new_tensor(vae.config.latents_std).view(1, 64, 1, 1, 1)
    pixels = vae.decode(latent * std + mean, return_dict=False)[0][0, :3, 0]
    return Image.fromarray(((pixels.float().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 0).cpu().numpy().astype(np.uint8))


def comparison_image(reference, generated):
    width, height = generated.size
    pair = Image.new("RGB", (width * 2, height + 28), "#eeeeee")
    pair.paste(ImageOps.contain(reference, (width, height)), (0, 28))
    pair.paste(generated, (width, 28))
    draw = ImageDraw.Draw(pair)
    draw.text((8, 8), "Reference", fill="black")
    draw.text((width + 8, 8), "Edited", fill="black")
    return pair


def multi_comparison_image(references, generated):
    """Show every reference and the result in bounded, equal-width columns."""
    cell_width = min(generated.width, 512)
    cell_height = max(1, round(generated.height * cell_width / generated.width))
    row = Image.new("RGB", (cell_width * (len(references) + 1), cell_height + 28), "#eeeeee")
    draw = ImageDraw.Draw(row)
    for index, source in enumerate([*references, generated]):
        image = ImageOps.contain(source, (cell_width, cell_height))
        x = index * cell_width
        row.paste(image, (x + (cell_width - image.width) // 2, 28 + (cell_height - image.height) // 2))
        if index < len(references):
            label = f"Ref {index + 1}" if cell_width < 96 else f"Reference {index + 1}"
        else:
            label = "Edited"
        draw.text((x + 8, 8), label, fill="black")
    return row


def decode_pending_samples(accelerator, args, vae):
    root = Path(args.output_dir) / "sample"
    files = sorted((root / "latents").glob("*.pt"))
    if not files:
        return
    try:
        for path in files:
            generated = image = None
            reference_images = []
            try:
                record = torch.load(path, map_location="cpu", weights_only=True)
                generated = decode_with_policy(
                    vae, record["latents"], accelerator.device,
                    decode_policy(args), decode_latent,
                    event_sink=decode_event_sink(args.output_dir, path),
                )
                metadata = record["metadata"]
                image = generated
                references = metadata.get("reference_images")
                if references is None:
                    references = [metadata["reference_image"]] if metadata.get("reference_image") else []
                if references:
                    metadata["reference_files"] = [
                        Path(reference).relative_to(Path(args.output_dir).resolve()).as_posix()
                        for reference in references
                    ]
                    metadata["reference_file"] = metadata["reference_files"][0]
                    results = root / "results"
                    results.mkdir(exist_ok=True)
                    result_path = results / f"{path.stem}.png"
                    generated.save(result_path)
                    metadata["result_file"] = result_path.relative_to(Path(args.output_dir)).as_posix()
                    for reference in references:
                        with Image.open(reference) as source:
                            reference_images.append(source.convert("RGB"))
                    image = (comparison_image(reference_images[0], generated) if len(reference_images) == 1
                             else multi_comparison_image(reference_images, generated))
                info = PngImagePlugin.PngInfo()
                info.add_text("qwen_preview", json.dumps(metadata, ensure_ascii=False))
                image.save(root / f"{path.stem}.png", pnginfo=info)
                path.unlink()
            except PreviewDecodeRestoreError:
                raise
            except Exception:
                logger.exception("Failed to decode Qwen preview %s; keeping latent for retry", path)
            finally:
                for reference_image in reference_images:
                    reference_image.close()
                if image is not None and image is not generated:
                    image.close()
                if generated is not None:
                    generated.close()
            clean_memory_on_device(accelerator.device)
    finally:
        clean_memory_on_device(accelerator.device)
