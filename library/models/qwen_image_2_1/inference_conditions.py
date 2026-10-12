"""Ordered, in-memory Edit conditions prepared before standalone DiT loading."""

from __future__ import annotations

import gc

import numpy as np
import torch
from PIL import Image

from library.datasets.qwen_image_edit import reference_size_for_bucket
from library.models.qwen_image_2_1.inference_text_encoder import encode_with_text_encoder
from library.models.qwen_image_2_1.latent import encode_qwen_image_2_1_latents
from library.models.qwen_image_2_1.strategy import QwenImage21EditTokenizeStrategy, encode_edit_prompt
from library.models.qwen_image_2_1.weights import (
    load_qwen_image_2_1_text_encoder,
    load_qwen_image_2_1_vae,
)
from library.runtime.device import clean_memory_on_device
from library.training.preview_spec import preview_references, read_reference, reference_path


def reference_paths(args) -> list[str]:
    prompt = {}
    if getattr(args, "reference_image", None) is not None:
        prompt["reference_image"] = args.reference_image
    if getattr(args, "reference_images", None) is not None:
        prompt["reference_images"] = args.reference_images
    return preview_references(prompt)


def validate_task(args) -> None:
    task = str(getattr(args, "qwen_image_2_1_task", None) or "t2i").lower()
    if task not in {"t2i", "edit"}:
        raise SystemExit("Qwen Image 2.1 standalone task must be t2i or edit.")
    references = reference_paths(args)
    if task == "edit" and not references:
        raise SystemExit("Qwen Image 2.1 edit requires --reference_image or --reference_images.")
    if task == "t2i" and references:
        raise SystemExit("Qwen Image 2.1 t2i does not accept reference images.")
    if len(references) > 4:
        raise SystemExit("Qwen Image 2.1 edit accepts 1..4 ordered reference images.")


def _freeze_references(args):
    height, width = (int(value) for value in args.image_size)
    images = []
    try:
        for path in reference_paths(args):
            source = read_reference(reference_path(path))
            try:
                size = reference_size_for_bucket(source.size, (width, height))
                images.append(source.resize(size, Image.Resampling.BICUBIC))
            finally:
                source.close()
    except BaseException:
        for image in images:
            image.close()
        raise
    return tuple(images)


def _prepare_edit_text(args, images, device, dtype):
    def encode(encoder):
        tokenizer = QwenImage21EditTokenizeStrategy(args.text_encoder)
        conditions = []
        for text in (args.prompt or "", getattr(args, "negative_prompt", "") or ""):
            values = encode_edit_prompt(encoder, tokenizer, text, images)
            conditions.append(tuple(
                value.unsqueeze(0).to(device=device, dtype=dtype if index == 0 else torch.bool)
                for index, value in enumerate(values)
            ))
        return tuple(conditions)

    return encode_with_text_encoder(
        args, device, encode, loader=load_qwen_image_2_1_text_encoder
    )


def _prepare_reference_latents(args, images, device):
    vae = load_qwen_image_2_1_vae(args.vae, dtype=torch.bfloat16, device=device)
    try:
        latents = []
        with torch.inference_mode():
            for image in images:
                pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0)
                pixels = pixels.to(device=device, dtype=torch.bfloat16) / 127.5 - 1
                latent = encode_qwen_image_2_1_latents(vae, pixels)
                if (latent.ndim != 4 or latent.shape[:2] != (1, 64)
                        or any(value <= 0 or value % 2 for value in latent.shape[-2:])):
                    raise ValueError("Qwen Edit reference latent must be [1,64,H,W] with even H/W")
                latents.append(latent.detach())
        return tuple(latents)
    finally:
        del vae
        gc.collect()
        clean_memory_on_device(device)


def _validate_slots(positive, negative, references):
    expected = sum(value.shape[-2] * value.shape[-1] // 4 for value in references)
    for values in (positive, negative):
        if len(values) != 3:
            raise ValueError("Qwen Edit requires hidden states, attention mask and image slot mask")
        hidden, mask, slots = values
        if (hidden.ndim != 3 or hidden.shape[0] != 1 or mask.shape != hidden.shape[:2]
                or slots.shape != mask.shape or mask.dtype != torch.bool or slots.dtype != torch.bool):
            raise ValueError("Qwen Edit condition mask shape or dtype mismatch")
        if int(slots.sum()) != expected or (slots & ~mask).any():
            raise ValueError("Qwen Edit reference VAE and Qwen3-VL image slots disagree")


def prepare_qwen_image_2_1_conditions(args, device, runtime_dtype):
    """TE encode/free -> reference VAE encode/free, retaining one shared resize."""
    validate_task(args)
    if str(getattr(args, "qwen_image_2_1_task", None) or "t2i").lower() == "t2i":
        from library.models.qwen_image_2_1.inference_runner import prepare_qwen_image_2_1_text

        positive, negative = prepare_qwen_image_2_1_text(args, device, runtime_dtype)
        return positive, negative, None
    images = _freeze_references(args)
    try:
        positive, negative = _prepare_edit_text(args, images, device, runtime_dtype)
        references = _prepare_reference_latents(args, images, device)
        _validate_slots(positive, negative, references)
        return positive, negative, references
    finally:
        for image in images:
            image.close()
