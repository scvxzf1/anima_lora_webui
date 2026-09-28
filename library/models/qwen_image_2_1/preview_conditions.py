"""Startup-only preview snapshots and reference conditions, before DiT loading."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import torch

from library import train_util
from library.datasets.qwen_image_edit import reference_size_for_bucket
from library.models.qwen_image_2_1.latent import encode_qwen_image_2_1_latents
from library.models.qwen_image_2_1.strategy import QwenImage21EditTokenizeStrategy, encode_edit_prompt
from library.runtime.device import clean_memory_on_device
from library.training.preview_spec import publish_reference, preview_references, read_reference, reference_path, validate_preview_prompt
from library.training.sample_preview_common import failed_on_any_process


def condition_key(prompt, *, negative=False):
    return f"qwen-preview:{prompt['enum']}:{'negative' if negative else 'positive'}"


def reference_key(prompt):
    return f"qwen-preview:{prompt['enum']}:reference"


def snapshot_prompts(args):
    prompts = train_util.load_prompts(args.sample_prompts)
    for prompt in prompts:
        validate_preview_prompt(prompt, "qwen_image_2_1", getattr(args, "sample_sampler", "euler"))
        references = preview_references(prompt)
        if not references:
            continue
        from PIL import Image

        root = Path(args.output_dir) / "sample" / "references"
        root.mkdir(parents=True, exist_ok=True)
        frozen = []
        for reference in references:
            image = read_reference(reference_path(reference))
            size = reference_size_for_bucket(image.size, (int(prompt.get("width", 512)), int(prompt.get("height", 512))))
            image = image.resize(size, Image.Resampling.BICUBIC)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            content = buffer.getvalue()
            path = root / (hashlib.sha256(content).hexdigest() + ".png")
            publish_reference(path, content)
            frozen.append(str(path.resolve()))
        if len(frozen) == 1:
            prompt.pop("reference_images", None)
            prompt["reference_image"] = frozen[0]
        else:
            prompt["reference_images"] = frozen
    return prompts


def cache_preview_text(trainer, args, model, tokenizer, encoder):
    prompts = snapshot_prompts(args)
    edit_tokenizer = None
    outputs = {}
    for prompt in prompts:
        references = preview_references(prompt)
        if references and edit_tokenizer is None:
            edit_tokenizer = tokenizer if isinstance(tokenizer, QwenImage21EditTokenizeStrategy) else QwenImage21EditTokenizeStrategy(args.qwen3)
        use_cfg = float(prompt.get("guidance_scale", prompt.get("scale", 1.0))) > 1 and prompt.get("negative_prompt") is not None
        for negative in ((False, True) if use_cfg else (False,)):
            text = prompt.get("negative_prompt" if negative else "prompt", "") or ""
            if references:
                images = [read_reference(reference) for reference in references]
                image_input = images[0] if len(images) == 1 else images
                encoded = tuple(value.unsqueeze(0) for value in encode_edit_prompt(model, edit_tokenizer, text, image_input))
            else:
                encoded = encoder.encode_tokens(tokenizer, [model], tokenizer.tokenize(text))
            outputs[condition_key(prompt, negative=negative)] = tuple(value.detach().cpu().contiguous() for value in encoded)
    trainer.sample_prompts_snapshot = prompts
    trainer.sample_prompts_te_outputs = outputs


def cache_preview_references(trainer, args, accelerator, vae):
    if not trainer.sample_prompts_snapshot:
        return
    prompts = [p for p in trainer.sample_prompts_snapshot if preview_references(p)]
    if not prompts:
        return
    if vae is None:
        raise ValueError("Qwen edit preview requires a VAE for reference caching")
    original_device = vae.device
    error = None
    try:
        vae.to(accelerator.device)
        with torch.no_grad():
            for prompt in prompts:
                latents = []
                for reference in preview_references(prompt):
                    image = read_reference(reference)
                    pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0)
                    pixels = pixels.to(device=accelerator.device, dtype=vae.dtype) / 127.5 - 1
                    latents.append(encode_qwen_image_2_1_latents(vae, pixels).cpu())
                expected_slots = sum(latent.shape[-2] * latent.shape[-1] // 4 for latent in latents)
                positive_slots = trainer.sample_prompts_te_outputs[condition_key(prompt)][2]
                negative = trainer.sample_prompts_te_outputs.get(condition_key(prompt, negative=True))
                if int(positive_slots.sum()) != expected_slots or (
                    negative is not None and int(negative[2].sum()) != expected_slots
                ):
                    raise ValueError("Qwen preview reference VAE and Qwen3-VL image slots disagree")
                trainer.sample_prompts_te_outputs[reference_key(prompt)] = tuple(latents)
    except Exception as exc:
        error = exc
    finally:
        vae.to(original_device)
        clean_memory_on_device(accelerator.device)
    if failed_on_any_process(accelerator, error is not None, getattr(accelerator, "num_processes", 1)):
        raise RuntimeError("Qwen preview reference caching failed on at least one process") from error
