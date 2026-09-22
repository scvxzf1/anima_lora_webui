"""Prepare real Krea text/latent fixtures separately from DiT training."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from safetensors.torch import save_file
import torch

from library.models.krea2_raw.strategy import (
    Krea2TextEncodingStrategy, Krea2TokenizeStrategy, load_krea2_text_encoder,
)
from library.models.qwen_vae import load_vae
from library.training.auto_block_swap.process import write_result


def encode(args, source, caption):
    device = torch.device("cuda:0")
    encoder, tokenizer = load_krea2_text_encoder(
        str(args.text_encoder), dtype=torch.bfloat16, device=device,
    )
    tokenize = Krea2TokenizeStrategy()
    with torch.no_grad():
        hidden, mask = Krea2TextEncodingStrategy().encode_tokens(
            tokenize, [encoder], tokenize.tokenize([caption]),
        )
    hidden, mask = hidden.float().cpu().contiguous(), mask.bool().cpu().contiguous()
    del encoder, tokenizer, tokenize
    gc.collect()
    torch.cuda.empty_cache()
    vae = load_vae(str(args.vae), device=device, dtype=torch.float32, eval=True)
    with Image.open(source) as image:
        resized = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"),
                               (args.resolution, args.resolution), method=Image.Resampling.LANCZOS)
        array = np.array(resized, dtype=np.float32) / 255.0
    pixels = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        latent = vae.encode_pixels_to_latents(pixels).float().cpu().unsqueeze(2).contiguous()
    del vae, pixels
    gc.collect()
    torch.cuda.empty_cache()
    noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(20260921))
    if not all(torch.isfinite(t).all() for t in (hidden, latent, noise)):
        raise ValueError("Nonfinite Krea input cache")
    return {"latents": latent, "noise": noise, "hidden": hidden, "mask": mask}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()
    if args.resolution < 32 or args.resolution % 16:
        parser.error("resolution must be divisible by 16 and >=32")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    if torch.cuda.get_device_capability()[0] < 8:
        raise ValueError("Prepare the shared BF16 text cache on Ampere+")
    torch.set_num_threads(4)
    torch.manual_seed(20260921)
    source = next((p for p in sorted(args.dataset.rglob("*"))
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                   and p.with_suffix(".txt").is_file()), None)
    if source is None:
        raise ValueError("Dataset has no paired image/caption")
    caption_path = source.with_suffix(".txt")
    caption = caption_path.read_text(encoding="utf-8").strip()
    if not caption:
        raise ValueError("Caption cannot be empty")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"schema": "adaptive_krea_inputs_v1", "image": str(source.resolve()),
                "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "caption_sha256": hashlib.sha256(caption_path.read_bytes()).hexdigest(),
                "resolution": str(args.resolution), "text_dtype": "bf16", "vae_dtype": "fp32"}
    write_result(args.output.with_suffix(".json"), {**metadata, "status": "preparing"})
    values = encode(args, source, caption)
    save_file(values, str(args.output), metadata)
    report = {**metadata, "status": "ok", "valid_tokens": int(values["mask"].sum()),
              "cache_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
    write_result(args.output.with_suffix(".json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
