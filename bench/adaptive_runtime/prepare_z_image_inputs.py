"""Create immutable real-image/text inputs without loading the DiT."""

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

from library.models.z_image.family import prepare_prompt_embeds
from library.models.z_image.latent import encode_z_image_latents
from library.models.z_image.strategy import ZImageTextEncodingStrategy, ZImageTokenizeStrategy
from library.models.z_image.weights import load_z_image_text_encoder, load_z_image_vae
from library.training.auto_block_swap.process import write_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()
    if args.resolution < 32 or args.resolution % 16:
        parser.error("resolution must be divisible by 16 and >=32")
    source = args.image
    if source is None and args.dataset:
        source = next((p for p in sorted(args.dataset.rglob("*"))
                       if p.suffix.lower() in {".jpg", ".png", ".jpeg"}
                       and p.with_suffix(".txt").is_file()), None)
    if source is None:
        parser.error("Provide --image or a dataset with paired captions")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    caption_path = source.with_suffix(".txt")
    caption = caption_path.read_text(encoding="utf-8").strip()
    if not caption:
        raise ValueError("Caption cannot be empty")
    device = torch.device("cuda:0")
    if torch.cuda.get_device_capability()[0] < 8:
        raise ValueError("Build the shared BF16 text reference on Ampere+ first")
    torch.set_num_threads(4)
    tokenize = ZImageTokenizeStrategy(str(args.weights))
    encoder = load_z_image_text_encoder(str(args.weights), dtype=torch.bfloat16, device=device)
    hidden, mask = ZImageTextEncodingStrategy().encode_tokens(
        tokenize, [encoder], tokenize.tokenize(caption),
    )
    prompt = prepare_prompt_embeds(hidden, mask)[0].float().cpu().contiguous()
    del hidden, mask, encoder, tokenize
    gc.collect()
    torch.cuda.empty_cache()
    vae = load_z_image_vae(str(args.weights), dtype=torch.float32, device=device)
    with Image.open(source) as image:
        resized = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"),
                               (args.resolution, args.resolution), method=Image.Resampling.LANCZOS)
        array = np.array(resized, dtype=np.float32) / 127.5 - 1
    pixels = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        latent = encode_z_image_latents(vae, pixels).float().cpu().unsqueeze(2).contiguous()
    del vae, pixels
    gc.collect()
    torch.cuda.empty_cache()
    noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(20260921))
    if not all(torch.isfinite(t).all() for t in (latent, prompt, noise)):
        raise ValueError("Nonfinite real input cache")
    metadata = {"schema": "adaptive_z_image_inputs_v1", "image": str(source.resolve()),
                "image_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "caption_sha256": hashlib.sha256(caption_path.read_bytes()).hexdigest(),
                "resolution": str(args.resolution), "text_dtype": "bf16", "vae_dtype": "fp32"}
    save_file({"latents": latent, "prompt": prompt, "noise": noise}, str(args.output), metadata)
    report = {**metadata, "prompt_tokens": len(prompt), "latent_shape": list(latent.shape),
              "cache_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
    write_result(args.output.with_suffix(".json"), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
