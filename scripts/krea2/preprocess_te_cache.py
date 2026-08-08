#!/usr/bin/env python3
"""Cache Krea-2 (Qwen3-VL) text encoder outputs for a captioned dataset.

Stage 6 preprocess entry for the Krea-2-Raw model family. Reads caption
sidecars (same logic as the anima cache_text_embeddings.py — captions.json /
.json / .txt), tokenizes with Krea2TokenizeStrategy (Qwen3-VL ChatML), encodes
through Qwen3-VL-4B, and writes ``{stem}_krea2_te.safetensors`` per image
(suffix isolation — does not pollute anima's ``_anima_te`` cache).

Single-variant only (Krea-2 first day): no shuffle variants, no LLM adapter,
no crossattn_emb, no uncond sidecar — those are anima-only. The DiT-side
txtfusion/txtmlp projectors live inside the DiT and are not part of the text
encoder, so the cached hiddens (B, L, 12, 2560) + mask (B, L) are the complete
text conditioning forward_for_loss needs.

Run::

    python -m scripts.krea2.preprocess_te_cache \
        --dir post_image_dataset/resized \
        --cache_dir post_image_dataset/lora \
        --qwen3 models/text_encoders/qwen3_vl_4b_instruct.safetensors
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from library.log import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def _iter_captioned_images(data_dir: Path, recursive: bool):
    """Yield (image_path, caption_str) for every image with a caption sidecar.

    Mirrors anima's caption source resolution: captions.json -> same-stem .json
    -> same-stem .txt. Krea-2 first day: .txt sidecars (the resize step writes
    them per-image from captions.json), so we just read .txt.
    """
    import json

    # captions.json master (filename -> [caption, ...])
    json_master = data_dir / "captions.json"
    master: dict = {}
    if json_master.exists():
        master = json.loads(json_master.read_text(encoding="utf-8"))

    globs = [data_dir.rglob("*")] if recursive else [data_dir.iterdir()]
    seen = set()
    for it in globs:
        for p in it:
            if not p.is_file():
                continue
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
                continue
            stem_key = p.name
            cap = None
            if stem_key in master:
                v = master[stem_key]
                cap = v[0] if isinstance(v, list) and v else str(v)
            else:
                txt = p.with_suffix(".txt")
                if txt.exists():
                    cap = txt.read_text(encoding="utf-8").strip()
            if cap:
                seen.add(p)
                yield p, cap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Resized image dir")
    parser.add_argument("--cache_dir", type=str, required=True, help="Output cache dir")
    parser.add_argument(
        "--qwen3", type=str, required=True, help="Qwen3-VL-4B safetensors path"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--recursive", action="store_true", default=True)
    args = parser.parse_args()

    from library.runtime.device import str_to_dtype
    from library.models.krea2_raw.strategy import (
        Krea2TextEncoderOutputsCachingStrategy,
        Krea2TextEncodingStrategy,
        Krea2TokenizeStrategy,
        load_krea2_text_encoder,
    )

    data_dir = Path(args.dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = str_to_dtype(args.dtype)

    logger.info(f"Loading Krea-2 Qwen3-VL text encoder from {args.qwen3} ...")
    text_encoder, _ = load_krea2_text_encoder(args.qwen3, dtype=dtype, device=str(device))
    text_encoder.eval()

    tokenize_strategy = Krea2TokenizeStrategy()
    encoding_strategy = Krea2TextEncodingStrategy()
    caching_strategy = Krea2TextEncoderOutputsCachingStrategy(
        cache_to_disk=True,
        batch_size=args.batch_size,
        skip_disk_cache_validity_check=False,
    )

    items = list(_iter_captioned_images(data_dir, args.recursive))
    logger.info(f"Found {len(items)} captioned images to cache")

    # Build a minimal info-like object per anima's cache_batch_outputs contract.
    from dataclasses import dataclass

    @dataclass
    class _Info:
        caption: str
        caption_dropout_rate: float
        absolute_path: str
        text_encoder_outputs_npz: str
        text_encoder_outputs: object = None

    n_written = 0
    for i in range(0, len(items), args.batch_size):
        chunk = items[i : i + args.batch_size]
        infos = []
        for img_path, cap in chunk:
            npz = caching_strategy.get_outputs_npz_path(
                str(img_path), cache_dir=str(cache_dir), image_dir=str(data_dir)
            )
            if Path(npz).exists() and caching_strategy.is_disk_cached_outputs_expected(
                npz
            ):
                n_written += 1
                continue
            infos.append(
                _Info(
                    caption=cap,
                    caption_dropout_rate=0.0,
                    absolute_path=str(img_path),
                    text_encoder_outputs_npz=npz,
                )
            )
        if not infos:
            continue
        caching_strategy.cache_batch_outputs(
            tokenize_strategy, [text_encoder], encoding_strategy, infos
        )
        n_written += len(infos)
        logger.info(f"  cached {n_written}/{len(items)}")

    logger.info(f"Krea-2 TE caching complete: {n_written}/{len(items)} cached")
    text_encoder.to("cpu")
    del text_encoder


if __name__ == "__main__":
    main()
