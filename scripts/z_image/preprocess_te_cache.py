#!/usr/bin/env python3
"""Cache official Z-Image Qwen3 penultimate-layer text embeddings."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from library.log import setup_logging
from scripts.krea2.preprocess_te_cache import _cache_items, _iter_caption_sources

setup_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--qwen3", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--path_pattern", default="*")
    parser.add_argument("--min_pixels", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--caption_shuffle_variants", type=int, default=0)
    parser.add_argument("--caption_tag_dropout_rate", type=float, default=0.0)
    parser.add_argument("--prefer_json_caption", action="store_true")
    parser.add_argument(
        "--caption_source_mode",
        choices=["auto", "txt", "json", "captions_json"],
        default=None,
    )
    parser.add_argument("--caption_extension", default=".txt")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    if args.caption_shuffle_variants < 0:
        parser.error("--caption_shuffle_variants must be >= 0")
    if not 0.0 <= args.caption_tag_dropout_rate <= 1.0:
        parser.error("--caption_tag_dropout_rate must be between 0 and 1")

    from library.models.z_image.strategy import (
        ZImageTextEncoderOutputsCachingStrategy,
        ZImageTextEncodingStrategy,
        ZImageTokenizeStrategy,
    )
    from library.models.z_image.weights import load_z_image_text_encoder
    from library.runtime.device import str_to_dtype

    data_dir = Path(args.dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = str_to_dtype(args.dtype)
    text_encoder = load_z_image_text_encoder(args.qwen3, dtype=dtype, device=device)
    tokenize_strategy = ZImageTokenizeStrategy(args.qwen3)
    encoding_strategy = ZImageTextEncodingStrategy()
    caching_strategy = ZImageTextEncoderOutputsCachingStrategy(
        cache_to_disk=True,
        batch_size=args.batch_size,
        skip_disk_cache_validity_check=False,
    )
    items = list(
        _iter_caption_sources(
            data_dir,
            recursive=args.recursive,
            path_pattern=args.path_pattern,
            min_pixels=max(0, args.min_pixels),
            prefer_json_caption=args.prefer_json_caption,
            caption_source_mode=args.caption_source_mode,
            caption_extension=args.caption_extension,
        )
    )
    written, skipped = _cache_items(
        items,
        data_dir=data_dir,
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        caption_shuffle_variants=args.caption_shuffle_variants,
        caption_tag_dropout_rate=args.caption_tag_dropout_rate,
        overwrite=args.overwrite,
        caching_strategy=caching_strategy,
        tokenize_strategy=tokenize_strategy,
        encoding_strategy=encoding_strategy,
        text_encoder=text_encoder,
    )
    logger.info(
        "Z-Image TE caching complete: %d written, %d reused, %d total",
        written,
        skipped,
        len(items),
    )
    text_encoder.to("cpu")


if __name__ == "__main__":
    main()
