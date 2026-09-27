#!/usr/bin/env python3
"""Cache Qwen Image 2.1 pre-norm Qwen3-VL text embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from library.preprocess.adaptive_batch import batch_size_arg
from library.runtime.device import str_to_dtype
from library.models.qwen_image_2_1.strategy import (
    QwenImage21TextCache,
    QwenImage21TextEncodingStrategy,
    QwenImage21TokenizeStrategy,
)
from library.models.qwen_image_2_1.weights import load_qwen_image_2_1_text_encoder
from scripts.krea2.preprocess_te_cache import _cache_items, _iter_caption_sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--qwen3", required=True)
    parser.add_argument("--batch_size", type=batch_size_arg, default="auto")
    parser.add_argument("--dtype", choices=["bfloat16"], default="bfloat16")
    parser.add_argument("--path_pattern", default="*")
    parser.add_argument("--min_pixels", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--caption_shuffle_variants", type=int, default=0)
    parser.add_argument("--caption_tag_dropout_rate", type=float, default=0.0)
    parser.add_argument("--prefer_json_caption", action="store_true")
    parser.add_argument("--caption_source_mode", choices=["auto", "txt", "json", "captions_json"])
    parser.add_argument("--caption_extension", default=".txt")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()
    if args.caption_tag_dropout_rate != 0:
        parser.error("Qwen Image 2.1 does not support caption tag dropout")

    data_dir = Path(args.dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder = load_qwen_image_2_1_text_encoder(
        args.qwen3, dtype=str_to_dtype(args.dtype), device=device
    )
    items = list(_iter_caption_sources(
        data_dir,
        recursive=args.recursive,
        path_pattern=args.path_pattern,
        min_pixels=max(0, args.min_pixels),
        prefer_json_caption=args.prefer_json_caption,
        caption_source_mode=args.caption_source_mode,
        caption_extension=args.caption_extension,
    ))
    written, skipped = _cache_items(
        items,
        data_dir=data_dir,
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        caption_shuffle_variants=args.caption_shuffle_variants,
        caption_tag_dropout_rate=0.0,
        overwrite=args.overwrite,
        caching_strategy=QwenImage21TextCache(True, args.batch_size),
        tokenize_strategy=QwenImage21TokenizeStrategy(args.qwen3),
        encoding_strategy=QwenImage21TextEncodingStrategy(),
        text_encoder=text_encoder,
    )
    print(f"Qwen Image 2.1 TE cache: {written} written, {skipped} reused")
    text_encoder.to("cpu")


if __name__ == "__main__":
    main()
