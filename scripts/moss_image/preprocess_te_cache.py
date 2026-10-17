#!/usr/bin/env python3
"""Cache frozen MOSS Qwen3.5 text states with explicit NF4/BF16 identity."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch

from library.log import setup_logging
from library.preprocess.adaptive_batch import batch_size_arg
from scripts.krea2.preprocess_te_cache import _cache_items, _iter_caption_sources

logger = logging.getLogger(__name__)


def run_task(row, extra, shuffle_variants=None, tag_dropout_rate=None):
    """Thin task adapter; preserve shared dataset selection and cache policy."""
    from scripts.tasks import preprocess as tasks
    from library.models.moss_image.cache_policy import validate_quantization

    shuffle = shuffle_variants or os.environ.get("CAPTION_SHUFFLE_VARIANTS", "0")
    dropout = tag_dropout_rate or os.environ.get("CAPTION_TAG_DROPOUT_RATE", "0")
    if int(shuffle) or float(dropout):
        raise ValueError("MOSS caption variants and tag dropout are unsupported")
    quantization = validate_quantization(
        tasks._path_overrides_value().get("moss_te_quantization", "nf4")
    )
    mp_args, extra = tasks._resolve_lowres_filter(extra)
    prefer_json_env = os.environ.get("CAPTION_PREFER_JSON")
    prefer_json = tasks._truthy(
        prefer_json_env if prefer_json_env is not None else row.get("prefer_json_caption")
    )
    source_mode = os.environ.get("CAPTION_SOURCE_MODE") or row.get("caption_source_mode")
    tasks.run([
        tasks.PY, "-m", "scripts.moss_image.preprocess_te_cache",
        "--dir", str(row.get("source_image_dir") or tasks._path("source_image_dir", "image_dataset")),
        "--cache_dir", tasks._text_cache_dir_for_row(row),
        "--qwen3", tasks._path("qwen3", "models/text_encoders/MOSS-Image/text_encoder"),
        "--batch_size", "1", "--dtype", "bfloat16",
        "--moss_te_quantization", quantization,
        "--caption_shuffle_variants", shuffle,
        "--caption_tag_dropout_rate", dropout,
        *tasks._caption_source_args(source_mode, prefer_json),
        *tasks._caption_extension_args_for_row(row),
        *(["--prefer_json_caption"] if prefer_json else []),
        *tasks._recursive_args(row), *tasks._path_pattern_args(row),
        *mp_args, *tasks._reuse_overwrite_args(row, kind="te"), *extra,
    ])


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--qwen3", required=True)
    parser.add_argument("--batch_size", type=batch_size_arg, default=1)
    parser.add_argument("--dtype", choices=["bfloat16"], default="bfloat16")
    parser.add_argument("--moss_te_quantization", choices=["nf4", "bf16"], default="nf4")
    parser.add_argument("--caption_shuffle_variants", type=int, default=0)
    parser.add_argument("--caption_tag_dropout_rate", type=float, default=0.0)
    parser.add_argument("--path_pattern", default="*")
    parser.add_argument("--min_pixels", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prefer_json_caption", action="store_true")
    parser.add_argument("--caption_source_mode", choices=["auto", "txt", "json", "captions_json"])
    parser.add_argument("--caption_extension", default=".txt")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()
    if args.batch_size not in (1, "auto"):
        parser.error("MOSS text caching requires --batch_size 1 (auto resolves to 1)")
    if args.caption_shuffle_variants or args.caption_tag_dropout_rate:
        parser.error("MOSS caption variants and tag dropout are unsupported")

    from library.models.moss_image.strategy import (
        MossImageTextCache, MossImageTextEncodingStrategy, MossImageTokenizeStrategy,
    )
    from library.models.moss_image.text_encoder_runtime import load_moss_text_encoder

    data_dir, cache_dir = Path(args.dir), Path(args.cache_dir)
    items = list(_iter_caption_sources(
        data_dir, recursive=args.recursive, path_pattern=args.path_pattern,
        min_pixels=max(0, args.min_pixels), prefer_json_caption=args.prefer_json_caption,
        caption_source_mode=args.caption_source_mode, caption_extension=args.caption_extension,
    ))
    if any(len(source.caption_texts()) > 1 for _, source in items):
        parser.error("MOSS multiple-caption variants are unsupported")
    caching = MossImageTextCache(True, 1, False, quantization=args.moss_te_quantization)
    tokenizer = MossImageTokenizeStrategy(args.qwen3, max_length=4096)
    encoder = load_moss_text_encoder(
        args.qwen3, dtype=torch.bfloat16, device="cpu", quantization=args.moss_te_quantization,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder.to(device)
    try:
        written, skipped = _cache_items(
            items, data_dir=data_dir, cache_dir=cache_dir, batch_size=1,
            caption_shuffle_variants=0, caption_tag_dropout_rate=0,
            overwrite=args.overwrite, caching_strategy=caching,
            tokenize_strategy=tokenizer, encoding_strategy=MossImageTextEncodingStrategy(),
            text_encoder=encoder,
        )
        logger.info("MOSS TE caching complete: %d written, %d reused", written, skipped)
    finally:
        encoder.to("cpu")


if __name__ == "__main__":
    main()
