#!/usr/bin/env python3
"""Cache Qwen Image 2.1 pre-norm Qwen3-VL text embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

from library.preprocess.adaptive_batch import batch_size_arg
from library.runtime.device import str_to_dtype
from library.models.qwen_image_2_1.strategy import (
    QwenImage21TextCache,
    QwenImage21TextEncodingStrategy,
    QwenImage21TokenizeStrategy,
)
from library.models.qwen_image_2_1.text_encoder_runtime import text_encoder_for_cache
from library.models.qwen_image_2_1.cache_policy import QWEN_TEXT_ENCODER_CACHE_POLICIES, resolve_cache_policy
from scripts.krea2.preprocess_te_cache import _cache_items, _iter_caption_sources, _caption_variants


def _pending_items(items, *, data_dir, cache_dir, strategy, variants, overwrite):
    """Avoid loading the encoder when all T2I outputs are already valid."""
    if overwrite:
        return items
    pending = []
    for image_path, source in items:
        captions = _caption_variants(source, variants, 0.0)
        variant_layout = variants > 0 or len(captions) > 1
        cache_path = strategy.get_outputs_npz_path(
            str(image_path), cache_dir=str(cache_dir), image_dir=str(data_dir),
        )
        if not strategy.is_disk_cached_outputs_expected(
            cache_path,
            expected_num_variants=len(captions) if variant_layout else 0,
            expected_caption_shuffle_variants=variants,
            expected_caption_tag_dropout_rate=0.0,
            expected_multi_source=bool(variant_layout and source.captions is not None),
        ):
            pending.append((image_path, source))
    return pending


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
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--offload", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--cache_policy", choices=QWEN_TEXT_ENCODER_CACHE_POLICIES, default="auto")
    args = parser.parse_args()
    if args.caption_tag_dropout_rate != 0:
        parser.error("Qwen Image 2.1 does not support caption tag dropout")
    resolve_cache_policy(args.cache_policy, device=args.device, offload=args.offload)

    data_dir = Path(args.dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    items = list(_iter_caption_sources(
        data_dir,
        recursive=args.recursive,
        path_pattern=args.path_pattern,
        min_pixels=max(0, args.min_pixels),
        prefer_json_caption=args.prefer_json_caption,
        caption_source_mode=args.caption_source_mode,
        caption_extension=args.caption_extension,
    ))
    strategy = QwenImage21TextCache(True, args.batch_size)
    pending = _pending_items(items, data_dir=data_dir, cache_dir=cache_dir,
                             strategy=strategy, variants=args.caption_shuffle_variants,
                             overwrite=args.overwrite)
    reused = len(items) - len(pending)
    if not pending:
        print(f"Qwen3-VL cache policy: requested={args.cache_policy}; valid text caches reused, encoder not loaded")
        print(f"Qwen Image 2.1 TE cache: 0 written, {reused} reused")
        return
    with text_encoder_for_cache(
        args.qwen3, dtype=str_to_dtype(args.dtype), device=args.device, offload=args.offload,
        cache_policy=args.cache_policy,
    ) as text_encoder:
        written, skipped = _cache_items(
            pending,
            data_dir=data_dir,
            cache_dir=cache_dir,
            batch_size=args.batch_size,
            caption_shuffle_variants=args.caption_shuffle_variants,
            caption_tag_dropout_rate=0.0,
            overwrite=args.overwrite,
            caching_strategy=strategy,
            tokenize_strategy=QwenImage21TokenizeStrategy(args.qwen3),
            encoding_strategy=QwenImage21TextEncodingStrategy(),
            text_encoder=text_encoder,
        )
    print(f"Qwen Image 2.1 TE cache: {written} written, {skipped + reused} reused")


if __name__ == "__main__":
    main()
