#!/usr/bin/env python3
"""Cache Krea-2 (Qwen3-VL) text encoder outputs for a captioned dataset.

Reads captions.json / same-stem JSON / text sidecars with the shared caption
resolver, tokenizes with Krea2TokenizeStrategy (Qwen3-VL ChatML), and writes
``{stem}_krea2_te.safetensors`` per image. Optional variants follow Anima's
contract: pristine v0, shuffled/tag-dropped v1+, and uniform selection for
multiple captions supplied by captions.json. Krea still has no T5 adapter or
crossattn sidecar; hiddens + mask are the complete DiT conditioning.

Run::

    python -m scripts.krea2.preprocess_te_cache \
        --dir image_dataset \
        --cache_dir post_image_dataset/lora \
        --qwen3 models/text_encoders/qwen3_vl_4b_instruct.safetensors
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from library.log import setup_logging
from library.preprocess._dataset import walk_images
from library.preprocess.captions import CaptionSource, read_caption_source
from library.preprocess.text import generate_caption_variants

setup_logging()
logger = logging.getLogger(__name__)


@dataclass
class _CacheInfo:
    caption: str
    caption_variants: list[str]
    cache_caption_variants: bool
    caption_multi_source: bool
    caption_shuffle_variants: int
    caption_tag_dropout_rate: float
    caption_dropout_rate: float
    absolute_path: str
    text_encoder_outputs_npz: str
    text_encoder_outputs: object = None


def _iter_caption_sources(
    data_dir: Path,
    *,
    recursive: bool,
    path_pattern: str | None,
    min_pixels: int,
    prefer_json_caption: bool,
    caption_source_mode: str | None,
    caption_extension: str,
):
    """Yield every selected image and its resolved caption source."""
    pattern = path_pattern if path_pattern and path_pattern != "*" else None
    for image_path in walk_images(data_dir, recursive=recursive, pattern=pattern):
        if min_pixels > 0:
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
            except Exception as exc:
                logger.warning("could not read %s: %s", image_path.name, exc)
                continue
            if width * height < min_pixels:
                continue
        yield (
            image_path,
            read_caption_source(
                image_path,
                prefer_json_caption=prefer_json_caption,
                caption_source_mode=caption_source_mode,
                caption_extension=caption_extension,
                captions_root=data_dir,
                warn=logger.warning,
            ),
        )


def _caption_variants(
    source: CaptionSource,
    count: int,
    tag_dropout_rate: float,
) -> list[str]:
    if source.captions is not None:
        return source.caption_texts() or [""]
    if count <= 0:
        return [source.render()]
    return generate_caption_variants(source, count, tag_dropout_rate)


def _cache_items(
    items: list[tuple[Path, CaptionSource]],
    *,
    data_dir: Path,
    cache_dir: Path,
    batch_size: int,
    caption_shuffle_variants: int,
    caption_tag_dropout_rate: float,
    overwrite: bool,
    caching_strategy,
    tokenize_strategy,
    encoding_strategy,
    text_encoder,
) -> tuple[int, int]:
    written = 0
    skipped = 0
    for start in range(0, len(items), batch_size):
        infos: list[_CacheInfo] = []
        for image_path, source in items[start : start + batch_size]:
            variants = _caption_variants(
                source,
                caption_shuffle_variants,
                caption_tag_dropout_rate,
            )
            use_variant_layout = caption_shuffle_variants > 0 or len(variants) > 1
            cache_path = caching_strategy.get_outputs_npz_path(
                str(image_path),
                cache_dir=str(cache_dir),
                image_dir=str(data_dir),
            )
            if not overwrite and caching_strategy.is_disk_cached_outputs_expected(
                cache_path,
                expected_num_variants=len(variants) if use_variant_layout else 0,
                expected_caption_shuffle_variants=caption_shuffle_variants,
                expected_caption_tag_dropout_rate=caption_tag_dropout_rate,
                expected_multi_source=bool(
                    use_variant_layout and source.captions is not None
                ),
            ):
                skipped += 1
                continue
            infos.append(
                _CacheInfo(
                    caption=variants[0],
                    caption_variants=variants,
                    cache_caption_variants=use_variant_layout,
                    caption_multi_source=bool(source.captions is not None),
                    caption_shuffle_variants=caption_shuffle_variants,
                    caption_tag_dropout_rate=caption_tag_dropout_rate,
                    # Whole-caption dropout is supplied by the active subset at
                    # load time; zero remains an old-layout compatibility slot.
                    caption_dropout_rate=0.0,
                    absolute_path=str(image_path),
                    text_encoder_outputs_npz=cache_path,
                )
            )
        if not infos:
            continue
        caching_strategy.cache_batch_outputs(
            tokenize_strategy,
            [text_encoder],
            encoding_strategy,
            infos,
        )
        written += len(infos)
        logger.info("  cached %d/%d", written + skipped, len(items))
    return written, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Source image dir")
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
    parser.add_argument("--path_pattern", type=str, default="*")
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
    parser.add_argument("--caption_extension", type=str, default=".txt")
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch_size must be >= 1")
    if args.caption_shuffle_variants < 0:
        parser.error("--caption_shuffle_variants must be >= 0")
    if not 0.0 <= args.caption_tag_dropout_rate <= 1.0:
        parser.error("--caption_tag_dropout_rate must be between 0 and 1")

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
    text_encoder, _ = load_krea2_text_encoder(
        args.qwen3, dtype=dtype, device=str(device)
    )
    text_encoder.eval()

    tokenize_strategy = Krea2TokenizeStrategy()
    encoding_strategy = Krea2TextEncodingStrategy()
    caching_strategy = Krea2TextEncoderOutputsCachingStrategy(
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
    logger.info("Found %d images to cache", len(items))
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
        "Krea-2 TE caching complete: %d written, %d reused, %d total",
        written,
        skipped,
        len(items),
    )
    text_encoder.to("cpu")
    del text_encoder


if __name__ == "__main__":
    main()
