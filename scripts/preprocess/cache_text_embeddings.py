#!/usr/bin/env python3
"""Cache text encoder (Qwen3) outputs for all captioned images in a dataset directory.

Reads .txt caption sidecars, tokenizes with Qwen3 + T5, encodes through the
Qwen3 text encoder, and optionally runs the LLM adapter to produce crossattn_emb.
Saves results as *_anima_te.safetensors alongside each image (or under
``--cache_dir``).

Supports caption shuffle variants: with --caption_shuffle_variants N, generates
N variants per image and caches them all in one file. v0 is the pristine
original caption (no shuffle, no dropout); v1..v{N-1} are smart-shuffled and,
if --caption_tag_dropout_rate > 0, have non-prefix tags independently dropped
at that rate. The strategy loader picks v0 with 20% probability and uniform
v1..v{N-1} with 80% probability when use_shuffled_caption_variants is on.
Pass --prefer_json_caption (or --prefer_json) to prefer same-stem .json
caption sidecars, falling back to .txt when JSON is missing or invalid.

The encode loop lives in ``library/preprocess/text.py``; this file is argparse +
model load + the one-time uncond sidecar staging.
"""

import argparse
from pathlib import Path

from PIL import Image
import torch


from library.preprocess import cache_text_embeddings, tqdm_progress
from library.runtime.cli import add_io_args


def _collect_image_caption_entries(
    image_paths,
    *,
    min_pixels: int = 500_000,
    prefer_json_caption: bool = False,
):
    """Compatibility helper for older tests/tooling.

    The real cache loop now lives in ``library.preprocess.text``; this helper
    keeps the old script-level inspection contract: return cacheable
    ``(image_path, caption)`` pairs while counting low-res, missing, and empty
    captions.
    """
    entries = []
    skipped_small = 0
    missing_captions = 0
    empty_caption_files = 0
    samples = []

    for image_path in [Path(p) for p in image_paths]:
        if min_pixels > 0:
            try:
                with Image.open(image_path) as image:
                    w, h = image.size
            except Exception:
                continue
            if w * h < min_pixels:
                skipped_small += 1
                continue

        if prefer_json_caption:
            from library.preprocess.captions import read_caption_source

            source = read_caption_source(
                image_path,
                prefer_json_caption=True,
                caption_extension=".txt",
            )
            caption_text = source.render()
            if source.path is None:
                missing_captions += 1
                samples.append(image_path.name)
            elif source.path.suffix.lower() == ".txt" and not caption_text:
                empty_caption_files += 1
                samples.append(image_path.name)
            entries.append((image_path, caption_text))
            continue

        caption_path = image_path.with_suffix(".txt")
        if not caption_path.exists():
            missing_captions += 1
            samples.append(image_path.name)
            entries.append((image_path, ""))
            continue

        caption = caption_path.read_text(encoding="utf-8").splitlines()
        caption_text = caption[0].strip() if caption else ""
        if not caption_text:
            empty_caption_files += 1
            samples.append(image_path.name)
        entries.append((image_path, caption_text))

    return entries, skipped_small, missing_captions, empty_caption_files, samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_io_args(
        parser,
        cache_noun="text-encoder caches",
        include_batch_size=True,
        batch_size_default=16,
    )
    parser.add_argument(
        "--qwen3", type=str, required=True, help="Path to Qwen3 text encoder"
    )
    parser.add_argument(
        "--dit",
        type=str,
        default=None,
        help="Path to DiT model for LLM adapter crossattn_emb caching",
    )
    parser.add_argument(
        "--t5_tokenizer_path",
        type=str,
        default=None,
        help="Path to T5 tokenizer (default: library/anima/configs/t5_old/)",
    )
    parser.add_argument(
        "--caption_shuffle_variants",
        type=int,
        default=0,
        help=(
            "Number of caption variants per image (0 = single caption). v0 is "
            "the pristine original; v1..v{N-1} are shuffled (and tag-dropped "
            "if --caption_tag_dropout_rate > 0)."
        ),
    )
    parser.add_argument(
        "--caption_tag_dropout_rate",
        type=float,
        default=0.0,
        help=(
            "Per-tag dropout probability applied to v1..v{N-1} only. Tags up "
            "to and including the first @artist marker are never dropped. "
            "Ignored when --caption_shuffle_variants <= 0."
        ),
    )
    parser.add_argument(
        "--prefer_json_caption",
        "--prefer_json",
        dest="prefer_json_caption",
        action="store_true",
        help=(
            "Prefer same-stem .json caption sidecars and fall back to .txt "
            "when JSON is missing or invalid. Disabled by default."
        ),
    )
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=500_000,
        help=(
            "Skip images with fewer than this many pixels (default: 500_000 "
            "= 0.5MP). Mirrors the same filter in scripts/preprocess/resize_images.py "
            "so TE caches don't accumulate for images that get dropped at "
            "resize time. Set to 0 to disable."
        ),
    )
    args = parser.parse_args()

    from library.anima import weights as anima_utils
    from library.anima.strategy import AnimaTextEncodingStrategy, AnimaTokenizeStrategy

    data_dir = Path(args.dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = args.caption_shuffle_variants

    # Load text encoder + tokenizers
    print(f"Loading Qwen3 text encoder from {args.qwen3} ...")
    text_encoder, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(
        args.qwen3, dtype=torch.bfloat16, device=str(device)
    )
    t5_tokenizer = anima_utils.load_t5_tokenizer(args.t5_tokenizer_path)

    # Optionally load LLM adapter for crossattn_emb caching
    llm_adapter = None
    if args.dit:
        print(f"Loading LLM adapter from {args.dit} ...")
        llm_adapter = anima_utils.load_llm_adapter(
            args.dit, dtype=torch.bfloat16, device=str(device)
        )

    tokenize_strategy = AnimaTokenizeStrategy(
        qwen3_tokenizer=qwen3_tokenizer, t5_tokenizer=t5_tokenizer
    )
    encoding_strategy = AnimaTextEncodingStrategy()

    # Stage the T5("") sidecar while Qwen3 + LLM adapter are already on
    # device. Every training/distill run reuses this one tiny file as the
    # CFG-uncond crossattn input -- matches `library/inference/text.py`.
    # Skipped when ``--dit`` is omitted (only TE outputs cached; no
    # llm_adapter, so we can't produce crossattn embeddings here).
    if llm_adapter is not None:
        from library.inference.uncond import (
            DEFAULT_UNCOND_DIR,
            stage_uncond_sidecar_with_models,
        )

        stage_uncond_sidecar_with_models(
            DEFAULT_UNCOND_DIR,
            text_encoder,
            tokenize_strategy,
            encoding_strategy,
            llm_adapter,
            device=device,
            overwrite=bool(getattr(args, "force_recache_uncond", False)),
        )

    tag_dropout_rate = float(args.caption_tag_dropout_rate)
    if N > 0:
        print(
            f"Caption shuffle variants: {N} "
            f"(v0=pristine, v1..v{N - 1}=shuffled"
            + (
                f" + tag dropout p={tag_dropout_rate:.3f}"
                if tag_dropout_rate > 0.0
                else ""
            )
            + ")"
        )
    elif tag_dropout_rate > 0.0:
        print(
            "warn: --caption_tag_dropout_rate ignored because "
            "--caption_shuffle_variants <= 0 (single-variant cache)."
        )
    if args.prefer_json_caption:
        print("Caption source: prefer JSON sidecars (.json -> .txt fallback)")

    stats = cache_text_embeddings(
        data_dir,
        tokenize_strategy,
        encoding_strategy,
        text_encoder,
        llm_adapter=llm_adapter,
        device=device,
        cache_dir=cache_dir,
        recursive=args.recursive,
        batch_size=args.batch_size,
        caption_shuffle_variants=N,
        caption_tag_dropout_rate=tag_dropout_rate,
        prefer_json_caption=args.prefer_json_caption,
        min_pixels=args.min_pixels,
        progress=tqdm_progress("Caching text embeddings"),
    )
    print(
        f"\nText embedding caching complete: {stats.written} cached, "
        f"{stats.skipped} skipped (already existed)"
    )

    text_encoder.to("cpu")
    del text_encoder, llm_adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
