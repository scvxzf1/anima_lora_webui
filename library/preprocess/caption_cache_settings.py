"""Effective settings for generated caption variants in TE caches."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

DEFAULT_CAPTION_SHUFFLE_VARIANTS = 4
DEFAULT_CAPTION_TAG_DROPOUT_RATE = 0.1


def resolve_caption_cache_settings(
    config: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[int, float]:
    """Resolve env > config > historical defaults for TE variant generation."""
    config = config or {}
    environ = os.environ if environ is None else environ

    variants_raw = environ.get("CAPTION_SHUFFLE_VARIANTS")
    if variants_raw is None or not str(variants_raw).strip():
        variants_raw = config.get(
            "caption_shuffle_variants",
            DEFAULT_CAPTION_SHUFFLE_VARIANTS,
        )
    tag_rate_raw = environ.get("CAPTION_TAG_DROPOUT_RATE")
    if tag_rate_raw is None or not str(tag_rate_raw).strip():
        tag_rate_raw = config.get(
            "caption_tag_dropout_rate",
            DEFAULT_CAPTION_TAG_DROPOUT_RATE,
        )

    try:
        variants = int(variants_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"caption_shuffle_variants must be an integer, got {variants_raw!r}"
        ) from exc
    try:
        tag_rate = float(tag_rate_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"caption_tag_dropout_rate must be a float, got {tag_rate_raw!r}"
        ) from exc

    if variants < 0:
        raise ValueError("caption_shuffle_variants must be >= 0")
    if not 0.0 <= tag_rate <= 1.0:
        raise ValueError("caption_tag_dropout_rate must be between 0 and 1")
    return variants, tag_rate
