"""Natural-language / tag mix caption helpers for dataset services."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from library.preprocess.captions import read_caption_source
from web.services.config.dataset_media import _dataset_image_files
from web.services.config.metadata import (
    CAPTION_SOURCE_AUTO,
    DATASET_IMAGE_EXTS,
    NL_TAG_MIX_CLASSIFICATION_METHOD,
)


def _nl_tag_mix_available_count(
    source_dir: Path,
    image_exts: set[str],
    *,
    recursive: bool = True,
    path_pattern: str = "*",
) -> int | None:
    if not source_dir.is_dir():
        return None
    return len(
        _nl_tag_mix_image_files(
            source_dir,
            image_exts,
            recursive=recursive,
            path_pattern=path_pattern,
        )
    )


def _nl_tag_mix_image_files(
    source_dir: Path,
    image_exts: set[str] | frozenset[str] = DATASET_IMAGE_EXTS,
    *,
    recursive: bool = True,
    path_pattern: str = "*",
) -> list[Path]:
    return _dataset_image_files(
        source_dir,
        image_exts,
        recursive=recursive,
        path_pattern=path_pattern,
    )


def _nl_tag_mix_caption_source(
    image_path: Path,
    *,
    caption_source_mode: str | None = None,
    caption_extension: str = ".txt",
    prefer_json_caption: bool = False,
    captions_root: Path | None = None,
):
    return read_caption_source(
        image_path,
        prefer_json_caption=prefer_json_caption,
        caption_source_mode=caption_source_mode or CAPTION_SOURCE_AUTO,
        caption_extension=caption_extension,
        captions_root=captions_root or image_path.parent,
    )


def _nl_tag_mix_caption_path_and_text(
    image_path: Path,
    *,
    caption_source_mode: str | None = None,
    caption_extension: str = ".txt",
    prefer_json_caption: bool = False,
    captions_root: Path | None = None,
) -> tuple[Path | None, str]:
    source = _nl_tag_mix_caption_source(
        image_path,
        caption_source_mode=caption_source_mode,
        caption_extension=caption_extension,
        prefer_json_caption=prefer_json_caption,
        captions_root=captions_root,
    )
    text = "\n".join(source.caption_texts())
    if source.path is not None and text.strip():
        return source.path, text
    return None, ""


def _nl_tag_mix_caption_counts(
    source_dir: Path,
    *,
    caption_source_mode: str | None = None,
    caption_extension: str = ".txt",
    prefer_json_caption: bool = False,
    recursive: bool = True,
    path_pattern: str = "*",
) -> tuple[int, int]:
    images = _nl_tag_mix_image_files(
        source_dir,
        recursive=recursive,
        path_pattern=path_pattern,
    )
    captioned = 0
    for image in images:
        _caption_path, text = _nl_tag_mix_caption_path_and_text(
            image,
            caption_source_mode=caption_source_mode,
            caption_extension=caption_extension,
            prefer_json_caption=prefer_json_caption,
            captions_root=source_dir,
        )
        if text.strip():
            captioned += 1
    return len(images), captioned


def _classify_nl_tag_caption_text(text: str) -> dict[str, Any]:
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return {
            "kind": "tag",
            "reason": "missing_or_empty_caption_default_tag",
            "method": NL_TAG_MIX_CLASSIFICATION_METHOD,
            "metrics": {"length": 0},
        }

    comma_parts = [part.strip() for part in re.split(r"[,，]", normalized) if part.strip()]
    word_groups = re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]+|\d+", normalized)
    sentence_marks = len(re.findall(r"[.!?。！？]", normalized))
    segment_word_counts = [
        len(re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]+|\d+", part))
        for part in comma_parts
    ]
    short_segments = sum(1 for count in segment_word_counts if 0 < count <= 4)
    short_segment_ratio = short_segments / len(segment_word_counts) if segment_word_counts else 0.0
    avg_segment_words = sum(segment_word_counts) / len(segment_word_counts) if segment_word_counts else float(len(word_groups))
    lower = normalized.lower()
    prose_markers = len(re.findall(
        r"\b(a|an|the|with|and|of|in|on|as|while|where|who|that|this|she|he|they|it|is|are|was|were|takes|place|scene|composition|rendered|illustration)\b",
        lower,
    ))

    is_nl = (
        sentence_marks >= 2
        or (sentence_marks >= 1 and len(word_groups) >= 24)
        or (len(word_groups) >= 35 and avg_segment_words >= 6 and prose_markers >= 3)
    )
    is_tag = (
        len(comma_parts) >= 4
        and short_segment_ratio >= 0.62
        and sentence_marks <= 1
    )
    metrics = {
        "length": len(normalized),
        "word_count": len(word_groups),
        "comma_part_count": len(comma_parts),
        "sentence_mark_count": sentence_marks,
        "short_segment_ratio": round(short_segment_ratio, 4),
        "avg_segment_words": round(avg_segment_words, 4),
        "prose_marker_count": prose_markers,
    }
    if is_nl:
        return {
            "kind": "nl",
            "reason": "caption_has_sentence_prose_shape",
            "method": NL_TAG_MIX_CLASSIFICATION_METHOD,
            "metrics": metrics,
        }
    if is_tag:
        return {
            "kind": "tag",
            "reason": "caption_has_comma_tag_shape",
            "method": NL_TAG_MIX_CLASSIFICATION_METHOD,
            "metrics": metrics,
        }
    return {
        "kind": "tag",
        "reason": "ambiguous_caption_default_tag",
        "method": NL_TAG_MIX_CLASSIFICATION_METHOD,
        "metrics": metrics,
    }


