"""Dataset image listing, preview metadata, and count helpers.

Extracted from ``web.services.config.datasets`` so media IO stays separate from
preset/editor orchestration. Path roots follow the config_service facade when
available, but pure helpers can still run without importing the facade.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from threading import RLock
from time import monotonic
from typing import Any
from urllib.parse import quote

import toml

from library.env import expand_env_vars, get_configs_root, load_dotenv
from library.preprocess._dataset import walk_images
from library.preprocess.captions import (
    normalize_caption_source_mode,
    read_caption_source_from_dirs,
)
from web.services.config import paths as _config_paths
from web.services.config.common import _nonnegative_int, _positive_int
from web.services.image_size import probe_image_size
from web.services.config.dataset_rows import (
    _normalize_path_pattern,
    _single_dataset_config_from_cfg,
)
from web.services.config.metadata import (
    CAPTION_SOURCE_AUTO,
    CAPTION_SOURCE_CAPTIONS_JSON,
    CAPTION_SOURCE_JSON,
    CAPTION_SOURCE_MODE_LABELS,
    CAPTION_SOURCE_TXT,
    DATASET_CAPTION_MAX_CHARS,
    DATASET_IMAGE_EXTS,
    DATASET_PREVIEW_LIMIT,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = get_configs_root()

load_dotenv()

_DATASET_IMAGE_LIST_CACHE_TTL_SECONDS = 2.0
_DATASET_IMAGE_LIST_CACHE_ITEMS = 64
_DATASET_IMAGE_LIST_CACHE: OrderedDict[
    tuple[str, tuple[str, ...], bool, str],
    tuple[float, int, tuple[Path, ...]],
] = OrderedDict()
_DATASET_IMAGE_LIST_CACHE_LOCK = RLock()


def _sync_from_facade() -> None:
    """Align path roots with datasets/config_service monkeypatches when present."""
    import sys

    global ROOT, CONFIGS_DIR
    for module_name in (
        "web.services.config.datasets",
        "web.services.config_service",
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        if hasattr(module, "ROOT"):
            ROOT = module.ROOT
        if hasattr(module, "CONFIGS_DIR"):
            CONFIGS_DIR = module.CONFIGS_DIR
        # datasets is the preferred owner for path monkeypatches.
        if module_name == "web.services.config.datasets":
            break


def _normalize_config_rel_path(rel_path: str) -> str:
    return _config_paths.normalize_config_rel_path(rel_path)


def _safe_resolve(rel_path: str) -> Path | None:
    _sync_from_facade()
    return _config_paths.safe_resolve(rel_path, root=ROOT, configs_dir=CONFIGS_DIR)


def _display_path(path: Path) -> str:
    _sync_from_facade()
    return _config_paths.display_path(path, root=ROOT, configs_dir=CONFIGS_DIR)


def _list_dataset_image_files(
    directory: Path,
    limit: int,
    *,
    offset: int = 0,
    recursive: bool = True,
    path_pattern: str = "*",
) -> dict[str, Any]:
    clean_limit = max(1, min(_positive_int(limit, DATASET_PREVIEW_LIMIT), DATASET_PREVIEW_LIMIT))
    clean_offset = _nonnegative_int(offset, 0)
    items = _dataset_image_files(
        directory,
        DATASET_IMAGE_EXTS,
        recursive=recursive,
        path_pattern=path_pattern,
    )
    total = len(items)
    resolved_offset = min(clean_offset, total)
    page = items[resolved_offset:resolved_offset + clean_limit]
    next_offset = min(total, resolved_offset + len(page))
    return {
        "items": page,
        "total": total,
        "offset": resolved_offset,
        "limit": clean_limit,
        "returned": len(page),
        "next_offset": next_offset,
        "has_more_before": resolved_offset > 0,
        "has_more_after": next_offset < total,
    }


def _dataset_image_preview_meta(
    path: Path,
    *,
    preset_file: str,
    dataset_index: int,
    source: str,
    caption_extension: str,
    prefer_json_caption: bool,
    caption_source_mode: str,
    source_dir: Path,
    train_dir: Path,
) -> dict[str, Any]:
    stat = path.stat()
    caption = _dataset_caption_meta(
        path,
        caption_extension,
        source_dir,
        train_dir,
        prefer_json_caption=prefer_json_caption,
        caption_source_mode=caption_source_mode,
    )
    dimensions = _dataset_image_dimensions(path)
    rel_path = _display_path(path)
    url = (
        "/api/config/dataset-presets/image"
        f"?file={quote(preset_file)}"
        f"&dataset_index={dataset_index}"
        f"&source={quote(source)}"
        f"&image={quote(rel_path)}"
    )
    thumbnail_url = (
        "/api/config/dataset-presets/thumbnail"
        f"?file={quote(preset_file)}"
        f"&dataset_index={dataset_index}"
        f"&source={quote(source)}"
        f"&image={quote(rel_path)}"
        f"&v={stat.st_mtime_ns:x}-{stat.st_size:x}"
    )
    return {
        "file": rel_path,
        "name": path.name,
        "url": url,
        "thumbnail_url": thumbnail_url,
        "mtime": stat.st_mtime,
        "mtime_text": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "size_bytes": stat.st_size,
        "width": dimensions.get("width"),
        "height": dimensions.get("height"),
        "total_pixels": dimensions.get("total_pixels"),
        "caption": caption,
    }


def _dataset_image_dimensions(path: Path) -> dict[str, int]:
    width, height = probe_image_size(path)
    if width is None or height is None:
        return {}
    return {
        "width": int(width),
        "height": int(height),
        "total_pixels": int(width) * int(height),
    }


def _dataset_caption_meta(
    path: Path,
    caption_extension: str,
    source_dir: Path,
    train_dir: Path,
    *,
    prefer_json_caption: bool = False,
    caption_source_mode: str | None = None,
) -> dict[str, Any]:
    extension = caption_extension if caption_extension.startswith(".") else f".{caption_extension}"
    source_mode = normalize_caption_source_mode(caption_source_mode, prefer_json_caption)
    directories: list[Path] = []
    for directory in (path.parent, source_dir, train_dir):
        if not directory:
            continue
        directory = directory.resolve()
        if directory not in directories:
            directories.append(directory)

    source = read_caption_source_from_dirs(
        path,
        directories,
        prefer_json_caption=prefer_json_caption,
        caption_source_mode=source_mode,
        caption_extension=extension,
        warn=None,
    )
    if source.path is not None:
        texts = source.caption_texts()
        text = _format_caption_preview_text(texts)
        truncated = len(text) > DATASET_CAPTION_MAX_CHARS
        if truncated:
            text = text[:DATASET_CAPTION_MAX_CHARS]
        return {
            "ok": True,
            "file": _display_path(source.path),
            "extension": _caption_extension_for_detected_mode(source.detected_mode, extension),
            "source_mode": source_mode,
            "source_label": _caption_source_mode_label(source_mode),
            "detected_mode": source.detected_mode,
            "format_label": _caption_source_mode_label(source.detected_mode),
            "caption_count": len(texts),
            "text": text,
            "truncated": truncated,
            "length": len(text),
        }
    return {
        "ok": False,
        "file": "",
        "extension": _caption_extension_for_detected_mode(source_mode, extension),
        "source_mode": source_mode,
        "source_label": _caption_source_mode_label(source_mode),
        "detected_mode": "",
        "format_label": "",
        "caption_count": 0,
        "text": "",
        "truncated": False,
        "length": 0,
    }


def _caption_source_mode_label(mode: str | None) -> str:
    return CAPTION_SOURCE_MODE_LABELS.get(str(mode or ""), str(mode or "自动识别"))


def _caption_extension_for_detected_mode(mode: str | None, fallback: str) -> str:
    if mode == CAPTION_SOURCE_CAPTIONS_JSON:
        return "captions.json"
    if mode == CAPTION_SOURCE_JSON:
        return ".json"
    if mode == CAPTION_SOURCE_TXT:
        return fallback
    if mode == CAPTION_SOURCE_AUTO:
        return "auto"
    return fallback


def _format_caption_preview_text(texts: list[str]) -> str:
    if len(texts) <= 1:
        return texts[0] if texts else ""
    return "\n".join(f"{idx}. {text}" for idx, text in enumerate(texts, start=1))


def _dataset_caption_detection_summary(images: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    caption_total = 0
    missing = 0
    for image in images:
        caption = image.get("caption") if isinstance(image, dict) else {}
        if not isinstance(caption, dict) or not caption.get("ok"):
            missing += 1
            continue
        mode = str(caption.get("detected_mode") or "")
        counts[mode] = counts.get(mode, 0) + 1
        caption_total += _positive_int(caption.get("caption_count"), 1)
    parts = [
        f"{_caption_source_mode_label(mode)} {count} 张"
        for mode, count in counts.items()
        if mode
    ]
    if missing:
        parts.append(f"缺少 {missing} 张")
    if caption_total and counts.get(CAPTION_SOURCE_CAPTIONS_JSON):
        parts.append(f"共 {caption_total} 条标注")
    return "，".join(parts)


def _caption_detection_counts_text(counts: dict[str, int], caption_total: int) -> str:
    parts = [
        f"{_caption_source_mode_label(mode)} {count} 张"
        for mode, count in counts.items()
        if mode
    ]
    if caption_total and counts.get(CAPTION_SOURCE_CAPTIONS_JSON):
        parts.append(f"共 {caption_total} 条标注")
    return "识别结果：" + "，".join(parts) if parts else ""


def _dataset_preview_empty_message(directory: Path, source: str) -> str:
    label = "原始图目录" if source == "source" else "训练图目录"
    if not directory.exists():
        return f"{label}不存在"
    if not directory.is_dir():
        return f"{label}不是目录"
    return f"{label}里没有可预览图片"


def _dataset_image_files(
    path: Path,
    image_exts: set[str] | frozenset[str],
    *,
    recursive: bool = True,
    path_pattern: str = "*",
) -> list[Path]:
    if not path.is_dir():
        return []
    normalized_pattern = _normalize_path_pattern(path_pattern)
    cache_key = (
        str(path.resolve()),
        tuple(sorted(image_exts)),
        bool(recursive),
        normalized_pattern,
    )
    directory_mtime_ns = path.stat().st_mtime_ns
    now = monotonic()
    with _DATASET_IMAGE_LIST_CACHE_LOCK:
        cached = _DATASET_IMAGE_LIST_CACHE.get(cache_key)
        if cached and cached[0] > now and cached[1] == directory_mtime_ns:
            _DATASET_IMAGE_LIST_CACHE.move_to_end(cache_key)
            return list(cached[2])

    items = tuple(sorted(
        item
        for item in walk_images(
            path,
            recursive=recursive,
            pattern=normalized_pattern,
        )
        if item.suffix.lower() in image_exts
    ))
    with _DATASET_IMAGE_LIST_CACHE_LOCK:
        _DATASET_IMAGE_LIST_CACHE[cache_key] = (
            monotonic() + _DATASET_IMAGE_LIST_CACHE_TTL_SECONDS,
            directory_mtime_ns,
            items,
        )
        _DATASET_IMAGE_LIST_CACHE.move_to_end(cache_key)
        while len(_DATASET_IMAGE_LIST_CACHE) > _DATASET_IMAGE_LIST_CACHE_ITEMS:
            _DATASET_IMAGE_LIST_CACHE.popitem(last=False)
    return list(items)


def _count_images(
    path: Path,
    image_exts: set[str],
    *,
    recursive: bool = True,
    path_pattern: str = "*",
) -> int:
    return len(
        _dataset_image_files(
            path,
            image_exts,
            recursive=recursive,
            path_pattern=path_pattern,
        )
    )


def _count_source_images(
    path: Path,
    image_exts: set[str],
    *,
    recursive: bool = True,
    path_pattern: str = "*",
) -> int:
    from web.services.config.dataset_nl_tag import _nl_tag_mix_image_files

    return len(
        _nl_tag_mix_image_files(
            path,
            image_exts,
            recursive=recursive,
            path_pattern=path_pattern,
        )
    )


def _dataset_num_repeats(cfg: dict[str, Any]) -> int:
    dataset_config = cfg.get("dataset_config")
    if dataset_config:
        path = _safe_resolve(_normalize_config_rel_path(str(dataset_config)))
        if path is not None and path.exists():
            try:
                data = toml.loads(path.read_text(encoding="utf-8"))
                repeats = []
                for dataset in data.get("datasets") or []:
                    for subset in dataset.get("subsets") or []:
                        repeats.append(_positive_int(subset.get("num_repeats"), 1))
                return max(1, sum(repeats) or 1)
            except Exception:
                return 1
    return 1
