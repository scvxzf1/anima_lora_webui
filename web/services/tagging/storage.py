"""Dataset image resolution and conservative caption write-back helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from library.env import expand_env_vars
from library.preprocess.captions import normalize_caption_source_mode, read_caption_source_from_dirs
from web.services.atomic_io import atomic_write_text
from web.services.config.dataset_presets_api import (
    load_dataset_preset,
    resolve_dataset_preview_image,
)
from web.services.config.paths import resolve_display_path
from web.services.config import dataset_presets_api as _preset_api


class CaptionWriteConflict(ValueError):
    """Raised when an existing structured caption should not be overwritten."""


def resolve_tagging_image(
    dataset_file: str,
    dataset_index: int,
    image_file: str,
    *,
    source: str = "source",
) -> dict[str, Any]:
    """Resolve one preview image and return its current caption metadata."""

    preset = load_dataset_preset(dataset_file)
    rows = preset.get("datasets") if isinstance(preset.get("datasets"), list) else []
    if dataset_index < 0 or dataset_index >= len(rows):
        raise ValueError("数据集序号不在范围内")
    row = rows[dataset_index] if isinstance(rows[dataset_index], dict) else {}
    source_kind = "source" if str(source or "").strip().lower() == "source" else "training"
    path = resolve_dataset_preview_image(dataset_file, dataset_index, image_file, source=source_kind)
    settings = _row_settings(row, preset.get("defaults"))
    caption = read_caption_for_image(path, row, settings)
    stat = path.stat()
    query = (
        f"file={quote(dataset_file)}"
        f"&dataset_index={dataset_index}"
        f"&source={quote(source_kind)}"
        f"&image={quote(_display_path(path))}"
    )
    return {
        "path": path,
        "file": _display_path(path),
        "name": path.name,
        "source": source_kind,
        "row": row,
        "settings": settings,
        "caption": caption,
        "url": f"/api/config/dataset-presets/image?{query}",
        "thumbnail_url": (
            f"/api/config/dataset-presets/thumbnail?{query}"
            f"&v={stat.st_mtime_ns:x}-{stat.st_size:x}"
        ),
    }


def read_caption_for_image(path: Path, row: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Read the same caption sources shown by the dataset preview."""

    source_dir = _resolve_optional_path(row.get("source_dir"))
    train_dir = _resolve_optional_path(row.get("image_dir"))
    directories = [path.parent, *(item for item in (source_dir, train_dir) if item is not None)]
    extension = _caption_extension(settings.get("caption_extension"))
    source_mode = normalize_caption_source_mode(
        settings.get("caption_source_mode"),
        _as_bool(settings.get("prefer_json_caption"), False),
    )
    source = read_caption_source_from_dirs(
        path,
        directories,
        prefer_json_caption=_as_bool(settings.get("prefer_json_caption"), False),
        caption_source_mode=source_mode,
        caption_extension=extension,
        warn=None,
    )
    texts = source.caption_texts()
    return {
        "ok": bool(source.path),
        "file": _display_path(source.path) if source.path else "",
        "path": source.path,
        "text": "\n".join(texts).strip(),
        "caption_count": len(texts),
        "source_mode": source_mode,
        "detected_mode": source.detected_mode or "",
        "extension": _extension_for_source(source.detected_mode, extension),
    }


def write_caption(
    dataset_file: str,
    dataset_index: int,
    image_file: str,
    text: str,
    *,
    source: str = "source",
) -> dict[str, Any]:
    """Atomically write one plain-text caption after re-validating its path."""

    resolved = resolve_tagging_image(dataset_file, dataset_index, image_file, source=source)
    path = resolved["path"]
    # This workbench deliberately writes one sidecar per image. Structured
    # sources remain readable input, but are never mutated by the review flow.
    target = path.with_suffix(".txt").resolve()
    _assert_caption_target(target, path, resolved["row"])
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(value) > 100_000:
        raise ValueError("caption 过长，最多支持 100000 个字符")
    atomic_write_text(target, f"{value}\n" if value else "")
    return {
        "ok": True,
        "file": resolved["file"],
        "caption_file": _display_path(target),
        "length": len(value),
        "text": value,
    }


def _row_settings(row: dict[str, Any], defaults: Any) -> dict[str, Any]:
    base = defaults if isinstance(defaults, dict) else {}
    local = row.get("settings") if isinstance(row.get("settings"), dict) else {}
    return {**base, **local}


def _resolve_optional_path(value: Any) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    return resolve_display_path(
        raw,
        root=Path(_preset_api.ROOT),
        configs_dir=Path(_preset_api.CONFIGS_DIR),
        expand_env_vars_fn=expand_env_vars,
    )


def _assert_caption_target(target: Path, image: Path, row: dict[str, Any]) -> None:
    allowed: list[Path] = [image.parent.resolve()]
    for key in ("source_dir", "image_dir"):
        candidate = _resolve_optional_path(row.get(key))
        if candidate is not None:
            allowed.append(candidate.resolve())
    for directory in allowed:
        try:
            target.relative_to(directory)
            return
        except ValueError:
            continue
    raise ValueError("caption 文件不属于当前数据集目录")


def _caption_extension(value: Any) -> str:
    raw = str(value or ".txt").strip()
    if not raw:
        return ".txt"
    if raw.lower() in {"captions.json", "jsonl"} or "/" in raw or "\\" in raw:
        return ".txt"
    extension = raw if raw.startswith(".") else f".{raw}"
    if len(extension) > 20 or not extension.replace(".", "").replace("-", "").replace("_", "").isalnum():
        return ".txt"
    return extension


def _extension_for_source(mode: str, fallback: str) -> str:
    if mode == "json":
        return ".json"
    if mode == "captions_json":
        return "captions.json"
    return _caption_extension(fallback)


def _display_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().relative_to(Path(_preset_api.ROOT).resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
