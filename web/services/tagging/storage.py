"""Dataset image resolution and conservative caption write-back helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

from library.env import expand_env_vars
from library.preprocess.captions import (
    CAPTION_SOURCE_AUTO,
    CAPTION_SOURCE_CAPTIONS_JSON,
    normalize_caption_source_mode,
    read_caption_source_from_dirs,
)
from web.services.config.dataset_presets_api import (
    load_dataset_preset,
    resolve_dataset_preview_image,
)
from web.services.config.paths import resolve_display_path
from web.services.config import dataset_presets_api as _preset_api

from . import caption_edit


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


def resolve_caption_directory(dataset_file: str, dataset_index: int, *, source: str = "source") -> Path:
    """Return the directory that holds caption sidecars for one dataset row."""

    preset = load_dataset_preset(dataset_file)
    rows = preset.get("datasets") if isinstance(preset.get("datasets"), list) else []
    if dataset_index < 0 or dataset_index >= len(rows):
        raise ValueError("数据集序号不在范围内")
    row = rows[dataset_index] if isinstance(rows[dataset_index], dict) else {}
    source_kind = "source" if str(source or "").strip().lower() == "source" else "training"
    primary = "source_dir" if source_kind == "source" else "image_dir"
    directory = (
        _resolve_optional_path(row.get(primary))
        or _resolve_optional_path(row.get("image_dir"))
        or _resolve_optional_path(row.get("source_dir"))
    )
    if directory is None or not directory.is_dir():
        raise FileNotFoundError("数据集图片目录不存在")
    return directory


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


def caption_text_suffix(settings: dict[str, Any] | None) -> str:
    """Return the text caption suffix that one dataset row writes to.

    The job skip pre-filter and write_caption both resolve the suffix here, so
    detection and write-back can never disagree about which sidecar counts.
    """

    row_settings = settings if isinstance(settings, dict) else {}
    return _caption_extension(row_settings.get("caption_extension"))


def write_caption(
    dataset_file: str,
    dataset_index: int,
    image_file: str,
    text: str,
    *,
    source: str = "source",
    policy: str = caption_edit.POLICY_OVERWRITE,
    trigger_word: str = "",
) -> dict[str, Any]:
    """Write one reviewed caption back to the authoritative sidecar.

    Structured JSON captions keep their unknown fields and receive the edited
    tags; plain text sidecars are rewritten in place.  ``skip`` leaves an
    already captioned image untouched and reports ``skipped``.
    """

    resolved = resolve_tagging_image(dataset_file, dataset_index, image_file, source=source)
    path = resolved["path"]
    row_settings = resolved.get("settings") if isinstance(resolved.get("settings"), dict) else {}
    text_suffix = caption_text_suffix(row_settings)
    mode = caption_edit.normalize_policy(policy)
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(value) > 100_000:
        raise ValueError("caption 过长，最多支持 100000 个字符")
    # Fail closed when the row stores captions in the captions.json aggregate:
    # a sidecar write would be invisible to the dataset reader (BAC-4).
    _assert_sidecar_write_allowed(path, resolved["row"], row_settings)
    # Detection and the write target must inspect the same sidecar: a custom
    # caption_extension changes both, so neither may fall back to the default.
    existing = caption_edit.read_caption(path, text_suffix=text_suffix)
    if mode == caption_edit.POLICY_SKIP and caption_edit.has_effective_caption(
        path, text_suffix=text_suffix
    ):
        return {
            "ok": True,
            "skipped": True,
            "file": resolved["file"],
            "caption_file": _display_path(existing["path"]),
            "length": len(existing["text"]),
            "text": existing["text"],
            "format": existing["kind"],
        }
    tags = caption_edit.split_tags(value)
    if mode == caption_edit.POLICY_APPEND:
        tags = caption_edit.merge_tags(existing["tags"], tags)
    target = caption_edit.sidecar_for(path, text_suffix=text_suffix)[0].resolve()
    _assert_caption_target(target, path, resolved["row"])
    written = caption_edit.write_tags(
        path,
        tags,
        text_suffix=text_suffix,
        trigger=trigger_word,
    )
    return {
        "ok": True,
        "skipped": False,
        "file": resolved["file"],
        "caption_file": _display_path(written["path"]),
        "length": len(written["text"]),
        "text": written["text"],
        "format": written["kind"],
        "tags": written["tags"],
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


def _caption_scope_directories(image: Path, row: dict[str, Any]) -> list[Path]:
    """Return the directories the dataset reader inspects for this image.

    Kept identical to read_caption_for_image: the image directory plus the row
    source/training roots, so a nested image still sees an aggregate caption
    that lives at the dataset root (BAC-4 / T9-1).
    """

    source_dir = _resolve_optional_path(row.get("source_dir"))
    train_dir = _resolve_optional_path(row.get("image_dir"))
    return [image.parent, *(item for item in (source_dir, train_dir) if item is not None)]


def _assert_sidecar_write_allowed(image: Path, row: dict[str, Any], settings: dict[str, Any]) -> None:
    """Refuse sidecar writes for rows whose captions live in captions.json.

    The decision is derived from the same reader the preview and training cache
    use (same directory set, same mode resolution), so a sidecar write can never
    be shadowed by an aggregate caption the reader would prefer.
    """

    source_mode = normalize_caption_source_mode(
        settings.get("caption_source_mode"),
        _as_bool(settings.get("prefer_json_caption"), False),
    )
    if source_mode == CAPTION_SOURCE_CAPTIONS_JSON:
        raise caption_edit.AggregateCaptionWriteError(
            "该数据集行使用 captions.json 聚合 caption，工作台不会新建或修改 sidecar；"
            "请把 caption 来源模式切换为 sidecar（.txt/.json），或直接编辑 captions.json 后再打标"
        )
    if source_mode != CAPTION_SOURCE_AUTO:
        # An explicit sidecar mode reads exactly that sidecar, so a stray
        # aggregate entry cannot shadow the write.
        return
    try:
        source = read_caption_source_from_dirs(
            image,
            _caption_scope_directories(image, row),
            prefer_json_caption=_as_bool(settings.get("prefer_json_caption"), False),
            caption_source_mode=source_mode,
            caption_extension=caption_text_suffix(settings),
            warn=None,
        )
    except (OSError, ValueError):
        return
    if source.detected_mode == CAPTION_SOURCE_CAPTIONS_JSON:
        raise caption_edit.AggregateCaptionWriteError(
            "captions.json 已包含该图片的 caption，自动来源模式会优先读取它；"
            "继续写 sidecar 会形成双真值。请切换 caption 来源模式，或先从聚合文件移除该条目"
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
