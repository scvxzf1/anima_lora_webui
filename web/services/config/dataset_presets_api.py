"""Dataset preset list/load/save API extracted from datasets service.

This module is loaded by ``web.services.config_service`` as part of the
compatibility facade.  It keeps facade access lazy so the module can also be
imported directly without pulling in the legacy facade.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import toml

from library.env import expand_env_vars, get_configs_root, load_dotenv
from library.preprocess.captions import normalize_caption_source_mode
from web.services.config import paths as _config_paths
from web.services.config.common import (
    _bool_value,
    _nonnegative_float,
    _positive_int,
)
from web.services.config.dataset_preset_paths import (
    _is_dataset_preset_readonly,
    _normalize_dataset_preset_path,
)
from web.services.config.dataset_rows import (
    _build_dataset_config_doc,
    _dataset_defaults_from_config,
    _dataset_rows_from_config,
    _dataset_summary_from_rows,
    _dataset_training_defaults,
    _ensure_training_dataset_rows,
    _fill_missing_dataset_row_settings,
    _first_training_dataset_row,
    _normalize_dataset_defaults,
    _normalize_dataset_rows,
    _normalize_path_pattern,
    _safe_file_stem,
)
from web.services.config.metadata import (
    DATASET_IMAGE_EXTS,
    DATASET_PREVIEW_LIMIT,
    HIDDEN_DATASET_PRESET_FILES,
    SYSTEM_DATASET_PRESET_FILES,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = get_configs_root()
DATASET_PRESETS_DIR = CONFIGS_DIR / "datasets"

load_dotenv()

LOGGER = logging.getLogger("web.services.config_service")


def _config_facade():
    from web.services import config_service as _facade

    return _facade


def _sync_from_facade() -> None:
    """Keep path roots aligned with the config_service facade / test patches.

    Prefer an already-imported facade so pure helper callers stay facade-free.
    """
    import sys

    global ROOT, CONFIGS_DIR, DATASET_PRESETS_DIR
    facade = sys.modules.get("web.services.config_service")
    if facade is None:
        return
    if hasattr(facade, "ROOT"):
        ROOT = facade.ROOT
    if hasattr(facade, "CONFIGS_DIR"):
        CONFIGS_DIR = facade.CONFIGS_DIR
    if hasattr(facade, "DATASET_PRESETS_DIR"):
        DATASET_PRESETS_DIR = facade.DATASET_PRESETS_DIR
    else:
        DATASET_PRESETS_DIR = CONFIGS_DIR / "datasets"


def save_raw_file(*args, **kwargs):
    return _config_facade().save_raw_file(*args, **kwargs)


def delete_raw_file(*args, **kwargs):
    return _config_facade().delete_raw_file(*args, **kwargs)


def get_config_file_meta(*args, **kwargs):
    return _config_facade().get_config_file_meta(*args, **kwargs)


def list_config_file_groups(*args, **kwargs):
    return _config_facade().list_config_file_groups(*args, **kwargs)


def _prepare_raw_file_patch(*args, **kwargs):
    return _config_facade()._prepare_raw_file_patch(*args, **kwargs)


def _normalize_config_rel_path(rel_path: str) -> str:
    return _config_paths.normalize_config_rel_path(rel_path)


def _safe_resolve(rel_path: str) -> Path | None:
    _sync_from_facade()
    return _config_paths.safe_resolve(rel_path, root=ROOT, configs_dir=CONFIGS_DIR)


def _resolve_project_path(value: str) -> Path:
    _sync_from_facade()
    return _config_paths.resolve_display_path(
        value,
        root=ROOT,
        configs_dir=CONFIGS_DIR,
        expand_env_vars_fn=expand_env_vars,
    )


def _display_path(path: Path) -> str:
    _sync_from_facade()
    return _config_paths.display_path(path, root=ROOT, configs_dir=CONFIGS_DIR)


def _system_dataset_preset_files() -> frozenset[str]:
    """Prefer the datasets module value so tests can monkeypatch the public surface."""
    from web.services.config import datasets as _datasets

    value = getattr(_datasets, "SYSTEM_DATASET_PRESET_FILES", SYSTEM_DATASET_PRESET_FILES)
    return frozenset(value)


def _hidden_dataset_preset_files() -> frozenset[str]:
    from web.services.config import datasets as _datasets

    value = getattr(_datasets, "HIDDEN_DATASET_PRESET_FILES", HIDDEN_DATASET_PRESET_FILES)
    return frozenset(value)


def _dataset_preset_summary(rel_path: str) -> dict[str, Any]:
    try:
        normalized = _normalize_dataset_preset_path(rel_path, must_exist=True)
        path = _safe_resolve(normalized)
        if path is None or not path.exists():
            raise ValueError("数据集预设不存在")
        _content, rows, defaults = _load_dataset_preset_content_rows_defaults(path)
    except Exception as e:
        return {"ok": False, "error": str(e), "dataset_count": 0}
    return _dataset_summary_from_rows(rows, defaults)


def _dataset_preset_groups_for_ui(
    presets_by_path: dict[str, dict[str, Any]],
    *,
    dataset_groups: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    covered: set[str] = set()
    for group in dataset_groups or list_config_file_groups(kind="dataset"):
        group_id = str(group.get("id") or "")
        files = [
            presets_by_path[item["path"]]
            for item in group.get("files", [])
            if item.get("path") in presets_by_path
        ]
        if not _is_dataset_group_for_ui(group, files):
            continue
        covered.update(item["path"] for item in files)
        groups.append({
            "id": group_id,
            "label": group.get("label") or group_id or "数据集分组",
            "open": group_id == "unfiled_datasets",
            "locked": bool(group.get("locked", False)),
            "group_locked": bool(group.get("group_locked", False)),
            "user_group_locked": bool(group.get("user_group_locked", False)),
            "system_locked": bool(group.get("system_locked", False)),
            "lockable": bool(group.get("lockable", False)),
            "user_managed": bool(group.get("user_managed", False)),
            "kind": group.get("kind") or "dataset",
            "renamable": bool(group.get("renamable", False)),
            "deletable": bool(group.get("deletable", False)),
            "movable": bool(group.get("movable", False)),
            "trainable": False,
            "methods_subdir": "",
            "files": files,
        })

    ungrouped = [
        presets_by_path[path]
        for path in sorted(presets_by_path)
        if path not in covered
    ]
    if ungrouped:
        groups.append({
            "id": "unfiled_datasets",
            "label": "未分组数据集配置",
            "open": True,
            "locked": False,
            "group_locked": False,
            "user_group_locked": False,
            "system_locked": False,
            "lockable": False,
            "user_managed": True,
            "kind": "dataset",
            "renamable": False,
            "deletable": False,
            "movable": True,
            "trainable": False,
            "methods_subdir": "",
            "files": ungrouped,
        })
    return sorted(groups, key=lambda group: 0 if group.get("id") == "unfiled_datasets" else 1)


def _is_dataset_group_for_ui(group: dict[str, Any], files: list[dict[str, Any]]) -> bool:
    group_id = str(group.get("id") or "")
    return bool(files) or group.get("kind") == "dataset" or group_id in {"datasets", "unfiled_datasets"}

def list_dataset_presets() -> dict[str, Any]:
    _sync_from_facade()
    DATASET_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_groups = list_config_file_groups(kind="dataset")
    grouped_meta_by_path: dict[str, dict[str, Any]] = {}
    for group in dataset_groups:
        for item in group.get("files", []):
            rel_path = _normalize_config_rel_path(str(item.get("path") or ""))
            if rel_path and rel_path not in grouped_meta_by_path:
                grouped_meta_by_path[rel_path] = item
    presets_by_path: dict[str, dict[str, Any]] = {}
    for path in sorted(DATASET_PRESETS_DIR.glob("*.toml")):
        rel_path = _normalize_config_rel_path(_display_path(path))
        if rel_path in _hidden_dataset_preset_files():
            continue
        meta = grouped_meta_by_path.get(rel_path) or get_config_file_meta(rel_path)
        summary = _dataset_preset_summary(rel_path)
        readonly = bool(meta.get("locked")) or rel_path in _system_dataset_preset_files()
        presets_by_path[rel_path] = {
            **meta,
            "readonly": readonly,
            "system_preset": rel_path in _system_dataset_preset_files(),
            "summary": summary,
        }

    groups = _dataset_preset_groups_for_ui(presets_by_path, dataset_groups=dataset_groups)
    ordered_paths: list[str] = []
    for group in groups:
        for item in group.get("files", []):
            path = item.get("path")
            if isinstance(path, str) and path not in ordered_paths:
                ordered_paths.append(path)

    presets = [presets_by_path[path] for path in ordered_paths if path in presets_by_path]
    return {"ok": True, "presets": presets, "groups": groups}


def diagnose_dataset_presets(rel_path: str = "") -> dict[str, Any]:
    """Return a compact scan report for debugging WebUI dataset preset visibility."""
    _sync_from_facade()
    DATASET_PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    target = ""
    if str(rel_path or "").strip():
        target = _normalize_dataset_preset_path(rel_path, must_exist=False)

    files: list[dict[str, Any]] = []
    for path in sorted(DATASET_PRESETS_DIR.glob("*.toml")):
        rel = _normalize_config_rel_path(_display_path(path))
        stat = path.stat()
        item: dict[str, Any] = {
            "path": rel,
            "absolute_path": str(path.resolve()),
            "size": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "hidden": rel in _hidden_dataset_preset_files(),
            "selected": bool(target and rel == target),
        }
        try:
            loaded = load_dataset_preset(rel)
            item.update({
                "ok": True,
                "readonly": bool(loaded.get("readonly")),
                "summary": loaded.get("summary") or {},
            })
        except Exception as exc:
            item.update({
                "ok": False,
                "error": str(exc),
            })
        files.append(item)

    try:
        listed = list_dataset_presets()
        listed_count = len(listed.get("presets", []))
        groups = [
            {
                "id": group.get("id"),
                "label": group.get("label"),
                "file_count": len(group.get("files", []) or []),
                "files": [item.get("path") for item in (group.get("files", []) or [])],
            }
            for group in listed.get("groups", [])
        ]
        list_error = ""
    except Exception as exc:
        listed_count = 0
        groups = []
        list_error = str(exc)

    return {
        "ok": not bool(list_error),
        "root": str(ROOT.resolve()),
        "dataset_dir": _display_path(DATASET_PRESETS_DIR),
        "absolute_dataset_dir": str(DATASET_PRESETS_DIR.resolve()),
        "target": target,
        "file_count": len(files),
        "listed_count": listed_count,
        "hidden_count": sum(1 for item in files if item.get("hidden")),
        "groups": groups,
        "files": files,
        "error": list_error,
    }


def load_dataset_preset(rel_path: str) -> dict[str, Any]:
    _sync_from_facade()
    normalized = _normalize_dataset_preset_path(rel_path, must_exist=True)
    path = _safe_resolve(normalized)
    if path is None or not path.exists():
        raise ValueError("数据集预设不存在")
    content, rows, defaults = _load_dataset_preset_content_rows_defaults(path)
    return {
        "ok": True,
        "file": normalized,
        "name": Path(normalized).stem,
        "content": content,
        "datasets": rows,
        "defaults": defaults,
        "readonly": _is_dataset_preset_readonly(normalized),
        "meta": get_config_file_meta(normalized),
        "summary": _dataset_summary_from_rows(rows, defaults),
    }


def _load_dataset_preset_content_rows_defaults(path: Path) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    content = path.read_text(encoding="utf-8")
    data = toml.loads(content)
    rows = _dataset_rows_from_config(data, {})
    defaults = _dataset_defaults_from_config(data)
    return content, rows, defaults


def save_dataset_preset(
    rel_path: str,
    rows: list[dict[str, Any]],
    defaults: dict[str, Any] | None = None,
    *,
    overwrite: bool = True,
) -> dict[str, Any]:
    _sync_from_facade()
    normalized = _normalize_dataset_preset_path(rel_path, must_exist=False)
    if _is_dataset_preset_readonly(normalized):
        raise ValueError("系统数据集预设为只读，请复制后编辑")
    path = _safe_resolve(normalized)
    if path is None:
        raise ValueError("数据集预设路径不合法")
    if path.exists() and not overwrite:
        raise ValueError("数据集预设已存在，请换一个名称")

    clean_rows = _fill_missing_dataset_row_settings(_normalize_dataset_rows(rows), _normalize_dataset_defaults(defaults or {}))
    if not clean_rows:
        raise ValueError("请至少填写一个数据集路径")
    _ensure_training_dataset_rows(clean_rows)
    cfg = _normalize_dataset_defaults(defaults or {})
    content = _build_dataset_config_doc(clean_rows, cfg)
    ok, msg = save_raw_file(normalized, content, overwrite=overwrite)
    if not ok:
        raise ValueError(msg)
    LOGGER.info(
        "saved dataset preset file=%s root=%s datasets=%d first_source=%s",
        normalized,
        ROOT.resolve(),
        len(clean_rows),
        clean_rows[0].get("source_dir") if clean_rows else "",
    )
    return {
        "ok": True,
        "message": f"已保存数据集预设 {Path(normalized).name}",
        "file": normalized,
        "datasets": clean_rows,
        "defaults": _normalize_dataset_defaults(cfg),
        "content": content,
        "summary": _dataset_summary_from_rows(clean_rows, _normalize_dataset_defaults(cfg)),
    }


def save_dataset_preset_as(
    name: str,
    rows: list[dict[str, Any]],
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stem = _safe_file_stem(name)
    return save_dataset_preset(f"configs/datasets/{stem}.toml", rows, defaults, overwrite=False)


def import_dataset_preset(
    name: str,
    content: str,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    _sync_from_facade()
    stem = _safe_file_stem(name)
    text = str(content or "")
    try:
        data = toml.loads(text)
    except toml.TomlDecodeError as exc:
        raise ValueError(f"导入失败，TOML 语法错误: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("导入失败，TOML 内容不合法")

    rows = _normalize_dataset_rows(_dataset_rows_from_config(data, data))
    if not rows or any(not str(row.get("source_dir") or "").strip() for row in rows):
        raise ValueError("导入失败，未找到可用的数据集路径")
    defaults = _dataset_defaults_from_config(data)
    return save_dataset_preset(f"configs/datasets/{stem}.toml", rows, defaults, overwrite=overwrite)


def delete_dataset_preset(rel_path: str) -> dict[str, Any]:
    _sync_from_facade()
    normalized = _normalize_dataset_preset_path(rel_path, must_exist=True)
    if _is_dataset_preset_readonly(normalized):
        raise ValueError("系统数据集预设为只读，不能删除")
    ok, msg = delete_raw_file(normalized)
    if not ok:
        raise ValueError(msg)
    return {"ok": True, "message": "数据集预设已删除", "file": normalized}


def apply_dataset_preset_to_training_config(
    dataset_file: str,
    train_file: str,
    train_content: str | None = None,
) -> dict[str, Any]:
    _sync_from_facade()
    dataset_rel = _normalize_dataset_preset_path(dataset_file, must_exist=True)
    train_rel = _normalize_config_rel_path(train_file)
    train_path = _safe_resolve(train_rel)
    if train_path is None or train_path.suffix.lower() != ".toml":
        raise ValueError("训练配置路径不合法")
    if not train_path.exists():
        raise ValueError("训练配置不存在")

    preset = load_dataset_preset(dataset_rel)
    rows = _normalize_dataset_rows(preset.get("datasets", []))
    if not rows:
        raise ValueError("数据集预设里没有可用路径")
    _ensure_training_dataset_rows(rows)
    first = _first_training_dataset_row(rows)
    defaults = _normalize_dataset_defaults(preset.get("defaults") or {})
    compatibility_defaults = _dataset_training_defaults(rows, defaults)
    values = {
        "dataset_config": dataset_rel,
        "source_image_dir": first["source_dir"],
        "resized_image_dir": first["image_dir"],
        "lora_cache_dir": first["cache_dir"],
        "prior_loss_weight": compatibility_defaults["prior_loss_weight"],
    }
    ok, msg, _path, next_content, changed = _prepare_raw_file_patch(train_rel, values, content=train_content)
    if not ok:
        raise ValueError(msg)
    ok, msg = save_raw_file(train_rel, next_content, overwrite=True)
    if not ok:
        raise ValueError(msg)
    return {
        "ok": True,
        "message": "已应用数据集预设",
        "dataset_config": dataset_rel,
        "datasets": rows,
        "defaults": defaults,
        "train_content": next_content,
        "changed": changed,
        "values": values,
        "summary": preset.get("summary") or _dataset_summary_from_rows(rows, defaults),
    }


def list_dataset_preset_images(
    dataset_file: str,
    dataset_index: int = 0,
    *,
    source: str = "training",
    limit: int = DATASET_PREVIEW_LIMIT,
) -> dict[str, Any]:
    _sync_from_facade()
    from web.services.config import datasets as _datasets

    _list_dataset_image_files = _datasets._list_dataset_image_files
    _dataset_image_preview_meta = _datasets._dataset_image_preview_meta
    _dataset_caption_detection_summary = _datasets._dataset_caption_detection_summary
    _caption_source_mode_label = _datasets._caption_source_mode_label
    _dataset_preview_empty_message = _datasets._dataset_preview_empty_message

    preset = load_dataset_preset(dataset_file)
    rows = _normalize_dataset_rows(preset.get("datasets", []))
    if not rows:
        raise ValueError("数据集预设里没有可预览路径")
    if dataset_index < 0 or dataset_index >= len(rows):
        raise ValueError("数据集序号不在范围内")

    row = rows[dataset_index]
    defaults = _normalize_dataset_defaults(preset.get("defaults") or {})
    settings = _normalize_dataset_defaults(row.get("settings") or defaults)
    caption_extension = str(settings.get("caption_extension") or ".txt").strip() or ".txt"
    if not caption_extension.startswith("."):
        caption_extension = f".{caption_extension}"
    prefer_json_caption = _bool_value(settings.get("prefer_json_caption"), False)
    caption_source_mode = normalize_caption_source_mode(
        settings.get("caption_source_mode"),
        prefer_json_caption,
    )
    source_kind = "source" if str(source or "").strip().lower() == "source" else "training"
    image_dir_raw = row.get("source_dir") if source_kind == "source" else row.get("image_dir")
    image_dir = _resolve_project_path(str(image_dir_raw or ""))
    source_dir = _resolve_project_path(str(row.get("source_dir") or ""))
    train_dir = _resolve_project_path(str(row.get("image_dir") or ""))
    recursive = _bool_value(row.get("recursive"), True)
    path_pattern = _normalize_path_pattern(row.get("path_pattern"))

    listing = _list_dataset_image_files(
        image_dir,
        limit,
        recursive=recursive,
        path_pattern=path_pattern,
    )
    images = [
        _dataset_image_preview_meta(
            path,
            preset_file=preset["file"],
            dataset_index=dataset_index,
            source=source_kind,
            caption_extension=caption_extension,
            prefer_json_caption=prefer_json_caption,
            caption_source_mode=caption_source_mode,
            source_dir=source_dir,
            train_dir=train_dir,
        )
        for path in listing["items"]
    ]
    caption_summary = _dataset_caption_detection_summary(images)
    directory_exists = image_dir.is_dir()
    return {
        "ok": True,
        "file": preset["file"],
        "dataset_index": dataset_index,
        "dataset_label": f"第 {dataset_index + 1} 组数据集",
        "source": source_kind,
        "source_label": "原始图目录" if source_kind == "source" else "训练图目录",
        "directory": _display_path(image_dir),
        "directory_exists": directory_exists,
        "caption_extension": caption_extension,
        "prefer_json_caption": prefer_json_caption,
        "caption_source_mode": caption_source_mode,
        "caption_source_label": _caption_source_mode_label(caption_source_mode),
        "caption_summary": caption_summary,
        "count": len(images),
        "total": listing["total"],
        "limit": listing["limit"],
        "images": images,
        "row": row,
        "settings": settings,
        "message": "" if images else _dataset_preview_empty_message(image_dir, source_kind),
    }


def resolve_dataset_preview_image(
    dataset_file: str,
    dataset_index: int,
    image_file: str,
    *,
    source: str = "training",
) -> Path:
    _sync_from_facade()
    preset = load_dataset_preset(dataset_file)
    rows = _normalize_dataset_rows(preset.get("datasets", []))
    if dataset_index < 0 or dataset_index >= len(rows):
        raise ValueError("数据集序号不在范围内")
    row = rows[dataset_index]
    source_kind = "source" if str(source or "").strip().lower() == "source" else "training"
    root = _resolve_project_path(str(row.get("source_dir") if source_kind == "source" else row.get("image_dir") or ""))
    if not root.is_dir():
        raise FileNotFoundError("数据集图片目录不存在")
    path = _resolve_project_path(str(image_file or ""))
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("图片不属于当前数据集路径") from exc
    if path.suffix.lower() not in DATASET_IMAGE_EXTS:
        raise ValueError("只允许读取数据集图片")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError("图片不存在")
    return path


