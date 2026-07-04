"""Dataset presets, dataset editor, and dataset runtime document helpers.

This module is loaded by ``web.services.config_service`` as part of the
compatibility facade.  It keeps facade access lazy so the module can also be
imported directly without pulling in the legacy facade.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any
from urllib.parse import quote

import toml
import tomlkit
from PIL import Image, UnidentifiedImageError

from library.env import expand_env_vars, expand_env_vars_in_obj, get_configs_root, load_dotenv
from library.preprocess._dataset import walk_images
from library.preprocess.captions import (
    normalize_caption_source_mode,
    read_caption_source,
    read_caption_source_from_dirs,
)
from web.services.config import paths as _config_paths
from web.services.config.metadata import (
    CAPTION_SOURCE_AUTO,
    CAPTION_SOURCE_CAPTIONS_JSON,
    CAPTION_SOURCE_JSON,
    CAPTION_SOURCE_MODE_LABELS,
    CAPTION_SOURCE_TXT,
    DATASET_CAPTION_MAX_CHARS,
    DATASET_IMAGE_EXTS,
    DATASET_PREVIEW_LIMIT,
    DATASET_SETTING_KEYS,
    DEFAULT_LORA_CACHE_DIR,
    DEFAULT_NL_TAG_MIX_TAG_RATIO,
    DEFAULT_RESIZED_IMAGE_DIR,
    HIDDEN_DATASET_PRESET_FILES,
    NL_TAG_MIX_ATTR_KEY,
    NL_TAG_MIX_CLASSIFICATION_METHOD,
    PREPROCESS_DATASET_SETTING_ORDER,
    RUNTIME_PREPROCESS_ATTR_KEY,
    SYSTEM_DATASET_PRESET_FILES,
    TRIGGER_CLONE_ATTR_KEY,
)
from web.services.settings_service import resolve_output_root

ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = get_configs_root()
GUI_METHODS_DIR = CONFIGS_DIR / "gui-methods"
IMPORTED_CONFIGS_DIR = CONFIGS_DIR / "imported"
PRESETS_FILE = CONFIGS_DIR / "presets.toml"
WEB_FILE_GROUPS_FILE = CONFIGS_DIR / "web-file-groups.toml"
WEB_USER_LOCKS_FILE = CONFIGS_DIR / "web-user-locks.toml"
DATASET_PRESETS_DIR = CONFIGS_DIR / "datasets"

load_dotenv()

LOGGER = logging.getLogger(__name__)


def _missing_facade_dependency(*args, **kwargs):
    raise RuntimeError("config facade dependency has not been synchronized")


save_raw_file = _missing_facade_dependency
load_raw_file = _missing_facade_dependency
delete_raw_file = _missing_facade_dependency
patch_raw_file_values = _missing_facade_dependency
preview_raw_file_patch = _missing_facade_dependency
get_config_file_meta = _missing_facade_dependency
list_config_file_groups = _missing_facade_dependency
move_config_file_to_group = _missing_facade_dependency
_inspect_network_weight = _missing_facade_dependency
load_merged_config = _missing_facade_dependency
apply_auto_data_dirs = _missing_facade_dependency
_prepare_raw_file_patch = _missing_facade_dependency
_load_training_config_for_web_run = _missing_facade_dependency


def _restore_dataset_config_after_failed_train_patch(path: Path, existed: bool, previous_content: str) -> None:
    if existed:
        path.write_text(previous_content, encoding="utf-8")
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _normalize_config_rel_path(rel_path: str) -> str:
    return _config_paths.normalize_config_rel_path(rel_path)


def _safe_resolve(rel_path: str) -> Path | None:
    return _config_paths.safe_resolve(rel_path, root=ROOT, configs_dir=CONFIGS_DIR)


def _safe_config_subdir(subdir: str) -> Path | None:
    return _config_paths.safe_config_subdir(subdir, configs_dir=CONFIGS_DIR)


def _resolve_project_path(value: str) -> Path:
    return _config_paths.resolve_display_path(
        value,
        root=ROOT,
        configs_dir=CONFIGS_DIR,
        expand_env_vars_fn=expand_env_vars,
    )


def _display_path(path: Path) -> str:
    return _config_paths.display_path(path, root=ROOT, configs_dir=CONFIGS_DIR)


def _derived_data_dir(source_path: Path, suffix: str) -> Path:
    parent = source_path.parent if source_path.name else source_path
    name = source_path.name or "dataset"
    return (parent / f"{name}_{suffix}").resolve()


def _positive_int(value: Any, fallback: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return fallback
    return n if n > 0 else fallback


def _nonnegative_int(value: Any, fallback: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return fallback
    return n if n >= 0 else fallback


def _nonnegative_float(value: Any, fallback: float) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return fallback
    return n if n >= 0 else fallback


def _bool_value(value: Any, fallback: bool = False) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_dataset_preset_path(rel_path: str, *, must_exist: bool) -> str:
    raw = str(rel_path or "").replace("\\", "/").strip()
    if not raw:
        raise ValueError("缺少数据集预设路径")
    path = Path(raw)
    if path.is_absolute():
        try:
            raw = path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError("数据集预设必须在项目目录内") from exc
        path = Path(raw)
    if ".." in path.parts:
        raise ValueError("数据集预设路径不能包含 ..")
    if path.suffix.lower() != ".toml":
        path = path.with_suffix(".toml")
    if len(path.parts) == 1:
        path = Path("configs") / "datasets" / path
    normalized = path.as_posix().lstrip("/")
    if not normalized.startswith("configs/datasets/"):
        raise ValueError("数据集预设必须保存在 configs/datasets/ 下")
    safe_path = _safe_resolve(normalized)
    if safe_path is None:
        raise ValueError("数据集预设路径不合法")
    if must_exist and not safe_path.exists():
        raise ValueError("数据集预设不存在")
    return normalized


def _is_dataset_preset_readonly(rel_path: str) -> bool:
    normalized = _normalize_config_rel_path(rel_path)
    if normalized in SYSTEM_DATASET_PRESET_FILES:
        return True
    try:
        return bool(get_config_file_meta(normalized).get("locked", False))
    except RuntimeError:
        return False


def _lock_reason_message(meta: dict[str, Any]) -> str:
    reason = str(meta.get("lock_reason") or meta.get("readonly_reason") or "").strip()
    return reason or "该配置由系统管理"

_SYNC_NAMES = (
    "ROOT",
    "CONFIGS_DIR",
    "GUI_METHODS_DIR",
    "IMPORTED_CONFIGS_DIR",
    "PRESETS_FILE",
    "WEB_FILE_GROUPS_FILE",
    "WEB_USER_LOCKS_FILE",
    "DATASET_PRESETS_DIR",
    "resolve_output_root",
    "_display_settings_path",
    "save_raw_file",
    "load_raw_file",
    "delete_raw_file",
    "patch_raw_file_values",
    "preview_raw_file_patch",
    "get_config_file_meta",
    "list_config_file_groups",
    "move_config_file_to_group",
    "_inspect_network_weight",
    "LOGGER",
)

_LEGACY_RAW_FILE_SHIM_NAMES = {
    "save_raw_file",
    "load_raw_file",
    "delete_raw_file",
    "patch_raw_file_values",
    "preview_raw_file_patch",
}
_LEGACY_SYNC_NAMES = tuple(
    _name for _name in _SYNC_NAMES
    if _name not in _LEGACY_RAW_FILE_SHIM_NAMES
)


def _sync_from_facade() -> None:
    from web.services import config_service as _facade

    _exported_names = set(globals().get("__all__", ()))
    _legacy_module = getattr(_facade, "_legacy", None)
    for _name in _SYNC_NAMES:
        if not hasattr(_facade, _name):
            continue
        _value = getattr(_facade, _name)
        if _name not in _exported_names:
            globals()[_name] = _value
        if _legacy_module is not None and _name in _LEGACY_SYNC_NAMES:
            setattr(_legacy_module, _name, _value)
    for _name in (
        "load_merged_config",
        "apply_auto_data_dirs",
        "_load_training_config_for_web_run",
        "_prepare_raw_file_patch",
        "_restore_dataset_config_after_failed_train_patch",
        "_is_dataset_preset_readonly",
        "_lock_reason_message",
    ):
        if hasattr(_facade, _name):
            globals()[_name] = getattr(_facade, _name)


def _exported(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        _sync_from_facade()
        return fn(*args, **kwargs)

    return wrapper

__all__ = ['list_dataset_presets', 'diagnose_dataset_presets', 'load_dataset_preset', 'save_dataset_preset', 'save_dataset_preset_as', 'import_dataset_preset', 'delete_dataset_preset', 'apply_dataset_preset_to_training_config', 'list_dataset_preset_images', 'resolve_dataset_preview_image', 'load_dataset_editor', 'save_dataset_editor', '_dataset_config_path_from_cfg', '_dataset_rows_for_estimate', '_dataset_rows_from_config', '_normalize_dataset_rows', '_normalize_dataset_defaults', '_normalize_nl_tag_mix', '_normalize_trigger_clone', '_normalize_path_pattern', '_build_dataset_config_doc', '_nl_tag_mix_caption_source', '_nl_tag_mix_image_files', '_classify_nl_tag_caption_text']

def list_dataset_presets() -> dict[str, Any]:
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
        if rel_path in HIDDEN_DATASET_PRESET_FILES:
            continue
        meta = grouped_meta_by_path.get(rel_path) or get_config_file_meta(rel_path)
        summary = _dataset_preset_summary(rel_path)
        readonly = bool(meta.get("locked")) or rel_path in SYSTEM_DATASET_PRESET_FILES
        presets_by_path[rel_path] = {
            **meta,
            "readonly": readonly,
            "system_preset": rel_path in SYSTEM_DATASET_PRESET_FILES,
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
            "hidden": rel in HIDDEN_DATASET_PRESET_FILES,
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


def load_dataset_editor(
    variant: str,
    preset: str,
    methods_subdir: str = "gui-methods",
    *,
    config_file: str | None = None,
    dataset_config: str | None = None,
) -> dict[str, Any]:
    cfg = _load_training_config_for_web_run(
        variant,
        preset,
        methods_subdir,
        config_file=config_file,
    )
    if dataset_config is not None:
        dataset_rel = _normalize_config_rel_path(str(dataset_config or ""))
        if dataset_rel:
            cfg["dataset_config"] = dataset_rel
        else:
            cfg.pop("dataset_config", None)
    dataset_path = _dataset_config_path_from_cfg(cfg)
    if dataset_path and dataset_path.exists():
        data = toml.loads(dataset_path.read_text(encoding="utf-8"))
    else:
        data = _single_dataset_config_from_cfg(cfg)
    rows = _dataset_rows_from_config(data, cfg)
    return {
        "ok": True,
        "dataset_config": _display_path(dataset_path) if dataset_path else "",
        "datasets": rows,
        "defaults": _dataset_defaults_from_config(data),
    }


def save_dataset_editor(
    variant: str,
    preset: str,
    methods_subdir: str,
    rows: list[dict[str, Any]],
    defaults: dict[str, Any] | None = None,
    config_values: dict[str, Any] | None = None,
    train_file: str | None = None,
    train_content: str | None = None,
    prefer_existing_dataset_config: bool = True,
) -> dict[str, Any]:
    cfg = apply_auto_data_dirs(load_merged_config(variant, preset, methods_subdir))
    if defaults:
        cfg.update(_normalize_dataset_defaults(defaults))
    clean_rows = _fill_missing_dataset_row_settings(_normalize_dataset_rows(rows), _normalize_dataset_defaults(cfg))
    if not clean_rows:
        raise ValueError("请至少填写一个数据集路径")
    _ensure_training_dataset_rows(clean_rows)

    train_rel = _normalize_config_rel_path(train_file) if train_file else _training_config_rel_path(variant, methods_subdir)
    dataset_variant = Path(train_rel).stem if train_rel else variant
    dataset_rel = _dataset_config_rel_path(
        cfg,
        dataset_variant,
        methods_subdir,
        prefer_existing=prefer_existing_dataset_config,
    )
    dataset_path = _safe_resolve(dataset_rel)
    if dataset_path is None:
        raise ValueError("数据集配置路径不合法")

    if train_rel and get_config_file_meta(train_rel).get("locked"):
        raise ValueError(f"{_lock_reason_message(get_config_file_meta(train_rel))}，请使用新名称保存新配置后编辑")

    next_content = ""
    if train_rel:
        first = _first_training_dataset_row(clean_rows)
        compatibility_defaults = _dataset_training_defaults(clean_rows, cfg)
        values = {
            "dataset_config": dataset_rel,
            "source_image_dir": first["source_dir"],
            "resized_image_dir": first["image_dir"],
            "lora_cache_dir": first["cache_dir"],
            "prior_loss_weight": compatibility_defaults["prior_loss_weight"],
        }
        ok, msg, _train_path, next_content, _changed = _prepare_raw_file_patch(train_rel, values, content=train_content)
        if not ok:
            raise ValueError(msg)

    doc_cfg = dict(cfg)
    if train_rel:
        try:
            doc_cfg.update(expand_env_vars_in_obj(toml.loads(next_content)))
        except toml.TomlDecodeError as exc:
            raise ValueError(f"训练配置 TOML 解析失败: {exc}") from exc
    elif train_content is not None:
        try:
            doc_cfg.update(expand_env_vars_in_obj(toml.loads(str(train_content or ""))))
        except toml.TomlDecodeError as exc:
            raise ValueError(f"训练配置 TOML 解析失败: {exc}") from exc
    if config_values:
        doc_cfg.update(expand_env_vars_in_obj(dict(config_values)))
    if defaults:
        doc_cfg.update(_normalize_dataset_defaults(defaults))
    doc_cfg = apply_auto_data_dirs(doc_cfg)

    dataset_doc = _build_dataset_config_doc(
        clean_rows,
        doc_cfg,
        prefer_train_batch_size=True,
    )
    dataset_existed = dataset_path.exists()
    previous_dataset_doc = dataset_path.read_text(encoding="utf-8") if dataset_existed else ""
    ok, msg = save_raw_file(dataset_rel, dataset_doc, overwrite=True)
    if not ok:
        raise ValueError(msg)
    if train_rel:
        ok, msg = save_raw_file(train_rel, next_content, overwrite=True)
        if not ok:
            _restore_dataset_config_after_failed_train_patch(dataset_path, dataset_existed, previous_dataset_doc)
            raise ValueError(msg)

    return {
        "ok": True,
        "message": f"已保存 {len(clean_rows)} 个数据集路径",
        "dataset_config": dataset_rel,
        "datasets": clean_rows,
        "defaults": _normalize_dataset_defaults(cfg),
        "train_content": next_content,
    }


def _dataset_config_path_from_cfg(cfg: dict[str, Any]) -> Path | None:
    rel_path = str(cfg.get("dataset_config") or "").strip()
    if not rel_path:
        return None
    path = _resolve_project_path(rel_path)
    if path.suffix.lower() != ".toml":
        return None
    if not _is_allowed_dataset_config_path(path):
        return None
    return path


def _is_allowed_dataset_config_path(path: Path) -> bool:
    resolved = path.resolve()
    for root in (ROOT.resolve(), CONFIGS_DIR.resolve(), resolve_output_root().resolve()):
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    if resolved.name == "dataset.runtime.toml":
        run_dir = resolved.parent
        return (
            resolved.is_file()
            and (run_dir / "config.runtime.toml").is_file()
            and (run_dir / "model_cache").is_dir()
            and (run_dir / "dataset_cache").is_dir()
            and (run_dir / "training_output").is_dir()
        )
    return False


def _dataset_config_rel_path(
    cfg: dict[str, Any],
    variant: str,
    methods_subdir: str,
    *,
    prefer_existing: bool = True,
) -> str:
    existing = str(cfg.get("dataset_config") or "").strip()
    if prefer_existing and existing:
        normalized = _normalize_config_rel_path(existing)
        path = _safe_resolve(normalized)
        if path is not None and normalized.startswith("configs/datasets/"):
            return normalized
    stem = _safe_file_stem(variant or methods_subdir or "dataset")
    return f"configs/datasets/{stem}.toml"


def _training_config_rel_path(variant: str, methods_subdir: str) -> str:
    methods_dir = _safe_config_subdir(methods_subdir)
    if methods_dir is None:
        return ""
    stem = _safe_file_stem(variant)
    path = methods_dir / f"{stem}.toml"
    if not path.exists():
        return ""
    return _display_path(path)


def _single_dataset_config_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    source_dir = str(cfg.get("source_image_dir") or "image_dataset")
    image_dir = str(cfg.get("resized_image_dir") or DEFAULT_RESIZED_IMAGE_DIR)
    cache_dir = str(cfg.get("lora_cache_dir") or DEFAULT_LORA_CACHE_DIR)
    return {
        "general": {
            "caption_extension": ".txt",
            "keep_tokens": 3,
            "prefer_json_caption": False,
            "caption_source_mode": CAPTION_SOURCE_AUTO,
        },
        "datasets": [
            {
                "resolution": 1024,
                "batch_size": 1,
                "enable_bucket": True,
                "min_bucket_reso": 256,
                "max_bucket_reso": 1024,
                "bucket_reso_steps": 64,
                "bucket_no_upscale": False,
                "validation_split": 0.0,
                "validation_seed": 42,
                "subsets": [
                    {
                        "image_dir": image_dir,
                        "cache_dir": cache_dir,
                        "num_repeats": 1,
                        "custom_attributes": {"source_dir": source_dir},
                    }
                ],
            }
        ],
    }


def _dataset_defaults_from_config(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "resolution": _positive_int(_first_dataset_value(data, "resolution"), 1024),
        "batch_size": _positive_int(_first_dataset_value(data, "batch_size"), 1),
        "prior_loss_weight": _nonnegative_float(_first_dataset_value(data, "prior_loss_weight", 1.0), 1.0),
        "enable_bucket": bool(_first_dataset_value(data, "enable_bucket", True)),
        "min_bucket_reso": _positive_int(_first_dataset_value(data, "min_bucket_reso"), 256),
        "max_bucket_reso": _positive_int(_first_dataset_value(data, "max_bucket_reso"), 1024),
        "bucket_reso_steps": _positive_int(_first_dataset_value(data, "bucket_reso_steps"), 64),
        "bucket_no_upscale": bool(_first_dataset_value(data, "bucket_no_upscale", False)),
        "validation_split": _nonnegative_float(_first_dataset_value(data, "validation_split", 0.0), 0.0),
        "validation_split_num": _nonnegative_int(_first_dataset_value(data, "validation_split_num", 0), 0),
        "validation_seed": _nonnegative_int(_first_dataset_value(data, "validation_seed", 42), 42),
        "caption_extension": str(_first_dataset_value(
            data,
            "caption_extension",
            (data.get("general") or {}).get("caption_extension") or ".txt",
        )),
        "keep_tokens": _positive_int((data.get("general") or {}).get("keep_tokens"), 3),
        "prefer_json_caption": _bool_value(_first_dataset_value(data, "prefer_json_caption"), False),
        "caption_source_mode": normalize_caption_source_mode(
            _first_dataset_value(data, "caption_source_mode"),
            _bool_value(_first_dataset_value(data, "prefer_json_caption"), False),
        ),
    }


def _dataset_defaults_from_dataset(dataset: dict[str, Any], data: dict[str, Any] | None = None) -> dict[str, Any]:
    source: dict[str, Any] = {"datasets": [dataset]}
    if isinstance(data, dict) and isinstance(data.get("general"), dict):
        source["general"] = data["general"]
    return _dataset_defaults_from_config(source)


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


def _dataset_summary_from_rows(rows: list[dict[str, Any]], defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    clean_rows = _normalize_dataset_rows(rows)
    clean_defaults = _normalize_dataset_defaults(defaults or _first_dataset_settings(clean_rows))
    first = _first_training_dataset_row(clean_rows) if clean_rows else {}
    repeats = sum(_positive_int(row.get("num_repeats"), 1) for row in clean_rows) if clean_rows else 0
    reg_rows = [row for row in clean_rows if _bool_value(row.get("is_reg"), False)]
    train_rows = [row for row in clean_rows if not _bool_value(row.get("is_reg"), False)]
    return {
        "ok": True,
        "dataset_count": len(clean_rows),
        "train_dataset_count": len(train_rows),
        "reg_dataset_count": len(reg_rows),
        "repeat_total": repeats,
        "reg_repeat_total": sum(_positive_int(row.get("num_repeats"), 1) for row in reg_rows),
        "source_dir": first.get("source_dir", ""),
        "image_dir": first.get("image_dir", ""),
        "cache_dir": first.get("cache_dir", ""),
        "resolution": clean_defaults.get("resolution", 1024),
        "batch_size": clean_defaults.get("batch_size", 1),
        "enable_bucket": clean_defaults.get("enable_bucket", True),
        "prior_loss_weight": clean_defaults.get("prior_loss_weight", 1.0),
    }


def _first_training_dataset_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        if not _bool_value(row.get("is_reg"), False):
            return row
    return rows[0] if rows else {}


def _ensure_training_dataset_rows(rows: list[dict[str, Any]]) -> None:
    if rows and not any(not _bool_value(row.get("is_reg"), False) for row in rows):
        raise ValueError("至少需要一组普通训练数据集，正则化数据集只能作为辅助保留集")


def _dataset_rows_for_estimate(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_path = _dataset_config_path_from_cfg(cfg)
    if dataset_path and dataset_path.exists():
        try:
            data = toml.loads(dataset_path.read_text(encoding="utf-8"))
        except toml.TomlDecodeError:
            data = _single_dataset_config_from_cfg(cfg)
    else:
        data = _single_dataset_config_from_cfg(cfg)
    return _dataset_rows_from_config(data, cfg)


def _dataset_rows_from_config(data: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    datasets = data.get("datasets") if isinstance(data, dict) else []
    if not isinstance(datasets, list):
        datasets = []

    fallback_source = str(cfg.get("source_image_dir") or "")
    fallback_image = str(cfg.get("resized_image_dir") or fallback_source)
    fallback_cache = str(cfg.get("lora_cache_dir") or "")
    fallback_path_pattern = cfg.get("path_pattern")

    for dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        subsets = dataset.get("subsets") or []
        if not isinstance(subsets, list):
            continue
        for subset in subsets:
            if not isinstance(subset, dict):
                continue
            attrs = subset.get("custom_attributes")
            if not isinstance(attrs, dict):
                attrs = {}
            image_dir = _dataset_path_value(subset.get("image_dir") or fallback_image, cfg)
            cache_dir = _dataset_path_value(subset.get("cache_dir") or fallback_cache, cfg)
            source_dir = _dataset_path_value(attrs.get("source_dir") or fallback_source or image_dir, cfg)
            settings = _dataset_defaults_from_dataset(dataset, data)
            settings.update(_preprocess_settings_from_custom_attributes(attrs))
            rows.append({
                "source_dir": source_dir,
                "image_dir": image_dir,
                "cache_dir": cache_dir,
                "num_repeats": _positive_int(subset.get("num_repeats"), 1),
                "is_reg": _bool_value(subset.get("is_reg"), False),
                "recursive": _bool_value(subset.get("recursive", dataset.get("recursive")), True),
                "path_pattern": _normalize_path_pattern(
                    subset.get("path_pattern", dataset.get("path_pattern", fallback_path_pattern))
                ),
                "nl_tag_mix": _normalize_nl_tag_mix(attrs.get(NL_TAG_MIX_ATTR_KEY)),
                "trigger_clone": _normalize_trigger_clone(attrs.get(TRIGGER_CLONE_ATTR_KEY)),
                "settings": settings,
            })

    if not rows:
        rows = _normalize_dataset_rows([
            {
                "source_dir": fallback_source,
                "image_dir": fallback_image,
                "cache_dir": fallback_cache,
                "num_repeats": 1,
            }
        ])
    return rows


def _normalize_dataset_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_rows: list[dict[str, Any]] = []
    for raw in rows or []:
        if not isinstance(raw, dict):
            continue
        source = str(raw.get("source_dir") or raw.get("source_image_dir") or "").strip()
        image = str(raw.get("image_dir") or raw.get("resized_image_dir") or "").strip()
        cache = str(raw.get("cache_dir") or raw.get("lora_cache_dir") or "").strip()
        if not source and not image and not cache:
            continue
        if not source:
            source = image
        source_path = _resolve_project_path(source)
        image_path = _resolve_project_path(image) if image else _derived_data_dir(source_path, "resized")
        cache_path = _resolve_project_path(cache) if cache else _derived_data_dir(source_path, "lora_cache")
        clean_rows.append({
            "source_dir": _display_path(source_path),
            "image_dir": _display_path(image_path),
            "cache_dir": _display_path(cache_path),
            "num_repeats": _positive_int(raw.get("num_repeats"), 1),
            "is_reg": _bool_value(raw.get("is_reg"), False),
            "recursive": _bool_value(raw.get("recursive"), True),
            "path_pattern": _normalize_path_pattern(raw.get("path_pattern")),
            "nl_tag_mix": _normalize_nl_tag_mix(raw.get(NL_TAG_MIX_ATTR_KEY) or raw.get("nl_tag_mix")),
            "trigger_clone": _normalize_trigger_clone(
                raw.get(TRIGGER_CLONE_ATTR_KEY) or raw.get("trigger_clone")
            ),
            "settings": _normalize_dataset_row_settings(raw),
        })
    return clean_rows


def _normalize_dataset_row_settings(raw: dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw.get("settings"), dict):
        settings = dict(raw["settings"])
        for key in DATASET_SETTING_KEYS:
            if key in raw and key not in settings:
                settings[key] = raw[key]
        return settings
    if any(key in raw for key in DATASET_SETTING_KEYS):
        return {key: raw[key] for key in DATASET_SETTING_KEYS if key in raw}
    return {}


def _fill_missing_dataset_row_settings(rows: list[dict[str, Any]], defaults: dict[str, Any]) -> list[dict[str, Any]]:
    fallback = _normalize_dataset_defaults(defaults)
    next_rows: list[dict[str, Any]] = []
    for row in rows:
        next_row = dict(row)
        settings = next_row.get("settings")
        if isinstance(settings, dict) and settings:
            merged = dict(fallback)
            merged.update(settings)
            next_row["settings"] = _normalize_dataset_defaults(merged)
        else:
            next_row["settings"] = fallback
        next_rows.append(next_row)
    return next_rows


def _normalize_dataset_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    out["resolution"] = _positive_int(raw.get("resolution"), 1024)
    out["batch_size"] = _positive_int(raw.get("batch_size"), 1)
    out["prior_loss_weight"] = _nonnegative_float(raw.get("prior_loss_weight"), 1.0)
    out["enable_bucket"] = str(raw.get("enable_bucket", True)).lower() not in {"0", "false", "no", "off"}
    out["min_bucket_reso"] = _positive_int(raw.get("min_bucket_reso"), 256)
    out["max_bucket_reso"] = _positive_int(raw.get("max_bucket_reso"), 1024)
    out["bucket_reso_steps"] = _positive_int(raw.get("bucket_reso_steps"), 64)
    out["bucket_no_upscale"] = str(raw.get("bucket_no_upscale", False)).lower() in {"1", "true", "yes", "on"}
    if raw.get("validation_split_num") not in (None, ""):
        out["validation_split_num"] = _nonnegative_int(raw.get("validation_split_num"), 0)
    out["validation_split"] = _nonnegative_float(raw.get("validation_split"), 0.0)
    out["validation_seed"] = _nonnegative_int(raw.get("validation_seed"), 42)
    out["caption_extension"] = str(raw.get("caption_extension") or ".txt").strip() or ".txt"
    out["keep_tokens"] = _positive_int(raw.get("keep_tokens"), 3)
    prefer_json = _bool_value(raw.get("prefer_json_caption"), False)
    out["prefer_json_caption"] = prefer_json
    out["caption_source_mode"] = normalize_caption_source_mode(
        raw.get("caption_source_mode"),
        prefer_json,
    )
    return out


def _normalize_preprocess_dataset_settings(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    if "resolution" in raw:
        out["resolution"] = _positive_int(raw.get("resolution"), 1024)
    if "enable_bucket" in raw:
        out["enable_bucket"] = str(raw.get("enable_bucket", True)).lower() not in {"0", "false", "no", "off"}
    if "min_bucket_reso" in raw:
        out["min_bucket_reso"] = _positive_int(raw.get("min_bucket_reso"), 256)
    if "max_bucket_reso" in raw:
        out["max_bucket_reso"] = _positive_int(raw.get("max_bucket_reso"), 1024)
    if "bucket_reso_steps" in raw:
        out["bucket_reso_steps"] = _positive_int(raw.get("bucket_reso_steps"), 64)
    if "bucket_no_upscale" in raw:
        out["bucket_no_upscale"] = str(raw.get("bucket_no_upscale", False)).lower() in {"1", "true", "yes", "on"}
    return out


def _normalize_nl_tag_mix(raw: Any) -> dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}
    enabled = _bool_value(source.get("enabled"), False)
    try:
        tag_ratio = float(source.get("tag_ratio", DEFAULT_NL_TAG_MIX_TAG_RATIO))
    except (TypeError, ValueError):
        tag_ratio = DEFAULT_NL_TAG_MIX_TAG_RATIO
    if tag_ratio > 1:
        tag_ratio = tag_ratio / 100
    tag_ratio = min(1.0, max(0.0, tag_ratio))
    return {
        "enabled": enabled,
        "tag_ratio": tag_ratio,
    }


def _normalize_trigger_clone(raw: Any) -> dict[str, Any]:
    source = raw if isinstance(raw, dict) else {}
    return {
        "enabled": _bool_value(source.get("enabled"), False),
        "prompt": str(source.get("prompt") or "").strip(),
        "num_repeats": _positive_int(source.get("num_repeats"), 1),
    }


def _trigger_clone_should_persist(clone: dict[str, Any]) -> bool:
    normalized = _normalize_trigger_clone(clone)
    return (
        bool(normalized["enabled"])
        or bool(normalized["prompt"])
        or _positive_int(normalized.get("num_repeats"), 1) != 1
    )


def _nl_tag_mix_enabled(row: dict[str, Any]) -> bool:
    return bool(_normalize_nl_tag_mix(row.get("nl_tag_mix")).get("enabled"))


def _preprocess_settings_from_custom_attributes(attrs: dict[str, Any]) -> dict[str, Any]:
    raw = attrs.get(RUNTIME_PREPROCESS_ATTR_KEY) if isinstance(attrs, dict) else None
    return _normalize_preprocess_dataset_settings(raw) if isinstance(raw, dict) else {}


def _preprocess_settings_for_runtime_attrs(row_cfg: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_dataset_defaults(row_cfg)
    return {key: normalized[key] for key in PREPROCESS_DATASET_SETTING_ORDER if key in normalized}


def _build_dataset_config_doc(
    clean_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    prefer_train_batch_size: bool = False,
    include_preprocess_settings: bool = True,
) -> str:
    doc = tomlkit.document()
    doc.add(tomlkit.comment("Web UI 自动生成的数据集配置。"))
    doc.add(tomlkit.comment("原始数据集路径保存在 custom_attributes.source_dir，训练读取 image_dir/cache_dir。"))

    general = tomlkit.table()
    general.add("caption_extension", str(cfg.get("caption_extension") or ".txt"))
    general.add("keep_tokens", _positive_int(cfg.get("keep_tokens"), 3))
    doc.add("general", general)

    datasets = tomlkit.aot()
    for row in clean_rows:
        row_cfg = _dataset_row_settings(row, cfg)
        dataset = tomlkit.table()
        if include_preprocess_settings:
            dataset.add("resolution", _positive_int(row_cfg.get("resolution"), 1024))
        batch_size = row_cfg.get("batch_size")
        if prefer_train_batch_size and cfg.get("train_batch_size") not in (None, ""):
            batch_size = cfg.get("train_batch_size")
        dataset.add("batch_size", _positive_int(batch_size, 1))
        dataset.add("prior_loss_weight", _nonnegative_float(row_cfg.get("prior_loss_weight"), 1.0))
        caption_source_mode = normalize_caption_source_mode(
            row_cfg.get("caption_source_mode"),
            _bool_value(row_cfg.get("prefer_json_caption"), False),
        )
        dataset.add("caption_source_mode", caption_source_mode)
        dataset.add("caption_extension", str(row_cfg.get("caption_extension") or cfg.get("caption_extension") or ".txt"))
        dataset.add(
            "prefer_json_caption",
            caption_source_mode == CAPTION_SOURCE_JSON
            or _bool_value(row_cfg.get("prefer_json_caption"), False),
        )
        if include_preprocess_settings:
            dataset.add("enable_bucket", bool(row_cfg.get("enable_bucket", True)))
            dataset.add("min_bucket_reso", _positive_int(row_cfg.get("min_bucket_reso"), 256))
            dataset.add("max_bucket_reso", _positive_int(row_cfg.get("max_bucket_reso"), 1024))
            dataset.add("bucket_reso_steps", _positive_int(row_cfg.get("bucket_reso_steps"), 64))
            dataset.add("bucket_no_upscale", bool(row_cfg.get("bucket_no_upscale", False)))
        validation_split_num = _nonnegative_int(row_cfg.get("validation_split_num"), 0)
        if validation_split_num > 0:
            dataset.add("validation_split_num", validation_split_num)
        dataset.add("validation_split", _nonnegative_float(row_cfg.get("validation_split"), 0.0))
        dataset.add("validation_seed", _nonnegative_int(row_cfg.get("validation_seed"), 42))

        subsets = tomlkit.aot()
        subset = tomlkit.table()
        subset.add("image_dir", row["image_dir"])
        subset.add("cache_dir", row["cache_dir"])
        subset.add("num_repeats", _positive_int(row.get("num_repeats"), 1))
        if _bool_value(row.get("is_reg"), False):
            subset.add("is_reg", True)
        if not _bool_value(row.get("recursive"), True):
            subset.add("recursive", False)
        path_pattern = _normalize_path_pattern(row.get("path_pattern"))
        if path_pattern != "*":
            subset.add("path_pattern", path_pattern)
        attrs = tomlkit.inline_table()
        attrs.add("source_dir", row["source_dir"])
        mix = _normalize_nl_tag_mix(row.get("nl_tag_mix"))
        if mix["enabled"]:
            mix_attrs = tomlkit.inline_table()
            mix_attrs.add("enabled", True)
            mix_attrs.add("tag_ratio", mix["tag_ratio"])
            attrs.add(NL_TAG_MIX_ATTR_KEY, mix_attrs)
        trigger_clone = _normalize_trigger_clone(row.get("trigger_clone"))
        if _trigger_clone_should_persist(trigger_clone):
            clone_attrs = tomlkit.inline_table()
            clone_attrs.add("enabled", bool(trigger_clone["enabled"]))
            clone_attrs.add("prompt", trigger_clone["prompt"])
            clone_attrs.add("num_repeats", _positive_int(trigger_clone.get("num_repeats"), 1))
            attrs.add(TRIGGER_CLONE_ATTR_KEY, clone_attrs)
        if not include_preprocess_settings:
            preprocess_attrs = tomlkit.inline_table()
            for key, value in _preprocess_settings_for_runtime_attrs(row_cfg).items():
                preprocess_attrs.add(key, value)
            attrs.add(RUNTIME_PREPROCESS_ATTR_KEY, preprocess_attrs)
        subset.add("custom_attributes", attrs)
        subsets.append(subset)
        dataset.add("subsets", subsets)
        datasets.append(dataset)
    doc.add("datasets", datasets)
    return tomlkit.dumps(doc)


def _dataset_row_settings(row: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("settings")
    if isinstance(raw, dict):
        return _normalize_dataset_defaults(raw)
    return _normalize_dataset_defaults(fallback)


def _dataset_training_defaults(rows: list[dict[str, Any]], fallback: dict[str, Any]) -> dict[str, Any]:
    defaults = _normalize_dataset_defaults(fallback)
    for row in rows:
        if not _bool_value(row.get("is_reg"), False):
            continue
        settings = row.get("settings")
        if isinstance(settings, dict) and "prior_loss_weight" in settings:
            defaults["prior_loss_weight"] = _nonnegative_float(settings.get("prior_loss_weight"), defaults["prior_loss_weight"])
            break
    return defaults


def _first_dataset_settings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if rows and isinstance(rows[0].get("settings"), dict):
        return rows[0]["settings"]
    return {}


def _first_dataset_value(data: dict[str, Any], key: str, default: Any = None) -> Any:
    datasets = data.get("datasets") if isinstance(data, dict) else []
    if isinstance(datasets, list) and datasets and isinstance(datasets[0], dict):
        if key in datasets[0]:
            return datasets[0].get(key)
    general = data.get("general") if isinstance(data, dict) else {}
    if isinstance(general, dict) and key in general:
        return general.get(key)
    return default


def _dataset_path_value(value: Any, cfg: dict[str, Any]) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for key, raw in cfg.items():
        if isinstance(raw, str):
            text = text.replace("{" + key + "}", raw)
    return _display_path(_resolve_project_path(expand_env_vars(text)))


def _list_dataset_image_files(
    directory: Path,
    limit: int,
    *,
    recursive: bool = True,
    path_pattern: str = "*",
) -> dict[str, Any]:
    clean_limit = max(1, min(_positive_int(limit, DATASET_PREVIEW_LIMIT), DATASET_PREVIEW_LIMIT))
    items = _dataset_image_files(
        directory,
        DATASET_IMAGE_EXTS,
        recursive=recursive,
        path_pattern=path_pattern,
    )
    return {"items": items[:clean_limit], "total": len(items), "limit": clean_limit}


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
    return {
        "file": rel_path,
        "name": path.name,
        "url": url,
        "mtime": stat.st_mtime,
        "mtime_text": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "size_bytes": stat.st_size,
        "width": dimensions.get("width"),
        "height": dimensions.get("height"),
        "total_pixels": dimensions.get("total_pixels"),
        "caption": caption,
    }


def _dataset_image_dimensions(path: Path) -> dict[str, int]:
    try:
        with Image.open(path) as image:
            width, height = image.size
    except (OSError, UnidentifiedImageError):
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


def _safe_file_stem(value: str) -> str:
    stem = Path(str(value or "").replace("\\", "/")).stem
    chars: list[str] = []
    for ch in stem:
        if ch.isascii() and (ch.isalnum() or ch in {"-", "_"}):
            chars.append(ch)
        elif ch.isspace():
            chars.append("_")
    return "".join(chars).strip("_-") or "dataset"


def _normalize_path_pattern(value: Any) -> str:
    return str(value or "*").strip() or "*"


def _dataset_image_files(
    path: Path,
    image_exts: set[str] | frozenset[str],
    *,
    recursive: bool = True,
    path_pattern: str = "*",
) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        item
        for item in walk_images(
            path,
            recursive=recursive,
            pattern=_normalize_path_pattern(path_pattern),
        )
        if item.suffix.lower() in image_exts
    )


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




for _name in __all__:
    globals()[_name] = _exported(globals()[_name])
