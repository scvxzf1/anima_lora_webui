"""Metadata inference helpers for config file groups."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from web.services.config.file_group_locks import (
    _is_system_locked_path,
    _is_system_preset_path,
    _is_user_group_locked,
    _is_user_locked,
    _lock_reason_label,
)
from web.services.config.file_group_paths import _normalize_config_rel_path, _strip_configs_prefix
from web.services.config.file_group_specs import _group_defaults
from web.services.config.metadata import CONFIG_FILE_LABELS_ZH, SYSTEM_DATASET_PRESET_FILES


def _list_config_file_groups_lazy(kind: str | None = None) -> list[dict[str, Any]]:
    """Lazy import to avoid core/ops circular dependency."""
    from web.services.config import file_group_ops as _ops

    return _ops.list_config_file_groups(kind=kind)


def get_config_file_meta(
    rel_path: str,
    group_id: str | None = None,
    group_label: str | None = None,
    locked: bool | None = None,
    trainable: bool | None = None,
    methods_subdir: str | None = None,
) -> dict[str, Any]:
    normalized = _normalize_config_rel_path(rel_path)
    inferred = (
        {
            "id": group_id,
            "label": group_label,
            "locked": locked,
            "open": True,
            "trainable": bool(trainable),
            "methods_subdir": methods_subdir or "",
        }
        if group_id and group_label and locked is not None and trainable is not None
        else _infer_config_file_group(normalized)
    )
    method = _config_method_name_for_path(normalized)
    group_locked = bool(inferred["locked"] if locked is None else locked)
    system_locked = _is_system_locked_path(normalized)
    user_locked = _is_user_locked(normalized)
    user_group_locked = _is_user_group_locked(group_id or inferred["id"])
    effective_locked = system_locked or user_locked or user_group_locked or group_locked
    lock_reason = ""
    if system_locked:
        lock_reason = "system"
    elif user_locked:
        lock_reason = "user"
    elif user_group_locked:
        lock_reason = "user_group"
    elif group_locked:
        lock_reason = "group"
    return {
        "path": normalized,
        "label": CONFIG_FILE_LABELS_ZH.get(normalized, Path(normalized).name),
        "filename": Path(normalized).name,
        "group": group_id or inferred["id"],
        "group_label": group_label or inferred["label"],
        "locked": effective_locked,
        "group_locked": group_locked,
        "user_group_locked": user_group_locked,
        "system_locked": system_locked,
        "user_locked": user_locked,
        "lock_reason": lock_reason,
        "lock_reason_label": _lock_reason_label(lock_reason),
        "restorable": _is_system_preset_path(normalized),
        "open": inferred["open"],
        "trainable": inferred["trainable"] if trainable is None else trainable,
        "method": method,
        "methods_subdir": methods_subdir or inferred["methods_subdir"],
    }


def _get_config_file_group(group_id: str) -> dict[str, Any] | None:
    from web.services.config.file_group_paths import _normalize_group_id

    normalized = _normalize_group_id(group_id)
    for group in _list_config_file_groups_lazy():
        if group.get("id") == normalized:
            return group
    return None


def _config_method_name_for_path(rel_path: str) -> str:
    normalized = _normalize_config_rel_path(rel_path)
    for prefix in ("configs/gui-methods/", "configs/methods/", "configs/imported/"):
        if not normalized.startswith(prefix):
            continue
        relative = normalized.removeprefix(prefix)
        if relative.lower().endswith(".toml"):
            relative = relative[:-5]
        return relative.strip("/")
    return Path(normalized).stem


def _infer_config_file_group(rel_path: str) -> dict[str, Any]:
    for group in _list_config_file_groups_lazy():
        for item in group["files"]:
            if item["path"] == rel_path:
                return {
                    "id": group["id"],
                    "label": group["label"],
                    "locked": group["locked"],
                    "open": group["open"],
                    "trainable": group["trainable"],
                    "methods_subdir": group["methods_subdir"],
                }
    normalized = _strip_configs_prefix(rel_path)
    if normalized.startswith("gui-methods/"):
        return _group_defaults("gui_methods", "可训练方法变体", False, True, "gui-methods", True)
    if normalized.startswith("methods/"):
        return _group_defaults("methods", "系统内置方法配置（锁定只读）", True, True, "methods", False)
    if normalized.startswith("imported/"):
        return _group_defaults("imported", "导入配置", False, True, "imported", True)
    if normalized.startswith("datasets/"):
        return _group_defaults("datasets", "数据集配置", False, False, "", False)
    if normalized in {"base.toml", "presets.toml"}:
        return _group_defaults("presets", "系统预设配置（锁定只读）", True, False, "", False)
    return _group_defaults("custom", "自定义配置", False, False, "", True)


def _is_dataset_preset_readonly(rel_path: str) -> bool:
    normalized = _normalize_config_rel_path(rel_path)
    return normalized in SYSTEM_DATASET_PRESET_FILES or get_config_file_meta(normalized).get("locked", False)
