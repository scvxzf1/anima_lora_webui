"""Spec load/save/build helpers for config file groups."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tomlkit

from web.services.atomic_io import atomic_write_text
from web.services.config.file_group_locks import _is_user_group_locked
from web.services.config.file_group_paths import (
    _config_group_path_list,
    _normalize_config_rel_path,
    _normalize_group_id,
    _string_list,
    _strip_configs_prefix,
)
from web.services.config.file_group_runtime import (
    CONFIGS_DIR,
    WEB_FILE_GROUPS_FILE,
    _display_path,
    _load,
    _owner_attr,
    _safe_resolve,
)
from web.services.config.metadata import (
    FILE_MOVE_TARGET_GROUPS,
    FIXED_SYSTEM_CONFIG_GROUP_IDS,
    HIDDEN_CONFIG_FILES,
    SYSTEM_CONFIG_GROUP_IDS,
    USER_LOCKABLE_GROUPS,
)


def _normalize_config_file_group_kind_filter(kind: str | None) -> str:
    clean = str(kind or "all").strip().lower()
    if clean in {"", "all", "*"}:
        return "all"
    if clean in {"training", "config", "configs"}:
        return "training"
    if clean in {"dataset", "datasets"}:
        return "dataset"
    raise ValueError("kind 参数只支持 training、dataset 或 all")


def _config_group_kind(raw: dict[str, Any]) -> str:
    explicit = str(raw.get("kind") or "").strip().lower()
    if explicit in {"dataset", "datasets"}:
        return "dataset"
    if explicit in {"training", "config", "configs"}:
        return "training"

    group_id = str(raw.get("id") or "").strip()
    paths = [*_string_list(raw.get("files")), *_string_list(raw.get("patterns"))]
    if group_id in {"datasets", "unfiled_datasets"}:
        return "dataset"
    if any(_strip_configs_prefix(str(item).replace("\\", "/")).startswith("datasets/") for item in paths):
        return "dataset"
    return "training"


def _load_config_file_group_specs() -> list[dict[str, Any]]:
    data = _load(_owner_attr("WEB_FILE_GROUPS_FILE", WEB_FILE_GROUPS_FILE))
    groups = data.get("groups")
    if not isinstance(groups, list):
        groups = _default_config_file_group_specs()
    specs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in groups:
        if not isinstance(raw, dict):
            continue
        group_id = str(raw.get("id") or "").strip()
        if not group_id or group_id in seen_ids:
            continue
        seen_ids.add(group_id)
        specs.append({
            "id": group_id,
            "label": str(raw.get("label") or group_id),
            "open": bool(raw.get("open", True)),
            "locked": bool(raw.get("locked", False)),
            "trainable": bool(raw.get("trainable", False)),
            "methods_subdir": str(raw.get("methods_subdir") or ""),
            "kind": _config_group_kind(raw),
            "user_managed": bool(raw.get("user_managed", False)),
            "files": _config_group_path_list(raw.get("files")),
            "order": _config_group_path_list(raw.get("order")),
            "patterns": _config_group_path_list(raw.get("patterns")),
            "exclude": set(_config_group_path_list(raw.get("exclude"))),
        })
    return specs


def _sort_config_file_group_specs_for_display(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item
        for _idx, item in sorted(
            enumerate(specs),
            key=lambda pair: (1 if _is_fixed_config_group(pair[1]) else 0, pair[0]),
        )
    ]


def _save_config_file_group_specs(specs: list[dict[str, Any]]) -> None:
    doc = tomlkit.document()
    doc.add(tomlkit.comment("Web UI 配置文件管理注册表，由 WebUI 自动维护。"))
    doc.add(tomlkit.comment("系统分组请谨慎修改；user_managed=true 的分组可在 WebUI 中重命名/删除。"))
    group_array = tomlkit.aot()
    for spec in specs:
        table = tomlkit.table()
        table.add("id", spec["id"])
        table.add("label", spec["label"])
        table.add("open", bool(spec.get("open", True)))
        table.add("locked", bool(spec.get("locked", False)))
        table.add("trainable", bool(spec.get("trainable", False)))
        if spec.get("kind") and spec.get("kind") != "training":
            table.add("kind", str(spec.get("kind") or ""))
        if spec.get("methods_subdir"):
            table.add("methods_subdir", str(spec.get("methods_subdir") or ""))
        if spec.get("user_managed"):
            table.add("user_managed", True)
        if spec.get("files"):
            table.add("files", _config_group_path_list(spec.get("files")))
        if spec.get("order"):
            table.add("order", _config_group_path_list(spec.get("order")))
        if spec.get("patterns"):
            table.add("patterns", _config_group_path_list(spec.get("patterns")))
        if spec.get("exclude"):
            table.add("exclude", _config_group_path_list(spec.get("exclude")))
        group_array.append(table)
    doc.add("groups", group_array)
    specs_file = _owner_attr("WEB_FILE_GROUPS_FILE", WEB_FILE_GROUPS_FILE)
    atomic_write_text(specs_file, tomlkit.dumps(doc))


def _glob_config_files(pattern: str) -> list[str]:
    normalized_pattern = _normalize_config_rel_path(pattern)
    if not normalized_pattern.startswith("configs/") or ".." in Path(normalized_pattern).parts:
        return []
    rel_pattern = normalized_pattern.removeprefix("configs/")
    return [
        _display_path(path)
        for path in sorted(_owner_attr("CONFIGS_DIR", CONFIGS_DIR).glob(rel_pattern))
        if path.is_file()
        and path.suffix == ".toml"
        and _safe_resolve(_display_path(path))
        and _display_path(path) not in HIDDEN_CONFIG_FILES
    ]


def _default_config_file_group_specs() -> list[dict[str, Any]]:
    return [
        {"id": "gui_methods", "label": "可训练方法变体", "open": True, "locked": False, "trainable": True, "methods_subdir": "gui-methods", "patterns": ["configs/gui-methods/*.toml"]},
        {"id": "imported", "label": "导入配置", "open": True, "locked": False, "trainable": True, "methods_subdir": "imported", "patterns": ["configs/imported/*.toml"]},
        {"id": "datasets", "label": "数据集配置", "open": False, "locked": False, "trainable": False, "patterns": ["configs/datasets/*.toml"]},
        {"id": "presets", "label": "系统预设配置（锁定只读）", "open": False, "locked": True, "trainable": False, "files": ["configs/base.toml", "configs/presets.toml"]},
    ]


def _group_defaults(
    group_id: str,
    label: str,
    locked: bool,
    trainable: bool,
    methods_subdir: str,
    open_by_default: bool,
) -> dict[str, Any]:
    return {
        "id": group_id,
        "label": label,
        "locked": locked,
        "open": open_by_default,
        "trainable": trainable,
        "methods_subdir": methods_subdir,
    }


def _find_config_group_spec(specs: list[dict[str, Any]], group_id: str) -> dict[str, Any] | None:
    normalized = _normalize_group_id(group_id)
    for spec in specs:
        if spec.get("id") == normalized:
            return spec
    return None


def _new_user_config_group_spec(group_id: str, label: str, kind: str = "training") -> dict[str, Any]:
    clean_kind = str(kind or "").strip().lower()
    if clean_kind in {"dataset", "datasets"}:
        return {
            "id": group_id,
            "label": label,
            "open": False,
            "locked": False,
            "trainable": False,
            "methods_subdir": "",
            "kind": "dataset",
            "user_managed": True,
            "files": [],
            "order": [],
            "patterns": [],
            "exclude": set(),
        }
    return {
        "id": group_id,
        "label": label,
        "open": True,
        "locked": False,
        "trainable": True,
        "methods_subdir": "imported",
        "kind": "training",
        "user_managed": True,
        "files": [],
        "order": [],
        "patterns": [],
        "exclude": set(),
    }


def _group_patterns_include_file(spec: dict[str, Any], rel_path: str) -> bool:
    path = _safe_resolve(rel_path)
    if path is None:
        return False
    normalized = _normalize_config_rel_path(rel_path)
    for pattern in spec.get("patterns") or []:
        if not str(pattern).startswith("configs/") or ".." in Path(str(pattern)).parts:
            continue
        if normalized in _glob_config_files(str(pattern)):
            return True
    return False


def _fallback_config_group_spec(rel_path: str) -> dict[str, Any]:
    # 支持外部 configs root：归一化后的路径不再包含 configs/ 前缀
    if rel_path.startswith("datasets/"):
        group_id = "unfiled_datasets"
        label = "未分组数据集配置"
        trainable = False
        methods_subdir = ""
    else:
        group_id = "unfiled_imported"
        label = "未分组导入配置"
        trainable = True
        methods_subdir = "imported"
    return {
        "id": group_id,
        "label": label,
        "open": True,
        "locked": False,
        "trainable": trainable,
        "methods_subdir": methods_subdir,
        "user_managed": True,
        "files": [],
        "order": [],
        "patterns": [],
        "exclude": set(),
    }


def _is_user_managed_group(spec: dict[str, Any]) -> bool:
    return bool(spec.get("user_managed")) and str(spec.get("id") or "") not in SYSTEM_CONFIG_GROUP_IDS


def _is_fixed_config_group(spec: dict[str, Any]) -> bool:
    group_id = str(spec.get("id") or "")
    if group_id in FIXED_SYSTEM_CONFIG_GROUP_IDS:
        return True
    if bool(spec.get("locked")) and not _is_user_managed_group(spec) and not _is_user_group_locked(group_id):
        return True
    return False


def _is_deletable_config_group(spec: dict[str, Any]) -> bool:
    group_id = str(spec.get("id") or "")
    return (
        group_id not in FIXED_SYSTEM_CONFIG_GROUP_IDS
        and not bool(spec.get("locked"))
        and not _is_user_group_locked(group_id)
    )


def _is_renamable_config_group(spec: dict[str, Any]) -> bool:
    group_id = str(spec.get("id") or "")
    return group_id not in FIXED_SYSTEM_CONFIG_GROUP_IDS and not bool(spec.get("locked"))


def _is_move_target_group(spec: dict[str, Any], rel_path: str = "") -> bool:
    group_id = str(spec.get("id") or "")
    normalized = _normalize_config_rel_path(rel_path)
    # 支持外部 configs root：归一化后可能包含或不包含 configs/ 前缀
    path_without_configs_prefix = normalized.removeprefix("configs/")
    if path_without_configs_prefix.startswith("datasets/"):
        return spec.get("kind") == "dataset" or group_id in {"datasets", "unfiled_datasets"}
    return _is_user_managed_group(spec) or group_id in FILE_MOVE_TARGET_GROUPS


def _is_sortable_config_group_for_place(spec: dict[str, Any], scope: str) -> bool:
    group_id = str(spec.get("id") or "")
    if group_id == "unfiled_datasets":
        return False
    if _config_group_kind(spec) != scope:
        return False
    if _is_fixed_config_group(spec):
        return False
    if bool(spec.get("locked")) or _is_user_group_locked(group_id):
        return False
    return True


def _build_config_file_group(spec: dict[str, Any]) -> dict[str, Any]:
    from web.services.config.file_group_meta import get_config_file_meta

    files: list[str] = []
    for file_path in spec["files"]:
        files.append(file_path)
    for pattern in spec["patterns"]:
        files.extend(_glob_config_files(pattern))

    unique_files: list[str] = []
    seen_files: set[str] = set()
    for file_path in files:
        normalized = _normalize_config_rel_path(file_path)
        if normalized in spec["exclude"] or normalized in seen_files:
            continue
        if normalized in HIDDEN_CONFIG_FILES:
            continue
        path = _safe_resolve(normalized)
        if path is None or not path.exists():
            continue
        seen_files.add(normalized)
        unique_files.append(normalized)

    order = [item for item in spec.get("order", []) if item in seen_files]
    if order:
        rank = {file_path: idx for idx, file_path in enumerate(order)}
        unique_files.sort(key=lambda item: (0, rank[item]) if item in rank else (1, 0))

    group_kind = spec.get("kind") or "training"
    group_id = spec["id"]
    is_locked = spec["locked"] or _is_user_group_locked(group_id)

    # movable 表示该分组是否可以接收文件拖放
    # 对于数据集分组，只要 kind 是 dataset，就应该可以接收数据集文件
    # 对于训练配置分组，依赖 _is_move_target_group 的判断
    if group_kind == "dataset":
        # 数据集分组始终可接收数据集配置文件（除非锁定）
        movable = not is_locked and group_id != "unfiled_datasets"
    else:
        # 训练配置分组的判断保持原逻辑
        movable = _is_move_target_group(spec)

    return {
        "id": group_id,
        "label": spec["label"],
        "open": spec["open"],
        "locked": is_locked,
        "group_locked": spec["locked"],
        "user_group_locked": _is_user_group_locked(group_id),
        "system_locked": group_id not in USER_LOCKABLE_GROUPS and spec["locked"],
        "lockable": group_id in USER_LOCKABLE_GROUPS or _is_user_managed_group(spec),
        "user_managed": _is_user_managed_group(spec),
        "kind": group_kind,
        "renamable": _is_renamable_config_group(spec),
        "deletable": _is_deletable_config_group(spec),
        "movable": movable,
        "trainable": spec["trainable"],
        "methods_subdir": spec["methods_subdir"],
        "files": [
            get_config_file_meta(
                file_path,
                group_id,
                spec["label"],
                spec["locked"],
                spec["trainable"],
                spec["methods_subdir"],
            )
            for file_path in unique_files
        ],
    }


def _config_file_is_covered_by_specs(specs: list[dict[str, Any]], rel_path: str) -> bool:
    normalized = _normalize_config_rel_path(rel_path)
    for spec in specs:
        if any(item.get("path") == normalized for item in _build_config_file_group(spec).get("files", [])):
            return True
    return False


def _move_orphaned_config_files_to_fallback_groups(specs: list[dict[str, Any]], files: list[str]) -> None:
    for file_path in files:
        normalized = _normalize_config_rel_path(file_path)
        path = _safe_resolve(normalized)
        if path is None or path.suffix.lower() != ".toml" or not path.exists():
            continue
        if _config_file_is_covered_by_specs(specs, normalized):
            continue

        fallback = _fallback_config_group_spec(normalized)
        target = _find_config_group_spec(specs, fallback["id"])
        if target is None:
            target = fallback
            specs.append(target)

        target.setdefault("files", [])
        if normalized not in target["files"]:
            target["files"].append(normalized)
        target.setdefault("order", [])
        target["order"] = [item for item in target["order"] if item != normalized]
        target["order"].append(normalized)
        target["exclude"] = set(item for item in target.get("exclude", set()) if item != normalized)
