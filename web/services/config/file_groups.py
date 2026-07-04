"""Config file grouping, locking, ordering, export, and restore helpers.

This module is loaded by ``web.services.config_service`` as part of the
compatibility facade.  It keeps facade access lazy so the module can also be
imported directly without pulling in the legacy facade.
"""

from __future__ import annotations

import io
import re
import shutil
import subprocess
import zipfile
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any

import toml
import tomlkit

from library.env import expand_env_vars_in_obj, get_configs_root, load_dotenv
from web.services.config import paths as _config_paths
from web.services.config.metadata import (
    CONFIG_FILE_LABELS_ZH,
    FILE_MOVE_TARGET_GROUPS,
    FIXED_SYSTEM_CONFIG_GROUP_IDS,
    HIDDEN_CONFIG_FILES,
    SYSTEM_CONFIG_GROUP_IDS,
    SYSTEM_DATASET_PRESET_FILES,
    SYSTEM_MANAGED_FILES,
    SYSTEM_PRESET_FILES,
    SYSTEM_PRESET_PREFIXES,
    USER_LOCKABLE_GROUPS,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = get_configs_root()
GUI_METHODS_DIR = CONFIGS_DIR / "gui-methods"
IMPORTED_CONFIGS_DIR = CONFIGS_DIR / "imported"
PRESETS_FILE = CONFIGS_DIR / "presets.toml"
WEB_FILE_GROUPS_FILE = CONFIGS_DIR / "web-file-groups.toml"
WEB_USER_LOCKS_FILE = CONFIGS_DIR / "web-user-locks.toml"
DATASET_PRESETS_DIR = CONFIGS_DIR / "datasets"

load_dotenv()

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
_LEGACY_FILE_GROUP_SHIM_NAMES = {
    "get_config_file_meta",
    "list_config_file_groups",
    "move_config_file_to_group",
}
_LEGACY_SYNC_NAMES = tuple(
    _name for _name in _SYNC_NAMES
    if _name not in _LEGACY_RAW_FILE_SHIM_NAMES
    and _name not in _LEGACY_FILE_GROUP_SHIM_NAMES
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


def _exported(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        _sync_from_facade()
        return fn(*args, **kwargs)

    return wrapper


def _load(p: Path) -> dict:
    if not p.exists():
        return {}
    return expand_env_vars_in_obj(toml.loads(p.read_text(encoding="utf-8")))


def _safe_resolve(rel_path: str) -> Path | None:
    return _config_paths.safe_resolve(rel_path, root=ROOT, configs_dir=CONFIGS_DIR)


def _display_path(path: Path) -> str:
    return _config_paths.display_path(path, root=ROOT, configs_dir=CONFIGS_DIR)

__all__ = ['set_user_file_lock', 'set_user_group_lock', 'create_config_file_group', 'rename_config_file_group', 'delete_config_file_group', 'reorder_config_file_group', 'move_config_file_to_group', 'place_config_file_in_group', 'place_config_file_group', 'reorder_config_file_in_group', 'restore_system_presets', 'list_config_files', 'list_config_file_groups', 'export_config_file_group_archive', 'get_config_file_meta', '_load_config_file_group_specs', '_save_config_file_group_specs', '_normalize_config_file_group_kind_filter', '_normalize_config_rel_path', '_normalize_dataset_preset_path', '_is_dataset_preset_readonly', '_is_user_locked', '_is_user_group_locked', '_load_user_locks', '_save_user_locks', '_lock_reason_message', '_lock_reason_label']

def set_user_file_lock(rel_path: str, locked: bool) -> tuple[bool, str, dict[str, Any]]:
    normalized = _normalize_config_rel_path(rel_path)
    path = _safe_resolve(normalized)
    if path is None or path.suffix != ".toml":
        return False, "路径不合法，只能锁定 configs/ 下的 TOML 文件", {}
    if not path.exists():
        return False, "只能锁定已经存在的 TOML 文件", {}

    meta = get_config_file_meta(normalized)
    if meta.get("system_locked"):
        return False, "系统预设为内置只读，不能手动锁定或解锁", meta
    if meta.get("group_locked"):
        return False, "该文件属于只读分组，不能手动锁定或解锁", meta
    if meta.get("user_group_locked"):
        return False, "该文件所在分组已锁定，请先解除分组锁定", meta

    user_locks, user_group_locks = _load_user_locks()
    if locked:
        user_locks.add(normalized)
    else:
        user_locks.discard(normalized)
    _save_user_locks(user_locks, user_group_locks)

    next_meta = get_config_file_meta(normalized)
    return True, ("已锁定当前文件" if locked else "已解除用户锁定"), next_meta


def set_user_group_lock(group_id: str, locked: bool) -> tuple[bool, str, dict[str, Any]]:
    normalized = _normalize_group_id(group_id)
    if not normalized:
        return False, "缺少 group 参数", {}

    group = _get_config_file_group(normalized)
    if group is None:
        return False, "分组不存在", {}
    if normalized not in _lockable_group_ids():
        return False, "该分组属于系统或只读参考，不能手动锁定或解锁", group
    if any(item.get("system_locked") for item in group.get("files", [])):
        return False, "该分组包含系统预设，不能手动锁定或解锁", group

    user_locks, user_group_locks = _load_user_locks()
    if locked:
        user_group_locks.add(normalized)
    else:
        user_group_locks.discard(normalized)
    _save_user_locks(user_locks, user_group_locks)

    next_group = _get_config_file_group(normalized) or group
    return True, ("已锁定当前分组" if locked else "已解除分组锁定"), next_group


def create_config_file_group(label: str, kind: str = "training") -> tuple[bool, str, dict[str, Any] | None]:
    clean_label = _normalize_group_label(label)
    if not clean_label:
        return False, "分组名称不能为空", None

    specs = _load_config_file_group_specs()
    group_id = _unique_group_id(_slugify_group_label(clean_label), specs)
    spec = _new_user_config_group_spec(group_id, clean_label, kind=kind)
    specs.append(spec)
    _save_config_file_group_specs(specs)
    return True, "分组已创建", _build_config_file_group(spec)


def rename_config_file_group(group_id: str, label: str) -> tuple[bool, str, dict[str, Any] | None]:
    normalized = _normalize_group_id(group_id)
    clean_label = _normalize_group_label(label)
    if not normalized:
        return False, "缺少 group 参数", None
    if not clean_label:
        return False, "分组名称不能为空", None

    specs = _load_config_file_group_specs()
    spec = _find_config_group_spec(specs, normalized)
    if spec is None:
        return False, "分组不存在", None
    if not _is_renamable_config_group(spec):
        return False, "系统固定或只读分组不能重命名", _build_config_file_group(spec)

    spec["label"] = clean_label
    _save_config_file_group_specs(specs)
    return True, "分组已重命名", _build_config_file_group(spec)


def delete_config_file_group(group_id: str) -> tuple[bool, str]:
    normalized = _normalize_group_id(group_id)
    if not normalized:
        return False, "缺少 group 参数"

    specs = _load_config_file_group_specs()
    spec = _find_config_group_spec(specs, normalized)
    if spec is None:
        return False, "分组不存在"
    if _is_user_group_locked(normalized):
        return False, "该分组已锁定，请先解除分组锁定后再删除"
    if not _is_deletable_config_group(spec):
        return False, "系统或只读分组不能删除"

    released_files = {item["path"] for item in _build_config_file_group(spec).get("files", [])}
    released_files.update(spec.get("files", []))
    released_files.update(spec.get("order", []))
    specs = [item for item in specs if item["id"] != normalized]
    if released_files:
        for item in specs:
            item["exclude"] = set(item.get("exclude", set())) - released_files
        _move_orphaned_config_files_to_fallback_groups(specs, sorted(released_files))
    user_locks, user_group_locks = _load_user_locks()
    if normalized in user_group_locks:
        user_group_locks.discard(normalized)
        _save_user_locks(user_locks, user_group_locks)
    _save_config_file_group_specs(specs)
    return True, "分组已删除，TOML 文件已保留在其他可见分组中"


def reorder_config_file_group(group_id: str, direction: str) -> tuple[bool, str, dict[str, Any] | None]:
    normalized = _normalize_group_id(group_id)
    clean_direction = str(direction or "").strip().lower()
    if not normalized:
        return False, "缺少 group 参数", None
    if clean_direction not in {"up", "down"}:
        return False, "排序方向必须是 up 或 down", None

    specs = _load_config_file_group_specs()
    spec = _find_config_group_spec(specs, normalized)
    if spec is None:
        return False, "分组不存在", None
    if _is_fixed_config_group(spec):
        return False, "系统分组不能调整顺序", _build_config_file_group(spec)

    movable_indices = [
        idx for idx, item in enumerate(specs)
        if not _is_fixed_config_group(item)
    ]
    current_pos = next((idx for idx, item_idx in enumerate(movable_indices) if specs[item_idx]["id"] == normalized), -1)
    if current_pos < 0:
        return False, "分组不在可排序列表中", _build_config_file_group(spec)

    next_pos = current_pos - 1 if clean_direction == "up" else current_pos + 1
    if next_pos < 0 or next_pos >= len(movable_indices):
        return True, "分组顺序未变化", _build_config_file_group(spec)

    current_index = movable_indices[current_pos]
    next_index = movable_indices[next_pos]
    specs[current_index], specs[next_index] = specs[next_index], specs[current_index]
    _save_config_file_group_specs(specs)
    moved = _find_config_group_spec(specs, normalized)
    return True, "分组顺序已更新", _build_config_file_group(moved) if moved else None


def move_config_file_to_group(rel_path: str, group_id: str) -> tuple[bool, str, dict[str, Any] | None]:
    normalized_file = _normalize_config_rel_path(rel_path)
    target_group_id = _normalize_group_id(group_id)
    path = _safe_resolve(normalized_file)
    if path is None or path.suffix.lower() != ".toml" or not path.exists():
        return False, "配置文件不存在或路径不合法", None
    if _is_system_locked_path(normalized_file):
        return False, "系统预设和 Web 管理配置不能移动分组", None

    # 支持外部 configs root：归一化后可能仍然保留 configs/ 前缀（默认行为），
    # 也可能不包含该前缀（外部 configs root 时从 display_path 得到的相对路径）
    # 检查时需要同时支持两种情况
    path_without_configs_prefix = normalized_file.removeprefix("configs/")
    if not (path_without_configs_prefix.startswith("imported/") or path_without_configs_prefix.startswith("datasets/")):
        return False, "当前仅支持移动导入配置和数据集配置", None

    specs = _load_config_file_group_specs()
    target = _find_config_group_spec(specs, target_group_id)
    if target is None:
        return False, "目标分组不存在", None
    if not _is_move_target_group(target, normalized_file):
        message = "数据集预设只能移动到数据集分组" if path_without_configs_prefix.startswith("datasets/") else "只能移动到导入配置、数据集配置或用户自定义分组"
        return False, message, _build_config_file_group(target)
    if target.get("locked") or _is_user_group_locked(target_group_id):
        return False, "目标分组已锁定，不能移入配置", _build_config_file_group(target)

    for spec in specs:
        spec["files"] = [item for item in _config_group_path_list(spec.get("files")) if item != normalized_file]
        spec["order"] = [item for item in _config_group_path_list(spec.get("order")) if item != normalized_file]
        exclude = [item for item in _config_group_path_list(spec.get("exclude")) if item != normalized_file]
        if _group_patterns_include_file(spec, normalized_file) and spec["id"] != target_group_id:
            exclude.append(normalized_file)
        spec["exclude"] = sorted(dict.fromkeys(exclude))

    target.setdefault("files", [])
    target["files"] = _config_group_path_list([*target["files"], normalized_file])
    target.setdefault("order", [])
    target["order"] = [item for item in _config_group_path_list(target["order"]) if item != normalized_file]
    target["order"].append(normalized_file)
    if normalized_file in target.get("exclude", []):
        target["exclude"] = [item for item in _config_group_path_list(target["exclude"]) if item != normalized_file]

    _save_config_file_group_specs(specs)
    return True, "配置已移动到分组", _build_config_file_group(target)


def place_config_file_in_group(
    rel_path: str,
    group_id: str,
    index: Any | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    normalized_file = _normalize_config_rel_path(rel_path)
    target_group_id = _normalize_group_id(group_id)
    path = _safe_resolve(normalized_file)
    if path is None or path.suffix.lower() != ".toml" or not path.exists():
        return False, "配置文件不存在或路径不合法", None
    if _is_system_locked_path(normalized_file):
        return False, "系统预设和 Web 管理配置不能移动分组", None

    path_without_configs_prefix = normalized_file.removeprefix("configs/")
    if not (path_without_configs_prefix.startswith("imported/") or path_without_configs_prefix.startswith("datasets/")):
        return False, "当前仅支持移动导入配置和数据集配置", None

    specs = _load_config_file_group_specs()
    target = _find_config_group_spec(specs, target_group_id)
    if target is None:
        return False, "目标分组不存在", None
    if not _is_move_target_group(target, normalized_file):
        message = "数据集预设只能移动到数据集分组" if path_without_configs_prefix.startswith("datasets/") else "只能移动到导入配置、数据集配置或用户自定义分组"
        return False, message, _build_config_file_group(target)
    if target.get("locked") or _is_user_group_locked(target_group_id):
        return False, "目标分组已锁定，不能移入配置", _build_config_file_group(target)

    for spec in specs:
        spec["files"] = [item for item in _config_group_path_list(spec.get("files")) if item != normalized_file]
        spec["order"] = [item for item in _config_group_path_list(spec.get("order")) if item != normalized_file]
        exclude = [item for item in _config_group_path_list(spec.get("exclude")) if item != normalized_file]
        if _group_patterns_include_file(spec, normalized_file) and spec["id"] != target_group_id:
            exclude.append(normalized_file)
        spec["exclude"] = sorted(dict.fromkeys(exclude))

    current_files = [
        item["path"]
        for item in _build_config_file_group(target).get("files", [])
        if item.get("path") != normalized_file
    ]
    target_index = _place_index(index, len(current_files))
    current_files.insert(target_index, normalized_file)

    target.setdefault("files", [])
    target["files"] = _config_group_path_list([*target["files"], normalized_file])
    target["order"] = current_files
    if normalized_file in target.get("exclude", []):
        target["exclude"] = [item for item in _config_group_path_list(target["exclude"]) if item != normalized_file]

    _save_config_file_group_specs(specs)
    return True, "配置位置已更新", _build_config_file_group(target)


def place_config_file_group(
    group_id: str,
    scope: str,
    index: Any | None,
) -> tuple[bool, str, dict[str, Any] | None]:
    normalized = _normalize_group_id(group_id)
    clean_scope = _normalize_config_file_group_kind_filter(scope or "")
    if clean_scope not in {"training", "dataset"}:
        return False, "scope 必须是 training 或 dataset", None
    if not normalized:
        return False, "缺少 group 参数", None

    specs = _load_config_file_group_specs()
    spec = _find_config_group_spec(specs, normalized)
    if spec is None:
        return False, "分组不存在", None
    if not _is_sortable_config_group_for_place(spec, clean_scope):
        return False, "该分组不能在当前范围内拖动排序", _build_config_file_group(spec)

    current_index = next((idx for idx, item in enumerate(specs) if item.get("id") == normalized), -1)
    if current_index < 0:
        return False, "分组不在可排序列表中", _build_config_file_group(spec)

    moving = specs.pop(current_index)
    remaining_indices = [
        idx for idx, item in enumerate(specs)
        if _is_sortable_config_group_for_place(item, clean_scope)
    ]
    target_index = _place_index(index, len(remaining_indices))
    if target_index >= len(remaining_indices):
        insert_at = (remaining_indices[-1] + 1) if remaining_indices else len(specs)
    else:
        insert_at = remaining_indices[target_index]
    specs.insert(insert_at, moving)

    _save_config_file_group_specs(specs)
    moved = _find_config_group_spec(specs, normalized)
    return True, "分组位置已更新", _build_config_file_group(moved) if moved else None


def reorder_config_file_in_group(
    rel_path: str,
    group_id: str,
    direction: str,
) -> tuple[bool, str, dict[str, Any] | None]:
    normalized_file = _normalize_config_rel_path(rel_path)
    target_group_id = _normalize_group_id(group_id)
    clean_direction = str(direction or "").strip().lower()
    if clean_direction not in {"up", "down"}:
        return False, "排序方向必须是 up 或 down", None

    path = _safe_resolve(normalized_file)
    if path is None or path.suffix.lower() != ".toml" or not path.exists():
        return False, "配置文件不存在或路径不合法", None

    specs = _load_config_file_group_specs()
    spec = _find_config_group_spec(specs, target_group_id)
    if spec is None:
        return False, "分组不存在", None

    files = [item["path"] for item in _build_config_file_group(spec).get("files", [])]
    if normalized_file not in files:
        return False, "配置文件不在该分组中", _build_config_file_group(spec)

    index = files.index(normalized_file)
    next_index = index - 1 if clean_direction == "up" else index + 1
    if next_index < 0 or next_index >= len(files):
        return True, "排序未变化", _build_config_file_group(spec)

    files[index], files[next_index] = files[next_index], files[index]
    spec["order"] = files
    _save_config_file_group_specs(specs)
    return True, "配置排序已更新", _build_config_file_group(spec)


def restore_system_presets(files: list[str] | None = None) -> dict[str, Any]:
    targets = _list_system_preset_files() if files is None else files
    normalized_targets: list[str] = []
    errors: list[dict[str, str]] = []
    seen: set[str] = set()

    for raw in targets:
        normalized = _normalize_config_rel_path(raw)
        path = _safe_resolve(normalized)
        if path is None or path.suffix != ".toml":
            errors.append({"file": normalized, "reason": "路径不合法"})
            continue
        if not _is_system_preset_path(normalized):
            errors.append({"file": normalized, "reason": "不是系统预设文件"})
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_targets.append(normalized)

    if errors:
        return {
            "ok": False,
            "error": "还原请求包含不合法文件",
            "restored": [],
            "skipped": [],
            "errors": errors,
            "backup_dir": "",
        }

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_root = CONFIGS_DIR / ".restore-backups" / timestamp
    restored: list[str] = []
    skipped: list[dict[str, str]] = []

    for rel_path in normalized_targets:
        path = _safe_resolve(rel_path)
        if path is None or not path.exists():
            skipped.append({"file": rel_path, "reason": "当前文件不存在"})
            continue

        baseline = _read_git_head_file(rel_path)
        if baseline is None:
            skipped.append({"file": rel_path, "reason": "没有可还原的系统基线"})
            continue

        backup_path = backup_root / _backup_relative_path(rel_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_path)
        path.write_text(baseline, encoding="utf-8")
        restored.append(rel_path)

    return {
        "ok": True,
        "restored": restored,
        "skipped": skipped,
        "errors": [],
        "backup_dir": _display_path(backup_root) if restored else "",
    }


def list_config_files() -> list[str]:
    return [item["path"] for group in list_config_file_groups() for item in group["files"]]


def list_config_file_groups(kind: str | None = None) -> list[dict[str, Any]]:
    specs = _sort_config_file_group_specs_for_display(_load_config_file_group_specs())
    groups = [_build_config_file_group(spec) for spec in specs]
    kind_filter = _normalize_config_file_group_kind_filter(kind)
    if kind_filter == "all":
        return groups
    return [group for group in groups if str(group.get("kind") or "training") == kind_filter]


def export_config_file_group_archive(group_id: str, kind: str | None = "training") -> dict[str, Any]:
    normalized_group_id = _normalize_group_id(group_id)
    groups = list_config_file_groups(kind=kind)
    group = next((item for item in groups if item.get("id") == normalized_group_id), None)
    if group is None:
        raise FileNotFoundError("配置分组不存在")

    files = [
        item for item in group.get("files", [])
        if item.get("path") and str(item.get("path")).lower().endswith(".toml")
    ]
    if not files:
        raise ValueError("该分组没有可导出的 TOML 文件")

    archive_stem = _safe_archive_name(str(group.get("label") or group.get("id") or "toml-group"))
    used_names: set[str] = set()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in files:
            rel_path = _normalize_config_rel_path(str(item.get("path") or ""))
            path = _safe_resolve(rel_path)
            if path is None or not path.is_file():
                raise FileNotFoundError(f"配置文件不存在: {rel_path}")
            archive_name = _unique_archive_member_name(
                _safe_archive_name(str(item.get("filename") or Path(rel_path).name)),
                used_names,
            )
            archive.writestr(archive_name, path.read_text(encoding="utf-8"))

    return {
        "filename": f"{archive_stem}.zip",
        "content": buffer.getvalue(),
        "count": len(files),
        "group": group,
    }


def _normalize_config_file_group_kind_filter(kind: str | None) -> str:
    clean = str(kind or "all").strip().lower()
    if clean in {"", "all", "*"}:
        return "all"
    if clean in {"training", "config", "configs"}:
        return "training"
    if clean in {"dataset", "datasets"}:
        return "dataset"
    raise ValueError("kind 参数只支持 training、dataset 或 all")


def _get_config_file_group(group_id: str) -> dict[str, Any] | None:
    normalized = _normalize_group_id(group_id)
    for group in list_config_file_groups():
        if group.get("id") == normalized:
            return group
    return None


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
    for group in list_config_file_groups():
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


def _strip_configs_prefix(rel_path: str) -> str:
    return _normalize_config_rel_path(rel_path).removeprefix("configs/")


def _load_config_file_group_specs() -> list[dict[str, Any]]:
    data = _load(WEB_FILE_GROUPS_FILE)
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
    WEB_FILE_GROUPS_FILE.parent.mkdir(parents=True, exist_ok=True)
    WEB_FILE_GROUPS_FILE.write_text(tomlkit.dumps(doc), encoding="utf-8")


def _build_config_file_group(spec: dict[str, Any]) -> dict[str, Any]:
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


def _glob_config_files(pattern: str) -> list[str]:
    normalized_pattern = _normalize_config_rel_path(pattern)
    if not normalized_pattern.startswith("configs/") or ".." in Path(normalized_pattern).parts:
        return []
    rel_pattern = normalized_pattern.removeprefix("configs/")
    return [
        _display_path(path)
        for path in sorted(CONFIGS_DIR.glob(rel_pattern))
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


def _config_file_is_covered_by_specs(specs: list[dict[str, Any]], rel_path: str) -> bool:
    normalized = _normalize_config_rel_path(rel_path)
    for spec in specs:
        if any(item.get("path") == normalized for item in _build_config_file_group(spec).get("files", [])):
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


def _place_index(value: Any | None, length: int) -> int:
    if value is None or value == "":
        return max(0, length)
    try:
        index = int(value)
    except (TypeError, ValueError):
        return max(0, length)
    return max(0, min(index, max(0, length)))


def _lockable_group_ids() -> set[str]:
    ids = set(USER_LOCKABLE_GROUPS)
    ids.update(
        spec["id"]
        for spec in _load_config_file_group_specs()
        if _is_user_managed_group(spec)
    )
    return ids


def _unique_group_id(base: str, specs: list[dict[str, Any]]) -> str:
    used = {str(spec.get("id") or "") for spec in specs}
    root = base or "custom_group"
    candidate = root
    idx = 2
    while candidate in used:
        candidate = f"{root}_{idx}"
        idx += 1
    return candidate


def _slugify_group_label(label: str) -> str:
    chars: list[str] = []
    for ch in label.strip().lower():
        if ch.isascii() and ch.isalnum():
            chars.append(ch)
        elif ch in {"-", "_"}:
            chars.append(ch)
        elif ch.isspace():
            chars.append("_")
    slug = "".join(chars).strip("_-")
    return slug or "custom_group"


def _normalize_group_label(label: str) -> str:
    return " ".join(str(label or "").strip().split())[:48]


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


def _normalize_config_rel_path(rel_path: str) -> str:
    return _config_paths.normalize_config_rel_path(rel_path)


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
        path = Path("datasets") / path
    normalized = path.as_posix().lstrip("/")
    # 支持外部 configs root：归一化后只保留相对于 configs root 的路径
    if not normalized.startswith("datasets/"):
        raise ValueError("数据集预设必须保存在 datasets/ 下")
    safe_path = _safe_resolve(normalized)
    if safe_path is None:
        raise ValueError("数据集预设路径不合法")
    if must_exist and not safe_path.exists():
        raise ValueError("数据集预设不存在")
    return normalized


def _normalize_group_id(group_id: str) -> str:
    return str(group_id or "").strip()


def _safe_archive_name(name: str) -> str:
    clean = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", str(name or "").strip())
    clean = re.sub(r"\s+", "_", clean).strip("._")
    return clean or "toml-group"


def _unique_archive_member_name(name: str, used_names: set[str]) -> str:
    clean = _safe_archive_name(name)
    if not clean.lower().endswith(".toml"):
        clean = f"{clean}.toml"
    candidate = clean
    stem = candidate[:-5]
    index = 2
    while candidate in used_names:
        candidate = f"{stem}-{index}.toml"
        index += 1
    used_names.add(candidate)
    return candidate


def _is_system_preset_path(rel_path: str) -> bool:
    normalized = _normalize_config_rel_path(rel_path)
    return normalized in SYSTEM_PRESET_FILES or normalized.startswith(SYSTEM_PRESET_PREFIXES)


def _is_system_locked_path(rel_path: str) -> bool:
    normalized = _normalize_config_rel_path(rel_path)
    return _is_system_preset_path(normalized) or normalized in SYSTEM_MANAGED_FILES


def _is_dataset_preset_readonly(rel_path: str) -> bool:
    normalized = _normalize_config_rel_path(rel_path)
    return normalized in SYSTEM_DATASET_PRESET_FILES or get_config_file_meta(normalized).get("locked", False)


def _is_user_locked(rel_path: str) -> bool:
    file_locks, _ = _load_user_locks()
    return _normalize_config_rel_path(rel_path) in file_locks


def _is_user_group_locked(group_id: str | None) -> bool:
    _, group_locks = _load_user_locks()
    return _normalize_group_id(group_id or "") in group_locks


def _load_user_locks() -> tuple[set[str], set[str]]:
    if not WEB_USER_LOCKS_FILE.exists():
        return set(), set()
    try:
        data = toml.loads(WEB_USER_LOCKS_FILE.read_text(encoding="utf-8"))
    except toml.TomlDecodeError:
        return set(), set()

    file_locks: set[str] = set()
    for raw in _string_list(data.get("locked")):
        normalized = _normalize_config_rel_path(raw)
        path = _safe_resolve(normalized)
        if path is None or path.suffix != ".toml":
            continue
        if _is_system_locked_path(normalized):
            continue
        file_locks.add(normalized)

    group_locks: set[str] = set()
    for raw in _string_list(data.get("locked_groups")):
        normalized = _normalize_group_id(raw)
        if normalized in _lockable_group_ids():
            group_locks.add(normalized)
    return file_locks, group_locks


def _save_user_locks(file_locks: set[str], group_locks: set[str]) -> None:
    WEB_USER_LOCKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    WEB_USER_LOCKS_FILE.write_text(
        toml.dumps({
            "locked": sorted(file_locks),
            "locked_groups": sorted(group_locks),
        }),
        encoding="utf-8",
    )


def _lock_reason_label(reason: str) -> str:
    labels = {
        "system": "系统只读",
        "user": "用户锁定",
        "user_group": "分组锁定",
        "group": "分组只读",
    }
    return labels.get(reason, "")


def _lock_reason_message(meta: dict[str, Any]) -> str:
    reason = str(meta.get("lock_reason") or "")
    if reason == "system":
        return "该配置文件是系统预设，已内置锁定"
    if reason == "user":
        return "该配置文件已被用户锁定"
    if reason == "user_group":
        return "该配置文件所在分组已被用户锁定"
    if reason == "group":
        return "该配置文件属于只读分组"
    return "该配置文件已锁定"


def _list_system_preset_files() -> list[str]:
    files: list[str] = []
    for rel_path in sorted(SYSTEM_PRESET_FILES):
        path = _safe_resolve(rel_path)
        if path is not None and path.exists():
            files.append(rel_path)
    for prefix in SYSTEM_PRESET_PREFIXES:
        folder = _safe_resolve(prefix.rstrip("/"))
        if folder is None or not folder.is_dir():
            continue
        files.extend(
            _display_path(path)
            for path in sorted(folder.glob("*.toml"))
            if path.is_file() and _display_path(path) not in HIDDEN_CONFIG_FILES
        )
    return sorted(dict.fromkeys(files))


def _read_git_head_file(rel_path: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"HEAD:{rel_path}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _backup_relative_path(rel_path: str) -> Path:
    path = Path(rel_path)
    try:
        return path.relative_to("configs")
    except ValueError:
        return path


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (set, frozenset)):
        return [str(item) for item in sorted(value) if item]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if item]


def _config_group_path_list(value: Any) -> list[str]:
    paths: list[str] = []
    for raw in _string_list(value):
        normalized = _normalize_config_rel_path(raw)
        if normalized:
            paths.append(normalized)
    return list(dict.fromkeys(paths))




for _name in __all__:
    globals()[_name] = _exported(globals()[_name])
