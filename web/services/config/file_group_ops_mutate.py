"""Create/rename/delete/reorder/move public operations for config file groups."""

from __future__ import annotations

from typing import Any

from web.services.config.file_group_locks import (
    _is_system_locked_path,
    _is_user_group_locked,
    _load_user_locks,
    _save_user_locks,
)
from web.services.config.file_group_paths import (
    _config_group_path_list,
    _normalize_config_rel_path,
    _normalize_group_id,
    _normalize_group_label,
    _place_index,
    _slugify_group_label,
    _unique_group_id,
)
from web.services.config.file_group_runtime import _exported, _safe_resolve
from web.services.config.file_group_specs import (
    _build_config_file_group,
    _find_config_group_spec,
    _group_patterns_include_file,
    _is_deletable_config_group,
    _is_fixed_config_group,
    _is_move_target_group,
    _is_renamable_config_group,
    _is_sortable_config_group_for_place,
    _load_config_file_group_specs,
    _move_orphaned_config_files_to_fallback_groups,
    _new_user_config_group_spec,
    _normalize_config_file_group_kind_filter,
    _save_config_file_group_specs,
)


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
    anchor: Any | None = None,
    position: Any | None = None,
    order: Any | None = None,
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

    existing_files = [
        item["path"]
        for item in _build_config_file_group(target).get("files", [])
    ]
    existing_set = set(existing_files)

    # 1) 前端若给出完整 DOM 新顺序，直接采用（同组排序最稳）。
    requested_order: list[str] = []
    if isinstance(order, (list, tuple)):
        for raw in order:
            path_item = _normalize_config_rel_path(str(raw or "").strip())
            if not path_item or path_item in requested_order:
                continue
            if path_item == normalized_file or path_item in existing_set:
                requested_order.append(path_item)
    if normalized_file not in requested_order and requested_order:
        # 容错：order 漏了拖动项时补到末尾
        requested_order.append(normalized_file)

    if requested_order:
        # 保留目标组里未出现在 DOM 顺序中的其余文件，追加在尾部，避免 pattern 组丢项。
        tail = [path for path in existing_files if path not in requested_order and path != normalized_file]
        current_files = [*requested_order, *tail]
        if normalized_file not in current_files:
            current_files.append(normalized_file)
    else:
        current_files = [path for path in existing_files if path != normalized_file]
        # 2) 锚点 before/after
        anchor_path = _normalize_config_rel_path(str(anchor or "").strip()) if anchor else ""
        place_pos = str(position or "").strip().lower()
        if anchor_path and anchor_path in current_files:
            anchor_index = current_files.index(anchor_path)
            target_index = anchor_index + (1 if place_pos == "after" else 0)
            target_index = max(0, min(target_index, len(current_files)))
        else:
            # 3) 兼容旧 index
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


create_config_file_group = _exported(create_config_file_group)
rename_config_file_group = _exported(rename_config_file_group)
delete_config_file_group = _exported(delete_config_file_group)
reorder_config_file_group = _exported(reorder_config_file_group)
move_config_file_to_group = _exported(move_config_file_to_group)
place_config_file_in_group = _exported(place_config_file_in_group)
place_config_file_group = _exported(place_config_file_group)
reorder_config_file_in_group = _exported(reorder_config_file_in_group)
