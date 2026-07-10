"""Lock/unlock public operations for config file groups."""

from __future__ import annotations

from typing import Any

from web.services.config.file_group_locks import (
    _load_user_locks,
    _lockable_group_ids,
    _save_user_locks,
)
from web.services.config.file_group_meta import (
    _get_config_file_group,
    get_config_file_meta,
)
from web.services.config.file_group_paths import (
    _normalize_config_rel_path,
    _normalize_group_id,
)
from web.services.config.file_group_runtime import _exported, _safe_resolve


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


# Keep public entrypoints facade-path aware for external configs roots.
set_user_file_lock = _exported(set_user_file_lock)
set_user_group_lock = _exported(set_user_group_lock)
