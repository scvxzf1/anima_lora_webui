"""User/system lock helpers for config file groups."""

from __future__ import annotations

from typing import Any

import toml

from web.services.config.file_group_paths import (
    _normalize_config_rel_path,
    _normalize_group_id,
    _string_list,
)
from web.services.config.file_group_runtime import (
    WEB_USER_LOCKS_FILE,
    _owner_attr,
    _safe_resolve,
)
from web.services.config.metadata import (
    HIDDEN_CONFIG_FILES,
    SYSTEM_DATASET_PRESET_FILES,
    SYSTEM_MANAGED_FILES,
    SYSTEM_PRESET_FILES,
    SYSTEM_PRESET_PREFIXES,
    USER_LOCKABLE_GROUPS,
)


def _is_system_preset_path(rel_path: str) -> bool:
    normalized = _normalize_config_rel_path(rel_path)
    return normalized in SYSTEM_PRESET_FILES or normalized.startswith(SYSTEM_PRESET_PREFIXES)


def _is_system_locked_path(rel_path: str) -> bool:
    normalized = _normalize_config_rel_path(rel_path)
    return _is_system_preset_path(normalized) or normalized in SYSTEM_MANAGED_FILES


def _is_user_locked(rel_path: str) -> bool:
    file_locks, _ = _load_user_locks()
    return _normalize_config_rel_path(rel_path) in file_locks


def _is_user_group_locked(group_id: str | None) -> bool:
    _, group_locks = _load_user_locks()
    return _normalize_group_id(group_id or "") in group_locks


def _lockable_group_ids() -> set[str]:
    # Lazy import avoids circular dependency with file_group_specs.
    from web.services.config.file_group_specs import (
        _is_user_managed_group,
        _load_config_file_group_specs,
    )

    ids = set(USER_LOCKABLE_GROUPS)
    ids.update(
        spec["id"]
        for spec in _load_config_file_group_specs()
        if _is_user_managed_group(spec)
    )
    return ids


def _load_user_locks() -> tuple[set[str], set[str]]:
    if not _owner_attr("WEB_USER_LOCKS_FILE", WEB_USER_LOCKS_FILE).exists():
        return set(), set()
    try:
        data = toml.loads(_owner_attr("WEB_USER_LOCKS_FILE", WEB_USER_LOCKS_FILE).read_text(encoding="utf-8"))
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
    _owner_attr("WEB_USER_LOCKS_FILE", WEB_USER_LOCKS_FILE).parent.mkdir(parents=True, exist_ok=True)
    _owner_attr("WEB_USER_LOCKS_FILE", WEB_USER_LOCKS_FILE).write_text(
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
    from web.services.config.file_group_runtime import _display_path

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
