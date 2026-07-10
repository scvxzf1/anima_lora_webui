"""Public operations for config file groups, locks, listing, and restore.

Compatibility re-export surface. Implementation lives in:

- ``file_group_ops_lock``: file/group lock toggles
- ``file_group_ops_mutate``: create/rename/delete/reorder/move/place
- ``file_group_ops_query``: list/export/restore

Public callers should import through ``file_groups`` so path monkeypatches keep
working.
"""

from __future__ import annotations

from web.services.config import file_group_core as _core
from web.services.config.file_group_ops_lock import (
    set_user_file_lock,
    set_user_group_lock,
)
from web.services.config.file_group_ops_mutate import (
    create_config_file_group,
    delete_config_file_group,
    move_config_file_to_group,
    place_config_file_group,
    place_config_file_in_group,
    rename_config_file_group,
    reorder_config_file_group,
    reorder_config_file_in_group,
)
from web.services.config.file_group_ops_query import (
    export_config_file_group_archive,
    list_config_file_groups,
    list_config_files,
    restore_system_presets,
)


def __getattr__(name: str):
    # Fall back to core helpers for internal private use and test discovery.
    if name.startswith("_") and hasattr(_core, name):
        return getattr(_core, name)
    raise AttributeError(name)


# Explicit local aliases for helpers used by tests/siblings; do not rely only on
# dynamic __getattr__ because assignment-time lookups need real names.
_sync_from_facade = _core._sync_from_facade
_load = _core._load
_safe_resolve = _core._safe_resolve
_display_path = _core._display_path
_normalize_config_rel_path = _core._normalize_config_rel_path
_normalize_group_id = _core._normalize_group_id
_normalize_group_label = _core._normalize_group_label
_normalize_config_file_group_kind_filter = _core._normalize_config_file_group_kind_filter
_load_user_locks = _core._load_user_locks
_save_user_locks = _core._save_user_locks
_load_config_file_group_specs = _core._load_config_file_group_specs
_save_config_file_group_specs = _core._save_config_file_group_specs
_find_config_group_spec = _core._find_config_group_spec
_new_user_config_group_spec = _core._new_user_config_group_spec
_unique_group_id = _core._unique_group_id
_slugify_group_label = _core._slugify_group_label
_is_user_managed_group = _core._is_user_managed_group
_is_fixed_config_group = _core._is_fixed_config_group
_is_deletable_config_group = _core._is_deletable_config_group
_is_renamable_config_group = _core._is_renamable_config_group
_is_move_target_group = _core._is_move_target_group
_is_sortable_config_group_for_place = _core._is_sortable_config_group_for_place
_is_user_locked = _core._is_user_locked
_is_user_group_locked = _core._is_user_group_locked
_is_system_locked_path = _core._is_system_locked_path
_is_system_preset_path = _core._is_system_preset_path
_is_dataset_preset_readonly = _core._is_dataset_preset_readonly
_lock_reason_message = _core._lock_reason_message
_lockable_group_ids = _core._lockable_group_ids
_build_config_file_group = _core._build_config_file_group
_sort_config_file_group_specs_for_display = _core._sort_config_file_group_specs_for_display
_get_config_file_group = _core._get_config_file_group
_place_index = _core._place_index
_safe_archive_name = _core._safe_archive_name
_unique_archive_member_name = _core._unique_archive_member_name
_config_group_path_list = _core._config_group_path_list
_group_patterns_include_file = _core._group_patterns_include_file
_move_orphaned_config_files_to_fallback_groups = _core._move_orphaned_config_files_to_fallback_groups
_list_system_preset_files = _core._list_system_preset_files
_read_git_head_file = _core._read_git_head_file
_backup_relative_path = _core._backup_relative_path
_normalize_dataset_preset_path = _core._normalize_dataset_preset_path
_config_method_name_for_path = _core._config_method_name_for_path
_infer_config_file_group = _core._infer_config_file_group
_strip_configs_prefix = _core._strip_configs_prefix
_config_group_kind = _core._config_group_kind
get_config_file_meta = _core.get_config_file_meta


def _export_public(fn):
    """Ensure path monkeypatches on config_service reach file_groups/core."""
    return _core._exported(fn)


__all__ = [
    "set_user_file_lock",
    "set_user_group_lock",
    "create_config_file_group",
    "rename_config_file_group",
    "delete_config_file_group",
    "reorder_config_file_group",
    "move_config_file_to_group",
    "place_config_file_in_group",
    "place_config_file_group",
    "reorder_config_file_in_group",
    "restore_system_presets",
    "list_config_files",
    "list_config_file_groups",
    "export_config_file_group_archive",
    "get_config_file_meta",
]
