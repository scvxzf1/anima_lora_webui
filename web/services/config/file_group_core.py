"""Compatibility re-export surface for config file-group internals.

Domain logic lives in:

- ``file_group_runtime``: owner lookup, facade sync, path roots
- ``file_group_paths``: normalize / archive / string list helpers
- ``file_group_locks``: system/user lock storage and messages
- ``file_group_specs``: group specs load/save/build/policy
- ``file_group_meta``: file metadata inference

Public callers should import through ``file_groups`` so path monkeypatches keep
working. Tests may still import helpers from this module.
"""

from __future__ import annotations

from web.services.config.file_group_locks import (
    _is_system_locked_path,
    _is_system_preset_path,
    _is_user_group_locked,
    _is_user_locked,
    _list_system_preset_files,
    _load_user_locks,
    _lock_reason_label,
    _lock_reason_message,
    _lockable_group_ids,
    _save_user_locks,
)
from web.services.config.file_group_meta import (
    _config_method_name_for_path,
    _get_config_file_group,
    _infer_config_file_group,
    _is_dataset_preset_readonly,
    _list_config_file_groups_lazy,
    get_config_file_meta,
)
from web.services.config.file_group_paths import (
    _backup_relative_path,
    _config_group_path_list,
    _normalize_config_rel_path,
    _normalize_dataset_preset_path,
    _normalize_group_id,
    _normalize_group_label,
    _place_index,
    _read_git_head_file,
    _safe_archive_name,
    _slugify_group_label,
    _string_list,
    _strip_configs_prefix,
    _unique_archive_member_name,
    _unique_group_id,
)
from web.services.config.file_group_runtime import (
    CONFIGS_DIR,
    DATASET_PRESETS_DIR,
    GUI_METHODS_DIR,
    IMPORTED_CONFIGS_DIR,
    PRESETS_FILE,
    ROOT,
    WEB_FILE_GROUPS_FILE,
    WEB_USER_LOCKS_FILE,
    _display_path,
    _exported,
    _load,
    _owner,
    _owner_attr,
    _safe_resolve,
    _sync_common_paths,
    _sync_from_facade,
)
from web.services.config.file_group_specs import (
    _build_config_file_group,
    _config_file_is_covered_by_specs,
    _config_group_kind,
    _default_config_file_group_specs,
    _fallback_config_group_spec,
    _find_config_group_spec,
    _glob_config_files,
    _group_defaults,
    _group_patterns_include_file,
    _is_deletable_config_group,
    _is_fixed_config_group,
    _is_move_target_group,
    _is_renamable_config_group,
    _is_sortable_config_group_for_place,
    _is_user_managed_group,
    _load_config_file_group_specs,
    _move_orphaned_config_files_to_fallback_groups,
    _new_user_config_group_spec,
    _normalize_config_file_group_kind_filter,
    _save_config_file_group_specs,
    _sort_config_file_group_specs_for_display,
)

__all__ = [
    "ROOT",
    "CONFIGS_DIR",
    "GUI_METHODS_DIR",
    "IMPORTED_CONFIGS_DIR",
    "PRESETS_FILE",
    "WEB_FILE_GROUPS_FILE",
    "WEB_USER_LOCKS_FILE",
    "DATASET_PRESETS_DIR",
    "_sync_from_facade",
    "_exported",
    "_sync_common_paths",
    "_load",
    "_safe_resolve",
    "_display_path",
    "get_config_file_meta",
]
