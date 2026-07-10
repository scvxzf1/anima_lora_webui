"""Config file grouping, locking, ordering, export, and restore helpers.

Compatibility facade. Implementation lives in ``file_group_core`` and
``file_group_ops``. Routes/tests should keep importing from this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from library.env import get_configs_root, load_dotenv
from web.services.config import file_group_core as _core
from web.services.config import file_group_ops as _ops

ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = get_configs_root()
GUI_METHODS_DIR = CONFIGS_DIR / "gui-methods"
IMPORTED_CONFIGS_DIR = CONFIGS_DIR / "imported"
PRESETS_FILE = CONFIGS_DIR / "presets.toml"
WEB_FILE_GROUPS_FILE = CONFIGS_DIR / "web-file-groups.toml"
WEB_USER_LOCKS_FILE = CONFIGS_DIR / "web-user-locks.toml"
DATASET_PRESETS_DIR = CONFIGS_DIR / "datasets"

load_dotenv()

# Re-export core helpers used by tests and sibling modules.
_sync_from_facade = _core._sync_from_facade
_exported = _core._exported
_sync_common_paths = _core._sync_common_paths
_load = _core._load
_safe_resolve = _core._safe_resolve
_display_path = _core._display_path
_safe_archive_name = _core._safe_archive_name
_place_index = _core._place_index
_normalize_config_file_group_kind_filter = _core._normalize_config_file_group_kind_filter
_get_config_file_group = _core._get_config_file_group
_config_group_kind = _core._config_group_kind
_config_method_name_for_path = _core._config_method_name_for_path
_infer_config_file_group = _core._infer_config_file_group
_strip_configs_prefix = _core._strip_configs_prefix
_load_config_file_group_specs = _core._load_config_file_group_specs
_sort_config_file_group_specs_for_display = _core._sort_config_file_group_specs_for_display
_save_config_file_group_specs = _core._save_config_file_group_specs
_build_config_file_group = _core._build_config_file_group
_glob_config_files = _core._glob_config_files
_default_config_file_group_specs = _core._default_config_file_group_specs
_group_defaults = _core._group_defaults
_find_config_group_spec = _core._find_config_group_spec
_new_user_config_group_spec = _core._new_user_config_group_spec
_move_orphaned_config_files_to_fallback_groups = _core._move_orphaned_config_files_to_fallback_groups
_config_file_is_covered_by_specs = _core._config_file_is_covered_by_specs
_fallback_config_group_spec = _core._fallback_config_group_spec
_is_user_managed_group = _core._is_user_managed_group
_is_fixed_config_group = _core._is_fixed_config_group
_is_deletable_config_group = _core._is_deletable_config_group
_is_renamable_config_group = _core._is_renamable_config_group
_is_move_target_group = _core._is_move_target_group
_is_sortable_config_group_for_place = _core._is_sortable_config_group_for_place
_lockable_group_ids = _core._lockable_group_ids
_unique_group_id = _core._unique_group_id
_slugify_group_label = _core._slugify_group_label
_normalize_group_label = _core._normalize_group_label
_group_patterns_include_file = _core._group_patterns_include_file
_normalize_config_rel_path = _core._normalize_config_rel_path
_normalize_dataset_preset_path = _core._normalize_dataset_preset_path
_normalize_group_id = _core._normalize_group_id
_unique_archive_member_name = _core._unique_archive_member_name
_is_system_preset_path = _core._is_system_preset_path
_is_system_locked_path = _core._is_system_locked_path
_is_dataset_preset_readonly = _core._is_dataset_preset_readonly
_is_user_locked = _core._is_user_locked
_is_user_group_locked = _core._is_user_group_locked
_load_user_locks = _core._load_user_locks
_save_user_locks = _core._save_user_locks
_lock_reason_label = _core._lock_reason_label
_lock_reason_message = _core._lock_reason_message
_list_system_preset_files = _core._list_system_preset_files
_read_git_head_file = _core._read_git_head_file
_backup_relative_path = _core._backup_relative_path
_string_list = _core._string_list
_config_group_path_list = _core._config_group_path_list

# Public operations
set_user_file_lock = _ops.set_user_file_lock
set_user_group_lock = _ops.set_user_group_lock
create_config_file_group = _ops.create_config_file_group
rename_config_file_group = _ops.rename_config_file_group
delete_config_file_group = _ops.delete_config_file_group
reorder_config_file_group = _ops.reorder_config_file_group
move_config_file_to_group = _ops.move_config_file_to_group
place_config_file_in_group = _ops.place_config_file_in_group
place_config_file_group = _ops.place_config_file_group
reorder_config_file_in_group = _ops.reorder_config_file_in_group
restore_system_presets = _ops.restore_system_presets
list_config_files = _ops.list_config_files
list_config_file_groups = _ops.list_config_file_groups
export_config_file_group_archive = _ops.export_config_file_group_archive
get_config_file_meta = _core.get_config_file_meta

__all__ = ['set_user_file_lock', 'set_user_group_lock', 'create_config_file_group', 'rename_config_file_group', 'delete_config_file_group', 'reorder_config_file_group', 'move_config_file_to_group', 'place_config_file_in_group', 'place_config_file_group', 'reorder_config_file_in_group', 'restore_system_presets', 'list_config_files', 'list_config_file_groups', 'export_config_file_group_archive', 'get_config_file_meta', '_load_config_file_group_specs', '_save_config_file_group_specs', '_normalize_config_file_group_kind_filter', '_normalize_config_rel_path', '_normalize_dataset_preset_path', '_is_dataset_preset_readonly', '_is_user_locked', '_is_user_group_locked', '_load_user_locks', '_save_user_locks', '_lock_reason_message', '_lock_reason_label']

# Ensure facade entrypoints always sync path monkeypatches from config_service.
set_user_file_lock = _core._exported(getattr(set_user_file_lock, "__wrapped__", set_user_file_lock))
set_user_group_lock = _core._exported(getattr(set_user_group_lock, "__wrapped__", set_user_group_lock))
create_config_file_group = _core._exported(getattr(create_config_file_group, "__wrapped__", create_config_file_group))
rename_config_file_group = _core._exported(getattr(rename_config_file_group, "__wrapped__", rename_config_file_group))
delete_config_file_group = _core._exported(getattr(delete_config_file_group, "__wrapped__", delete_config_file_group))
reorder_config_file_group = _core._exported(getattr(reorder_config_file_group, "__wrapped__", reorder_config_file_group))
move_config_file_to_group = _core._exported(getattr(move_config_file_to_group, "__wrapped__", move_config_file_to_group))
place_config_file_in_group = _core._exported(getattr(place_config_file_in_group, "__wrapped__", place_config_file_in_group))
place_config_file_group = _core._exported(getattr(place_config_file_group, "__wrapped__", place_config_file_group))
reorder_config_file_in_group = _core._exported(getattr(reorder_config_file_in_group, "__wrapped__", reorder_config_file_in_group))
restore_system_presets = _core._exported(getattr(restore_system_presets, "__wrapped__", restore_system_presets))
list_config_files = _core._exported(getattr(list_config_files, "__wrapped__", list_config_files))
list_config_file_groups = _core._exported(getattr(list_config_file_groups, "__wrapped__", list_config_file_groups))
export_config_file_group_archive = _core._exported(getattr(export_config_file_group_archive, "__wrapped__", export_config_file_group_archive))

