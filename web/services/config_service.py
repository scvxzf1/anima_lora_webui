"""Compatibility facade for WebUI configuration services.

The implementation is split across ``web.services.config`` modules.  This file
keeps the historical ``web.services.config_service`` import surface stable for
routes, services, tests, and third-party scripts.
"""

from __future__ import annotations

import logging

from web.services.config import _legacy as _legacy

for _name, _value in _legacy.__dict__.items():
    if _name.startswith("__") and _name.endswith("__"):
        continue
    globals()[_name] = _value

LOGGER = logging.getLogger(__name__)
_legacy.LOGGER = LOGGER


_SYNC_LEGACY_NAMES = (
    "ROOT",
    "CONFIGS_DIR",
    "GUI_METHODS_DIR",
    "IMPORTED_CONFIGS_DIR",
    "PRESETS_FILE",
    "WEB_FILE_GROUPS_FILE",
    "WEB_USER_LOCKS_FILE",
    "DEFAULT_SAMPLE_PROMPTS_FILE",
    "DATASET_PRESETS_DIR",
    "resolve_output_root",
    "_display_settings_path",
    "LOGGER",
)


def _sync_legacy_from_facade() -> None:
    for _name in _SYNC_LEGACY_NAMES:
        if _name in globals():
            setattr(_legacy, _name, globals()[_name])


def _load(p):
    _sync_legacy_from_facade()
    return _legacy._load(p)


def _safe_resolve(rel_path: str):
    _sync_legacy_from_facade()
    return _legacy._safe_resolve(rel_path)


def _safe_config_subdir(subdir: str):
    _sync_legacy_from_facade()
    return _legacy._safe_config_subdir(subdir)


def _resolve_project_path(value: str):
    _sync_legacy_from_facade()
    return _legacy._resolve_project_path(value)


def _display_path(path):
    _sync_legacy_from_facade()
    return _legacy._display_path(path)


def _derived_data_dir(source_path, suffix: str):
    _sync_legacy_from_facade()
    return _legacy._derived_data_dir(source_path, suffix)


def _auto_data_dir_for_key(value, source_path, suffix: str):
    _sync_legacy_from_facade()
    return _legacy._auto_data_dir_for_key(value, source_path, suffix)


def _is_builtin_default_data_dir(value: str):
    _sync_legacy_from_facade()
    return _legacy._is_builtin_default_data_dir(value)


# Compatibility facade exports.  Keep this module as the stable import surface
# while implementation responsibility moves into web.services.config.* modules.
from web.services.config.datasets import (  # noqa: E402,F401
    _build_dataset_config_doc,
    _classify_nl_tag_caption_text,
    _dataset_config_path_from_cfg,
    _dataset_rows_for_estimate,
    _dataset_rows_from_config,
    _nl_tag_mix_caption_source,
    _nl_tag_mix_image_files,
    _normalize_dataset_defaults,
    _normalize_dataset_rows,
    _normalize_nl_tag_mix,
    _normalize_path_pattern,
    _normalize_trigger_clone,
    apply_dataset_preset_to_training_config,
    delete_dataset_preset,
    diagnose_dataset_presets,
    import_dataset_preset,
    list_dataset_preset_images,
    list_dataset_presets,
    load_dataset_editor,
    load_dataset_preset,
    resolve_dataset_preview_image,
    save_dataset_editor,
    save_dataset_preset,
    save_dataset_preset_as,
)
from web.services.config.estimation import estimate_training_steps  # noqa: E402,F401
from web.services.config.file_groups import (  # noqa: E402,F401
    _is_dataset_preset_readonly,
    _is_user_group_locked,
    _is_user_locked,
    _load_config_file_group_specs,
    _load_user_locks,
    _lock_reason_label,
    _lock_reason_message,
    _normalize_config_file_group_kind_filter,
    _normalize_config_rel_path,
    _normalize_dataset_preset_path,
    _save_config_file_group_specs,
    _save_user_locks,
    create_config_file_group,
    delete_config_file_group,
    export_config_file_group_archive,
    get_config_file_meta,
    list_config_file_groups,
    list_config_files,
    move_config_file_to_group,
    place_config_file_group,
    place_config_file_in_group,
    rename_config_file_group,
    reorder_config_file_group,
    reorder_config_file_in_group,
    restore_system_presets,
    set_user_file_lock,
    set_user_group_lock,
)
from web.services.config.merge import (  # noqa: E402,F401
    apply_auto_data_dirs,
    list_all_variants,
    list_methods,
    list_presets,
    list_variants,
    load_merged_config,
    suggest_data_dirs,
    suggest_dataset_dirs,
)
from web.services.config.output_runs import (  # noqa: E402,F401
    _normalize_output_run_name,
    _resolve_output_run_dir,
    list_output_runs,
    load_output_run_config,
    save_output_run_config_as,
)
from web.services.config.preflight import (  # noqa: E402,F401
    _check_cache_sidecars,
    _check_dataset_paths,
    _check_dataset_source_paths,
    _check_training_images,
    _config_file_path,
    _load_training_config_for_web_run,
    apply_global_model_path_defaults,
    is_web_runtime_config,
    preflight_training_config,
    training_sample_sampler_status,
)
from web.services.config.raw_files import (  # noqa: E402,F401
    _is_blank_output_name,
    _normalize_patch_value,
    _normalize_saved_raw_config_content,
    _normalize_saved_raw_config_content_with_changed_keys,
    _patch_toml_top_level,
    _prepare_raw_file_patch,
    _restore_dataset_config_after_failed_train_patch,
    delete_raw_file,
    load_raw_file,
    patch_raw_file_values,
    preview_raw_file_patch,
    save_raw_file,
)
from web.services.config.sample_prompts import (  # noqa: E402,F401
    _normalize_prompt_file_path,
    _sample_prompts_path_for_config,
    load_sample_prompts_file,
    save_sample_prompts_file,
)
