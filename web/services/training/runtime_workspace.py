"""Compatibility facade for split runtime workspace helpers."""

from __future__ import annotations

from web.services.training import runtime_datasets as _runtime_datasets
from web.services.training.runtime_common import (
    RUNTIME_META_KEYS,
    _build_dataset_config_doc,
    _classify_nl_tag_caption_text,
    _cli_arg_overrides,
    _dataset_rows_for_estimate,
    _default_preprocess_precision_preference,
    _load_config_file_config,
    _nl_tag_mix_caption_source,
    _nl_tag_mix_image_files,
    _normalize_nl_tag_mix,
    _normalize_path_pattern,
    _normalize_trigger_clone,
    _sample_config_from_cfg,
    apply_auto_data_dirs,
    apply_global_model_path_defaults,
    load_merged_config,
    toml_dumps_sorted,
    training_sample_sampler_status,
)
from web.services.training.runtime_paths import (
    _display_project_path,
    _display_settings_path,
    _path_exists,
    _path_is_relative_to,
    _resolve_display_path,
    _safe_run_stem,
    _unique_runtime_dir,
    resolve_output_root,
)
from web.services.training.runtime_prepare import (
    _ensure_training_data_dirs,
    _prepare_web_runtime_config,
    _resolve_training_runtime_info,
)
from web.services.training.runtime_resume import (
    _apply_resume_duration_overrides,
    _clone_frozen_runtime_config,
    _count_resume_images,
    _drop_resume_hotstart_overrides,
    _estimate_resume_steps_per_epoch,
    _normalize_resume_duration_overrides,
    _positive_float_value,
    _positive_int_value,
)
from web.services.training.runtime_state import (
    _apply_runtime_env,
    _delete_queue_item_runtime_dir,
    _is_web_runtime_dir,
    _queue_item_runtime_delete_dir,
    _queue_item_runtime_dir_label,
    _read_runtime_run_meta,
    _runtime_from_config_file,
    _runtime_meta,
    _validate_queue_runtime_dir_match,
    _write_runtime_run_meta,
)

_clone_runtime_dataset_rows = _runtime_datasets._clone_runtime_dataset_rows
_runtime_dataset_child_name = _runtime_datasets._runtime_dataset_child_name
_bool_value_for_row = _runtime_datasets._bool_value_for_row
_prepare_runtime_nl_tag_mix_source = _runtime_datasets._prepare_runtime_nl_tag_mix_source
_prepare_runtime_trigger_clone_source = _runtime_datasets._prepare_runtime_trigger_clone_source
_nl_tag_mix_caption_settings = _runtime_datasets._nl_tag_mix_caption_settings
_build_nl_tag_mix_source = _runtime_datasets._build_nl_tag_mix_source
_classify_nl_tag_mix_samples = _runtime_datasets._classify_nl_tag_mix_samples
_nl_tag_mix_caption_entries = _runtime_datasets._nl_tag_mix_caption_entries
_nl_tag_mix_source_counts = _runtime_datasets._nl_tag_mix_source_counts
_nl_tag_mix_dominant_source = _runtime_datasets._nl_tag_mix_dominant_source
_cycle_nl_tag_entries = _runtime_datasets._cycle_nl_tag_entries
_select_nl_tag_caption_entries = _runtime_datasets._select_nl_tag_caption_entries
_nl_tag_mix_relative_image_path = _runtime_datasets._nl_tag_mix_relative_image_path
_select_nl_tag_mix_samples = _runtime_datasets._select_nl_tag_mix_samples
_copy_nl_tag_caption_sidecars = _runtime_datasets._copy_nl_tag_caption_sidecars
_copy_runtime_dataset_dir = _runtime_datasets._copy_runtime_dataset_dir
_is_materialized_runtime_source_dir = _runtime_datasets._is_materialized_runtime_source_dir
