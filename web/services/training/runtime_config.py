"""Runtime configuration helpers for WebUI training runs."""

from __future__ import annotations

from web.services.training import runtime_datasets as _runtime_datasets
from web.services.training import runtime_common as _runtime_common_impl
from web.services.training import runtime_prepare as _runtime_prepare_impl
from web.services.training import runtime_resume as _runtime_resume_impl
from web.services.training import runtime_state as _runtime_state_impl
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

RUNTIME_META_KEYS = _runtime_common_impl.RUNTIME_META_KEYS

_LOCAL_IMPL_NAMES = {
    "_resolve_training_runtime_info",
    "_ensure_training_data_dirs",
    "_load_config_file_config",
    "toml_dumps_sorted",
    "_prepare_web_runtime_config",
    "_apply_runtime_env",
    "_runtime_meta",
    "_delete_queue_item_runtime_dir",
    "_queue_item_runtime_dir_label",
    "_queue_item_runtime_delete_dir",
    "_validate_queue_runtime_dir_match",
    "_path_is_relative_to",
    "_write_runtime_run_meta",
    "_read_runtime_run_meta",
    "_runtime_from_config_file",
    "_clone_frozen_runtime_config",
    "_apply_resume_duration_overrides",
    "_normalize_resume_duration_overrides",
    "_estimate_resume_steps_per_epoch",
    "_count_resume_images",
    "_positive_int_value",
    "_positive_float_value",
    "_drop_resume_hotstart_overrides",
    "_clone_runtime_dataset_rows",
    "_runtime_dataset_child_name",
    "_bool_value_for_row",
    "_prepare_runtime_nl_tag_mix_source",
    "_prepare_runtime_trigger_clone_source",
    "_nl_tag_mix_caption_settings",
    "_build_nl_tag_mix_source",
    "_classify_nl_tag_mix_samples",
    "_nl_tag_mix_caption_entries",
    "_nl_tag_mix_source_counts",
    "_nl_tag_mix_dominant_source",
    "_cycle_nl_tag_entries",
    "_select_nl_tag_caption_entries",
    "_nl_tag_mix_relative_image_path",
    "_select_nl_tag_mix_samples",
    "_copy_nl_tag_caption_sidecars",
    "_copy_runtime_dataset_dir",
    "_is_materialized_runtime_source_dir",
    "_unique_runtime_dir",
    "_safe_run_stem",
    "_is_web_runtime_dir",
    "_path_exists",
    "_sample_config_from_cfg",
    "_cli_arg_overrides",
    "_resolve_display_path",
    "_display_project_path",
}


load_merged_config = _runtime_common_impl.load_merged_config
apply_auto_data_dirs = _runtime_common_impl.apply_auto_data_dirs
apply_global_model_path_defaults = _runtime_common_impl.apply_global_model_path_defaults
_dataset_rows_for_estimate = _runtime_common_impl._dataset_rows_for_estimate
_build_dataset_config_doc = _runtime_common_impl._build_dataset_config_doc
_normalize_path_pattern = _runtime_common_impl._normalize_path_pattern
_normalize_trigger_clone = _runtime_common_impl._normalize_trigger_clone
_normalize_nl_tag_mix = _runtime_common_impl._normalize_nl_tag_mix
_nl_tag_mix_image_files = _runtime_common_impl._nl_tag_mix_image_files
_nl_tag_mix_caption_source = _runtime_common_impl._nl_tag_mix_caption_source
_classify_nl_tag_caption_text = _runtime_common_impl._classify_nl_tag_caption_text
training_sample_sampler_status = _runtime_common_impl.training_sample_sampler_status
_load_config_file_config = _runtime_common_impl._load_config_file_config
toml_dumps_sorted = _runtime_common_impl.toml_dumps_sorted
_default_preprocess_precision_preference = _runtime_common_impl._default_preprocess_precision_preference
_sample_config_from_cfg = _runtime_common_impl._sample_config_from_cfg
_cli_arg_overrides = _runtime_common_impl._cli_arg_overrides

_resolve_training_runtime_info = _runtime_prepare_impl._resolve_training_runtime_info
_ensure_training_data_dirs = _runtime_prepare_impl._ensure_training_data_dirs
_prepare_web_runtime_config = _runtime_prepare_impl._prepare_web_runtime_config

_apply_runtime_env = _runtime_state_impl._apply_runtime_env
_runtime_meta = _runtime_state_impl._runtime_meta
_delete_queue_item_runtime_dir = _runtime_state_impl._delete_queue_item_runtime_dir
_queue_item_runtime_dir_label = _runtime_state_impl._queue_item_runtime_dir_label
_queue_item_runtime_delete_dir = _runtime_state_impl._queue_item_runtime_delete_dir
_validate_queue_runtime_dir_match = _runtime_state_impl._validate_queue_runtime_dir_match
_write_runtime_run_meta = _runtime_state_impl._write_runtime_run_meta
_read_runtime_run_meta = _runtime_state_impl._read_runtime_run_meta
_runtime_from_config_file = _runtime_state_impl._runtime_from_config_file
_is_web_runtime_dir = _runtime_state_impl._is_web_runtime_dir

_clone_frozen_runtime_config = _runtime_resume_impl._clone_frozen_runtime_config
_apply_resume_duration_overrides = _runtime_resume_impl._apply_resume_duration_overrides
_normalize_resume_duration_overrides = _runtime_resume_impl._normalize_resume_duration_overrides
_estimate_resume_steps_per_epoch = _runtime_resume_impl._estimate_resume_steps_per_epoch
_count_resume_images = _runtime_resume_impl._count_resume_images
_positive_int_value = _runtime_resume_impl._positive_int_value
_positive_float_value = _runtime_resume_impl._positive_float_value
_drop_resume_hotstart_overrides = _runtime_resume_impl._drop_resume_hotstart_overrides

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
