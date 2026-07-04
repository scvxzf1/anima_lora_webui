"""Configuration loading, merging, and saving."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from library.env import load_dotenv, get_configs_root
from library.preprocess.captions import (
    CAPTION_SOURCE_AUTO,
)
from web.services.config.metadata import (
    CAPTION_SOURCE_MODE_LABELS,  # noqa: F401 - re-exported for legacy facade compatibility
    DATASET_CAPTION_EXTS,  # noqa: F401 - re-exported for legacy facade compatibility
    DATASET_CAPTION_MAX_CHARS,  # noqa: F401 - re-exported for legacy facade compatibility
    DATASET_IMAGE_EXTS,  # noqa: F401 - re-exported for legacy facade compatibility
    DATASET_PREVIEW_LIMIT,  # noqa: F401 - re-exported for legacy facade compatibility
    DATASET_SETTING_KEYS,  # noqa: F401 - re-exported for legacy facade compatibility
    DEFAULT_LORA_CACHE_DIR,  # noqa: F401 - re-exported for legacy facade compatibility
    DEFAULT_NL_TAG_MIX_TAG_RATIO,  # noqa: F401 - re-exported for legacy facade compatibility
    DEFAULT_RESIZED_IMAGE_DIR,  # noqa: F401 - re-exported for legacy facade compatibility
    FILE_MOVE_TARGET_GROUPS,  # noqa: F401 - re-exported for legacy facade compatibility
    FIXED_SYSTEM_CONFIG_GROUP_IDS,  # noqa: F401 - re-exported for legacy facade compatibility
    HIDDEN_CONFIG_FILES,  # noqa: F401 - re-exported for legacy facade compatibility
    HIDDEN_DATASET_PRESET_FILES,  # noqa: F401 - re-exported for legacy facade compatibility
    LEGACY_TRAINING_SAMPLE_SAMPLERS,  # noqa: F401 - re-exported for legacy facade compatibility
    NL_TAG_MIX_ATTR_KEY,  # noqa: F401 - re-exported for legacy facade compatibility
    NL_TAG_MIX_CLASSIFICATION_METHOD,  # noqa: F401 - re-exported for legacy facade compatibility
    OUTPUT_RUN_CONFIG_FILES,  # noqa: F401 - re-exported for legacy facade compatibility
    PREPROCESS_DATASET_SETTING_KEYS,  # noqa: F401 - re-exported for legacy facade compatibility
    PREPROCESS_DATASET_SETTING_ORDER,  # noqa: F401 - re-exported for legacy facade compatibility
    PREPROCESS_ENV_CHECK_KEY,  # noqa: F401 - re-exported for legacy facade compatibility
    PREPROCESS_ENV_REQUIRED_FILES,  # noqa: F401 - re-exported for legacy facade compatibility
    RETIRED_TOP_LEVEL_CONFIG_FIELDS,  # noqa: F401 - re-exported for legacy facade compatibility
    RUNTIME_PREPROCESS_ATTR_KEY,  # noqa: F401 - re-exported for legacy facade compatibility
    SPD_NESTED_PATCH_FIELDS,  # noqa: F401 - re-exported for legacy facade compatibility
    SUPPORTED_TRAINING_SAMPLE_SAMPLERS,  # noqa: F401 - re-exported for legacy facade compatibility
    SYSTEM_CONFIG_GROUP_IDS,  # noqa: F401 - re-exported for legacy facade compatibility
    SYSTEM_DATASET_PRESET_FILES,  # noqa: F401 - re-exported for legacy facade compatibility
    SYSTEM_MANAGED_FILES,  # noqa: F401 - re-exported for legacy facade compatibility
    SYSTEM_PRESET_FILES,  # noqa: F401 - re-exported for legacy facade compatibility
    SYSTEM_PRESET_PREFIXES,  # noqa: F401 - re-exported for legacy facade compatibility
    TRIGGER_CLONE_ATTR_KEY,  # noqa: F401 - re-exported for legacy facade compatibility
    UI_ONLY_CONFIG_FIELDS,  # noqa: F401 - re-exported for legacy facade compatibility
    USER_LOCKABLE_GROUPS,  # noqa: F401 - re-exported for legacy facade compatibility
    get_field_help,  # noqa: F401 - re-exported for legacy facade compatibility
    get_groups,  # noqa: F401 - re-exported for legacy facade compatibility
)
from web.services.settings_service import display_path as _display_settings_path  # noqa: F401 - synced into split modules
from web.services.settings_service import resolve_output_root  # noqa: F401 - synced into split modules

ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = get_configs_root()
GUI_METHODS_DIR = CONFIGS_DIR / "gui-methods"
IMPORTED_CONFIGS_DIR = CONFIGS_DIR / "imported"
PRESETS_FILE = CONFIGS_DIR / "presets.toml"
WEB_FILE_GROUPS_FILE = CONFIGS_DIR / "web-file-groups.toml"
WEB_USER_LOCKS_FILE = CONFIGS_DIR / "web-user-locks.toml"
DEFAULT_SAMPLE_PROMPTS_FILE = str(CONFIGS_DIR / "sample_prompts.txt")
DEFAULT_MAX_TRAIN_STEPS = 0
DATASET_PRESETS_DIR = CONFIGS_DIR / "datasets"
_DELETE_TOML_KEY = object()

load_dotenv()

LOGGER = logging.getLogger(__name__)


def list_methods() -> list[str]:
    return _call_merge_impl("list_methods")


def list_variants(method: str) -> list[str]:
    return _call_merge_impl("list_variants", method)


def _builtin_variants_by_family() -> dict[str, list[tuple[int, str]]]:
    return _call_merge_impl("_builtin_variants_by_family")


def _read_variant_metadata(path: Path) -> dict[str, Any]:
    return _call_merge_impl("_read_variant_metadata", path)


def _legacy_exact_variant_for_method(method: str) -> list[str]:
    return _call_merge_impl("_legacy_exact_variant_for_method", method)


def _custom_gui_variants() -> list[str]:
    return _call_merge_impl("_custom_gui_variants")


def list_all_variants() -> list[str]:
    return _call_merge_impl("list_all_variants")


def list_presets() -> list[str]:
    return _call_merge_impl("list_presets")


def load_merged_config(variant: str, preset: str, methods_subdir: str = "gui-methods") -> dict[str, Any]:
    return _call_merge_impl("load_merged_config", variant, preset, methods_subdir)


def suggest_data_dirs(source_image_dir: str) -> dict[str, Any]:
    return _call_merge_impl("suggest_data_dirs", source_image_dir)


def suggest_dataset_dirs(source_dirs: list[str]) -> dict[str, Any]:
    return _call_merge_impl("suggest_dataset_dirs", source_dirs)


def list_dataset_presets() -> dict[str, Any]:
    return _call_dataset_impl("list_dataset_presets")


def diagnose_dataset_presets(rel_path: str = "") -> dict[str, Any]:
    return _call_dataset_impl("diagnose_dataset_presets", rel_path)


def load_dataset_preset(rel_path: str) -> dict[str, Any]:
    return _call_dataset_impl("load_dataset_preset", rel_path)


def save_dataset_preset(
    rel_path: str,
    rows: list[dict[str, Any]],
    defaults: dict[str, Any] | None = None,
    *,
    overwrite: bool = True,
) -> dict[str, Any]:
    return _call_dataset_impl(
        "save_dataset_preset",
        rel_path,
        rows,
        defaults,
        overwrite=overwrite,
    )


def save_dataset_preset_as(
    name: str,
    rows: list[dict[str, Any]],
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _call_dataset_impl("save_dataset_preset_as", name, rows, defaults)


def import_dataset_preset(
    name: str,
    content: str,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    return _call_dataset_impl("import_dataset_preset", name, content, overwrite=overwrite)


def delete_dataset_preset(rel_path: str) -> dict[str, Any]:
    return _call_dataset_impl("delete_dataset_preset", rel_path)


def apply_dataset_preset_to_training_config(
    dataset_file: str,
    train_file: str,
    train_content: str | None = None,
) -> dict[str, Any]:
    return _call_dataset_impl(
        "apply_dataset_preset_to_training_config",
        dataset_file,
        train_file,
        train_content,
    )


def list_dataset_preset_images(
    dataset_file: str,
    dataset_index: int = 0,
    *,
    source: str = "training",
    limit: int = DATASET_PREVIEW_LIMIT,
) -> dict[str, Any]:
    return _call_dataset_impl(
        "list_dataset_preset_images",
        dataset_file,
        dataset_index,
        source=source,
        limit=limit,
    )


def resolve_dataset_preview_image(
    dataset_file: str,
    dataset_index: int,
    image_file: str,
    *,
    source: str = "training",
) -> Path:
    return _call_dataset_impl(
        "resolve_dataset_preview_image",
        dataset_file,
        dataset_index,
        image_file,
        source=source,
    )


def load_dataset_editor(
    variant: str,
    preset: str,
    methods_subdir: str = "gui-methods",
    *,
    config_file: str | None = None,
    dataset_config: str | None = None,
) -> dict[str, Any]:
    return _call_dataset_impl(
        "load_dataset_editor",
        variant,
        preset,
        methods_subdir,
        config_file=config_file,
        dataset_config=dataset_config,
    )


def save_dataset_editor(
    variant: str,
    preset: str,
    methods_subdir: str,
    rows: list[dict[str, Any]],
    defaults: dict[str, Any] | None = None,
    config_values: dict[str, Any] | None = None,
    train_file: str | None = None,
    train_content: str | None = None,
    prefer_existing_dataset_config: bool = True,
) -> dict[str, Any]:
    return _call_dataset_impl(
        "save_dataset_editor",
        variant,
        preset,
        methods_subdir,
        rows,
        defaults,
        config_values,
        train_file,
        train_content,
        prefer_existing_dataset_config,
    )


def apply_auto_data_dirs(cfg: dict[str, Any], *, create: bool = False) -> dict[str, Any]:
    return _call_merge_impl("apply_auto_data_dirs", cfg, create=create)


def preflight_training_config(
    variant: str,
    preset: str,
    methods_subdir: str = "gui-methods",
    *,
    config_file: str | None = None,
) -> dict[str, Any]:
    return _call_preflight_impl(
        "preflight_training_config",
        variant,
        preset,
        methods_subdir,
        config_file=config_file,
    )


def _inspect_network_weight(
    path: str,
    *,
    variant: str,
    preset: str,
    methods_subdir: str,
    config_file: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _call_preflight_impl(
        "_inspect_network_weight_impl",
        path,
        variant=variant,
        preset=preset,
        methods_subdir=methods_subdir,
        config_file=config_file,
        cfg=cfg,
    )


def _check_network_weights(
    cfg: dict[str, Any],
    add,
    variant: str,
    preset: str,
    methods_subdir: str,
    config_file: str | None,
) -> None:
    return _call_preflight_impl(
        "_check_network_weights",
        cfg,
        add,
        variant,
        preset,
        methods_subdir,
        config_file,
    )


def _check_training_sample_config(cfg: dict[str, Any], add) -> None:
    return _call_preflight_impl("_check_training_sample_config", cfg, add)


def _load_training_config_for_web_run(
    variant: str,
    preset: str,
    methods_subdir: str,
    *,
    config_file: str | None = None,
) -> dict[str, Any]:
    return _call_preflight_impl(
        "_load_training_config_for_web_run",
        variant,
        preset,
        methods_subdir,
        config_file=config_file,
    )


def _config_file_path(config_file: str | None) -> Path | None:
    return _call_preflight_impl("_config_file_path", config_file)


def _config_path_from_display_path(normalized: str) -> Path | None:
    return _call_preflight_impl("_config_path_from_display_path", normalized)


def _is_allowed_training_config_path(path: Path) -> bool:
    return _call_preflight_impl("_is_allowed_training_config_path", path)


def _is_web_runtime_config_tree(path: Path) -> bool:
    return _call_preflight_impl("_is_web_runtime_config_tree", path)


def _is_output_run_snapshot_config(path: Path) -> bool:
    return _call_preflight_impl("_is_output_run_snapshot_config", path)


def _has_web_runtime_dirs(run_dir: Path) -> bool:
    return _call_preflight_impl("_has_web_runtime_dirs", run_dir)


def is_web_runtime_config(config_file: str | None) -> bool:
    return _call_preflight_impl("is_web_runtime_config", config_file)


def _looks_like_web_runtime_config(cfg: dict[str, Any]) -> bool:
    return _call_preflight_impl("_looks_like_web_runtime_config", cfg)


def _check_web_preprocess_environment(add) -> None:
    return _call_preflight_impl("_check_web_preprocess_environment", add)


def _web_python_executable() -> str:
    return _call_preflight_impl("_web_python_executable")


def estimate_training_steps(
    variant: str,
    preset: str,
    methods_subdir: str = "gui-methods",
    *,
    config_file: str | None = None,
    dataset_config: str | None = None,
) -> dict[str, Any]:
    return _call_estimation_impl(
        "estimate_training_steps",
        variant,
        preset,
        methods_subdir,
        config_file=config_file,
        dataset_config=dataset_config,
    )


def _dataset_config_path_from_cfg(cfg: dict[str, Any]) -> Path | None:
    return _call_dataset_impl("_dataset_config_path_from_cfg", cfg)


def _is_allowed_dataset_config_path(path: Path) -> bool:
    return _call_dataset_impl("_is_allowed_dataset_config_path", path)


def _dataset_config_rel_path(
    cfg: dict[str, Any],
    variant: str,
    methods_subdir: str,
    *,
    prefer_existing: bool = True,
) -> str:
    return _call_dataset_impl(
        "_dataset_config_rel_path",
        cfg,
        variant,
        methods_subdir,
        prefer_existing=prefer_existing,
    )


def _training_config_rel_path(variant: str, methods_subdir: str) -> str:
    return _call_dataset_impl("_training_config_rel_path", variant, methods_subdir)


def _single_dataset_config_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    return _call_dataset_impl("_single_dataset_config_from_cfg", cfg)


def _dataset_defaults_from_config(data: dict[str, Any]) -> dict[str, Any]:
    return _call_dataset_impl("_dataset_defaults_from_config", data)


def _dataset_defaults_from_dataset(dataset: dict[str, Any], data: dict[str, Any] | None = None) -> dict[str, Any]:
    return _call_dataset_impl("_dataset_defaults_from_dataset", dataset, data)


def _dataset_preset_summary(rel_path: str) -> dict[str, Any]:
    return _call_dataset_impl("_dataset_preset_summary", rel_path)


def _dataset_preset_groups_for_ui(presets_by_path: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return _call_dataset_impl("_dataset_preset_groups_for_ui", presets_by_path)


def _is_dataset_group_for_ui(group: dict[str, Any], files: list[dict[str, Any]]) -> bool:
    return _call_dataset_impl("_is_dataset_group_for_ui", group, files)


def _dataset_summary_from_rows(rows: list[dict[str, Any]], defaults: dict[str, Any] | None = None) -> dict[str, Any]:
    return _call_dataset_impl("_dataset_summary_from_rows", rows, defaults)


def _dataset_rows_for_estimate(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return _call_dataset_impl("_dataset_rows_for_estimate", cfg)


def _dataset_rows_from_config(data: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return _call_dataset_impl("_dataset_rows_from_config", data, cfg)


def _normalize_dataset_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _call_dataset_impl("_normalize_dataset_rows", rows)


def _normalize_dataset_row_settings(raw: dict[str, Any]) -> dict[str, Any]:
    return _call_dataset_impl("_normalize_dataset_row_settings", raw)


def _fill_missing_dataset_row_settings(rows: list[dict[str, Any]], defaults: dict[str, Any]) -> list[dict[str, Any]]:
    return _call_dataset_impl("_fill_missing_dataset_row_settings", rows, defaults)


def _normalize_dataset_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    return _call_dataset_impl("_normalize_dataset_defaults", raw)


def _normalize_preprocess_dataset_settings(raw: dict[str, Any]) -> dict[str, Any]:
    return _call_dataset_impl("_normalize_preprocess_dataset_settings", raw)


def _normalize_nl_tag_mix(raw: Any) -> dict[str, Any]:
    return _call_dataset_impl("_normalize_nl_tag_mix", raw)


def _normalize_trigger_clone(raw: Any) -> dict[str, Any]:
    return _call_dataset_impl("_normalize_trigger_clone", raw)


def _trigger_clone_should_persist(clone: dict[str, Any]) -> bool:
    return _call_dataset_impl("_trigger_clone_should_persist", clone)


def _nl_tag_mix_enabled(row: dict[str, Any]) -> bool:
    return _call_dataset_impl("_nl_tag_mix_enabled", row)


def _preprocess_settings_from_custom_attributes(attrs: dict[str, Any]) -> dict[str, Any]:
    return _call_dataset_impl("_preprocess_settings_from_custom_attributes", attrs)


def _preprocess_settings_for_runtime_attrs(row_cfg: dict[str, Any]) -> dict[str, Any]:
    return _call_dataset_impl("_preprocess_settings_for_runtime_attrs", row_cfg)


def _build_dataset_config_doc(
    clean_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    *,
    prefer_train_batch_size: bool = False,
    include_preprocess_settings: bool = True,
) -> str:
    return _call_dataset_impl(
        "_build_dataset_config_doc",
        clean_rows,
        cfg,
        prefer_train_batch_size=prefer_train_batch_size,
        include_preprocess_settings=include_preprocess_settings,
    )


def _dataset_row_settings(row: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    return _call_dataset_impl("_dataset_row_settings", row, fallback)


def _first_dataset_settings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return _call_dataset_impl("_first_dataset_settings", rows)


def _first_dataset_value(data: dict[str, Any], key: str, default: Any = None) -> Any:
    return _call_dataset_impl("_first_dataset_value", data, key, default)


def _dataset_path_value(value: Any, cfg: dict[str, Any]) -> str:
    return _call_dataset_impl("_dataset_path_value", value, cfg)


def _list_dataset_image_files(
    directory: Path,
    limit: int,
    *,
    recursive: bool = True,
    path_pattern: str = "*",
) -> dict[str, Any]:
    return _call_dataset_impl(
        "_list_dataset_image_files",
        directory,
        limit,
        recursive=recursive,
        path_pattern=path_pattern,
    )


def _dataset_image_preview_meta(
    path: Path,
    *,
    preset_file: str,
    dataset_index: int,
    source: str,
    caption_extension: str,
    prefer_json_caption: bool,
    caption_source_mode: str,
    source_dir: Path,
    train_dir: Path,
) -> dict[str, Any]:
    return _call_dataset_impl(
        "_dataset_image_preview_meta",
        path,
        preset_file=preset_file,
        dataset_index=dataset_index,
        source=source,
        caption_extension=caption_extension,
        prefer_json_caption=prefer_json_caption,
        caption_source_mode=caption_source_mode,
        source_dir=source_dir,
        train_dir=train_dir,
    )


def _dataset_image_dimensions(path: Path) -> dict[str, int]:
    return _call_dataset_impl("_dataset_image_dimensions", path)


def _dataset_caption_meta(
    path: Path,
    caption_extension: str,
    source_dir: Path,
    train_dir: Path,
    *,
    prefer_json_caption: bool = False,
    caption_source_mode: str | None = None,
) -> dict[str, Any]:
    return _call_dataset_impl(
        "_dataset_caption_meta",
        path,
        caption_extension,
        source_dir,
        train_dir,
        prefer_json_caption=prefer_json_caption,
        caption_source_mode=caption_source_mode,
    )


def _caption_source_mode_label(mode: str | None) -> str:
    return _call_dataset_impl("_caption_source_mode_label", mode)


def _caption_extension_for_detected_mode(mode: str | None, fallback: str) -> str:
    return _call_dataset_impl("_caption_extension_for_detected_mode", mode, fallback)


def _format_caption_preview_text(texts: list[str]) -> str:
    return _call_dataset_impl("_format_caption_preview_text", texts)


def _dataset_caption_detection_summary(images: list[dict[str, Any]]) -> str:
    return _call_dataset_impl("_dataset_caption_detection_summary", images)


def _caption_detection_counts_text(counts: dict[str, int], caption_total: int) -> str:
    return _call_dataset_impl("_caption_detection_counts_text", counts, caption_total)


def _dataset_preview_empty_message(directory: Path, source: str) -> str:
    return _call_dataset_impl("_dataset_preview_empty_message", directory, source)


def _safe_file_stem(value: str) -> str:
    return _call_dataset_impl("_safe_file_stem", value)


def _normalize_path_pattern(value: Any) -> str:
    return _call_dataset_impl("_normalize_path_pattern", value)


def _dataset_image_files(
    path: Path,
    image_exts: set[str] | frozenset[str],
    *,
    recursive: bool = True,
    path_pattern: str = "*",
) -> list[Path]:
    return _call_dataset_impl(
        "_dataset_image_files",
        path,
        image_exts,
        recursive=recursive,
        path_pattern=path_pattern,
    )


def _count_images(
    path: Path,
    image_exts: set[str],
    *,
    recursive: bool = True,
    path_pattern: str = "*",
) -> int:
    return _call_dataset_impl(
        "_count_images",
        path,
        image_exts,
        recursive=recursive,
        path_pattern=path_pattern,
    )


def _count_source_images(
    path: Path,
    image_exts: set[str],
    *,
    recursive: bool = True,
    path_pattern: str = "*",
) -> int:
    return _call_dataset_impl(
        "_count_source_images",
        path,
        image_exts,
        recursive=recursive,
        path_pattern=path_pattern,
    )


def _dataset_num_repeats(cfg: dict[str, Any]) -> int:
    return _call_dataset_impl("_dataset_num_repeats", cfg)


def _positive_int(value: Any, fallback: int) -> int:
    return _call_common_impl("_positive_int", value, fallback)


def _positive_int_or_none(value: Any) -> int | None:
    return _call_common_impl("_positive_int_or_none", value)


def _nonnegative_int(value: Any, fallback: int) -> int:
    return _call_common_impl("_nonnegative_int", value, fallback)


def _nonnegative_float(value: Any, fallback: float) -> float:
    return _call_common_impl("_nonnegative_float", value, fallback)


def _positive_float(value: Any, fallback: float) -> float:
    return _call_common_impl("_positive_float", value, fallback)


def _bool_value(value: Any, fallback: bool = False) -> bool:
    return _call_common_impl("_bool_value", value, fallback)


def training_sample_sampler_status(value: Any) -> tuple[str, str]:
    return _call_preflight_impl("training_sample_sampler_status", value)


def list_output_runs(limit: int = 200) -> dict[str, Any]:
    return _call_output_runs_impl("list_output_runs", limit)


def load_output_run_config(run: str, kind: str) -> dict[str, Any]:
    return _call_output_runs_impl("load_output_run_config", run, kind)


def save_output_run_config_as(run: str, name: str, target_group: str | None = None) -> dict[str, Any]:
    return _call_output_runs_impl("save_output_run_config_as", run, name, target_group)


def _output_run_summary(run_dir: Path) -> dict[str, Any]:
    return _call_output_runs_impl("_output_run_summary", run_dir)


def _resolve_output_run_dir(run: str) -> Path:
    return _call_output_runs_impl("_resolve_output_run_dir", run)


def _normalize_output_run_name(run: str) -> str:
    return _call_output_runs_impl("_normalize_output_run_name", run)


def _output_run_config_path(run_dir: Path, kind: str) -> Path:
    return _call_output_runs_impl("_output_run_config_path", run_dir, kind)


def _normalize_output_run_save_as_path(value: str, *, fallback_stem: str) -> str:
    return _call_output_runs_impl(
        "_normalize_output_run_save_as_path",
        value,
        fallback_stem=fallback_stem,
    )


def _safe_mtime(path: Path) -> float:
    return _call_output_runs_impl("_safe_mtime", path)


def _format_file_time(value: float) -> str:
    return _call_output_runs_impl("_format_file_time", value)


def load_raw_file(rel_path: str) -> str:
    return _call_raw_files_impl("load_raw_file", rel_path)


def load_sample_prompts_file(rel_path: str | None = None) -> dict[str, Any]:
    return _call_sample_prompts_impl("load_sample_prompts_file", rel_path)


def save_sample_prompts_file(
    content: str,
    rel_path: str | None = None,
    *,
    train_config_file: str | None = None,
) -> dict[str, Any]:
    return _call_sample_prompts_impl(
        "save_sample_prompts_file",
        content,
        rel_path,
        train_config_file=train_config_file,
    )


def save_raw_file(
    rel_path: str,
    content: str,
    *,
    allow_locked: bool = False,
    overwrite: bool = True,
) -> tuple[bool, str]:
    return _call_raw_files_impl(
        "save_raw_file",
        rel_path,
        content,
        allow_locked=allow_locked,
        overwrite=overwrite,
    )


def delete_raw_file(rel_path: str) -> tuple[bool, str]:
    return _call_raw_files_impl("delete_raw_file", rel_path)


def patch_raw_file_values(
    rel_path: str,
    values: dict[str, Any],
    *,
    content: str | None = None,
) -> tuple[bool, str, str, list[str]]:
    return _call_raw_files_impl("patch_raw_file_values", rel_path, values, content=content)


def preview_raw_file_patch(
    rel_path: str,
    values: dict[str, Any],
    *,
    content: str | None = None,
) -> tuple[bool, str, str, list[str]]:
    return _call_raw_files_impl("preview_raw_file_patch", rel_path, values, content=content)


def _prepare_raw_file_patch(
    rel_path: str,
    values: dict[str, Any],
    *,
    content: str | None = None,
) -> tuple[bool, str, Path | None, str, list[str]]:
    return _call_raw_files_impl(
        "_prepare_raw_file_patch",
        rel_path,
        values,
        content=content,
    )


def _restore_dataset_config_after_failed_train_patch(path: Path, existed: bool, previous_content: str) -> None:
    return _call_raw_files_impl(
        "_restore_dataset_config_after_failed_train_patch",
        path,
        existed,
        previous_content,
    )


def set_user_file_lock(rel_path: str, locked: bool) -> tuple[bool, str, dict[str, Any]]:
    return _call_file_groups_impl("set_user_file_lock", rel_path, locked)


def set_user_group_lock(group_id: str, locked: bool) -> tuple[bool, str, dict[str, Any]]:
    return _call_file_groups_impl("set_user_group_lock", group_id, locked)


def create_config_file_group(label: str, kind: str = "training") -> tuple[bool, str, dict[str, Any] | None]:
    return _call_file_groups_impl("create_config_file_group", label, kind)


def rename_config_file_group(group_id: str, label: str) -> tuple[bool, str, dict[str, Any] | None]:
    return _call_file_groups_impl("rename_config_file_group", group_id, label)


def delete_config_file_group(group_id: str) -> tuple[bool, str]:
    return _call_file_groups_impl("delete_config_file_group", group_id)


def reorder_config_file_group(group_id: str, direction: str) -> tuple[bool, str, dict[str, Any] | None]:
    return _call_file_groups_impl("reorder_config_file_group", group_id, direction)


def move_config_file_to_group(rel_path: str, group_id: str) -> tuple[bool, str, dict[str, Any] | None]:
    return _call_file_groups_impl("move_config_file_to_group", rel_path, group_id)


def place_config_file_in_group(
    rel_path: str,
    group_id: str,
    index: Any | None = None,
) -> tuple[bool, str, dict[str, Any] | None]:
    return _call_file_groups_impl("place_config_file_in_group", rel_path, group_id, index)


def place_config_file_group(
    group_id: str,
    scope: str,
    index: Any | None,
) -> tuple[bool, str, dict[str, Any] | None]:
    return _call_file_groups_impl("place_config_file_group", group_id, scope, index)


def reorder_config_file_in_group(
    rel_path: str,
    group_id: str,
    direction: str,
) -> tuple[bool, str, dict[str, Any] | None]:
    return _call_file_groups_impl("reorder_config_file_in_group", rel_path, group_id, direction)


def restore_system_presets(files: list[str] | None = None) -> dict[str, Any]:
    return _call_file_groups_impl("restore_system_presets", files)


def list_config_files() -> list[str]:
    return _call_file_groups_impl("list_config_files")


def list_config_file_groups(kind: str | None = None) -> list[dict[str, Any]]:
    return _call_file_groups_impl("list_config_file_groups", kind)


def export_config_file_group_archive(group_id: str, kind: str | None = "training") -> dict[str, Any]:
    return _call_file_groups_impl("export_config_file_group_archive", group_id, kind)


def _normalize_config_file_group_kind_filter(kind: str | None) -> str:
    return _call_file_groups_impl("_normalize_config_file_group_kind_filter", kind)


def _get_config_file_group(group_id: str) -> dict[str, Any] | None:
    return _call_file_groups_impl("_get_config_file_group", group_id)


def _config_group_kind(raw: dict[str, Any]) -> str:
    return _call_file_groups_impl("_config_group_kind", raw)


def get_config_file_meta(
    rel_path: str,
    group_id: str | None = None,
    group_label: str | None = None,
    locked: bool | None = None,
    trainable: bool | None = None,
    methods_subdir: str | None = None,
) -> dict[str, Any]:
    return _call_file_groups_impl(
        "get_config_file_meta",
        rel_path,
        group_id,
        group_label,
        locked,
        trainable,
        methods_subdir,
    )


def _config_method_name_for_path(rel_path: str) -> str:
    return _call_file_groups_impl("_config_method_name_for_path", rel_path)


def _infer_config_file_group(rel_path: str) -> dict[str, Any]:
    return _call_file_groups_impl("_infer_config_file_group", rel_path)


def _strip_configs_prefix(rel_path: str) -> str:
    return _call_file_groups_impl("_strip_configs_prefix", rel_path)


def _load_config_file_group_specs() -> list[dict[str, Any]]:
    return _call_file_groups_impl("_load_config_file_group_specs")


def _sort_config_file_group_specs_for_display(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _call_file_groups_impl("_sort_config_file_group_specs_for_display", specs)


def _save_config_file_group_specs(specs: list[dict[str, Any]]) -> None:
    return _call_file_groups_impl("_save_config_file_group_specs", specs)


def _build_config_file_group(spec: dict[str, Any]) -> dict[str, Any]:
    return _call_file_groups_impl("_build_config_file_group", spec)


def _glob_config_files(pattern: str) -> list[str]:
    return _call_file_groups_impl("_glob_config_files", pattern)


def _default_config_file_group_specs() -> list[dict[str, Any]]:
    return _call_file_groups_impl("_default_config_file_group_specs")


def _group_defaults(
    group_id: str,
    label: str,
    locked: bool,
    trainable: bool,
    methods_subdir: str,
    open_by_default: bool,
) -> dict[str, Any]:
    return _call_file_groups_impl(
        "_group_defaults",
        group_id,
        label,
        locked,
        trainable,
        methods_subdir,
        open_by_default,
    )


def _find_config_group_spec(specs: list[dict[str, Any]], group_id: str) -> dict[str, Any] | None:
    return _call_file_groups_impl("_find_config_group_spec", specs, group_id)


def _new_user_config_group_spec(group_id: str, label: str, kind: str = "training") -> dict[str, Any]:
    return _call_file_groups_impl("_new_user_config_group_spec", group_id, label, kind)


def _move_orphaned_config_files_to_fallback_groups(specs: list[dict[str, Any]], files: list[str]) -> None:
    return _call_file_groups_impl("_move_orphaned_config_files_to_fallback_groups", specs, files)


def _config_file_is_covered_by_specs(specs: list[dict[str, Any]], rel_path: str) -> bool:
    return _call_file_groups_impl("_config_file_is_covered_by_specs", specs, rel_path)


def _fallback_config_group_spec(rel_path: str) -> dict[str, Any]:
    return _call_file_groups_impl("_fallback_config_group_spec", rel_path)


def _is_user_managed_group(spec: dict[str, Any]) -> bool:
    return _call_file_groups_impl("_is_user_managed_group", spec)


def _is_fixed_config_group(spec: dict[str, Any]) -> bool:
    return _call_file_groups_impl("_is_fixed_config_group", spec)


def _is_deletable_config_group(spec: dict[str, Any]) -> bool:
    return _call_file_groups_impl("_is_deletable_config_group", spec)


def _is_renamable_config_group(spec: dict[str, Any]) -> bool:
    return _call_file_groups_impl("_is_renamable_config_group", spec)


def _is_move_target_group(spec: dict[str, Any], rel_path: str = "") -> bool:
    return _call_file_groups_impl("_is_move_target_group", spec, rel_path)


def _is_sortable_config_group_for_place(spec: dict[str, Any], scope: str) -> bool:
    return _call_file_groups_impl("_is_sortable_config_group_for_place", spec, scope)


def _place_index(value: Any | None, length: int) -> int:
    return _call_file_groups_impl("_place_index", value, length)


def _lockable_group_ids() -> set[str]:
    return _call_file_groups_impl("_lockable_group_ids")


def _unique_group_id(base: str, specs: list[dict[str, Any]]) -> str:
    return _call_file_groups_impl("_unique_group_id", base, specs)


def _slugify_group_label(label: str) -> str:
    return _call_file_groups_impl("_slugify_group_label", label)


def _normalize_group_label(label: str) -> str:
    return _call_file_groups_impl("_normalize_group_label", label)


def _group_patterns_include_file(spec: dict[str, Any], rel_path: str) -> bool:
    return _call_file_groups_impl("_group_patterns_include_file", spec, rel_path)


def _normalize_config_rel_path(rel_path: str) -> str:
    return _call_file_groups_impl("_normalize_config_rel_path", rel_path)


def _normalize_dataset_preset_path(rel_path: str, *, must_exist: bool) -> str:
    return _call_file_groups_impl("_normalize_dataset_preset_path", rel_path, must_exist=must_exist)


def _normalize_group_id(group_id: str) -> str:
    return _call_file_groups_impl("_normalize_group_id", group_id)


def _safe_archive_name(name: str) -> str:
    return _call_file_groups_impl("_safe_archive_name", name)


def _unique_archive_member_name(name: str, used_names: set[str]) -> str:
    return _call_file_groups_impl("_unique_archive_member_name", name, used_names)


def _is_system_preset_path(rel_path: str) -> bool:
    return _call_file_groups_impl("_is_system_preset_path", rel_path)


def _is_system_locked_path(rel_path: str) -> bool:
    return _call_file_groups_impl("_is_system_locked_path", rel_path)


def _is_dataset_preset_readonly(rel_path: str) -> bool:
    return _call_file_groups_impl("_is_dataset_preset_readonly", rel_path)


def _is_user_locked(rel_path: str) -> bool:
    return _call_file_groups_impl("_is_user_locked", rel_path)


def _is_user_group_locked(group_id: str | None) -> bool:
    return _call_file_groups_impl("_is_user_group_locked", group_id)


def _load_user_locks() -> tuple[set[str], set[str]]:
    return _call_file_groups_impl("_load_user_locks")


def _save_user_locks(file_locks: set[str], group_locks: set[str]) -> None:
    return _call_file_groups_impl("_save_user_locks", file_locks, group_locks)


def _lock_reason_label(reason: str) -> str:
    return _call_file_groups_impl("_lock_reason_label", reason)


def _lock_reason_message(meta: dict[str, Any]) -> str:
    return _call_file_groups_impl("_lock_reason_message", meta)


def _list_system_preset_files() -> list[str]:
    return _call_file_groups_impl("_list_system_preset_files")


def _read_git_head_file(rel_path: str) -> str | None:
    return _call_file_groups_impl("_read_git_head_file", rel_path)


def _backup_relative_path(rel_path: str) -> Path:
    return _call_file_groups_impl("_backup_relative_path", rel_path)


def _string_list(value: Any) -> list[str]:
    return _call_file_groups_impl("_string_list", value)


def _config_group_path_list(value: Any) -> list[str]:
    return _call_file_groups_impl("_config_group_path_list", value)


def _patch_toml_top_level(content: str, values: dict[str, Any], *, rel_path: str = "") -> str:
    return _call_raw_files_impl("_patch_toml_top_level", content, values, rel_path=rel_path)


def _is_spd_patch_target(rel_path: str, doc: dict[str, Any]) -> bool:
    return _call_raw_files_impl("_is_spd_patch_target", rel_path, doc)


def _remove_retired_top_level_fields(content: str) -> tuple[str, list[str]]:
    return _call_raw_files_impl("_remove_retired_top_level_fields", content)


def _normalize_patch_value(key: str, value: Any) -> Any:
    return _call_raw_files_impl("_normalize_patch_value", key, value)


def _is_blank_output_name(value: Any) -> bool:
    return _call_raw_files_impl("_is_blank_output_name", value)


def _normalize_saved_raw_config_content(content: str) -> str:
    return _call_raw_files_impl("_normalize_saved_raw_config_content", content)


def _normalize_saved_raw_config_content_with_changed_keys(content: str) -> tuple[str, list[str]]:
    return _call_raw_files_impl("_normalize_saved_raw_config_content_with_changed_keys", content)


def _load(p: Path) -> dict:
    return _call_common_impl("_load", p)


def _safe_resolve(rel_path: str) -> Path | None:
    return _call_common_impl("_safe_resolve", rel_path)


def _normalize_prompt_file_path(value: str) -> str:
    return _call_sample_prompts_impl("_normalize_prompt_file_path", value)


def _sample_prompts_path_for_config(train_config_file: str) -> str:
    return _call_sample_prompts_impl("_sample_prompts_path_for_config", train_config_file)


def _safe_config_subdir(subdir: str) -> Path | None:
    return _call_common_impl("_safe_config_subdir", subdir)


def _resolve_project_path(value: str) -> Path:
    return _call_common_impl("_resolve_project_path", value)


def _auto_data_dir_for_key(value: Any, source_path: Path, suffix: str) -> Path:
    return _call_common_impl("_auto_data_dir_for_key", value, source_path, suffix)


def _derived_data_dir(source_path: Path, suffix: str) -> Path:
    return _call_common_impl("_derived_data_dir", source_path, suffix)


def _is_builtin_default_data_dir(value: str) -> bool:
    return _call_common_impl("_is_builtin_default_data_dir", value)


def _display_path(path: Path) -> str:
    return _call_common_impl("_display_path", path)


def _nl_tag_mix_available_count(
    source_dir: Path,
    image_exts: set[str],
    *,
    recursive: bool = True,
    path_pattern: str = "*",
) -> int | None:
    return _call_dataset_impl(
        "_nl_tag_mix_available_count",
        source_dir,
        image_exts,
        recursive=recursive,
        path_pattern=path_pattern,
    )


def _nl_tag_mix_image_files(
    source_dir: Path,
    image_exts: set[str] | frozenset[str] = DATASET_IMAGE_EXTS,
    *,
    recursive: bool = True,
    path_pattern: str = "*",
) -> list[Path]:
    return _call_dataset_impl(
        "_nl_tag_mix_image_files",
        source_dir,
        image_exts,
        recursive=recursive,
        path_pattern=path_pattern,
    )


def _nl_tag_mix_caption_source(
    image_path: Path,
    *,
    caption_source_mode: str | None = None,
    caption_extension: str = ".txt",
    prefer_json_caption: bool = False,
    captions_root: Path | None = None,
):
    return _call_dataset_impl(
        "_nl_tag_mix_caption_source",
        image_path,
        caption_source_mode=caption_source_mode or CAPTION_SOURCE_AUTO,
        caption_extension=caption_extension,
        prefer_json_caption=prefer_json_caption,
        captions_root=captions_root,
    )


def _nl_tag_mix_caption_path_and_text(
    image_path: Path,
    *,
    caption_source_mode: str | None = None,
    caption_extension: str = ".txt",
    prefer_json_caption: bool = False,
    captions_root: Path | None = None,
) -> tuple[Path | None, str]:
    return _call_dataset_impl(
        "_nl_tag_mix_caption_path_and_text",
        image_path,
        caption_source_mode=caption_source_mode,
        caption_extension=caption_extension,
        prefer_json_caption=prefer_json_caption,
        captions_root=captions_root,
    )


def _nl_tag_mix_caption_counts(
    source_dir: Path,
    *,
    caption_source_mode: str | None = None,
    caption_extension: str = ".txt",
    prefer_json_caption: bool = False,
    recursive: bool = True,
    path_pattern: str = "*",
) -> tuple[int, int]:
    return _call_dataset_impl(
        "_nl_tag_mix_caption_counts",
        source_dir,
        caption_source_mode=caption_source_mode,
        caption_extension=caption_extension,
        prefer_json_caption=prefer_json_caption,
        recursive=recursive,
        path_pattern=path_pattern,
    )


def _classify_nl_tag_caption_text(text: str) -> dict[str, Any]:
    return _call_dataset_impl("_classify_nl_tag_caption_text", text)


def _check_training_images(cfg: dict[str, Any], add) -> None:
    return _call_preflight_impl("_check_training_images", cfg, add)


def _check_dataset_source_paths(cfg: dict[str, Any], add) -> None:
    return _call_preflight_impl("_check_dataset_source_paths", cfg, add)


def _check_dataset_paths(cfg: dict[str, Any], add, *, check_runtime_dirs: bool = True) -> None:
    return _call_preflight_impl(
        "_check_dataset_paths",
        cfg,
        add,
        check_runtime_dirs=check_runtime_dirs,
    )


def _check_cache_sidecars(cfg: dict[str, Any], add) -> None:
    return _call_preflight_impl("_check_cache_sidecars", cfg, add)


def _check_cache_sidecar_pattern(
    add,
    cache_dirs: list[tuple[int, Path, bool]],
    pattern: str,
    key: str,
    label: str,
    missing_message: str,
) -> None:
    return _call_preflight_impl(
        "_check_cache_sidecar_pattern",
        add,
        cache_dirs,
        pattern,
        key,
        label,
        missing_message,
    )


_COMMON_SYNC_NAMES = (
    "ROOT",
    "CONFIGS_DIR",
)


def _call_common_impl(name: str, *args, **kwargs):
    from web.services.config import common as _common

    previous = {
        sync_name: getattr(_common, sync_name)
        for sync_name in _COMMON_SYNC_NAMES
        if hasattr(_common, sync_name)
    }
    for sync_name in _COMMON_SYNC_NAMES:
        if sync_name in globals():
            setattr(_common, sync_name, globals()[sync_name])
    try:
        return getattr(_common, name)(*args, **kwargs)
    finally:
        for sync_name, value in previous.items():
            setattr(_common, sync_name, value)


_PREFLIGHT_SHIM_SYNC_NAMES = (
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
    "_display_path",
    "_resolve_project_path",
    "_is_blank_output_name",
    "_dataset_config_path_from_cfg",
    "_bool_value",
    "_positive_int_or_none",
    "_safe_resolve",
    "_normalize_config_rel_path",
    "apply_auto_data_dirs",
    "load_merged_config",
    "_dataset_rows_for_estimate",
    "_normalize_path_pattern",
    "_dataset_image_files",
    "_caption_detection_counts_text",
    "_normalize_trigger_clone",
    "_count_source_images",
    "_nl_tag_mix_enabled",
    "_nl_tag_mix_caption_counts",
)


def _call_preflight_impl(name: str, *args, **kwargs):
    from web.services import config_service as _facade
    from web.services.config import preflight as _preflight

    sync_state = {
        sync_name: globals()[sync_name]
        for sync_name in _PREFLIGHT_SHIM_SYNC_NAMES
        if sync_name in globals()
    }
    facade_previous = {
        sync_name: getattr(_facade, sync_name)
        for sync_name in sync_state
        if hasattr(_facade, sync_name)
    }
    facade_missing = set(sync_state) - set(facade_previous)
    for sync_name, value in sync_state.items():
        setattr(_facade, sync_name, value)
    _preflight._sync_from_facade()
    for sync_name, value in sync_state.items():
        setattr(_preflight, sync_name, value)
    _restore_raw_files_shims()
    exported = getattr(_preflight, name)
    impl = getattr(exported, "__wrapped__", exported)
    try:
        return impl(*args, **kwargs)
    finally:
        for sync_name, value in facade_previous.items():
            setattr(_facade, sync_name, value)
        for sync_name in facade_missing:
            if hasattr(_facade, sync_name):
                delattr(_facade, sync_name)
        _restore_raw_files_shims()


def _make_preflight_shim(name: str):
    def shim(*args, **kwargs):
        return _call_preflight_impl(name, *args, **kwargs)

    shim.__name__ = name
    shim.__qualname__ = name
    shim.__doc__ = f"Compatibility shim forwarding to web.services.config.preflight.{name}."
    return shim


_PREFLIGHT_SHIM_NAMES = (
    "preflight_training_config",
    "_load_training_config_for_web_run",
    "_config_file_path",
    "is_web_runtime_config",
    "training_sample_sampler_status",
    "apply_global_model_path_defaults",
    "_check_training_images",
    "_check_dataset_source_paths",
    "_check_dataset_paths",
    "_check_cache_sidecars",
)

_PREFLIGHT_SHIMS = {
    _preflight_name: _make_preflight_shim(_preflight_name)
    for _preflight_name in _PREFLIGHT_SHIM_NAMES
}

for _preflight_name, _preflight_shim in _PREFLIGHT_SHIMS.items():
    globals()[_preflight_name] = _preflight_shim


_MERGE_SHIM_SYNC_NAMES = (
    "ROOT",
    "CONFIGS_DIR",
    "GUI_METHODS_DIR",
    "IMPORTED_CONFIGS_DIR",
    "PRESETS_FILE",
    "WEB_FILE_GROUPS_FILE",
    "WEB_USER_LOCKS_FILE",
    "DATASET_PRESETS_DIR",
    "DEFAULT_MAX_TRAIN_STEPS",
    "resolve_output_root",
    "_display_settings_path",
    "LOGGER",
)


def _call_merge_impl(name: str, *args, **kwargs):
    from web.services import config_service as _facade
    from web.services.config import merge as _merge

    sync_state = {
        sync_name: globals()[sync_name]
        for sync_name in _MERGE_SHIM_SYNC_NAMES
        if sync_name in globals()
    }
    for sync_name, value in sync_state.items():
        setattr(_facade, sync_name, value)
    _merge._sync_from_facade()
    _restore_raw_files_shims()
    for sync_name, value in sync_state.items():
        setattr(_merge, sync_name, value)
    exported = getattr(_merge, name)
    impl = getattr(exported, "__wrapped__", exported)
    try:
        return impl(*args, **kwargs)
    finally:
        _restore_raw_files_shims()


def _make_merge_shim(name: str):
    def shim(*args, **kwargs):
        return _call_merge_impl(name, *args, **kwargs)

    shim.__name__ = name
    shim.__qualname__ = name
    shim.__doc__ = f"Compatibility shim forwarding to web.services.config.merge.{name}."
    return shim


_MERGE_SHIM_NAMES = (
    "list_methods",
    "list_variants",
    "list_all_variants",
    "list_presets",
    "load_merged_config",
    "suggest_data_dirs",
    "suggest_dataset_dirs",
    "apply_auto_data_dirs",
)

_MERGE_SHIMS = {
    _merge_name: _make_merge_shim(_merge_name)
    for _merge_name in _MERGE_SHIM_NAMES
}

for _merge_name, _merge_shim in _MERGE_SHIMS.items():
    globals()[_merge_name] = _merge_shim


_OUTPUT_RUNS_SHIM_SYNC_NAMES = (
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
    "get_config_file_meta",
    "list_config_file_groups",
    "move_config_file_to_group",
    "LOGGER",
)

_OUTPUT_RUNS_LEGACY_HELPER_NAMES = (
    "_safe_resolve",
    "_normalize_group_id",
)


def _call_output_runs_impl(name: str, *args, **kwargs):
    from web.services import config_service as _facade
    from web.services.config import output_runs as _output_runs

    sync_state = {
        sync_name: globals()[sync_name]
        for sync_name in _OUTPUT_RUNS_SHIM_SYNC_NAMES
        if sync_name in globals()
    }
    for sync_name, value in sync_state.items():
        setattr(_facade, sync_name, value)
    _output_runs._sync_from_facade()
    for sync_name, value in sync_state.items():
        setattr(_output_runs, sync_name, value)
    for helper_name in _OUTPUT_RUNS_LEGACY_HELPER_NAMES:
        if helper_name in globals():
            setattr(_output_runs, helper_name, globals()[helper_name])
    _restore_raw_files_shims()
    exported = getattr(_output_runs, name)
    impl = getattr(exported, "__wrapped__", exported)
    try:
        return impl(*args, **kwargs)
    finally:
        _restore_raw_files_shims()


def _make_output_runs_shim(name: str):
    def shim(*args, **kwargs):
        return _call_output_runs_impl(name, *args, **kwargs)

    shim.__name__ = name
    shim.__qualname__ = name
    shim.__doc__ = f"Compatibility shim forwarding to web.services.config.output_runs.{name}."
    return shim


_OUTPUT_RUNS_SHIM_NAMES = (
    "list_output_runs",
    "load_output_run_config",
    "save_output_run_config_as",
    "_resolve_output_run_dir",
    "_normalize_output_run_name",
)

_OUTPUT_RUNS_SHIMS = {
    _output_name: _make_output_runs_shim(_output_name)
    for _output_name in _OUTPUT_RUNS_SHIM_NAMES
}

for _output_name, _output_shim in _OUTPUT_RUNS_SHIMS.items():
    globals()[_output_name] = _output_shim


_ESTIMATION_SHIM_SYNC_NAMES = (
    "ROOT",
    "CONFIGS_DIR",
    "GUI_METHODS_DIR",
    "IMPORTED_CONFIGS_DIR",
    "PRESETS_FILE",
    "WEB_FILE_GROUPS_FILE",
    "WEB_USER_LOCKS_FILE",
    "DATASET_PRESETS_DIR",
    "DEFAULT_MAX_TRAIN_STEPS",
    "resolve_output_root",
    "_display_settings_path",
    "LOGGER",
    "_load_training_config_for_web_run",
    "_normalize_config_rel_path",
    "_dataset_rows_for_estimate",
    "_resolve_project_path",
    "_display_path",
    "_positive_int",
    "_positive_float",
    "_nonnegative_int",
    "_bool_value",
    "_normalize_nl_tag_mix",
    "_normalize_trigger_clone",
    "_normalize_path_pattern",
    "_nl_tag_mix_available_count",
    "_count_source_images",
    "_count_images",
)


def _call_estimation_impl(name: str, *args, **kwargs):
    from web.services import config_service as _facade
    from web.services.config import estimation as _estimation

    sync_state = {
        sync_name: globals()[sync_name]
        for sync_name in _ESTIMATION_SHIM_SYNC_NAMES
        if sync_name in globals()
    }
    for sync_name, value in sync_state.items():
        setattr(_facade, sync_name, value)
    _estimation._sync_from_facade()
    for sync_name, value in sync_state.items():
        setattr(_estimation, sync_name, value)
    exported = getattr(_estimation, name)
    impl = getattr(exported, "__wrapped__", exported)
    return impl(*args, **kwargs)


def _make_estimation_shim(name: str):
    def shim(*args, **kwargs):
        return _call_estimation_impl(name, *args, **kwargs)

    shim.__name__ = name
    shim.__qualname__ = name
    shim.__doc__ = f"Compatibility shim forwarding to web.services.config.estimation.{name}."
    return shim


_ESTIMATION_SHIM_NAMES = ("estimate_training_steps",)

_ESTIMATION_SHIMS = {
    _estimation_name: _make_estimation_shim(_estimation_name)
    for _estimation_name in _ESTIMATION_SHIM_NAMES
}

for _estimation_name, _estimation_shim in _ESTIMATION_SHIMS.items():
    globals()[_estimation_name] = _estimation_shim


_DATASET_SHIM_SYNC_NAMES = (
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


def _restore_raw_files_shims() -> None:
    for shim_name, shim in globals().get("_FILE_GROUPS_SHIMS", {}).items():
        globals()[shim_name] = shim
    for shim_name, shim in globals().get("_RAW_FILES_SHIMS", {}).items():
        globals()[shim_name] = shim


def _call_dataset_impl(name: str, *args, **kwargs):
    from web.services import config_service as _facade
    from web.services.config import datasets as _datasets

    sync_state = {
        sync_name: globals()[sync_name]
        for sync_name in _DATASET_SHIM_SYNC_NAMES
        if sync_name in globals()
    }
    facade_previous = {
        sync_name: getattr(_facade, sync_name)
        for sync_name in sync_state
        if hasattr(_facade, sync_name)
    }
    facade_missing = set(sync_state) - set(facade_previous)
    try:
        for sync_name, value in sync_state.items():
            setattr(_facade, sync_name, value)
        _datasets._sync_from_facade()
        _restore_raw_files_shims()
        return getattr(_datasets, name)(*args, **kwargs)
    finally:
        for sync_name, value in facade_previous.items():
            setattr(_facade, sync_name, value)
        for sync_name in facade_missing:
            if hasattr(_facade, sync_name):
                delattr(_facade, sync_name)
        _restore_raw_files_shims()


def _make_dataset_shim(name: str):
    def shim(*args, **kwargs):
        return _call_dataset_impl(name, *args, **kwargs)

    shim.__name__ = name
    shim.__qualname__ = name
    shim.__doc__ = f"Compatibility shim forwarding to web.services.config.datasets.{name}."
    return shim


_DATASET_SHIM_NAMES = (
    "list_dataset_presets",
    "diagnose_dataset_presets",
    "load_dataset_preset",
    "save_dataset_preset",
    "save_dataset_preset_as",
    "import_dataset_preset",
    "delete_dataset_preset",
    "apply_dataset_preset_to_training_config",
    "list_dataset_preset_images",
    "resolve_dataset_preview_image",
    "load_dataset_editor",
    "save_dataset_editor",
    "_dataset_config_path_from_cfg",
    "_dataset_rows_for_estimate",
    "_dataset_rows_from_config",
    "_normalize_dataset_rows",
    "_normalize_dataset_defaults",
    "_dataset_preset_summary",
    "_dataset_preset_groups_for_ui",
    "_is_dataset_group_for_ui",
    "_dataset_summary_from_rows",
    "_normalize_nl_tag_mix",
    "_normalize_trigger_clone",
    "_normalize_path_pattern",
    "_build_dataset_config_doc",
    "_nl_tag_mix_caption_source",
    "_nl_tag_mix_image_files",
    "_classify_nl_tag_caption_text",
)

for _dataset_name in _DATASET_SHIM_NAMES:
    globals()[_dataset_name] = _make_dataset_shim(_dataset_name)


_SAMPLE_PROMPTS_SHIM_SYNC_NAMES = (
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


def _call_sample_prompts_impl(name: str, *args, **kwargs):
    from web.services import config_service as _facade
    from web.services.config import sample_prompts as _sample_prompts

    sync_state = {
        sync_name: globals()[sync_name]
        for sync_name in _SAMPLE_PROMPTS_SHIM_SYNC_NAMES
        if sync_name in globals()
    }
    facade_previous = {
        sync_name: getattr(_facade, sync_name)
        for sync_name in sync_state
        if hasattr(_facade, sync_name)
    }
    facade_missing = set(sync_state) - set(facade_previous)
    for sync_name, value in sync_state.items():
        setattr(_facade, sync_name, value)
    _sample_prompts._sync_from_facade()
    for sync_name, value in sync_state.items():
        setattr(_sample_prompts, sync_name, value)
    _restore_raw_files_shims()
    exported = getattr(_sample_prompts, name)
    impl = getattr(exported, "__wrapped__", exported)
    try:
        return impl(*args, **kwargs)
    finally:
        for sync_name, value in facade_previous.items():
            setattr(_facade, sync_name, value)
        for sync_name in facade_missing:
            if hasattr(_facade, sync_name):
                delattr(_facade, sync_name)
        _restore_raw_files_shims()


def _make_sample_prompts_shim(name: str):
    def shim(*args, **kwargs):
        return _call_sample_prompts_impl(name, *args, **kwargs)

    shim.__name__ = name
    shim.__qualname__ = name
    shim.__doc__ = f"Compatibility shim forwarding to web.services.config.sample_prompts.{name}."
    return shim


_SAMPLE_PROMPTS_SHIM_NAMES = (
    "load_sample_prompts_file",
    "save_sample_prompts_file",
    "_normalize_prompt_file_path",
    "_sample_prompts_path_for_config",
)

_SAMPLE_PROMPTS_SHIMS = {
    _sample_prompt_name: _make_sample_prompts_shim(_sample_prompt_name)
    for _sample_prompt_name in _SAMPLE_PROMPTS_SHIM_NAMES
}

for _sample_prompt_name, _sample_prompt_shim in _SAMPLE_PROMPTS_SHIMS.items():
    globals()[_sample_prompt_name] = _sample_prompt_shim


_FILE_GROUPS_SHIM_SYNC_NAMES = (
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
    "_inspect_network_weight",
    "LOGGER",
)


def _call_file_groups_impl(name: str, *args, **kwargs):
    from web.services import config_service as _facade
    from web.services.config import file_groups as _file_groups

    sync_state = {
        sync_name: globals()[sync_name]
        for sync_name in _FILE_GROUPS_SHIM_SYNC_NAMES
        if sync_name in globals()
    }
    for sync_name, value in sync_state.items():
        setattr(_facade, sync_name, value)
    _file_groups._sync_from_facade()
    for sync_name, value in sync_state.items():
        setattr(_file_groups, sync_name, value)
    _restore_raw_files_shims()
    exported = getattr(_file_groups, name)
    impl = getattr(exported, "__wrapped__", exported)
    try:
        return impl(*args, **kwargs)
    finally:
        _restore_raw_files_shims()


def _make_file_groups_shim(name: str):
    def shim(*args, **kwargs):
        return _call_file_groups_impl(name, *args, **kwargs)

    shim.__name__ = name
    shim.__qualname__ = name
    shim.__doc__ = f"Compatibility shim forwarding to web.services.config.file_groups.{name}."
    return shim


_FILE_GROUPS_SHIM_NAMES = (
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
    "_load_config_file_group_specs",
    "_save_config_file_group_specs",
    "_normalize_config_file_group_kind_filter",
    "_normalize_config_rel_path",
    "_normalize_dataset_preset_path",
    "_is_dataset_preset_readonly",
    "_is_user_locked",
    "_is_user_group_locked",
    "_load_user_locks",
    "_save_user_locks",
    "_lock_reason_message",
    "_lock_reason_label",
)

_FILE_GROUPS_SHIMS = {
    _file_group_name: _make_file_groups_shim(_file_group_name)
    for _file_group_name in _FILE_GROUPS_SHIM_NAMES
}

for _file_group_name, _file_group_shim in _FILE_GROUPS_SHIMS.items():
    globals()[_file_group_name] = _file_group_shim


_RAW_FILES_SHIM_SYNC_NAMES = (
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
    "get_config_file_meta",
    "list_config_file_groups",
    "move_config_file_to_group",
    "_inspect_network_weight",
    "LOGGER",
)

_RAW_FILES_LEGACY_HELPER_NAMES = (
    "_safe_resolve",
    "_normalize_config_rel_path",
    "_load_user_locks",
    "_save_user_locks",
    "_lock_reason_message",
)

_RAW_FILES_FACADE_HELPER_NAMES = (
    "get_config_file_meta",
    "list_config_file_groups",
    "move_config_file_to_group",
)


def _call_raw_files_impl(name: str, *args, **kwargs):
    from web.services import config_service as _facade
    from web.services.config import raw_files as _raw_files

    sync_state = {
        sync_name: globals()[sync_name]
        for sync_name in _RAW_FILES_SHIM_SYNC_NAMES
        if sync_name in globals()
    }
    facade_previous = {
        sync_name: getattr(_facade, sync_name)
        for sync_name in sync_state
        if hasattr(_facade, sync_name)
    }
    facade_missing = set(sync_state) - set(facade_previous)
    try:
        for sync_name, value in sync_state.items():
            setattr(_facade, sync_name, value)
        _raw_files._sync_from_facade()
        for sync_name, value in sync_state.items():
            setattr(_raw_files, sync_name, value)
        for helper_name in _RAW_FILES_LEGACY_HELPER_NAMES:
            if helper_name in globals():
                setattr(_raw_files, helper_name, globals()[helper_name])
        for helper_name in _RAW_FILES_FACADE_HELPER_NAMES:
            if hasattr(_facade, helper_name):
                setattr(_raw_files, helper_name, getattr(_facade, helper_name))
            elif helper_name in globals():
                setattr(_raw_files, helper_name, globals()[helper_name])
        for sync_name, value in sync_state.items():
            globals()[sync_name] = value
        # raw_files 同步时会把 _legacy 里的同名导出回填成 facade 版本，
        # 这里立刻恢复 shim，保证旧模块后续调用仍然走 lazy forwarding。
        _restore_raw_files_shims()
        exported = getattr(_raw_files, name)
        impl = getattr(exported, "__wrapped__", exported)
        return impl(*args, **kwargs)
    finally:
        for sync_name, value in facade_previous.items():
            setattr(_facade, sync_name, value)
        for sync_name in facade_missing:
            if hasattr(_facade, sync_name):
                delattr(_facade, sync_name)
        _restore_raw_files_shims()


def _make_raw_files_shim(name: str):
    def shim(*args, **kwargs):
        return _call_raw_files_impl(name, *args, **kwargs)

    shim.__name__ = name
    shim.__qualname__ = name
    shim.__doc__ = f"Compatibility shim forwarding to web.services.config.raw_files.{name}."
    return shim


_RAW_FILES_SHIM_NAMES = (
    "load_raw_file",
    "save_raw_file",
    "delete_raw_file",
    "patch_raw_file_values",
    "preview_raw_file_patch",
    "_prepare_raw_file_patch",
    "_restore_dataset_config_after_failed_train_patch",
    "_patch_toml_top_level",
    "_is_spd_patch_target",
    "_remove_retired_top_level_fields",
    "_normalize_patch_value",
    "_normalize_saved_raw_config_content",
    "_normalize_saved_raw_config_content_with_changed_keys",
    "_is_blank_output_name",
)

_RAW_FILES_SHIMS = {
    _raw_name: _make_raw_files_shim(_raw_name)
    for _raw_name in _RAW_FILES_SHIM_NAMES
}

for _raw_name, _raw_shim in _RAW_FILES_SHIMS.items():
    globals()[_raw_name] = _raw_shim
