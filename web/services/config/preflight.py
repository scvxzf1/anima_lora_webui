"""Training preflight checks and runtime config path validation.

Compatibility facade. Implementation lives in:

- ``preflight_runtime``: roots, facade sync, path wrappers
- ``preflight_paths``: config path validation / web runtime detection
- ``preflight_compat``: checkpoint / network / sample / preprocess env checks
- ``preflight_history``: history output-dir reuse checks
- ``preflight_dataset_checks``: dataset image / source / cache checks

This module is loaded by ``web.services.config_service`` as part of the
compatibility facade.  It keeps facade access lazy so the module can also be
imported directly without pulling in the facade cycle.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from web.services.config.preflight_compat import (
    _check_checkpointing_config,
    _check_network_weights,
    _check_no_dataset_regularization_config,
    _check_training_sample_config,
    _check_web_preprocess_environment,
    _compat_web_message,
    _inspect_network_weight,
    _inspect_network_weight_impl,
    _web_python_executable,
    training_sample_sampler_status,
)
from web.services.config.preflight_stage_schedule import check_stage_schedule
from web.services.config.dataset_rows import merge_stage_schedule_from_dataset_config
from web.services.config.schema_gate import validate_config_mapping
from web.services.config.preflight_dataset_checks import (
    _check_cache_sidecar_pattern,
    _check_cache_sidecars,
    _check_dataset_bucket_settings,
    _check_dataset_paths,
    _check_dataset_source_paths,
    _check_training_images,
)
from web.services.config.preflight_history import (
    _check_output_dir_history_reuse,
    _history_output_match_label,
    _history_task_output_candidates,
    _history_task_reuses_output_dir,
    _history_training_tasks_for_output_dir,
    _is_web_runtime_training_output_dir,
    _read_history_meta_for_output_reuse,
)
from web.services.config.preflight_paths import (
    _blank_model_path,
    _config_file_path,
    _config_path_from_display_path,
    _global_model_path_defaults,
    _has_web_runtime_dirs,
    _is_allowed_training_config_path,
    _is_output_run_snapshot_config,
    _is_web_runtime_config_tree,
    _load_training_config_for_web_run,
    _looks_like_web_runtime_config,
    apply_global_model_path_defaults,
    is_web_runtime_config,
)
from web.services.config.preflight_runtime import (
    CONFIGS_DIR,
    DATASET_PRESETS_DIR,
    GLOBAL_MODEL_PATH_KEYS,
    GUI_METHODS_DIR,
    IMPORTED_CONFIGS_DIR,
    PRESETS_FILE,
    ROOT,
    WEB_FILE_GROUPS_FILE,
    WEB_USER_LOCKS_FILE,
    _display_path,
    _exported,
    _missing_facade_dependency,
    _nonnegative_float_value,
    _nonnegative_int_value,
    _normalize_config_rel_path,
    _resolve_project_path,
    _safe_resolve,
    _sync_from_facade,
)

# Keep facade-filled placeholders re-exported for monkeypatch / legacy sync.
from web.services.config.preflight_runtime import (  # noqa: F401
    LOGGER,
    _bool_value,
    _caption_detection_counts_text,
    _count_source_images,
    _dataset_config_path_from_cfg,
    _dataset_image_files,
    _dataset_rows_for_estimate,
    _display_settings_path,
    _is_blank_output_name,
    _nl_tag_mix_caption_counts,
    _nl_tag_mix_enabled,
    _normalize_path_pattern,
    _normalize_trigger_clone,
    _positive_int_or_none,
    apply_auto_data_dirs,
    delete_raw_file,
    get_config_file_meta,
    list_config_file_groups,
    load_merged_config,
    load_raw_file,
    move_config_file_to_group,
    patch_raw_file_values,
    preview_raw_file_patch,
    resolve_output_root,
    save_raw_file,
)


__all__ = [
    "preflight_training_config",
    "_load_training_config_for_web_run",
    "_config_file_path",
    "is_web_runtime_config",
    "training_sample_sampler_status",
    "apply_global_model_path_defaults",
    "_check_training_images",
    "_check_dataset_source_paths",
    "_check_dataset_bucket_settings",
    "_check_dataset_paths",
    "_check_cache_sidecars",
]


def _validate_lokr_config(cfg: dict) -> list[str]:
    """Extra LoKR rules that are not pure argparse choices."""
    errors: list[str] = []
    use_lokr = cfg.get("use_lokr")
    if isinstance(use_lokr, str):
        use_lokr = use_lokr.strip().lower() in {"1", "true", "yes", "on"}
    if not use_lokr:
        return errors
    full_factor = cfg.get("lokr_full_factor")
    if isinstance(full_factor, str):
        full_factor = full_factor.strip().lower() in {"1", "true", "yes", "on"}
    decompose_w2 = cfg.get("lokr_decompose_w2")
    if isinstance(decompose_w2, str):
        decompose_w2 = decompose_w2.strip().lower() in {"1", "true", "yes", "on"}
    if full_factor and decompose_w2:
        errors.append("lokr_full_factor=true conflicts with lokr_decompose_w2=true")
    network_dim = cfg.get("network_dim")
    try:
        network_dim_i = int(network_dim) if network_dim is not None else None
    except (TypeError, ValueError):
        network_dim_i = None
    allow_legacy = cfg.get("lokr_allow_legacy_dim")
    if isinstance(allow_legacy, str):
        allow_legacy = allow_legacy.strip().lower() in {"1", "true", "yes", "on"}
    if network_dim_i == 114514 and not allow_legacy:
        errors.append(
            "LoKR network_dim=114514 is a deprecated full-factor sentinel "
            "that suppresses training via alpha/dim. Use network_dim=32, "
            "network_alpha=32, lokr_full_factor=true instead."
        )
    return errors



def preflight_training_config(
    variant: str,
    preset: str,
    methods_subdir: str = "gui-methods",
    *,
    config_file: str | None = None,
) -> dict[str, Any]:
    cfg = _load_training_config_for_web_run(variant, preset, methods_subdir, config_file=config_file)
    cfg = merge_stage_schedule_from_dataset_config(cfg)
    checks: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    runtime_config = is_web_runtime_config(config_file) or _looks_like_web_runtime_config(cfg)

    def add(level: str, key: str, message: str, path: Path | None = None) -> None:
        item = {
            "level": level,
            "key": key,
            "message": message,
        }
        if path is not None:
            item["path"] = _display_path(path)
        checks.append(item)
        if level == "error":
            errors.append(item)
        elif level == "warning":
            warnings.append(item)

    def check_file(key: str, label: str, suffixes: tuple[str, ...] = ()) -> None:
        raw = cfg.get(key)
        if not raw:
            add("error", key, f"{label} 未填写")
            return
        path = _resolve_project_path(str(raw))
        if not path.exists():
            add("error", key, f"{label} 不存在", path)
            return
        if not path.is_file():
            add("error", key, f"{label} 不是文件", path)
            return
        if suffixes and path.suffix.lower() not in suffixes:
            add("warning", key, f"{label} 后缀不是常见格式 {', '.join(suffixes)}", path)
            return
        add("ok", key, f"{label} 存在", path)

    def check_dir(key: str, label: str, *, must_exist: bool, warn_empty: bool = False) -> None:
        raw = cfg.get(key)
        if not raw:
            add("error", key, f"{label} 未填写")
            return
        path = _resolve_project_path(str(raw))
        if not path.exists():
            if must_exist:
                add("error", key, f"{label} 不存在", path)
            else:
                add("warning", key, f"{label} 不存在，训练/预处理可能会创建它", path)
            return
        if not path.is_dir():
            add("error", key, f"{label} 不是目录", path)
            return
        if warn_empty and not any(path.iterdir()):
            add("warning", key, f"{label} 为空", path)
            return
        add("ok", key, f"{label} 存在", path)

    if "output_name" in cfg and _is_blank_output_name(cfg.get("output_name")):
        add("error", "output_name", "输出名称未填写")
    _check_checkpointing_config(cfg, add)
    _check_no_dataset_regularization_config(cfg, add)
    _check_output_dir_history_reuse(cfg, add)
    check_file("pretrained_model_name_or_path", "基础 DiT 模型", (".safetensors", ".pt", ".pth", ".ckpt"))
    check_file("qwen3", "Qwen3 文本编码器", (".safetensors", ".pt", ".pth", ".bin"))
    check_file("vae", "VAE 模型", (".safetensors", ".pt", ".pth", ".ckpt"))
    _check_network_weights(cfg, add, variant, preset, methods_subdir, config_file)
    dataset_config_path = _dataset_config_path_from_cfg(cfg)
    if cfg.get("dataset_config") and (runtime_config or (dataset_config_path and dataset_config_path.exists())):
        check_file("dataset_config", "数据集配置", (".toml",))

    _check_dataset_source_paths(cfg, add)
    _check_dataset_bucket_settings(cfg, add)
    _check_dataset_paths(cfg, add, check_runtime_dirs=runtime_config)
    _check_training_sample_config(cfg, add)
    check_stage_schedule(
        cfg,
        dataset_rows=_dataset_rows_for_estimate(cfg),
        add=add,
    )
    schema_errors, schema_warnings = validate_config_mapping(cfg)
    schema_errors.extend(_validate_lokr_config(cfg))
    for msg in schema_errors:
        add("error", "schema", msg)
    for msg in schema_warnings:
        add("warning", "schema", msg)
    if not runtime_config:
        _check_web_preprocess_environment(add)
    if runtime_config:
        _check_training_images(cfg, add)
        _check_cache_sidecars(cfg, add)

    return {
        "ok": not errors,
        "variant": variant,
        "preset": preset,
        "methods_subdir": methods_subdir,
        "summary": {
            "errors": len(errors),
            "warnings": len(warnings),
            "checks": len(checks),
        },
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }


for _name in __all__:
    globals()[_name] = _exported(globals()[_name])
