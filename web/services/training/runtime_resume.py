"""Runtime resume helpers for WebUI training runs."""

from __future__ import annotations

import shutil
from typing import Any

from web.services.config.metadata import DATASET_IMAGE_EXTS
from web.services.training.runtime_common import (
    _build_dataset_config_doc,
    _build_runtime_payload,
    _dataset_rows_for_estimate,
    _ensure_runtime_dir_layout,
    _load_config_file_config,
    _nl_tag_mix_image_files,
    _normalize_path_pattern,
    _sample_config_from_cfg,
    toml_dumps_sorted,
)
from web.services.training.runtime_paths import (
    _display_project_path,
    _display_settings_path,
    _path_exists,
    _resolve_display_path,
    _safe_run_stem,
    _unique_runtime_dir,
    resolve_output_root,
)
from web.services.training import runtime_datasets as _runtime_datasets
from web.services.training.runtime_state import (
    _read_runtime_run_meta,
    _write_runtime_run_meta,
)

from library.training.stage_schedule import (
    parse_stage_specs,
    progress_from_steps,
    resolve_stage_index,
)

_bool_value_for_row = _runtime_datasets._bool_value_for_row
_clone_runtime_dataset_rows = _runtime_datasets._clone_runtime_dataset_rows


def _clone_frozen_runtime_config(
    config_file: str,
    *,
    source_config_file: str = "",
    reset_data_dirs: bool = False,
    resume_step: int | None = None,
    duration_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config_path = _resolve_display_path(config_file)
    if config_path is None or not _path_exists(config_path) or not config_path.is_file():
        raise FileNotFoundError(f"冻结运行配置不存在: {config_file}")

    cfg = _load_config_file_config(_display_settings_path(config_path))
    if not cfg:
        raise ValueError("冻结运行配置为空或无法解析")
    cfg = _drop_resume_hotstart_overrides(cfg)

    previous_run_dir = config_path.parent
    run_stem = _safe_run_stem(f"{previous_run_dir.name or config_path.stem}-retry")
    run_dir = _unique_runtime_dir(resolve_output_root(), run_stem)
    layout = _ensure_runtime_dir_layout(run_dir)

    original_config_path = run_dir / "config.original.toml"
    old_original = previous_run_dir / "config.original.toml"
    shutil.copy2(old_original if _path_exists(old_original) else config_path, original_config_path)

    dataset_config_path = run_dir / "dataset.runtime.toml"
    runtime_rows = _dataset_rows_for_estimate(cfg)
    if not runtime_rows:
        raise ValueError(
            "冻结运行配置缺少数据集路径，无法重新预处理"
            if reset_data_dirs
            else "冻结运行配置缺少数据集路径，无法重新入队"
        )

    cloned_rows = _clone_runtime_dataset_rows(
        runtime_rows,
        layout["dataset_cache_dir"],
        copy_existing=not reset_data_dirs,
    )
    dataset_config_path.write_text(
        _build_dataset_config_doc(
            cloned_rows,
            cfg,
            prefer_train_batch_size=True,
            include_preprocess_settings=False,
        ),
        encoding="utf-8",
    )

    first_row = cloned_rows[0]
    data_dirs = {
        "source_image_dir": first_row["source_dir"],
        "resized_image_dir": first_row["image_dir"],
        "lora_cache_dir": first_row["cache_dir"],
    }

    runtime_cfg = dict(cfg)
    runtime_cfg.update({
        "output_dir": _display_settings_path(layout["training_output_dir"]),
        "logging_dir": _display_settings_path(layout["logs_dir"]),
        "dataset_config": _display_settings_path(dataset_config_path),
    })
    runtime_cfg.update({key: value for key, value in data_dirs.items() if value})
    resume_duration = _apply_resume_duration_overrides(
        runtime_cfg,
        cloned_rows,
        resume_step=resume_step,
        duration_overrides=duration_overrides,
    )

    runtime_config_path = run_dir / "config.runtime.toml"
    runtime_config_path.write_text(toml_dumps_sorted(runtime_cfg), encoding="utf-8")

    run_meta = _read_runtime_run_meta(previous_run_dir)
    history_source_config_file = _display_project_path(
        source_config_file
        or str(run_meta.get("history_source_config_file") or run_meta.get("source_config_file") or "")
    )
    _write_runtime_run_meta(
        run_dir,
        {
            "history_source_config_file": history_source_config_file,
            "source_config_file": history_source_config_file,
            "run_dir": _display_settings_path(run_dir),
            "runtime_config_file": _display_settings_path(runtime_config_path),
            "original_config_file": _display_settings_path(original_config_path),
            "dataset_config_file": _display_settings_path(dataset_config_path),
            "resume_duration": resume_duration,
        },
    )
    return _build_runtime_payload(
        run_dir=run_dir,
        layout=layout,
        runtime_config_file=runtime_config_path,
        original_config_file=original_config_path,
        dataset_config_file=dataset_config_path,
        output_dir=runtime_cfg["output_dir"],
        logs_dir=runtime_cfg["logging_dir"],
        history_source_config_file=history_source_config_file,
        data_dirs=data_dirs,
        sample_config=_sample_config_from_cfg(runtime_cfg, []),
        resume_duration=resume_duration,
    )


def _apply_resume_duration_overrides(
    runtime_cfg: dict[str, Any],
    runtime_rows: list[dict[str, Any]],
    *,
    resume_step: int | None,
    duration_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    overrides = _normalize_resume_duration_overrides(duration_overrides)
    if not overrides:
        return {}
    if resume_step is None or resume_step < 0:
        raise ValueError("无法读取检查点已完成步数，不能按当前配置页训练时长追加续训")

    current_step = int(resume_step)
    if "max_train_epochs" in overrides:
        epochs = int(overrides["max_train_epochs"])
        steps_per_epoch = _estimate_resume_steps_per_epoch(runtime_cfg, runtime_rows)
        if steps_per_epoch <= 0:
            raise ValueError("无法根据历史数据集估算每轮步数，不能按 max_train_epochs 追加续训")
        append_steps = steps_per_epoch * epochs
        info = {
            "mode": "epochs",
            "max_train_epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
        }
    else:
        append_steps = int(overrides["max_train_steps"])
        info = {
            "mode": "steps",
            "max_train_steps": append_steps,
            "steps_per_epoch": None,
        }

    # Capture pre-override max steps for stage % diagnosis (global step / max_train_steps).
    previous_max_steps = _positive_int_value(runtime_cfg.get("max_train_steps"))
    target_total_steps = current_step + append_steps
    runtime_cfg["max_train_steps"] = target_total_steps
    runtime_cfg.pop("max_train_epochs", None)
    info.update({
        "resume_step": current_step,
        "append_steps": append_steps,
        "target_total_steps": target_total_steps,
    })
    diagnosis = diagnose_resume_stage_shift(
        runtime_cfg,
        resume_step=current_step,
        previous_max_steps=previous_max_steps,
        target_total_steps=target_total_steps,
    )
    if diagnosis:
        info.update(diagnosis)
    return info



def diagnose_resume_stage_shift(
    runtime_cfg: dict[str, Any],
    *,
    resume_step: int,
    previous_max_steps: int | None,
    target_total_steps: int,
) -> dict[str, Any]:
    """Return stage_before / stage_after / warning when schedule is enabled.

    Stage progress stays global: ``global_step / max_train_steps``. Appending
    resume steps changes the denominator, so the same checkpoint step can land
    in a different stage after the override.
    """
    if not bool(runtime_cfg.get("stage_schedule_enabled")):
        return {}
    stages = parse_stage_specs(runtime_cfg.get("stage_schedule"))
    if not stages:
        return {}

    before_total = int(previous_max_steps) if previous_max_steps and previous_max_steps > 0 else int(target_total_steps)
    after_total = max(1, int(target_total_steps))
    step = max(0, int(resume_step))

    progress_before = progress_from_steps(step, before_total)
    progress_after = progress_from_steps(step, after_total)
    idx_before = resolve_stage_index(stages, progress_before)
    idx_after = resolve_stage_index(stages, progress_after)
    stage_before = {
        "index": idx_before,
        "name": stages[idx_before].name or f"阶段{idx_before + 1}",
        "progress": progress_before,
    }
    stage_after = {
        "index": idx_after,
        "name": stages[idx_after].name or f"阶段{idx_after + 1}",
        "progress": progress_after,
    }
    # Duration override rewrites max_train_steps; always surface the recomputed boundary.
    warning = "追加步数后阶段边界已按新总步数重算"
    return {
        "stage_before": stage_before,
        "stage_after": stage_after,
        "warning": warning,
    }

def _normalize_resume_duration_overrides(duration_overrides: dict[str, Any] | None) -> dict[str, int]:
    if not isinstance(duration_overrides, dict):
        return {}
    epochs = _positive_int_value(duration_overrides.get("max_train_epochs"))
    if epochs is not None:
        return {"max_train_epochs": epochs}
    steps = _positive_int_value(duration_overrides.get("max_train_steps"))
    if steps is not None:
        return {"max_train_steps": steps}
    return {}


def _estimate_resume_steps_per_epoch(
    runtime_cfg: dict[str, Any],
    runtime_rows: list[dict[str, Any]],
) -> int:
    weighted_images = 0
    for row in runtime_rows:
        recursive = _bool_value_for_row(row.get("recursive"), True)
        path_pattern = _normalize_path_pattern(row.get("path_pattern"))
        resized_count = _count_resume_images(row.get("image_dir"), recursive=recursive, path_pattern=path_pattern)
        source_count = _count_resume_images(row.get("source_dir"), recursive=recursive, path_pattern=path_pattern)
        used_count = resized_count or source_count
        repeats = _positive_int_value(row.get("num_repeats")) or 1
        weighted_images += used_count * repeats

    sample_ratio = _positive_float_value(runtime_cfg.get("sample_ratio"), 1.0)
    repeated_images = int(weighted_images * sample_ratio)
    batch_size = _positive_int_value(runtime_cfg.get("train_batch_size")) or 1
    grad_accum = _positive_int_value(runtime_cfg.get("gradient_accumulation_steps")) or 1
    effective_batch = max(1, batch_size * grad_accum)
    return (repeated_images + effective_batch - 1) // effective_batch if repeated_images else 0


def _count_resume_images(value: Any, *, recursive: bool, path_pattern: str) -> int:
    path = _resolve_display_path(str(value or ""))
    if path is None or not path.is_dir():
        return 0
    try:
        return len(
            _nl_tag_mix_image_files(
                path,
                DATASET_IMAGE_EXTS,
                recursive=recursive,
                path_pattern=path_pattern,
            )
        )
    except OSError:
        return 0


def _positive_int_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _positive_float_value(value: Any, fallback: float) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return fallback
    return n if n > 0 else fallback


def _drop_resume_hotstart_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(cfg)
    cleaned.pop("network_weights", None)
    cleaned.pop("dim_from_weights", None)
    return cleaned
