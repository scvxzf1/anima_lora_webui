"""Runtime configuration helpers for WebUI training runs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import json
    import re
    import shutil
    import toml
    from datetime import datetime
    from pathlib import Path
    from typing import Any

    from library.preprocess.captions import (
        CAPTION_SOURCE_CAPTIONS_JSON,
        CAPTIONS_JSON_FILE,
        normalize_caption_source_mode,
    )
    from web.services.config_service import (
        NL_TAG_MIX_CLASSIFICATION_METHOD,
        _build_dataset_config_doc,
        _classify_nl_tag_caption_text,
        _dataset_rows_for_estimate,
        _nl_tag_mix_caption_source,
        _nl_tag_mix_image_files,
        _normalize_nl_tag_mix,
        _normalize_path_pattern,
        _normalize_trigger_clone,
        apply_auto_data_dirs,
        apply_global_model_path_defaults,
        load_merged_config,
        training_sample_sampler_status,
    )
    from web.services.settings_service import display_path as _display_settings_path
    from web.services.settings_service import resolve_output_root

    from web.services.training_service import (
        DATASET_CAPTION_EXTS,
        DATASET_IMAGE_EXTS,
        ROOT,
        RUNTIME_META_KEYS,
        RUN_META_FILE,
        _positive_int_or_none,
        _read_json,
        _write_json,
    )


_LOCAL_IMPL_NAMES = {
    "_bind_legacy",
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


def _bind_legacy() -> None:
    """Bind training_service globals lazily, preserving local runtime helpers."""
    from web.services import training_service as legacy

    _sync_config_facade_paths()
    for name, value in vars(legacy).items():
        if name.startswith("__") or name in _LOCAL_IMPL_NAMES:
            continue
        globals()[name] = value


def _sync_config_facade_paths() -> None:
    """Keep config helper modules aligned with config_service monkeypatches."""
    from web.services import config_service as facade
    from web.services.config import _legacy as config_legacy
    from web.services.config import datasets as config_datasets

    sync_names = (
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
    )
    for name in sync_names:
        if not hasattr(facade, name):
            continue
        value = getattr(facade, name)
        setattr(config_legacy, name, value)
        setattr(config_datasets, name, value)


def _resolve_training_runtime_info(
    variant: str,
    preset: str,
    methods_subdir: str,
    extra_args: list[str],
    config_file: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    _bind_legacy()
    output_dir = "output/ckpt"
    cfg: dict[str, Any] = {}
    try:
        if config_file:
            cfg = _load_config_file_config(config_file)
        else:
            cfg = load_merged_config(variant, preset, methods_subdir)
        output_dir = str(cfg.get("output_dir") or output_dir)
    except Exception:
        pass

    for idx, arg in enumerate(extra_args):
        if arg == "--output_dir" and idx + 1 < len(extra_args):
            output_dir = str(extra_args[idx + 1])
            break
        if arg.startswith("--output_dir="):
            output_dir = arg.split("=", 1)[1]
            break

    rel_output = _display_project_path(output_dir) or "output/ckpt"
    return rel_output, f"{rel_output.rstrip('/')}/sample", _sample_config_from_cfg(cfg, extra_args)

def _ensure_training_data_dirs(
    variant: str,
    preset: str,
    methods_subdir: str,
    *,
    config_file: str | None = None,
) -> dict[str, str]:
    _bind_legacy()
    if config_file:
        cfg = apply_auto_data_dirs(_load_config_file_config(config_file), create=True)
    else:
        cfg = apply_auto_data_dirs(load_merged_config(variant, preset, methods_subdir), create=True)
    return {
        "source_image_dir": str(cfg.get("source_image_dir") or ""),
        "resized_image_dir": str(cfg.get("resized_image_dir") or ""),
        "lora_cache_dir": str(cfg.get("lora_cache_dir") or ""),
    }

def _load_config_file_config(config_file: str) -> dict[str, Any]:
    _bind_legacy()
    path = _resolve_display_path(config_file)
    if path is None or not _path_exists(path):
        return {}
    try:
        return toml.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def toml_dumps_sorted(data: dict[str, Any]) -> str:
    _bind_legacy()
    try:
        import toml
        return toml.dumps({key: data[key] for key in sorted(data)})
    except Exception:
        return json.dumps(data, ensure_ascii=False, indent=2)

def _prepare_web_runtime_config(
    variant: str,
    preset: str,
    methods_subdir: str,
    *,
    source_config_file: str | None,
) -> dict[str, Any]:
    _bind_legacy()
    source_path = _resolve_display_path(source_config_file or "") if source_config_file else None
    if source_config_file and (source_path is None or not _path_exists(source_path) or not source_path.is_file()):
        raise FileNotFoundError(f"训练配置不存在: {source_config_file}")

    stem_source = source_path.stem if source_path is not None else variant
    run_stem = _safe_run_stem(stem_source or variant or "run")
    run_dir = _unique_runtime_dir(resolve_output_root(), run_stem)

    model_cache_dir = run_dir / "model_cache"
    dataset_cache_dir = run_dir / "dataset_cache"
    training_output_dir = run_dir / "training_output"
    sample_dir = training_output_dir / "sample"
    logs_dir = model_cache_dir / "logs"
    torchinductor_dir = model_cache_dir / "torchinductor"
    triton_dir = model_cache_dir / "triton"

    for path in (
        model_cache_dir,
        dataset_cache_dir,
        training_output_dir,
        sample_dir,
        logs_dir,
        torchinductor_dir,
        triton_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    fallback_cfg = load_merged_config(variant, preset, methods_subdir)
    cfg = dict(fallback_cfg)
    if source_path is not None:
        source_cfg = _load_config_file_config(_display_settings_path(source_path))
        if source_cfg:
            cfg.update(source_cfg)
    cfg = apply_global_model_path_defaults(cfg, fallback=fallback_cfg)
    source_rows = _dataset_rows_for_estimate(cfg)
    if not source_rows:
        raise ValueError("请先配置至少一个数据集路径")

    runtime_rows: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows, start=1):
        group_dir = dataset_cache_dir / f"dataset-{index:02d}"
        resized_dir = group_dir / "resized"
        lora_dir = group_dir / "lora"
        resized_dir.mkdir(parents=True, exist_ok=True)
        lora_dir.mkdir(parents=True, exist_ok=True)
        source_dir = str(
            row.get("source_dir")
            or row.get("source_image_dir")
            or row.get("image_dir")
            or ""
        ).strip()
        source_dir = _prepare_runtime_nl_tag_mix_source(row, group_dir, source_dir)
        runtime_rows.append({
            "source_dir": source_dir,
            "image_dir": _display_settings_path(resized_dir),
            "cache_dir": _display_settings_path(lora_dir),
            "num_repeats": row.get("num_repeats") or 1,
            "recursive": _bool_value_for_row(row.get("recursive"), True),
            "path_pattern": _normalize_path_pattern(row.get("path_pattern")),
            "settings": row.get("settings") if isinstance(row.get("settings"), dict) else {},
        })
        trigger_clone = _normalize_trigger_clone(row.get("trigger_clone"))
        if trigger_clone["enabled"]:
            clone_source_dir = _prepare_runtime_trigger_clone_source(row, group_dir, source_dir)
            clone_resized_dir = group_dir / "trigger-clone-resized"
            clone_lora_dir = group_dir / "trigger-clone-lora"
            clone_resized_dir.mkdir(parents=True, exist_ok=True)
            clone_lora_dir.mkdir(parents=True, exist_ok=True)
            clone_settings = dict(row.get("settings") if isinstance(row.get("settings"), dict) else {})
            clone_settings["caption_source_mode"] = CAPTION_SOURCE_CAPTIONS_JSON
            clone_settings["prefer_json_caption"] = False
            runtime_rows.append({
                "source_dir": clone_source_dir,
                "image_dir": _display_settings_path(clone_resized_dir),
                "cache_dir": _display_settings_path(clone_lora_dir),
                "num_repeats": trigger_clone["num_repeats"],
                "recursive": _bool_value_for_row(row.get("recursive"), True),
                "path_pattern": _normalize_path_pattern(row.get("path_pattern")),
                "settings": clone_settings,
            })

    original_config_path = run_dir / "config.original.toml"
    if source_path is not None:
        shutil.copy2(source_path, original_config_path)
    else:
        original_config_path.write_text(toml_dumps_sorted(cfg), encoding="utf-8")

    dataset_config_path = run_dir / "dataset.runtime.toml"
    runtime_cfg = dict(cfg)
    dataset_doc = _build_dataset_config_doc(
        runtime_rows,
        runtime_cfg,
        prefer_train_batch_size=True,
        include_preprocess_settings=False,
    )
    dataset_config_path.write_text(dataset_doc, encoding="utf-8")

    first_row = runtime_rows[0]
    runtime_cfg.update({
        "output_dir": _display_settings_path(training_output_dir),
        "logging_dir": _display_settings_path(logs_dir),
        "dataset_config": _display_settings_path(dataset_config_path),
        "source_image_dir": first_row["source_dir"],
        "resized_image_dir": first_row["image_dir"],
        "lora_cache_dir": first_row["cache_dir"],
    })
    runtime_config_path = run_dir / "config.runtime.toml"
    runtime_config_path.write_text(toml_dumps_sorted(runtime_cfg), encoding="utf-8")

    data_dirs = {
        "source_image_dir": first_row["source_dir"],
        "resized_image_dir": first_row["image_dir"],
        "lora_cache_dir": first_row["cache_dir"],
    }
    history_source_config_file = _display_settings_path(source_path) if source_path is not None else ""
    _write_runtime_run_meta(
        run_dir,
        {
            "history_source_config_file": history_source_config_file,
            "source_config_file": history_source_config_file,
            "run_dir": _display_settings_path(run_dir),
            "runtime_config_file": _display_settings_path(runtime_config_path),
            "original_config_file": _display_settings_path(original_config_path),
            "dataset_config_file": _display_settings_path(dataset_config_path),
        },
    )
    return {
        "run_dir": _display_settings_path(run_dir),
        "runtime_config_file": _display_settings_path(runtime_config_path),
        "original_config_file": _display_settings_path(original_config_path),
        "dataset_config_file": _display_settings_path(dataset_config_path),
        "output_dir": runtime_cfg["output_dir"],
        "sample_dir": _display_settings_path(sample_dir),
        "model_cache_dir": _display_settings_path(model_cache_dir),
        "dataset_cache_dir": _display_settings_path(dataset_cache_dir),
        "training_output_dir": runtime_cfg["output_dir"],
        "logs_dir": runtime_cfg["logging_dir"],
        "torchinductor_cache_dir": _display_settings_path(torchinductor_dir),
        "triton_cache_dir": _display_settings_path(triton_dir),
        "history_source_config_file": history_source_config_file,
        "data_dirs": data_dirs,
        "dataset_dirs": runtime_rows,
        "sample_config": _sample_config_from_cfg(runtime_cfg, []),
    }

def _apply_runtime_env(env: dict[str, str], runtime: dict[str, Any] | None) -> None:
    _bind_legacy()
    if not runtime:
        return
    env["ANIMA_RUNTIME_CONFIG"] = str(runtime.get("runtime_config_file") or "")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(runtime.get("torchinductor_cache_dir") or "")
    env["TRITON_CACHE_DIR"] = str(runtime.get("triton_cache_dir") or "")

def _runtime_meta(runtime: dict[str, Any] | None) -> dict[str, str]:
    _bind_legacy()
    if not isinstance(runtime, dict):
        return {}
    return {
        key: str(runtime.get(key) or "")
        for key in RUNTIME_META_KEYS
        if str(runtime.get(key) or "").strip()
    }

def _delete_queue_item_runtime_dir(item: dict[str, Any]) -> dict[str, Any]:
    _bind_legacy()
    run_dir = _queue_item_runtime_delete_dir(item)
    if run_dir is None:
        return {"deleted": False, "runtime_dir": ""}

    runtime_dir = _display_settings_path(run_dir)
    if not _path_exists(run_dir):
        return {"deleted": False, "runtime_dir": runtime_dir}
    if not run_dir.is_dir():
        raise ValueError("运行缓存路径不是目录，已阻止删除")

    output_root = resolve_output_root()
    if run_dir == output_root or not _path_is_relative_to(run_dir, output_root):
        raise ValueError("运行缓存目录不在 WebUI 输出根目录内，已阻止删除")
    if not _is_web_runtime_dir(run_dir):
        raise ValueError("运行缓存目录缺少 WebUI runtime 标记，已阻止删除")
    _validate_queue_runtime_dir_match(item, run_dir)

    shutil.rmtree(run_dir)
    return {"deleted": True, "runtime_dir": runtime_dir}

def _queue_item_runtime_dir_label(item: dict[str, Any]) -> str:
    _bind_legacy()
    run_dir = _queue_item_runtime_delete_dir(item)
    return _display_settings_path(run_dir) if run_dir is not None else ""

def _queue_item_runtime_delete_dir(item: dict[str, Any]) -> Path | None:
    _bind_legacy()
    runtime_info = item.get("runtime_info") if isinstance(item.get("runtime_info"), dict) else {}
    run_dir = _resolve_display_path(str(runtime_info.get("run_dir") or ""))
    if run_dir is not None:
        return run_dir
    for value in (
        str(runtime_info.get("runtime_config_file") or ""),
        str(item.get("runtime_config_file") or ""),
    ):
        path = _resolve_display_path(value)
        if path is not None and path.name == "config.runtime.toml":
            return path.parent
    output_dir = _resolve_display_path(str(runtime_info.get("training_output_dir") or runtime_info.get("output_dir") or ""))
    if output_dir is not None and output_dir.name == "training_output":
        return output_dir.parent
    return None

def _validate_queue_runtime_dir_match(item: dict[str, Any], run_dir: Path) -> None:
    _bind_legacy()
    expected_config = _resolve_display_path(str(item.get("runtime_config_file") or ""))
    runtime_info = item.get("runtime_info") if isinstance(item.get("runtime_info"), dict) else {}
    info_config = _resolve_display_path(str(runtime_info.get("runtime_config_file") or ""))
    valid_configs = [path.resolve() for path in (expected_config, info_config) if path is not None]
    actual_config = (run_dir / "config.runtime.toml").resolve()
    if not valid_configs:
        raise ValueError("队列记录缺少 runtime 配置，已阻止删除")
    if actual_config not in valid_configs:
        raise ValueError("运行缓存目录与队列记录的 runtime 配置不匹配，已阻止删除")
    run_meta = _read_runtime_run_meta(run_dir)
    meta_config = _resolve_display_path(str(run_meta.get("runtime_config_file") or ""))
    if meta_config is not None and meta_config.resolve() != actual_config:
        raise ValueError("运行缓存目录的 runtime 元数据不匹配，已阻止删除")

def _path_is_relative_to(path: Path, parent: Path) -> bool:
    _bind_legacy()
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False

def _write_runtime_run_meta(run_dir: Path, payload: dict[str, Any]) -> None:
    _bind_legacy()
    meta = {key: value for key, value in payload.items() if str(value or "").strip()}
    _write_json(run_dir / RUN_META_FILE, meta)

def _read_runtime_run_meta(run_dir: Path) -> dict[str, Any]:
    _bind_legacy()
    meta = _read_json(run_dir / RUN_META_FILE)
    return meta if isinstance(meta, dict) else {}

def _runtime_from_config_file(
    config_file: str | None,
    *,
    source_config_file: str | None = None,
) -> dict[str, Any] | None:
    _bind_legacy()
    if not config_file:
        return None
    config_path = _resolve_display_path(config_file)
    if config_path is None or not _path_exists(config_path) or not config_path.is_file():
        return None
    run_dir = config_path.parent
    model_cache_dir = run_dir / "model_cache"
    training_output_dir = run_dir / "training_output"
    dataset_cache_dir = run_dir / "dataset_cache"
    if not model_cache_dir.is_dir() or not training_output_dir.is_dir():
        return None

    cfg = _load_config_file_config(_display_settings_path(config_path))
    run_meta = _read_runtime_run_meta(run_dir)
    source_config_path = _resolve_display_path(source_config_file or "") if source_config_file else None
    history_source_config_file = (
        _display_settings_path(source_config_path)
        if source_config_path is not None
        else str(
            run_meta.get("history_source_config_file")
            or run_meta.get("source_config_file")
            or ""
        )
    )
    history_source_config_file = _display_project_path(history_source_config_file)
    source_dir = str(cfg.get("source_image_dir") or "")
    resized_dir = str(cfg.get("resized_image_dir") or "")
    lora_dir = str(cfg.get("lora_cache_dir") or "")
    sample_dir = training_output_dir / "sample"
    logs_dir = model_cache_dir / "logs"
    torchinductor_dir = model_cache_dir / "torchinductor"
    triton_dir = model_cache_dir / "triton"
    for path in (sample_dir, logs_dir, torchinductor_dir, triton_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": _display_settings_path(run_dir),
        "runtime_config_file": _display_settings_path(config_path),
        "original_config_file": _display_settings_path(run_dir / "config.original.toml"),
        "dataset_config_file": str(cfg.get("dataset_config") or ""),
        "output_dir": str(cfg.get("output_dir") or _display_settings_path(training_output_dir)),
        "sample_dir": _display_settings_path(sample_dir),
        "model_cache_dir": _display_settings_path(model_cache_dir),
        "dataset_cache_dir": _display_settings_path(dataset_cache_dir),
        "training_output_dir": str(cfg.get("output_dir") or _display_settings_path(training_output_dir)),
        "logs_dir": str(cfg.get("logging_dir") or _display_settings_path(logs_dir)),
        "torchinductor_cache_dir": _display_settings_path(torchinductor_dir),
        "triton_cache_dir": _display_settings_path(triton_dir),
        "history_source_config_file": history_source_config_file,
        "data_dirs": {
            "source_image_dir": source_dir,
            "resized_image_dir": resized_dir,
            "lora_cache_dir": lora_dir,
        },
    }

def _clone_frozen_runtime_config(
    config_file: str,
    *,
    source_config_file: str = "",
    reset_data_dirs: bool = False,
    resume_step: int | None = None,
    duration_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _bind_legacy()
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
    model_cache_dir = run_dir / "model_cache"
    dataset_cache_dir = run_dir / "dataset_cache"
    training_output_dir = run_dir / "training_output"
    sample_dir = training_output_dir / "sample"
    logs_dir = model_cache_dir / "logs"
    torchinductor_dir = model_cache_dir / "torchinductor"
    triton_dir = model_cache_dir / "triton"
    for path in (
        model_cache_dir,
        dataset_cache_dir,
        training_output_dir,
        sample_dir,
        logs_dir,
        torchinductor_dir,
        triton_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    original_config_path = run_dir / "config.original.toml"
    old_original = previous_run_dir / "config.original.toml"
    shutil.copy2(old_original if _path_exists(old_original) else config_path, original_config_path)

    dataset_config_path = run_dir / "dataset.runtime.toml"
    runtime_rows = _dataset_rows_for_estimate(cfg)
    if not runtime_rows:
        raise ValueError("冻结运行配置缺少数据集路径，无法重新预处理" if reset_data_dirs else "冻结运行配置缺少数据集路径，无法重新入队")
    cloned_rows = _clone_runtime_dataset_rows(
        runtime_rows,
        dataset_cache_dir,
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
        "output_dir": _display_settings_path(training_output_dir),
        "logging_dir": _display_settings_path(logs_dir),
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
    return {
        "run_dir": _display_settings_path(run_dir),
        "runtime_config_file": _display_settings_path(runtime_config_path),
        "original_config_file": _display_settings_path(original_config_path),
        "dataset_config_file": _display_settings_path(dataset_config_path),
        "output_dir": runtime_cfg["output_dir"],
        "sample_dir": _display_settings_path(sample_dir),
        "model_cache_dir": _display_settings_path(model_cache_dir),
        "dataset_cache_dir": _display_settings_path(dataset_cache_dir),
        "training_output_dir": runtime_cfg["output_dir"],
        "logs_dir": runtime_cfg["logging_dir"],
        "torchinductor_cache_dir": _display_settings_path(torchinductor_dir),
        "triton_cache_dir": _display_settings_path(triton_dir),
        "history_source_config_file": history_source_config_file,
        "data_dirs": data_dirs,
        "sample_config": _sample_config_from_cfg(runtime_cfg, []),
        "resume_duration": resume_duration,
    }

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

    target_total_steps = current_step + append_steps
    runtime_cfg["max_train_steps"] = target_total_steps
    runtime_cfg.pop("max_train_epochs", None)
    info.update({
        "resume_step": current_step,
        "append_steps": append_steps,
        "target_total_steps": target_total_steps,
    })
    return info

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

def _clone_runtime_dataset_rows(
    runtime_rows: list[dict[str, Any]],
    dataset_cache_dir: Path,
    *,
    copy_existing: bool,
) -> list[dict[str, Any]]:
    _bind_legacy()
    cloned_rows: list[dict[str, Any]] = []
    for index, row in enumerate(runtime_rows, start=1):
        group_dir = dataset_cache_dir / f"dataset-{index:02d}"
        resized_dir = group_dir / _runtime_dataset_child_name(
            str(row.get("image_dir") or row.get("resized_image_dir") or ""),
            default="resized",
            allowed={"resized", "trigger-clone-resized"},
        )
        lora_dir = group_dir / _runtime_dataset_child_name(
            str(row.get("cache_dir") or row.get("lora_cache_dir") or ""),
            default="lora",
            allowed={"lora", "trigger-clone-lora"},
        )
        resized_dir.mkdir(parents=True, exist_ok=True)
        lora_dir.mkdir(parents=True, exist_ok=True)
        if copy_existing:
            _copy_runtime_dataset_dir(str(row.get("image_dir") or row.get("resized_image_dir") or ""), resized_dir)
            _copy_runtime_dataset_dir(str(row.get("cache_dir") or row.get("lora_cache_dir") or ""), lora_dir)
        source_dir = str(row.get("source_dir") or row.get("source_image_dir") or row.get("image_dir") or "")
        source_path = _resolve_display_path(source_dir)
        source_target = group_dir / _runtime_dataset_child_name(
            source_dir,
            default="source",
            allowed={"source", "trigger-clone-source"},
        )
        if (
            copy_existing
            and source_path
            and _is_materialized_runtime_source_dir(source_path)
            and source_path.resolve() != source_target.resolve()
        ):
            _copy_runtime_dataset_dir(source_dir, source_target)
            source_dir = _display_settings_path(source_target)
        cloned_rows.append({
            "source_dir": source_dir,
            "image_dir": _display_settings_path(resized_dir),
            "cache_dir": _display_settings_path(lora_dir),
            "num_repeats": row.get("num_repeats") or 1,
            "recursive": _bool_value_for_row(row.get("recursive"), True),
            "path_pattern": _normalize_path_pattern(row.get("path_pattern")),
            "settings": row.get("settings") if isinstance(row.get("settings"), dict) else {},
        })
    return cloned_rows

def _runtime_dataset_child_name(value: str, *, default: str, allowed: set[str]) -> str:
    _bind_legacy()
    path = _resolve_display_path(value)
    name = path.name if path is not None else ""
    return name if name in allowed else default

def _bool_value_for_row(value: Any, fallback: bool = False) -> bool:
    _bind_legacy()
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

def _prepare_runtime_nl_tag_mix_source(row: dict[str, Any], group_dir: Path, source_dir: str) -> str:
    _bind_legacy()
    mix = _normalize_nl_tag_mix(row.get("nl_tag_mix"))
    if not mix.get("enabled"):
        return source_dir
    source_path = _resolve_display_path(source_dir)
    if source_path is None:
        raise ValueError("captions格式nl/tag权重调整需要填写原始数据集路径")
    if not source_path.is_dir():
        raise ValueError(f"captions格式nl/tag权重调整失败: {source_dir} 不是目录")
    target_dir = group_dir / "source"
    target_dir.mkdir(parents=True, exist_ok=True)
    caption_settings = _nl_tag_mix_caption_settings(row)
    manifest = _build_nl_tag_mix_source(
        source_path,
        target_dir,
        tag_ratio=float(mix.get("tag_ratio") or 0.0),
        recursive=_bool_value_for_row(row.get("recursive"), True),
        path_pattern=_normalize_path_pattern(row.get("path_pattern")),
        **caption_settings,
    )
    (target_dir / "results.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return _display_settings_path(target_dir)

def _prepare_runtime_trigger_clone_source(row: dict[str, Any], group_dir: Path, source_dir: str) -> str:
    _bind_legacy()
    clone = _normalize_trigger_clone(row.get("trigger_clone"))
    if not clone["enabled"]:
        return source_dir
    prompt = clone["prompt"]
    if not prompt:
        raise ValueError("触发提示词图像克隆需要填写触发提示词")
    source_path = _resolve_display_path(source_dir)
    if source_path is None:
        raise ValueError("触发提示词图像克隆需要填写原始数据集路径")
    if not source_path.is_dir():
        raise ValueError(f"触发提示词图像克隆失败: {source_dir} 不是目录")
    target_dir = group_dir / "trigger-clone-source"
    target_dir.mkdir(parents=True, exist_ok=True)
    recursive = _bool_value_for_row(row.get("recursive"), True)
    path_pattern = _normalize_path_pattern(row.get("path_pattern"))
    images = _nl_tag_mix_image_files(
        source_path,
        DATASET_IMAGE_EXTS,
        recursive=recursive,
        path_pattern=path_pattern,
    )
    if not images:
        raise ValueError("触发提示词图像克隆失败: 数据集目录里没有可训练图片")

    captions_json: dict[str, list[str]] = {}
    items: list[dict[str, str]] = []
    for image_path in images:
        rel_image = _nl_tag_mix_relative_image_path(image_path, source_path)
        target_image = target_dir / rel_image
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_image)
        rel_key = rel_image.as_posix()
        captions_json[rel_key] = [prompt]
        items.append({
            "image": _display_settings_path(image_path),
            "target": _display_settings_path(target_image),
            "caption_key": rel_key,
        })

    (target_dir / CAPTIONS_JSON_FILE).write_text(
        json.dumps(captions_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "prompt": prompt,
        "num_repeats": clone["num_repeats"],
        "recursive": recursive,
        "source_dir": _display_settings_path(source_path),
        "target_dir": _display_settings_path(target_dir),
        "caption_source_mode": CAPTION_SOURCE_CAPTIONS_JSON,
        "total": len(items),
        "items": items,
    }
    (target_dir / "trigger-clone-results.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return _display_settings_path(target_dir)

def _nl_tag_mix_caption_settings(row: dict[str, Any]) -> dict[str, Any]:
    _bind_legacy()
    settings = row.get("settings") if isinstance(row.get("settings"), dict) else {}
    prefer_json_caption = bool(settings.get("prefer_json_caption"))
    return {
        "caption_source_mode": normalize_caption_source_mode(
            settings.get("caption_source_mode"),
            prefer_json_caption,
        ),
        "caption_extension": str(settings.get("caption_extension") or ".txt"),
        "prefer_json_caption": prefer_json_caption,
    }

def _build_nl_tag_mix_source(
    source_dir: Path,
    target_dir: Path,
    *,
    tag_ratio: float,
    recursive: bool = True,
    path_pattern: str = "*",
    caption_source_mode: str = "auto",
    caption_extension: str = ".txt",
    prefer_json_caption: bool = False,
) -> dict[str, Any]:
    _bind_legacy()
    samples = _classify_nl_tag_mix_samples(
        source_dir,
        recursive=recursive,
        path_pattern=path_pattern,
        caption_source_mode=caption_source_mode,
        caption_extension=caption_extension,
        prefer_json_caption=prefer_json_caption,
    )
    if not samples:
        raise ValueError("captions格式nl/tag权重调整失败: 数据集目录里没有可训练图片")
    captions_json_samples = [sample for sample in samples if sample.get("caption_entries")]
    plain_samples = [sample for sample in samples if not sample.get("caption_entries")]
    selected = [
        *captions_json_samples,
        *_select_nl_tag_mix_samples(plain_samples, tag_ratio=tag_ratio),
    ]
    items: list[dict[str, Any]] = []
    counts = {"tag": 0, "nl": 0}
    available_counts = {"tag": 0, "nl": 0}
    caption_available_counts = {"tag": 0, "nl": 0}
    caption_counts = {"tag": 0, "nl": 0}
    missing_caption_count = 0
    captions_json: dict[str, list[str]] = {}
    captions_json_target = target_dir / CAPTIONS_JSON_FILE
    for sample in samples:
        available_counts[sample["source"]] += 1
        available_entry_counts = _nl_tag_mix_source_counts(sample.get("caption_entries") or [])
        if available_entry_counts["tag"] or available_entry_counts["nl"]:
            caption_available_counts["tag"] += available_entry_counts["tag"]
            caption_available_counts["nl"] += available_entry_counts["nl"]
        elif sample.get("caption_path"):
            caption_available_counts[sample["source"]] += 1
        if not sample.get("caption_path"):
            missing_caption_count += 1
    for sample in sorted(selected, key=lambda item: item["image_path"].as_posix()):
        image_path = sample["image_path"]
        rel_image = _nl_tag_mix_relative_image_path(image_path, source_dir)
        target_image = target_dir / rel_image
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_image)
        caption_source = sample.get("caption_source")
        selected_entries: list[dict[str, Any]] = []
        selected_entry_counts = {"tag": 0, "nl": 0}
        if getattr(caption_source, "from_captions_json", False):
            selected_entries = _select_nl_tag_caption_entries(
                sample.get("caption_entries") or [],
                tag_ratio=tag_ratio,
            )
            selected_entry_counts = _nl_tag_mix_source_counts(selected_entries)
            caption_counts["tag"] += selected_entry_counts["tag"]
            caption_counts["nl"] += selected_entry_counts["nl"]
            captions_json[rel_image.as_posix()] = [
                str(entry.get("text") or "")
                for entry in selected_entries
                if str(entry.get("text") or "").strip()
            ]
        elif sample.get("caption_path"):
            caption_counts[sample["source"]] += 1
        copied_captions = _copy_nl_tag_caption_sidecars(
            image_path,
            target_image,
            target_dir,
            caption_source,
            captions_json_path=captions_json_target,
        )
        source_kind = (
            _nl_tag_mix_dominant_source(selected_entry_counts)
            if selected_entries
            else sample["source"]
        )
        counts[source_kind] += 1
        item = {
            "stem": image_path.stem,
            "source": source_kind,
            "classification": sample["classification"],
            "image": _display_settings_path(image_path),
            "target": _display_settings_path(target_image),
            "caption": _display_settings_path(sample["caption_path"]) if sample.get("caption_path") else "",
            "captions": copied_captions,
            "caption_source_mode": sample.get("caption_source_mode", ""),
        }
        if selected_entries:
            item["caption_entry_count"] = len(sample.get("caption_entries") or [])
            item["weighted_caption_count"] = len(selected_entries)
            item["available_caption_counts"] = _nl_tag_mix_source_counts(sample.get("caption_entries") or [])
            item["actual_caption_counts"] = selected_entry_counts
            item["selected_caption_indices"] = [int(entry.get("index", 0)) for entry in selected_entries]
        items.append(item)
    if captions_json:
        captions_json_target.write_text(
            json.dumps(captions_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return {
        "tag_ratio": min(1.0, max(0.0, tag_ratio)),
        "classification_method": NL_TAG_MIX_CLASSIFICATION_METHOD,
        "caption_source_mode": caption_source_mode,
        "recursive": bool(recursive),
        "path_pattern": _normalize_path_pattern(path_pattern),
        "source_dir": _display_settings_path(source_dir),
        "available_tag_count": available_counts["tag"],
        "available_nl_count": available_counts["nl"],
        "actual_tag_count": counts["tag"],
        "actual_nl_count": counts["nl"],
        "available_tag_caption_count": caption_available_counts["tag"],
        "available_nl_caption_count": caption_available_counts["nl"],
        "actual_tag_caption_count": caption_counts["tag"],
        "actual_nl_caption_count": caption_counts["nl"],
        "total": len(items),
        "missing_caption_count": missing_caption_count,
        "items": items,
    }

def _classify_nl_tag_mix_samples(
    source_dir: Path,
    *,
    recursive: bool = True,
    path_pattern: str = "*",
    caption_source_mode: str = "auto",
    caption_extension: str = ".txt",
    prefer_json_caption: bool = False,
) -> list[dict[str, Any]]:
    _bind_legacy()
    samples: list[dict[str, Any]] = []
    for image_path in _nl_tag_mix_image_files(
        source_dir,
        DATASET_IMAGE_EXTS,
        recursive=recursive,
        path_pattern=path_pattern,
    ):
        caption_source = _nl_tag_mix_caption_source(
            image_path,
            caption_source_mode=caption_source_mode,
            caption_extension=caption_extension,
            prefer_json_caption=prefer_json_caption,
            captions_root=source_dir,
        )
        caption_texts = caption_source.caption_texts()
        caption_entries = (
            _nl_tag_mix_caption_entries(caption_texts)
            if getattr(caption_source, "from_captions_json", False)
            else []
        )
        if caption_entries:
            entry_counts = _nl_tag_mix_source_counts(caption_entries)
            source_kind = _nl_tag_mix_dominant_source(entry_counts)
            classification = {
                "kind": source_kind,
                "reason": "captions_json_caption_entries_majority",
                "method": NL_TAG_MIX_CLASSIFICATION_METHOD,
                "metrics": {
                    "caption_count": len(caption_entries),
                    "tag_caption_count": entry_counts["tag"],
                    "nl_caption_count": entry_counts["nl"],
                },
            }
        else:
            caption_text = "\n".join(caption_texts)
            classification = _classify_nl_tag_caption_text(caption_text)
            source_kind = classification["kind"]
        samples.append({
            "image_path": image_path,
            "caption_path": caption_source.path,
            "caption_source": caption_source,
            "caption_source_mode": caption_source.detected_mode,
            "caption_texts": caption_texts,
            "caption_entries": caption_entries,
            "source": source_kind,
            "classification": classification,
        })
    return samples

def _nl_tag_mix_caption_entries(caption_texts: list[str]) -> list[dict[str, Any]]:
    _bind_legacy()
    entries: list[dict[str, Any]] = []
    for index, text in enumerate(caption_texts):
        clean_text = str(text or "").strip()
        if not clean_text:
            continue
        classification = _classify_nl_tag_caption_text(clean_text)
        entries.append({
            "index": index,
            "text": clean_text,
            "source": classification["kind"],
            "classification": classification,
        })
    return entries

def _nl_tag_mix_source_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    _bind_legacy()
    counts = {"tag": 0, "nl": 0}
    for entry in entries:
        source = str(entry.get("source") or "")
        if source in counts:
            counts[source] += 1
    return counts

def _nl_tag_mix_dominant_source(counts: dict[str, int]) -> str:
    _bind_legacy()
    return "nl" if int(counts.get("nl") or 0) > int(counts.get("tag") or 0) else "tag"

def _cycle_nl_tag_entries(entries: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    _bind_legacy()
    if count <= 0 or not entries:
        return []
    return [entries[index % len(entries)] for index in range(count)]

def _select_nl_tag_caption_entries(
    entries: list[dict[str, Any]],
    *,
    tag_ratio: float,
) -> list[dict[str, Any]]:
    _bind_legacy()
    if not entries:
        return []
    ratio = min(1.0, max(0.0, tag_ratio))
    total = len(entries)
    tag_entries = [entry for entry in entries if entry["source"] == "tag"]
    nl_entries = [entry for entry in entries if entry["source"] == "nl"]
    tag_quota = int(round(total * ratio))
    nl_quota = total - tag_quota
    selected = [
        *_cycle_nl_tag_entries(tag_entries, tag_quota),
        *_cycle_nl_tag_entries(nl_entries, nl_quota),
    ]
    if len(selected) < total:
        fallback = tag_entries + nl_entries or entries
        selected.extend(_cycle_nl_tag_entries(fallback, total - len(selected)))
    return selected[:total]

def _nl_tag_mix_relative_image_path(image_path: Path, source_dir: Path) -> Path:
    _bind_legacy()
    try:
        return image_path.resolve().relative_to(source_dir.resolve())
    except ValueError:
        return Path(image_path.name)

def _select_nl_tag_mix_samples(samples: list[dict[str, Any]], *, tag_ratio: float) -> list[dict[str, Any]]:
    _bind_legacy()
    ratio = min(1.0, max(0.0, tag_ratio))
    tag_samples = [sample for sample in samples if sample["source"] == "tag"]
    nl_samples = [sample for sample in samples if sample["source"] == "nl"]
    tag_quota = int(round(len(samples) * ratio))
    nl_quota = len(samples) - tag_quota
    selected = [*tag_samples[:tag_quota], *nl_samples[:nl_quota]]
    if len(selected) < len(samples):
        selected_ids = {id(sample) for sample in selected}
        fill = [sample for sample in samples if id(sample) not in selected_ids]
        selected.extend(fill[:len(samples) - len(selected)])
    return sorted(selected, key=lambda sample: sample["image_path"].name)

def _copy_nl_tag_caption_sidecars(
    image_path: Path,
    target_image: Path,
    target_dir: Path,
    caption_source=None,
    *,
    captions_json_path: Path | None = None,
) -> list[str]:
    _bind_legacy()
    if getattr(caption_source, "from_captions_json", False):
        return [_display_settings_path(captions_json_path)] if captions_json_path is not None else []
    copied: list[str] = []
    copied_sources: set[Path] = set()
    if getattr(caption_source, "path", None) is not None:
        source_path = caption_source.path
        if source_path.is_file():
            target = target_image.with_suffix(source_path.suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
            copied.append(_display_settings_path(target))
            copied_sources.add(source_path.resolve())
    for ext in DATASET_CAPTION_EXTS:
        source = image_path.with_suffix(ext)
        if not source.is_file() or source.resolve() in copied_sources:
            continue
        target = target_image.with_suffix(ext)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(_display_settings_path(target))
    return copied

def _copy_runtime_dataset_dir(source: str, target: Path) -> None:
    _bind_legacy()
    source_path = _resolve_display_path(source)
    if source_path is None or not _path_exists(source_path) or not source_path.is_dir():
        return
    if source_path.resolve() == target.resolve():
        return
    shutil.copytree(source_path, target, dirs_exist_ok=True)

def _is_materialized_runtime_source_dir(path: Path) -> bool:
    _bind_legacy()
    parts = {part.lower() for part in path.parts}
    return path.name in {"source", "trigger-clone-source"} and "dataset_cache" in parts

def _unique_runtime_dir(output_root: Path, stem: str) -> Path:
    _bind_legacy()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = output_root / f"{stem}-{timestamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = output_root / f"{stem}-{timestamp}-{suffix}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate

def _safe_run_stem(value: str) -> str:
    _bind_legacy()
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip()).strip(".-")
    return clean[:80] or "run"

def _is_web_runtime_dir(path: Path) -> bool:
    _bind_legacy()
    return (
        ((path / "config.runtime.toml").is_file() or (path / RUN_META_FILE).is_file())
        and (path / "model_cache").is_dir()
        and (path / "dataset_cache").is_dir()
        and (path / "training_output").is_dir()
    )

def _path_exists(path: Path) -> bool:
    _bind_legacy()
    try:
        return path.exists()
    except OSError:
        return False

def _sample_config_from_cfg(cfg: dict[str, Any], extra_args: list[str]) -> dict[str, Any]:
    _bind_legacy()
    sample_prompts = cfg.get("sample_prompts")
    sample_every_n_epochs = cfg.get("sample_every_n_epochs")
    sample_every_n_steps = cfg.get("sample_every_n_steps")
    sample_at_first = bool(cfg.get("sample_at_first", False))
    sample_sampler_raw = str(cfg.get("sample_sampler") or "euler")

    overrides = _cli_arg_overrides(extra_args)
    if "sample_prompts" in overrides:
        sample_prompts = overrides["sample_prompts"]
    if "sample_every_n_epochs" in overrides:
        sample_every_n_epochs = overrides["sample_every_n_epochs"]
    if "sample_every_n_steps" in overrides:
        sample_every_n_steps = overrides["sample_every_n_steps"]
    if "sample_at_first" in overrides:
        sample_at_first = True
    if "sample_sampler" in overrides:
        sample_sampler_raw = str(overrides["sample_sampler"] or sample_sampler_raw)

    epoch_freq = _positive_int_or_none(sample_every_n_epochs)
    step_freq = _positive_int_or_none(sample_every_n_steps)
    sample_sampler, sample_sampler_status = training_sample_sampler_status(sample_sampler_raw)
    prompt_path = _resolve_display_path(str(sample_prompts or ""))
    prompt_exists = prompt_path.is_file() if prompt_path else False
    enabled = bool(prompt_path and prompt_exists and (epoch_freq is not None or step_freq is not None or sample_at_first))

    if not sample_prompts:
        message = "未设置 sample_prompts，训练不会生成样张"
    elif not prompt_exists:
        message = f"sample_prompts 文件不存在: {sample_prompts}"
    elif epoch_freq is None and step_freq is None and not sample_at_first:
        message = "未设置 sample_every_n_epochs 或 sample_every_n_steps，训练不会生成样张"
    else:
        message = "训练中采样已配置"

    return {
        "enabled": enabled,
        "sample_prompts": str(sample_prompts or ""),
        "sample_prompts_exists": prompt_exists,
        "sample_every_n_epochs": epoch_freq,
        "sample_every_n_steps": step_freq,
        "sample_at_first": sample_at_first,
        "sample_sampler": sample_sampler,
        "sample_sampler_raw": sample_sampler_raw,
        "sample_sampler_status": sample_sampler_status,
        "message": message,
    }

def _cli_arg_overrides(extra_args: list[str]) -> dict[str, Any]:
    _bind_legacy()
    out: dict[str, Any] = {}
    keys = {
        "--sample_prompts": "sample_prompts",
        "--sample_every_n_epochs": "sample_every_n_epochs",
        "--sample_every_n_steps": "sample_every_n_steps",
        "--sample_sampler": "sample_sampler",
    }
    for idx, arg in enumerate(extra_args):
        if arg == "--sample_at_first":
            out["sample_at_first"] = True
            continue
        if arg in keys and idx + 1 < len(extra_args):
            out[keys[arg]] = extra_args[idx + 1]
            continue
        for cli_key, config_key in keys.items():
            prefix = cli_key + "="
            if arg.startswith(prefix):
                out[config_key] = arg.split("=", 1)[1]
                break
    return out

def _resolve_display_path(value: str) -> Path | None:
    _bind_legacy()
    raw = str(value or "").replace("\\", "/").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()

def _display_project_path(value: str) -> str:
    _bind_legacy()
    raw = str(value or "").replace("\\", "/").strip()
    if not raw:
        return ""
    path = Path(raw)
    if not path.is_absolute():
        return path.as_posix().strip("/")
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return raw
