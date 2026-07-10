"""Runtime state helpers for WebUI training runs."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

from web.services.training.common import RUN_META_FILE
from web.services.training.runtime_common import (
    RUNTIME_META_KEYS,
    _build_runtime_payload,
    _ensure_runtime_dir_layout,
    _load_config_file_config,
)
from web.services.training.runtime_paths import (
    _display_project_path,
    _display_settings_path,
    _path_exists,
    _path_is_relative_to,
    _resolve_display_path,
    resolve_output_root,
)
from web.services.training.storage import _read_json, _write_json


def _apply_runtime_env(env: dict[str, str], runtime: dict[str, Any] | None) -> None:
    if not runtime:
        return
    env["ANIMA_RUNTIME_CONFIG"] = str(runtime.get("runtime_config_file") or "")
    env["TORCHINDUCTOR_CACHE_DIR"] = str(runtime.get("torchinductor_cache_dir") or "")
    env["TRITON_CACHE_DIR"] = str(runtime.get("triton_cache_dir") or "")


def _runtime_meta(runtime: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(runtime, dict):
        return {}
    return {
        key: str(runtime.get(key) or "")
        for key in RUNTIME_META_KEYS
        if str(runtime.get(key) or "").strip()
    }


def _delete_queue_item_runtime_dir(item: dict[str, Any]) -> dict[str, Any]:
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
    run_dir = _queue_item_runtime_delete_dir(item)
    return _display_settings_path(run_dir) if run_dir is not None else ""


def _queue_item_runtime_delete_dir(item: dict[str, Any]) -> Path | None:
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


def _write_runtime_run_meta(run_dir: Path, payload: dict[str, Any]) -> None:
    meta = {key: value for key, value in payload.items() if str(value or "").strip()}
    _write_json(run_dir / RUN_META_FILE, meta)


def _read_runtime_run_meta(run_dir: Path) -> dict[str, Any]:
    meta = _read_json(run_dir / RUN_META_FILE)
    return meta if isinstance(meta, dict) else {}


def _runtime_from_config_file(
    config_file: str | None,
    *,
    source_config_file: str | None = None,
) -> dict[str, Any] | None:
    if not config_file:
        return None
    config_path = _resolve_display_path(config_file)
    if config_path is None or not _path_exists(config_path) or not config_path.is_file():
        return None

    run_dir = config_path.parent
    layout = _ensure_runtime_dir_layout(run_dir)
    if not layout["model_cache_dir"].is_dir() or not layout["training_output_dir"].is_dir():
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
    return _build_runtime_payload(
        run_dir=run_dir,
        layout=layout,
        runtime_config_file=config_path,
        original_config_file=run_dir / "config.original.toml",
        dataset_config_file=str(cfg.get("dataset_config") or ""),
        output_dir=str(cfg.get("output_dir") or _display_settings_path(layout["training_output_dir"])),
        logs_dir=str(cfg.get("logging_dir") or _display_settings_path(layout["logs_dir"])),
        history_source_config_file=history_source_config_file,
        data_dirs={
            "source_image_dir": str(cfg.get("source_image_dir") or ""),
            "resized_image_dir": str(cfg.get("resized_image_dir") or ""),
            "lora_cache_dir": str(cfg.get("lora_cache_dir") or ""),
        },
    )


def _is_web_runtime_dir(path: Path) -> bool:
    return (
        ((path / "config.runtime.toml").is_file() or (path / RUN_META_FILE).is_file())
        and (path / "model_cache").is_dir()
        and (path / "dataset_cache").is_dir()
        and (path / "training_output").is_dir()
    )
