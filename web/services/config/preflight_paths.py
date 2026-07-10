"""Config path validation and web-runtime detection for preflight."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import toml

from library.env import expand_env_vars, expand_env_vars_in_obj
from web.services.config.metadata import OUTPUT_RUN_CONFIG_FILES
from web.services.config.preflight_runtime import (
    CONFIGS_DIR,
    GLOBAL_MODEL_PATH_KEYS,
    ROOT,
    _normalize_config_rel_path,
    _resolve_project_path,
    _safe_resolve,
    apply_auto_data_dirs,
    load_merged_config,
    resolve_output_root,
)


def _blank_model_path(value: Any) -> bool:
    return str(value or "").strip() == ""


def apply_global_model_path_defaults(
    cfg: dict[str, Any],
    *,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fill blank model path scalars from Web global settings, preserving explicit config values."""

    defaults = _global_model_path_defaults(fallback or {})
    if not defaults:
        return cfg
    next_cfg = dict(cfg)
    for key in GLOBAL_MODEL_PATH_KEYS:
        if _blank_model_path(next_cfg.get(key)) and not _blank_model_path(defaults.get(key)):
            next_cfg[key] = defaults[key]
    return next_cfg


def _global_model_path_defaults(fallback: dict[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    settings_file = CONFIGS_DIR / "web-ui-settings.toml"
    if settings_file.exists():
        try:
            raw = toml.loads(settings_file.read_text(encoding="utf-8"))
        except toml.TomlDecodeError:
            raw = {}
        section = raw.get("global") if isinstance(raw, dict) else {}
        if isinstance(section, dict):
            for key in GLOBAL_MODEL_PATH_KEYS:
                value = str(section.get(key) or "").strip()
                if value:
                    values[key] = expand_env_vars(value)
    for key in GLOBAL_MODEL_PATH_KEYS:
        if key not in values and not _blank_model_path(fallback.get(key)):
            values[key] = expand_env_vars(str(fallback[key]).strip())
    return values


def _load_training_config_for_web_run(
    variant: str,
    preset: str,
    methods_subdir: str,
    *,
    config_file: str | None = None,
) -> dict[str, Any]:
    fallback_cfg = apply_auto_data_dirs(load_merged_config(variant, preset, methods_subdir))
    cfg = dict(fallback_cfg)
    source = _config_file_path(config_file)
    if source is not None:
        try:
            cfg.update(expand_env_vars_in_obj(toml.loads(source.read_text(encoding="utf-8"))))
        except toml.TomlDecodeError as exc:
            raise ValueError(f"训练配置 TOML 解析失败: {config_file}") from exc
    cfg = apply_global_model_path_defaults(cfg, fallback=fallback_cfg)
    return apply_auto_data_dirs(cfg)


def _config_file_path(config_file: str | None) -> Path | None:
    raw = str(config_file or "").strip()
    if not raw:
        return None
    path = Path(raw.replace("\\", "/"))
    if ".." in path.parts:
        raise ValueError("训练配置路径不能包含 ..")
    if path.is_absolute():
        resolved = path.resolve()
    else:
        normalized = _normalize_config_rel_path(raw)
        resolved = _config_path_from_display_path(normalized)
        if resolved is None:
            resolved = (ROOT / normalized).resolve()
    if _is_output_run_snapshot_config(resolved) and resolved.name != OUTPUT_RUN_CONFIG_FILES["runtime"][0]:
        raise ValueError("训练输出目录只能使用 config.runtime.toml 作为训练配置")
    if not _is_allowed_training_config_path(resolved):
        raise ValueError("训练配置必须在项目目录或全局输出文件夹内")
    if not resolved.is_file():
        raise FileNotFoundError(f"训练配置不存在: {config_file}")
    if resolved.suffix.lower() != ".toml":
        raise ValueError("训练配置必须是 TOML 文件")
    return resolved

def _config_path_from_display_path(normalized: str) -> Path | None:
    if normalized == "configs" or normalized.startswith("configs/"):
        return _safe_resolve(normalized)
    return None

def _is_allowed_training_config_path(path: Path) -> bool:
    resolved = path.resolve()

    # 1. 检查是否在项目根目录下
    try:
        resolved.relative_to(ROOT.resolve())
        return True
    except ValueError:
        pass

    # 2. 检查是否在配置目录下（支持外置配置）
    # 使用同步后的 CONFIGS_DIR，它会在 _exported 装饰器中更新
    # 注意：即使 configs/ 是指向外部目录的符号链接，resolve() 也会返回相同的真实路径
    # 因此这个检查能覆盖符号链接和真实外部目录两种情况
    try:
        configs_root = CONFIGS_DIR.resolve()
        resolved.relative_to(configs_root)
        return True
    except (ValueError, AttributeError):
        pass

    # 3. 检查是否在全局输出目录下
    try:
        rel_to_output = resolved.relative_to(resolve_output_root().resolve())
    except ValueError:
        return _is_web_runtime_config_tree(resolved)
    return (
        len(rel_to_output.parts) == 2
        and rel_to_output.name == "config.runtime.toml"
        and _is_web_runtime_config_tree(resolved)
    )

def _is_web_runtime_config_tree(path: Path) -> bool:
    run_dir = path.parent
    return (
        path.name == "config.runtime.toml"
        and path.is_file()
        and _has_web_runtime_dirs(run_dir)
    )

def _is_output_run_snapshot_config(path: Path) -> bool:
    resolved = path.resolve()
    if resolved.name not in {filename for filename, _label in OUTPUT_RUN_CONFIG_FILES.values()}:
        return False
    try:
        rel_to_output = resolved.relative_to(resolve_output_root().resolve())
        if len(rel_to_output.parts) == 2:
            return True
    except ValueError:
        pass
    return _has_web_runtime_dirs(resolved.parent)

def _has_web_runtime_dirs(run_dir: Path) -> bool:
    return (
        (run_dir / "model_cache").is_dir()
        and (run_dir / "dataset_cache").is_dir()
        and (run_dir / "training_output").is_dir()
    )

def is_web_runtime_config(config_file: str | None) -> bool:
    path = _config_file_path(config_file)
    if path is None:
        return False
    run_dir = path.parent
    return (
        path.name == "config.runtime.toml"
        and (run_dir / "model_cache").is_dir()
        and (run_dir / "dataset_cache").is_dir()
        and (run_dir / "training_output").is_dir()
    )

def _looks_like_web_runtime_config(cfg: dict[str, Any]) -> bool:
    output_root = resolve_output_root().resolve()
    for key in ("output_dir", "logging_dir", "dataset_config", "resized_image_dir", "lora_cache_dir"):
        raw = str(cfg.get(key) or "").strip()
        if not raw:
            continue
        path = _resolve_project_path(raw)
        try:
            path.relative_to(output_root)
            return True
        except ValueError:
            continue
    return False
