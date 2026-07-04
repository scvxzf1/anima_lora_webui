"""Shared scalar and path helpers for WebUI config services."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import toml

from library.env import expand_env_vars, expand_env_vars_in_obj, get_configs_root
from web.services.config import paths as _config_paths
from web.services.config.metadata import DEFAULT_LORA_CACHE_DIR, DEFAULT_RESIZED_IMAGE_DIR

ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = get_configs_root()


def _load(p: Path) -> dict:
    if not p.exists():
        return {}
    return expand_env_vars_in_obj(toml.loads(p.read_text(encoding="utf-8")))


def _safe_resolve(rel_path: str) -> Path | None:
    return _config_paths.safe_resolve(rel_path, root=ROOT, configs_dir=CONFIGS_DIR)


def _safe_config_subdir(subdir: str) -> Path | None:
    return _config_paths.safe_config_subdir(subdir, configs_dir=CONFIGS_DIR)


def _resolve_project_path(value: str) -> Path:
    return _config_paths.resolve_display_path(
        value,
        root=ROOT,
        configs_dir=CONFIGS_DIR,
        expand_env_vars_fn=expand_env_vars,
    )


def _auto_data_dir_for_key(value: Any, source_path: Path, suffix: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        return _derived_data_dir(source_path, suffix)
    path = _resolve_project_path(raw)
    if _is_builtin_default_data_dir(raw) or not path.exists():
        return _derived_data_dir(source_path, suffix)
    return path


def _derived_data_dir(source_path: Path, suffix: str) -> Path:
    parent = source_path.parent if source_path.name else source_path
    name = source_path.name or "dataset"
    return (parent / f"{name}_{suffix}").resolve()


def _is_builtin_default_data_dir(value: str) -> bool:
    clean = str(value or "").replace("\\", "/").strip().strip("/")
    return clean in {DEFAULT_RESIZED_IMAGE_DIR, DEFAULT_LORA_CACHE_DIR}


def _display_path(path: Path) -> str:
    return _config_paths.display_path(path, root=ROOT, configs_dir=CONFIGS_DIR)


def _positive_int(value: Any, fallback: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return fallback
    return n if n > 0 else fallback


def _positive_int_or_none(value: Any) -> int | None:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _nonnegative_int(value: Any, fallback: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return fallback
    return n if n >= 0 else fallback


def _nonnegative_float(value: Any, fallback: float) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return fallback
    return n if n >= 0 else fallback


def _positive_float(value: Any, fallback: float) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return fallback
    return n if n > 0 else fallback


def _bool_value(value: Any, fallback: bool = False) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
