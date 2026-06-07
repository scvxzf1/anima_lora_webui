"""Shared path normalization helpers for WebUI config services."""

from __future__ import annotations

from pathlib import Path
from typing import Callable


ExpandEnvVars = Callable[[str], str]


def normalize_config_rel_path(rel_path: str) -> str:
    return str(rel_path or "").strip().replace("\\", "/").lstrip("/")


def safe_resolve(rel_path: str, *, root: Path, configs_dir: Path) -> Path | None:
    resolved = (root / normalize_config_rel_path(rel_path)).resolve()
    try:
        resolved.relative_to(configs_dir.resolve())
    except ValueError:
        return None
    return resolved


def safe_config_subdir(subdir: str, *, configs_dir: Path) -> Path | None:
    clean = str(subdir or "").replace("\\", "/").strip("/")
    if not clean or ".." in Path(clean).parts:
        return None
    resolved = (configs_dir / clean).resolve()
    try:
        resolved.relative_to(configs_dir.resolve())
    except ValueError:
        return None
    return resolved


def resolve_project_path(value: str, *, root: Path, expand_env_vars_fn: ExpandEnvVars) -> Path:
    path = Path(expand_env_vars_fn(value))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def display_path(path: Path, *, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except (TypeError, ValueError):
        return path.as_posix()
