"""Shared path normalization helpers for WebUI config services."""

from __future__ import annotations

from pathlib import Path
from typing import Callable


ExpandEnvVars = Callable[[str], str]


def normalize_config_rel_path(rel_path: str) -> str:
    return str(rel_path or "").strip().replace("\\", "/").lstrip("/")


def safe_resolve(rel_path: str, *, root: Path, configs_dir: Path) -> Path | None:
    raw = str(rel_path or "").strip()
    if not raw:
        return None
    raw_path = Path(raw).expanduser()
    if raw_path.is_absolute():
        resolved = raw_path.resolve()
    else:
        normalized = normalize_config_rel_path(raw)
        if normalized == "configs":
            config_relative = ""
        elif normalized.startswith("configs/"):
            config_relative = normalized.removeprefix("configs/")
        else:
            config_relative = normalized
        resolved = (configs_dir / config_relative).resolve()
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


def resolve_display_path(
    value: str,
    *,
    root: Path,
    configs_dir: Path,
    expand_env_vars_fn: ExpandEnvVars,
) -> Path:
    raw = str(value or "").strip()
    path = Path(expand_env_vars_fn(raw))
    if path.is_absolute():
        return path.resolve()
    normalized = normalize_config_rel_path(raw)
    if normalized == "configs":
        return configs_dir.resolve()
    if normalized.startswith("configs/"):
        return (configs_dir / normalized.removeprefix("configs/")).resolve()
    return (root / normalized).resolve()


def display_path(path: Path, *, root: Path, configs_dir: Path | None = None) -> str:
    resolved = path.resolve()
    if configs_dir is not None:
        try:
            rel_to_configs = resolved.relative_to(configs_dir.resolve()).as_posix()
            return "configs" if rel_to_configs == "." else f"configs/{rel_to_configs}"
        except (TypeError, ValueError):
            pass
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except (TypeError, ValueError):
        return resolved.as_posix()
