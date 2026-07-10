"""Shared runtime path helpers for WebUI training modules."""

from __future__ import annotations

from pathlib import Path
import re

from web.services import config_service
from web.services.training.context import training_facade as _training_facade


def _project_root() -> Path:
    return _training_facade().ROOT


def _history_dir():
    return _training_facade().HISTORY_DIR


def _configs_dir() -> Path:
    return config_service.CONFIGS_DIR


def _display_settings_path(path: str | Path | None) -> str:
    return _training_facade()._display_settings_path(path)


def _training_datetime():
    return _training_facade().datetime


def resolve_output_root() -> Path:
    return _training_facade().resolve_output_root()


def _resolve_display_path(value: str) -> Path | None:
    raw = str(value or "").replace("\\", "/").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    normalized = path.as_posix().strip("/")
    if normalized == "configs":
        return _configs_dir().resolve()
    if normalized.startswith("configs/"):
        return (_configs_dir() / normalized.removeprefix("configs/")).resolve()
    return (_project_root() / path).resolve()


def _display_project_path(value: str) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    if not raw:
        return ""
    path = Path(raw)
    if not path.is_absolute():
        normalized = path.as_posix().strip("/")
        if normalized == "configs" or normalized.startswith("configs/"):
            return normalized
        path = (_project_root() / path).resolve()
    try:
        rel_to_configs = path.resolve().relative_to(_configs_dir().resolve()).as_posix()
        return "configs" if rel_to_configs == "." else f"configs/{rel_to_configs}"
    except ValueError:
        pass
    try:
        return path.resolve().relative_to(_project_root().resolve()).as_posix()
    except ValueError:
        return raw


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _unique_runtime_dir(output_root: Path, stem: str) -> Path:
    timestamp = _training_datetime().now().strftime("%Y%m%d-%H%M%S")
    base = output_root / f"{stem}-{timestamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = output_root / f"{stem}-{timestamp}-{suffix}"
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _safe_run_stem(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "").strip()).strip(".-")
    return clean[:80] or "run"
