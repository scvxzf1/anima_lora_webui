"""Cross-platform project venv Python resolution for WebUI."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from web.services import config_service as _config_service

ROOT = _config_service.ROOT


def venv_python_path(root: Path | None = None, *, windowless: bool = False) -> Path | None:
    """Return the project ``.venv`` interpreter if it exists."""
    base = (root or ROOT).resolve()
    if sys.platform == "win32":
        name = "pythonw.exe" if windowless else "python.exe"
        cand = base / ".venv" / "Scripts" / name
        if cand.is_file():
            return cand
        if windowless:
            cand = base / ".venv" / "Scripts" / "python.exe"
            if cand.is_file():
                return cand
        return None
    cand = base / ".venv" / "bin" / "python"
    return cand if cand.is_file() else None


def resolve_web_python_executable(root: Path | None = None) -> str:
    """Interpreter Web training/preprocess should use (venv first)."""
    found = venv_python_path(root, windowless=False)
    if found is not None:
        return str(found)
    return sys.executable


def venv_bin_dir(root: Path | None = None) -> Path | None:
    """Directory to prepend to PATH for venv CLI tools."""
    base = (root or ROOT).resolve()
    if sys.platform == "win32":
        scripts = base / ".venv" / "Scripts"
        return scripts if scripts.is_dir() else None
    bindir = base / ".venv" / "bin"
    return bindir if bindir.is_dir() else None


def prepend_path(env: dict[str, str], directory: Path | str) -> None:
    value = str(directory)
    sep = os.pathsep
    parts = [part for part in env.get("PATH", "").split(sep) if part]
    if value not in parts:
        env["PATH"] = sep.join([value, *parts])
