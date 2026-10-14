"""Read-only, bounded browsing of server-configured model directories."""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath
from typing import Any

from library.env import anima_home
from web.services.model_config_service import ModelConfigFileError, get_model_configs

MODEL_SUFFIXES = frozenset({".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf"})
MODEL_PATH_KEYS = ("pretrained_model_name_or_path", "qwen3", "vae")
MAX_DIRECTORY_ENTRIES = 5000


def _resolve_path(value: str, *, follow_symlinks: bool = True) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    windows_path = PureWindowsPath(expanded)
    if windows_path.drive and len(windows_path.parts) == 1:
        raise ValueError("不能浏览盘根目录")
    path = Path(expanded)
    if ".." in path.parts or ".." in PureWindowsPath(expanded).parts:
        raise ValueError("模型目录路径不能包含 ..")
    if not path.is_absolute():
        path = anima_home() / path
    return path.resolve() if follow_symlinks else path.absolute()


def _model_roots() -> list[Path]:
    # Only durable server settings grant access; client paths never add roots.
    try:
        items = get_model_configs()["items"]
    except ValueError as exc:
        raise ModelConfigFileError(str(exc)) from exc
    roots = []
    default = _resolve_path(str(anima_home() / "models"))
    candidates = [default]
    for item in items:
        for key in MODEL_PATH_KEYS:
            value = str(item.get(key) or "").strip()
            if not value:
                continue
            try:
                path = _resolve_path(value)
                candidates.append(path if path.is_dir() else path.parent)
            except (ValueError, OSError, RuntimeError):
                continue
    for path in candidates:
        if path == Path(path.anchor) or path in roots:
            continue
        roots.append(path)
    return roots


def _is_allowed(path: Path, roots: list[Path]) -> bool:
    return any(path.is_relative_to(root) for root in roots)


def _is_hidden(path: Path, roots: list[Path]) -> bool:
    return all(
        any(part.startswith(".") for part in path.relative_to(root).parts)
        for root in roots
        if path.is_relative_to(root)
    )


def _scan_entries(directory: Path, roots: list[Path]) -> tuple[list[dict[str, str]], bool]:
    entries = []
    truncated = False
    with os.scandir(directory) as children:
        for index, child in enumerate(children):
            if index >= MAX_DIRECTORY_ENTRIES:
                truncated = True
                break
            if child.name.startswith("."):
                continue
            try:
                path = Path(child.path).resolve()
                if not _is_allowed(path, roots) or _is_hidden(path, roots):
                    continue
                if path.is_dir():
                    kind = "directory"
                elif (
                    Path(child.name).suffix.lower() in MODEL_SUFFIXES
                    and path.suffix.lower() in MODEL_SUFFIXES
                    and path.is_file()
                ):
                    kind = "file"
                else:
                    continue
            except (OSError, RuntimeError):
                continue
            entries.append({"name": child.name, "path": str(path), "kind": kind})
    entries.sort(key=lambda entry: (entry["kind"] != "directory", entry["name"].casefold(), entry["name"]))
    return entries, truncated


def list_model_files(
    directory: str | None = None, *, offset: int = 0, limit: int = 100,
) -> dict[str, Any]:
    """List directories and model weights without reading or loading weights."""
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset 必须是非负整数")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
        raise ValueError("limit 必须是 1..200 的整数")
    roots = _model_roots()
    if directory is None:
        path = next((root for root in roots if root.is_dir()), None)
    else:
        if not isinstance(directory, str) or not directory.strip():
            raise ValueError("directory 必须是非空目录路径")
        lexical_path = _resolve_path(directory, follow_symlinks=False)
        if _is_allowed(lexical_path, roots) and _is_hidden(lexical_path, roots):
            raise PermissionError("只能浏览服务器配置的模型目录")
        path = lexical_path.resolve()
    if path is not None and (not _is_allowed(path, roots) or _is_hidden(path, roots)):
        raise PermissionError("只能浏览服务器配置的模型目录")
    entries, truncated = [], False
    if path is not None:
        if directory is not None or path.exists():
            if not path.is_dir():
                raise FileNotFoundError("模型目录不存在")
            entries, truncated = _scan_entries(path, roots)
    end = min(offset + limit, len(entries))
    parent = path.parent if path is not None else None
    return {
        "ok": True,
        "roots": [{"name": root.name, "path": str(root)} for root in roots],
        "directory": str(path) if path is not None else "",
        "parent": str(parent) if parent is not None and _is_allowed(parent, roots) and not _is_hidden(parent, roots) else None,
        "entries": entries[offset:end],
        "offset": offset,
        "next_offset": end if end < len(entries) else None,
        "total": len(entries),
        "truncated": truncated,
    }
