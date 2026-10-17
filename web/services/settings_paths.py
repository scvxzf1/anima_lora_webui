"""Read-only path inspection and bounded directory browsing for settings UI."""

from __future__ import annotations

import os
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any

from library.env import anima_home, get_configs_root, get_training_history_root, get_training_queue_root
from web.services import model_file_service, settings_service

MAX_DIRECTORY_ENTRIES = 5000
MAX_PAGE_SIZE = 200


def _resolve_input(value: str, *, follow_symlinks: bool) -> Path:
    """Resolve a user path using the same lexical restrictions as model files."""
    expanded = os.path.expandvars(os.path.expanduser(value))
    if PureWindowsPath(expanded).drive and len(PureWindowsPath(expanded).parts) == 1:
        raise ValueError("不能浏览盘根目录")
    path = Path(expanded)
    if ".." in path.parts or ".." in PureWindowsPath(expanded).parts:
        raise ValueError("路径不能包含 ..")
    if not path.is_absolute():
        path = anima_home() / path
    return path.resolve() if follow_symlinks else path.absolute()


def _effective_roots() -> list[Path]:
    """Return project root and configured storage roots, without adding arbitrary parents."""
    candidates: list[Path] = [anima_home().resolve()]
    resolvers = (
        get_configs_root,
        get_training_history_root,
        get_training_queue_root,
        settings_service.resolve_output_root,
    )
    for resolver in resolvers:
        try:
            candidates.append(Path(resolver()).expanduser().resolve())
        except (OSError, RuntimeError, ValueError):
            continue
    try:
        image_test = settings_service.resolve_image_test_save_root()
        candidates.append(_resolve_input(str(image_test), follow_symlinks=True))
    except (OSError, RuntimeError, ValueError):
        pass
    roots: list[Path] = []
    for path in candidates:
        if path == Path(path.anchor):
            continue
        if any(path == existing for existing in roots):
            continue
        roots.append(path)
    return roots


def _is_allowed(path: Path, roots: list[Path]) -> bool:
    return any(path == root or path.is_relative_to(root) for root in roots)


def _is_hidden(path: Path, roots: list[Path]) -> bool:
    for root in roots:
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if any(part.startswith(".") for part in rel.parts):
            return True
    return False


def _root_payload(roots: list[Path]) -> list[dict[str, str]]:
    return [{"name": root.name, "path": str(root)} for root in roots]


def _validate_page(offset: int, limit: int) -> None:
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset 必须是非负整数")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_PAGE_SIZE:
        raise ValueError(f"limit 必须是 1..{MAX_PAGE_SIZE} 的整数")


def check_model_path(path: str | None) -> dict[str, Any]:
    """Inspect a model path without loading it; access is limited to model roots."""
    if path is None or not str(path).strip():
        return {"path": "", "kind": "empty"}
    value = str(path)
    lexical = model_file_service._resolve_path(value, follow_symlinks=False)
    roots = model_file_service._model_roots()
    if not model_file_service._is_allowed(lexical, roots) or model_file_service._is_hidden(lexical, roots):
        raise PermissionError("只能检查服务器配置的模型路径")
    resolved = lexical.resolve()
    if not model_file_service._is_allowed(resolved, roots) or model_file_service._is_hidden(resolved, roots):
        raise PermissionError("只能检查服务器配置的模型路径")
    result: dict[str, Any] = {"path": str(resolved), "kind": "missing"}
    try:
        if resolved.is_file():
            result["kind"] = "file"
            result["size_bytes"] = resolved.stat().st_size
        elif resolved.is_dir():
            result["kind"] = "directory"
    except OSError:
        raise
    return result


def list_directories(
    directory: str | None = None, *, offset: int = 0, limit: int = 100,
) -> dict[str, Any]:
    """List visible subdirectories under project/configured roots with bounded pagination."""
    _validate_page(offset, limit)
    roots = _effective_roots()
    if directory is None:
        path = roots[0] if roots else None
    else:
        if not isinstance(directory, str) or not directory.strip():
            raise ValueError("directory 必须是非空目录路径")
        lexical = _resolve_input(directory, follow_symlinks=False)
        if not _is_allowed(lexical, roots) or _is_hidden(lexical, roots):
            raise PermissionError("只能浏览服务器配置的目录")
        path = lexical.resolve()
        if not _is_allowed(path, roots) or _is_hidden(path, roots):
            raise PermissionError("只能浏览服务器配置的目录")
    if path is None or not path.exists():
        if directory is None:
            entries: list[dict[str, str]] = []
            return {
                "ok": True, "roots": _root_payload(roots), "directory": "", "parent": None,
                "entries": entries, "offset": offset, "next_offset": None,
                "total": 0, "truncated": False,
            }
        raise FileNotFoundError("目录不存在")
    if not path.is_dir():
        raise NotADirectoryError("不是目录")

    entries: list[dict[str, str]] = []
    truncated = False
    with os.scandir(path) as children:
        for index, child in enumerate(children):
            if index >= MAX_DIRECTORY_ENTRIES:
                truncated = True
                break
            if child.name.startswith("."):
                continue
            try:
                child_path = Path(child.path).resolve()
                if not child_path.is_dir() or not _is_allowed(child_path, roots) or _is_hidden(child_path, roots):
                    continue
            except (OSError, RuntimeError):
                continue
            entries.append({"name": child.name, "path": str(child_path), "kind": "directory"})
    entries.sort(key=lambda item: (item["name"].casefold(), item["name"]))
    end = min(offset + limit, len(entries))
    parent = path.parent
    if not _is_allowed(parent, roots) or _is_hidden(parent, roots):
        parent_value = None
    else:
        parent_value = str(parent)
    return {
        "ok": True,
        "roots": _root_payload(roots),
        "directory": str(path),
        "parent": parent_value,
        "entries": entries[offset:end],
        "offset": offset,
        "next_offset": end if end < len(entries) else None,
        "total": len(entries),
        "truncated": truncated,
    }
