"""Shared JSON / JSONL storage helpers for WebUI training services."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from web.services.training.common import _positive_int_or_none

def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _read_json(path: Path) -> dict[str, Any]:
    data = _read_json_object(path)
    return data if isinstance(data, dict) else {}


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not _path_exists(path):
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        _fsync_parent_dir(path)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _fsync_parent_dir(path: Path) -> None:
    try:
        fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl_limited(path)[0]


def _read_jsonl_limited(path: Path, *, limit: int | None = None) -> tuple[list[dict[str, Any]], int, bool]:
    if not _path_exists(path):
        return [], 0, False
    out: list[dict[str, Any]] = []
    try:
        lines = [line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    except OSError:
        return [], 0, False
    total = len(lines)
    safe_limit = _positive_int_or_none(limit)
    truncated = bool(safe_limit and total > safe_limit)
    if safe_limit:
        lines = lines[-safe_limit:]
    for line in lines:
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                out.append(value)
        except Exception:
            continue
    return out, total, truncated


def _count_jsonl(path: Path) -> int:
    if not _path_exists(path):
        return 0
    try:
        return sum(1 for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip())
    except Exception:
        return 0


def _read_text_file(path: Path) -> str:
    if not _path_exists(path):
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
