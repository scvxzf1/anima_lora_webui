"""Shared cache pool cleanup helpers."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from library.cache_pool.refs import list_orphans, release_ref
from library.cache_pool.store import default_pool_root, read_manifest


def release_refs_for_run_meta(
    meta: dict[str, Any],
    *,
    run_id: str,
    resolve_path=None,
) -> list[str]:
    """Release pool refs recorded in run.meta / history meta bindings."""
    released: list[str] = []
    bindings = meta.get("dataset_cache_bindings")
    if not isinstance(bindings, list):
        return released
    for binding in bindings:
        if not isinstance(binding, dict):
            continue
        pool_path = str(binding.get("pool_path") or "").strip()
        if not pool_path:
            continue
        entry = Path(pool_path)
        if not entry.is_absolute() and resolve_path is not None:
            resolved = resolve_path(pool_path)
            if resolved is not None:
                entry = Path(resolved)
        if not entry.is_dir():
            continue
        release_ref(entry, run_id)
        released.append(str(entry))
    return released


def unlink_dataset_cache_mounts(run_dir: Path) -> int:
    """Remove symlink mounts under run_dir/dataset_cache without following them.

    Returns number of symlink mounts removed.
    """
    root = Path(run_dir) / "dataset_cache"
    if not root.exists():
        return 0
    removed = 0
    for path in sorted(root.rglob("*"), reverse=True):
        try:
            if path.is_symlink():
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def safe_rmtree_run_dir(run_dir: Path) -> None:
    """Delete a WebUI run directory without following pool symlinks."""
    run_dir = Path(run_dir)
    unlink_dataset_cache_mounts(run_dir)
    if run_dir.exists() or run_dir.is_symlink():
        if run_dir.is_symlink():
            run_dir.unlink()
        else:
            shutil.rmtree(run_dir)


def cleanup_orphan_cache_pool(pool_root: Path | None = None) -> dict[str, Any]:
    """Delete pool entries with empty refs. Returns summary."""
    root = Path(pool_root) if pool_root is not None else default_pool_root()
    deleted: list[str] = []
    errors: dict[str, str] = {}
    for entry in list_orphans(root):
        # only delete real pool entries with manifest
        if read_manifest(entry) is None:
            continue
        try:
            shutil.rmtree(entry)
            deleted.append(str(entry))
        except OSError as exc:
            errors[str(entry)] = str(exc)
    return {
        "ok": True,
        "pool_root": str(root),
        "deleted": deleted,
        "errors": errors,
        "deleted_count": len(deleted),
    }
