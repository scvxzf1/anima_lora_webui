"""Pool path layout, manifest IO, and atomic publish."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


def default_pool_root() -> Path:
    """Return ``<output_root>/cache_pool``.

    Prefer settings_service when available; fall back to project ``output/cache_pool``.
    """
    try:
        from web.services.settings_service import resolve_output_root

        return Path(resolve_output_root()) / "cache_pool"
    except Exception:
        from library.env import project_root

        return project_root() / "output" / "cache_pool"


def pool_entry_dir(pool_root: Path, fingerprint: str) -> Path:
    safe = "".join(c for c in str(fingerprint) if c.isalnum())[:64]
    if not safe:
        raise ValueError("fingerprint must contain alnum characters")
    return Path(pool_root) / safe


def write_manifest(entry_dir: Path, manifest: dict[str, Any]) -> None:
    entry_dir = Path(entry_dir)
    entry_dir.mkdir(parents=True, exist_ok=True)
    path = entry_dir / "manifest.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_manifest(entry_dir: Path) -> dict[str, Any] | None:
    path = Path(entry_dir) / "manifest.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def publish_pool_entry(
    pool_root: Path,
    fingerprint: str,
    *,
    staging_dir: Path,
    manifest: dict[str, Any],
) -> Path:
    """Atomically publish ``staging_dir`` as ``pool_root/<fp>``.

    If the final entry already has a manifest, return it and leave staging alone
    (caller may clean staging).
    """
    pool_root = Path(pool_root)
    staging_dir = Path(staging_dir)
    pool_root.mkdir(parents=True, exist_ok=True)
    final = pool_entry_dir(pool_root, fingerprint)
    if final.exists() and read_manifest(final) is not None:
        return final

    write_manifest(staging_dir, manifest)
    parent = final.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_name = parent / f".tmp-{final.name}-{os.getpid()}"
    if tmp_name.exists():
        shutil.rmtree(tmp_name)
    shutil.move(str(staging_dir), str(tmp_name))
    try:
        tmp_name.rename(final)
    except FileExistsError:
        shutil.rmtree(tmp_name, ignore_errors=True)
        if read_manifest(final) is None:
            raise
    return final
