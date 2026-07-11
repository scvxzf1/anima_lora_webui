"""Per-pool-entry run_id reference tracking."""

from __future__ import annotations

import json
from pathlib import Path


def _refs_path(entry_dir: Path) -> Path:
    return Path(entry_dir) / "refs.json"


def _load(entry_dir: Path) -> list[str]:
    path = _refs_path(entry_dir)
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        refs = data.get("run_ids") or []
    else:
        refs = data or []
    return [str(x) for x in refs]


def _save(entry_dir: Path, run_ids: list[str]) -> None:
    path = _refs_path(entry_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    uniq = sorted(set(run_ids))
    path.write_text(
        json.dumps({"run_ids": uniq}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def acquire_ref(entry_dir: Path, run_id: str) -> None:
    run_id = str(run_id).strip()
    if not run_id:
        return
    refs = _load(entry_dir)
    if run_id not in refs:
        refs.append(run_id)
    _save(entry_dir, refs)


def release_ref(entry_dir: Path, run_id: str) -> None:
    run_id = str(run_id).strip()
    refs = [r for r in _load(entry_dir) if r != run_id]
    _save(entry_dir, refs)


def list_orphans(pool_root: Path) -> list[Path]:
    pool_root = Path(pool_root)
    if not pool_root.is_dir():
        return []
    out: list[Path] = []
    for child in pool_root.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        if not (child / "manifest.json").is_file():
            continue
        if not _load(child):
            out.append(child)
    return out
