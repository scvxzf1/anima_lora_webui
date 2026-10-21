"""Restore points for caption sidecars written by the tagging workbench.

Every caption write-back first stores the current sidecar bytes of the dataset
scope, so a bad batch can be reverted without touching a single image.  The
module is deliberately standalone: it only needs a dataset preset row and the
captioning runtime directory.

Behavior contract: `docs/findings/tagging-port-spec.md`.  This is an
independent implementation for this repository's dataset-preset architecture;
no third-party source, prompt text or test fixture is reproduced here.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from library.datasets.image_utils import glob_images
from web.services.atomic_io import atomic_write_bytes, atomic_write_text

from . import settings as tagging_settings
from .storage import resolve_caption_directory

SNAPSHOT_DIRNAME = "snapshots"
SCHEMA_VERSION = 1
CAPTION_SUFFIXES = (".txt", ".caption", ".json")
AGGREGATE_CAPTION_NAME = "captions.json"
ENCODING_UTF8 = "utf-8"
ENCODING_BASE64 = "base64"
MAX_SNAPSHOT_FILES = 20_000
MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_RETAINED = 20
REASON_MANUAL = "manual"
REASON_COMMIT = "commit"
_SNAPSHOT_ID_PATTERN = re.compile(r"^[0-9A-Za-z_-]{8,64}$")


class CaptionSnapshotError(ValueError):
    """A caption restore point could not be created, read or applied."""


class CaptionSnapshotNotFound(CaptionSnapshotError):
    """The requested restore point does not exist."""


class CaptionSnapshotTooLarge(CaptionSnapshotError):
    """The dataset scope is larger than one restore point may hold."""


def snapshot_root() -> Path:
    """Return the directory that stores snapshot documents."""

    return Path(tagging_settings.SETTINGS_FILE).parent / SNAPSHOT_DIRNAME


def create_snapshot(
    dataset_file: str,
    dataset_index: int,
    *,
    source: str = "source",
    reason: str = REASON_MANUAL,
    label: str = "",
    max_retained: int = DEFAULT_MAX_RETAINED,
) -> dict[str, Any]:
    """Capture the current caption sidecars of one dataset scope.

    Scope rule (spec 2.2): only caption sidecars at the top level of the
    dataset image directory are captured.  Datasets that recurse into
    subdirectories can therefore write sidecars outside a restore point;
    extending the snapshot scope to nested directories needs its own design
    review (performance and restore-scope rules), see docs/findings/
    tagging-port-spec.md 2.2.
    """

    directory = resolve_caption_directory(dataset_file, dataset_index, source=source)
    document_id = _new_snapshot_id()
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for path in _caption_files(directory):
        payload = path.read_bytes()
        total_bytes += len(payload)
        if len(entries) + 1 > MAX_SNAPSHOT_FILES or total_bytes > MAX_SNAPSHOT_BYTES:
            raise CaptionSnapshotTooLarge(
                f"还原点范围过大（超过 {MAX_SNAPSHOT_FILES} 个文件或 "
                f"{MAX_SNAPSHOT_BYTES // (1024 * 1024)} MiB），已跳过自动备份"
            )
        entries.append(_encode_entry(path.name, payload))
    manifest = {
        "schema": SCHEMA_VERSION,
        "id": document_id,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "created_at_ms": int(time.time() * 1000),
        "reason": str(reason or REASON_MANUAL),
        "label": str(label or ""),
        "dataset_file": str(dataset_file or ""),
        "dataset_index": int(dataset_index),
        "source": str(source or "source"),
        "scope_dir": directory.resolve().as_posix(),
        "totals": {"files": len(entries), "bytes": total_bytes},
        "entries": entries,
    }
    document = _document_path(document_id)
    document.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(document, json.dumps(manifest, ensure_ascii=False, indent=2))
    _prune(int(dataset_index), str(dataset_file or ""), max_retained)
    return _summary(manifest)


def list_snapshots(*, dataset_file: str = "", dataset_index: int | None = None) -> list[dict[str, Any]]:
    """List restore points newest first, tolerating damaged documents."""

    root = snapshot_root()
    if not root.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        if not _SNAPSHOT_ID_PATTERN.match(path.stem):
            continue
        manifest = _read_manifest(path)
        if manifest is None:
            items.append(_broken_entry(path))
            continue
        items.append(_summary(manifest))
    filtered = [
        item
        for item in items
        if not dataset_file
        or (item.get("dataset_file") == dataset_file and (dataset_index is None or item.get("dataset_index") == dataset_index))
    ]
    filtered.sort(key=lambda item: int(item.get("created_at_ms") or 0), reverse=True)
    return filtered


def read_snapshot(snapshot_id: str) -> dict[str, Any]:
    """Return the full manifest of one restore point, payload included."""

    path = _document_path(snapshot_id)
    manifest = _read_manifest(path)
    if manifest is None:
        if not path.is_file():
            raise CaptionSnapshotNotFound(f"找不到还原点：{snapshot_id}")
        raise CaptionSnapshotError(f"还原点文件已损坏：{snapshot_id}")
    if manifest.get("schema") != SCHEMA_VERSION:
        raise CaptionSnapshotError(f"还原点版本不受支持：{snapshot_id}")
    return manifest


def restore_snapshot(snapshot_id: str) -> dict[str, Any]:
    """Put the scope back to the captured state, and only that scope.

    The manifest is validated as a whole before the first file is touched: a
    damaged document must never delete captions that it cannot restore.  A
    restore plan is built first and only a fully valid plan mutates the disk.
    """

    manifest = read_snapshot(snapshot_id)
    directory = _scope_directory(manifest)
    plan = _restore_plan(manifest, directory)
    planned = {target.name for target, _payload in plan}
    current = {path.name: path for path in _caption_files(directory)}
    removed = 0
    for name, path in sorted(current.items()):
        if name in planned:
            continue
        try:
            path.unlink()
        except OSError:
            continue
        removed += 1
    written = 0
    for target, payload in plan:
        atomic_write_bytes(target, payload)
        written += 1
    return {
        "ok": True,
        "id": str(manifest.get("id") or snapshot_id),
        "written": written,
        "removed": removed,
        # Kept for response compatibility: invalid entries now abort the whole
        # restore instead of being skipped one by one.
        "skipped": [],
        "snapshot": _summary(manifest),
    }


def _scope_directory(manifest: dict[str, Any]) -> Path:
    """Return the captured scope, refusing anything but a live absolute directory."""

    raw = str(manifest.get("scope_dir") or "").strip()
    if not raw:
        raise CaptionSnapshotError("还原点缺少数据集目录，已取消还原")
    directory = Path(raw)
    if not directory.is_absolute():
        raise CaptionSnapshotError("还原点的数据集目录不是绝对路径，已取消还原")
    if not directory.is_dir():
        raise CaptionSnapshotError("数据集图片目录不存在，已取消还原")
    return directory


def _restore_plan(manifest: dict[str, Any], directory: Path) -> list[tuple[Path, bytes]]:
    """Validate every entry and return the writes to perform.

    Nothing is deleted or written until this function accepts the whole
    manifest, so one unusable entry aborts the restore with zero file changes.
    """

    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise CaptionSnapshotError("还原点数据损坏：entries 不是列表，已取消还原")
    plan: list[tuple[Path, bytes]] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise CaptionSnapshotError(f"还原点第 {index} 条记录不是对象，已取消还原")
        name = entry.get("name")
        if not isinstance(name, str) or not name or name in {".", ".."}:
            raise CaptionSnapshotError(f"还原点第 {index} 条记录缺少合法文件名，已取消还原")
        if Path(name).suffix.lower() not in CAPTION_SUFFIXES:
            raise CaptionSnapshotError(f"还原点第 {index} 条记录不是 caption 文件，已取消还原")
        if not _is_bare_caption_name(name):
            raise CaptionSnapshotError(f"还原点第 {index} 条记录文件名不合法，已取消还原")
        if entry.get("encoding") not in {ENCODING_UTF8, ENCODING_BASE64}:
            raise CaptionSnapshotError(f"还原点第 {index} 条记录编码不受支持，已取消还原")
        payload = _decode_entry(entry)
        if payload is None:
            raise CaptionSnapshotError(f"还原点第 {index} 条记录内容无法解码，已取消还原")
        target = _child_of(directory, name)
        if target is None:
            raise CaptionSnapshotError(f"还原点第 {index} 条记录超出数据集目录，已取消还原")
        if name in seen:
            raise CaptionSnapshotError(f"还原点包含重复文件名：{name}")
        seen.add(name)
        plan.append((target, payload))
    return plan


def delete_snapshot(snapshot_id: str) -> dict[str, Any]:
    """Remove one restore point."""

    path = _document_path(snapshot_id)
    if not path.is_file():
        raise CaptionSnapshotNotFound(f"找不到还原点：{snapshot_id}")
    path.unlink()
    return {"ok": True, "deleted": path.stem}


def _caption_files(directory: Path) -> list[Path]:
    """Return caption sidecars that pair with an image in this directory.

    Deliberately top level only (spec 2.2): nested caption sidecars of a
    recursive dataset stay outside the restore point, and restoring never
    touches them.  Expanding to nested directories is a separate decision.
    """

    if not directory.is_dir():
        return []
    stems = {Path(item).stem for item in glob_images(str(directory))}
    found: list[Path] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in CAPTION_SUFFIXES:
            continue
        # Suffix matching is case-insensitive, so the aggregate name compare must
        # be as well: CAPTIONS.JSON is the same caption file on a case-sensitive fs.
        if path.name.lower() == AGGREGATE_CAPTION_NAME or path.stem in stems:
            found.append(path)
    return found


def _encode_entry(name: str, payload: bytes) -> dict[str, Any]:
    digest = hashlib.sha256(payload).hexdigest()
    base = {"name": name, "size": len(payload), "sha256": digest}
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return {**base, "encoding": ENCODING_BASE64, "data": base64.b64encode(payload).decode("ascii")}
    return {**base, "encoding": ENCODING_UTF8, "data": text}


def _decode_entry(entry: Any) -> bytes | None:
    if not isinstance(entry, dict):
        return None
    data = entry.get("data")
    if entry.get("encoding") == ENCODING_BASE64:
        try:
            return base64.b64decode(str(data or ""), validate=True)
        except (binascii.Error, ValueError):
            return None
    if isinstance(data, str):
        return data.encode("utf-8")
    return None


def _is_bare_caption_name(name: str) -> bool:
    """Report whether name is a plain caption file name with no path part."""

    if not name or name in {".", ".."} or "\x00" in name:
        return False
    if "/" in name or "\\" in name or (os.sep and os.sep in name):
        return False
    try:
        if Path(name).name != name:
            return False
    except ValueError:
        return False
    return Path(name).suffix.lower() in CAPTION_SUFFIXES


def _child_of(directory: Path, name: str) -> Path | None:
    if not _is_bare_caption_name(name):
        return None
    try:
        target = (directory / name).resolve()
        parent = directory.resolve()
    except (OSError, ValueError):
        return None
    if target.parent != parent:
        return None
    return target


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return manifest if isinstance(manifest, dict) else None


def _broken_entry(path: Path) -> dict[str, Any]:
    try:
        created_at_ms = int(path.stat().st_mtime * 1000)
    except OSError:
        created_at_ms = 0
    return {"id": path.stem, "broken": True, "created_at_ms": created_at_ms, "totals": {"files": 0, "bytes": 0}}


def _summary(manifest: dict[str, Any]) -> dict[str, Any]:
    totals = manifest.get("totals") if isinstance(manifest.get("totals"), dict) else {}
    return {
        "id": str(manifest.get("id") or ""),
        "created_at": str(manifest.get("created_at") or ""),
        "created_at_ms": int(manifest.get("created_at_ms") or 0),
        "reason": str(manifest.get("reason") or ""),
        "label": str(manifest.get("label") or ""),
        "dataset_file": str(manifest.get("dataset_file") or ""),
        "dataset_index": int(manifest.get("dataset_index") or 0),
        "source": str(manifest.get("source") or ""),
        "scope_dir": str(manifest.get("scope_dir") or ""),
        "file_count": int(totals.get("files") or 0),
        "bytes": int(totals.get("bytes") or 0),
    }


def _document_path(snapshot_id: str) -> Path:
    raw = str(snapshot_id or "").strip()
    if not _SNAPSHOT_ID_PATTERN.match(raw):
        raise CaptionSnapshotError("还原点 ID 不合法")
    return snapshot_root() / f"{raw}.json"


def _new_snapshot_id() -> str:
    return f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"


def _prune(dataset_index: int, dataset_file: str, max_retained: int) -> None:
    if not str(dataset_file or "").strip():
        # Without a dataset scope the filter below matches every snapshot, so
        # pruning would delete other datasets.  Callers always pass a scope.
        return
    limit = max(1, int(max_retained))
    owned = list_snapshots(dataset_file=dataset_file, dataset_index=dataset_index)
    for item in owned[limit:]:
        try:
            _document_path(str(item.get("id") or "")).unlink()
        except (CaptionSnapshotError, OSError):
            continue
