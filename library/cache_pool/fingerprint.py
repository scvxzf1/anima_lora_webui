"""Light / content fingerprints for shared preprocess cache pool entries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from library.preprocess._dataset import walk_images

SCHEMA_VERSION = "1"
FingerprintMode = Literal["light", "content"]

# Keys that affect resized / VAE / TE outputs. Adapter method, lr, seed must not appear.
_PREPROCESS_KEYS = (
    "resolution",
    "drop_lowres_images",
    "min_pixels",
    "model_family",
    "enable_bucket",
    "min_bucket_reso",
    "max_bucket_reso",
    "bucket_reso_steps",
    "caption_extension",
    "keep_tokens",
)

_CAPTIONS_JSON = "captions.json"


def build_preprocess_signature(
    cfg: dict[str, Any], subset_settings: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Normalize preprocess-facing config into a stable signature dict.

    Training-only knobs (``network_module``, ``learning_rate``, seeds, …) are
    ignored so LoRA/LoKr variants share the same data pool entry.
    """
    merged: dict[str, Any] = dict(cfg)
    if subset_settings:
        merged = {**merged, **subset_settings}
    out: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    for key in _PREPROCESS_KEYS:
        if key in merged:
            out[key] = merged[key]
    return out


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stat_entry(path: Path, *, relpath: str, kind: str) -> dict[str, Any]:
    st = path.stat()
    return {
        "relpath": relpath,
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "kind": kind,
    }


def scan_input_inventory(
    source_dir: Path,
    *,
    recursive: bool,
    path_pattern: str | None,
    caption_mode: str | None,
) -> list[dict[str, Any]]:
    """Enumerate images + paired captions under ``source_dir``.

    Each item has ``relpath``, ``size``, ``mtime_ns``, ``kind``.
    ``kind`` is ``image``, ``caption``, or ``caption_index`` (captions.json).
    """
    source_dir = Path(source_dir).resolve()
    if not source_dir.is_dir():
        return []

    pattern = path_pattern if path_pattern and path_pattern != "*" else None
    try:
        image_paths = walk_images(source_dir, recursive=recursive, pattern=pattern)
    except ValueError:
        # Stem collision: still fingerprint what we can via non-assert walk.
        from library.datasets.image_utils import glob_images_pathlib
        from library.datasets.subsets import filter_paths_by_glob

        image_paths = glob_images_pathlib(source_dir, recursive)
        if pattern:
            keep = filter_paths_by_glob(
                [str(p) for p in image_paths], str(source_dir), pattern
            )
            image_paths = [p for p, k in zip(image_paths, keep) if k]

    items: list[dict[str, Any]] = []
    mode = (caption_mode or "txt").strip().lower()

    captions_json = source_dir / _CAPTIONS_JSON
    if captions_json.is_file() and mode in {"captions_json", "auto", "json"}:
        items.append(
            _stat_entry(
                captions_json,
                relpath=_CAPTIONS_JSON,
                kind="caption_index",
            )
        )

    for img in image_paths:
        rel = img.relative_to(source_dir).as_posix()
        items.append(_stat_entry(img, relpath=rel, kind="image"))
        if mode in {None, "", "txt", "sidecar", "auto"}:
            # Prefer .txt sidecar; caption_extension overrides could be added later.
            cap = img.with_suffix(".txt")
            if cap.is_file():
                items.append(
                    _stat_entry(
                        cap,
                        relpath=cap.relative_to(source_dir).as_posix(),
                        kind="caption",
                    )
                )

    items.sort(key=lambda x: (x["relpath"], x["kind"]))
    return items


def compute_fingerprint(
    *,
    mode: FingerprintMode,
    source_dir: Path,
    inventory: list[dict[str, Any]],
    preprocess_signature: dict[str, Any],
    normalized_source: str,
) -> str:
    """Return a short hex fingerprint (sha256 truncated to 16)."""
    if mode not in ("light", "content"):
        raise ValueError(f"unsupported fingerprint mode: {mode!r}")

    source_dir = Path(source_dir).resolve()
    payload_inventory: list[dict[str, Any]] = []
    for item in inventory:
        row: dict[str, Any] = {
            "relpath": item["relpath"],
            "size": item["size"],
            "mtime_ns": item["mtime_ns"],
            "kind": item.get("kind"),
        }
        if mode == "content":
            digest = item.get("digest")
            if not digest:
                p = source_dir / str(item["relpath"])
                digest = _file_digest(p) if p.is_file() else ""
            row["digest"] = digest
        payload_inventory.append(row)

    blob = {
        "schema_version": SCHEMA_VERSION,
        "mode": mode,
        "normalized_source": normalized_source,
        "preprocess_signature": preprocess_signature,
        "inventory": payload_inventory,
    }
    raw = json.dumps(
        blob, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]
