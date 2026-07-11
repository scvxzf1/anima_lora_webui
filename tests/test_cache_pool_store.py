"""Tests for cache_pool store, refs, and mount."""

from __future__ import annotations

from pathlib import Path

from library.cache_pool.mount import mount_dir
from library.cache_pool.refs import acquire_ref, list_orphans, release_ref
from library.cache_pool.store import publish_pool_entry, read_manifest


def test_publish_and_ref_lifecycle(tmp_path: Path) -> None:
    pool = tmp_path / "cache_pool"
    staging = tmp_path / "staging"
    (staging / "resized").mkdir(parents=True)
    (staging / "lora").mkdir(parents=True)
    (staging / "resized" / "a.png").write_bytes(b"x")
    manifest = {"schema_version": "1", "fingerprint": "abc123", "mode": "light"}
    entry = publish_pool_entry(pool, "abc123", staging_dir=staging, manifest=manifest)
    assert entry.is_dir()
    assert (entry / "resized" / "a.png").is_file()
    assert read_manifest(entry)["fingerprint"] == "abc123"

    acquire_ref(entry, "run-1")
    acquire_ref(entry, "run-2")
    release_ref(entry, "run-1")
    assert list_orphans(pool) == []
    release_ref(entry, "run-2")
    orphans = list_orphans(pool)
    assert entry in orphans


def test_publish_idempotent_when_exists(tmp_path: Path) -> None:
    pool = tmp_path / "cache_pool"
    staging1 = tmp_path / "staging1"
    (staging1 / "resized").mkdir(parents=True)
    (staging1 / "lora").mkdir(parents=True)
    (staging1 / "resized" / "a.png").write_bytes(b"one")
    entry1 = publish_pool_entry(
        pool,
        "deadbeef",
        staging_dir=staging1,
        manifest={"schema_version": "1", "fingerprint": "deadbeef"},
    )
    staging2 = tmp_path / "staging2"
    (staging2 / "resized").mkdir(parents=True)
    (staging2 / "lora").mkdir(parents=True)
    (staging2 / "resized" / "a.png").write_bytes(b"two")
    entry2 = publish_pool_entry(
        pool,
        "deadbeef",
        staging_dir=staging2,
        manifest={"schema_version": "1", "fingerprint": "deadbeef"},
    )
    assert entry1 == entry2
    assert (entry1 / "resized" / "a.png").read_bytes() == b"one"


def test_mount_dir_fallback(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "f.bin").write_bytes(b"data")
    dst = tmp_path / "dst"
    mode = mount_dir(src, dst)
    assert mode in {"symlink", "hardlink", "copy"}
    assert (dst / "f.bin").read_bytes() == b"data"
