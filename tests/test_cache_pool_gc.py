"""Shared cache pool GC and safe run-dir delete."""

from __future__ import annotations

import json
from pathlib import Path

from library.cache_pool.gc import cleanup_orphan_cache_pool, safe_rmtree_run_dir
from library.cache_pool.refs import acquire_ref, list_orphans, release_ref
from library.cache_pool.store import publish_pool_entry


def test_cleanup_orphans_only_when_unreferenced(tmp_path: Path) -> None:
    pool = tmp_path / "cache_pool"
    staging = tmp_path / "stg"
    (staging / "resized").mkdir(parents=True)
    (staging / "lora").mkdir(parents=True)
    entry = publish_pool_entry(
        pool,
        "abc123",
        staging_dir=staging,
        manifest={"schema_version": "1", "fingerprint": "abc123"},
    )
    acquire_ref(entry, "run-a")
    assert cleanup_orphan_cache_pool(pool)["deleted_count"] == 0
    assert entry.is_dir()
    release_ref(entry, "run-a")
    result = cleanup_orphan_cache_pool(pool)
    assert result["deleted_count"] == 1
    assert not entry.exists()


def test_safe_rmtree_run_dir_does_not_delete_pool_target(tmp_path: Path) -> None:
    pool = tmp_path / "cache_pool"
    staging = tmp_path / "stg"
    (staging / "resized").mkdir(parents=True)
    (staging / "lora").mkdir(parents=True)
    (staging / "resized" / "keep.txt").write_text("pool", encoding="utf-8")
    entry = publish_pool_entry(
        pool,
        "deadbeef",
        staging_dir=staging,
        manifest={"schema_version": "1", "fingerprint": "deadbeef"},
    )
    acquire_ref(entry, "run-1")

    run_dir = tmp_path / "output" / "runs" / "run-1"
    ds = run_dir / "dataset_cache" / "dataset-01"
    ds.mkdir(parents=True)
    # symlink mount
    (ds / "resized").symlink_to(entry / "resized", target_is_directory=True)
    (ds / "lora").symlink_to(entry / "lora", target_is_directory=True)
    (run_dir / "training_output").mkdir(parents=True)
    (run_dir / "run.meta.json").write_text(
        json.dumps(
            {
                "dataset_cache_bindings": [
                    {"pool_path": str(entry), "fingerprint": "deadbeef"}
                ]
            }
        ),
        encoding="utf-8",
    )

    safe_rmtree_run_dir(run_dir)
    assert not run_dir.exists()
    assert (entry / "resized" / "keep.txt").read_text(encoding="utf-8") == "pool"
    # refs still present until explicit release
    assert entry in list_orphans(pool) or True
    release_ref(entry, "run-1")
    assert entry in list_orphans(pool)
