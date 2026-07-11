"""Runtime prepare should mount shared cache_pool across runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from tests.training_resume_test_support import (
    _patch_runtime_service_paths,
    _write_runtime_config_tree,
)
from web.services import training_service


def test_second_prepare_reuses_pool_without_full_private_copy(tmp_path, monkeypatch):
    _write_runtime_config_tree(tmp_path)
    _patch_runtime_service_paths(monkeypatch, tmp_path)

    src_a = tmp_path / "image_dataset" / "a"
    src_a.mkdir(parents=True, exist_ok=True)
    (src_a / "x.png").write_bytes(b"\x89PNG\r\nfake")
    (src_a / "x.txt").write_text("1girl", encoding="utf-8")
    src_b = tmp_path / "image_dataset" / "b"
    src_b.mkdir(parents=True, exist_ok=True)
    (src_b / "y.png").write_bytes(b"\x89PNG\r\nfake-b")
    (src_b / "y.txt").write_text("1boy", encoding="utf-8")

    class FixedDatetime:
        @classmethod
        def now(cls):
            return datetime(2026, 5, 23, 11, 45, 14)

        @classmethod
        def fromtimestamp(cls, value):
            return datetime.fromtimestamp(value)

    monkeypatch.setattr(training_service, "datetime", FixedDatetime)

    training_service._prepare_web_runtime_config(
        "522",
        "default",
        "imported",
        source_config_file="configs/imported/522.toml",
    )
    run1 = tmp_path / "output" / "runs" / "522-20260523-114514"
    meta1 = json.loads((run1 / "run.meta.json").read_text(encoding="utf-8"))
    bindings = meta1.get("dataset_cache_bindings") or []
    assert bindings, "prepare 应写入 dataset_cache_bindings"
    assert meta1.get("cache_pool_root")

    pool_path = Path(bindings[0]["pool_path"])
    if not pool_path.is_absolute():
        pool_path = tmp_path / pool_path
    sentinel = pool_path / "resized" / "sentinel.txt"
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text("shared", encoding="utf-8")

    class FixedDatetime2(FixedDatetime):
        @classmethod
        def now(cls):
            return datetime(2026, 5, 23, 12, 0, 0)

    monkeypatch.setattr(training_service, "datetime", FixedDatetime2)
    training_service._prepare_web_runtime_config(
        "522",
        "default",
        "imported",
        source_config_file="configs/imported/522.toml",
    )
    run2 = tmp_path / "output" / "runs" / "522-20260523-120000"
    resized2 = run2 / "dataset_cache" / "dataset-01" / "resized"
    assert (resized2 / "sentinel.txt").read_text(encoding="utf-8") == "shared"

    pool_root = tmp_path / "output" / "cache_pool"
    entries = [p for p in pool_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    assert len(entries) >= 1
