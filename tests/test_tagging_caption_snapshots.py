from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from web.routes.tagging import setup_tagging_routes
from web.services.tagging import caption_snapshots, jobs, settings, storage

DATASET_FILE = "configs/datasets/example.toml"


def _patch_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    images: tuple[str, ...] = ("a.png", "b.png"),
    captions: dict[str, str] | None = None,
) -> Path:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for name in images:
        (dataset_dir / name).write_bytes(b"png")
    for name, text in (captions or {}).items():
        (dataset_dir / name).write_text(text, encoding="utf-8")
    monkeypatch.setattr(
        storage,
        "load_dataset_preset",
        lambda _file: {"datasets": [{"source_dir": dataset_dir.as_posix()}], "defaults": {}},
    )
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "configs" / "captioning" / "settings.toml")
    return dataset_dir


def _job(dataset_dir: Path, images: list[str]) -> dict:
    return {
        "id": "job-1",
        "state": "completed",
        "created_at": 1.0,
        "started_at": 1.0,
        "finished_at": 2.0,
        "dataset_file": DATASET_FILE,
        "dataset_index": 0,
        "source": "source",
        "prompt": "prompt",
        "system_prompt": "system",
        "total": len(images),
        "completed": len(images),
        "failed": 0,
        "canceled": 0,
        "items": [
            {
                "id": f"item-{index}",
                "file": (dataset_dir / name).as_posix(),
                "name": name,
                "state": "ready",
                "caption": "",
                "proposed_caption": "proposed caption",
                "error": "",
                "commit_error": "",
            }
            for index, name in enumerate(images)
        ],
        "error": "",
        "settings": {"model": "vision"},
    }


def test_restore_returns_scope_to_the_snapshot_state(tmp_path, monkeypatch) -> None:
    dataset_dir = _patch_scope(tmp_path, monkeypatch, captions={"a.txt": "1girl\n", "b.txt": "solo\n"})

    snapshot = caption_snapshots.create_snapshot(DATASET_FILE, 0)
    assert snapshot["file_count"] == 2
    assert snapshot["reason"] == "manual"

    (dataset_dir / "a.txt").write_text("rewritten\n", encoding="utf-8")
    (dataset_dir / "c.png").write_bytes(b"png")
    (dataset_dir / "c.txt").write_text("added later\n", encoding="utf-8")

    result = caption_snapshots.restore_snapshot(snapshot["id"])

    assert result["written"] == 2
    assert result["removed"] == 1
    assert result["skipped"] == []
    assert (dataset_dir / "a.txt").read_text(encoding="utf-8") == "1girl\n"
    assert (dataset_dir / "b.txt").read_text(encoding="utf-8") == "solo\n"
    assert not (dataset_dir / "c.txt").exists()
    assert (dataset_dir / "c.png").exists()


def test_scope_only_contains_caption_sidecars_paired_with_images(tmp_path, monkeypatch) -> None:
    dataset_dir = _patch_scope(tmp_path, monkeypatch, captions={"a.txt": "keep\n"})
    (dataset_dir / "notes.txt").write_text("readme\n", encoding="utf-8")
    (dataset_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (dataset_dir / "captions.json").write_text('{"a.png": ["keep"]}\n', encoding="utf-8")

    snapshot = caption_snapshots.create_snapshot(DATASET_FILE, 0)
    assert snapshot["file_count"] == 2

    (dataset_dir / "a.txt").write_text("changed\n", encoding="utf-8")
    (dataset_dir / "notes.txt").write_text("still mine\n", encoding="utf-8")
    caption_snapshots.restore_snapshot(snapshot["id"])

    assert (dataset_dir / "a.txt").read_text(encoding="utf-8") == "keep\n"
    assert (dataset_dir / "notes.txt").read_text(encoding="utf-8") == "still mine\n"
    assert (dataset_dir / "config.json").read_text(encoding="utf-8") == "{}\n"


def test_snapshot_scope_stays_top_level_for_recursive_dataset_layouts(tmp_path, monkeypatch) -> None:
    # Documented limitation (spec 2.2 / t9 review): a recursive dataset row can
    # hold images in subdirectories, but a restore point only covers top-level
    # caption sidecars, and restoring never touches the nested ones.  Widening
    # the scope to nested directories needs its own review of performance and
    # restore-scope rules, so this test pins today's behaviour on purpose.
    dataset_dir = _patch_scope(tmp_path, monkeypatch, captions={"a.txt": "a-old\n"})
    (dataset_dir / "sub").mkdir()
    (dataset_dir / "sub" / "img.png").write_bytes(b"png")
    (dataset_dir / "sub" / "img.txt").write_text("nested-old\n", encoding="utf-8")

    snapshot = caption_snapshots.create_snapshot(DATASET_FILE, 0)

    assert snapshot["file_count"] == 1
    manifest = caption_snapshots.read_snapshot(snapshot["id"])
    assert [entry["name"] for entry in manifest["entries"]] == ["a.txt"]

    (dataset_dir / "a.txt").write_text("a-reviewed\n", encoding="utf-8")
    (dataset_dir / "sub" / "img.txt").write_text("nested-reviewed\n", encoding="utf-8")
    caption_snapshots.restore_snapshot(snapshot["id"])

    assert (dataset_dir / "a.txt").read_text(encoding="utf-8") == "a-old\n"
    assert (dataset_dir / "sub" / "img.txt").read_text(encoding="utf-8") == "nested-reviewed\n"


def test_snapshot_keeps_non_utf8_payloads_byte_exact(tmp_path, monkeypatch) -> None:
    dataset_dir = _patch_scope(tmp_path, monkeypatch)
    payload = b"legacy\xff\xfe caption"
    (dataset_dir / "a.txt").write_bytes(payload)

    snapshot = caption_snapshots.create_snapshot(DATASET_FILE, 0)
    (dataset_dir / "a.txt").write_text("replaced\n", encoding="utf-8")
    caption_snapshots.restore_snapshot(snapshot["id"])

    assert (dataset_dir / "a.txt").read_bytes() == payload


@pytest.mark.parametrize("snapshot_id", ["", "short", "../escape", "a/b", "x" * 80, None])
def test_invalid_snapshot_ids_are_rejected(snapshot_id) -> None:
    with pytest.raises(caption_snapshots.CaptionSnapshotError):
        caption_snapshots.read_snapshot(snapshot_id)


def test_missing_snapshot_is_reported_as_not_found(tmp_path, monkeypatch) -> None:
    _patch_scope(tmp_path, monkeypatch)
    with pytest.raises(caption_snapshots.CaptionSnapshotNotFound):
        caption_snapshots.restore_snapshot("20260101-000000-abcdef")


def test_oversized_scope_is_rejected_without_writing_a_document(tmp_path, monkeypatch) -> None:
    _patch_scope(tmp_path, monkeypatch, captions={"a.txt": "x", "b.txt": "y"})
    monkeypatch.setattr(caption_snapshots, "MAX_SNAPSHOT_FILES", 1)

    with pytest.raises(caption_snapshots.CaptionSnapshotTooLarge):
        caption_snapshots.create_snapshot(DATASET_FILE, 0)

    assert not caption_snapshots.list_snapshots()


def test_retention_keeps_the_newest_snapshots_of_one_scope(tmp_path, monkeypatch) -> None:
    _patch_scope(tmp_path, monkeypatch, captions={"a.txt": "x"})
    created = []
    for _ in range(3):
        created.append(caption_snapshots.create_snapshot(DATASET_FILE, 0, max_retained=2)["id"])
        time.sleep(0.02)

    remaining = [item["id"] for item in caption_snapshots.list_snapshots()]

    assert remaining == [created[2], created[1]]


def test_damaged_documents_are_listed_without_breaking_the_surface(tmp_path, monkeypatch) -> None:
    _patch_scope(tmp_path, monkeypatch)
    root = caption_snapshots.snapshot_root()
    root.mkdir(parents=True, exist_ok=True)
    (root / "20260101-000000-abcdef.json").write_text("{not json", encoding="utf-8")

    items = caption_snapshots.list_snapshots()

    assert items and items[0]["broken"] is True
    with pytest.raises(caption_snapshots.CaptionSnapshotError):
        caption_snapshots.read_snapshot("20260101-000000-abcdef")


def test_delete_removes_one_document_and_reports_missing_ids(tmp_path, monkeypatch) -> None:
    _patch_scope(tmp_path, monkeypatch, captions={"a.txt": "x"})
    snapshot = caption_snapshots.create_snapshot(DATASET_FILE, 0)

    assert caption_snapshots.delete_snapshot(snapshot["id"])["deleted"] == snapshot["id"]
    assert caption_snapshots.list_snapshots() == []
    with pytest.raises(caption_snapshots.CaptionSnapshotNotFound):
        caption_snapshots.delete_snapshot(snapshot["id"])


def _patch_manifest(snapshot_id: str, mutate) -> None:
    document = caption_snapshots._document_path(snapshot_id)
    manifest = json.loads(document.read_text(encoding="utf-8"))
    mutate(manifest)
    document.write_text(json.dumps(manifest), encoding="utf-8")


def _scope_with_post_snapshot_caption(tmp_path, monkeypatch):
    dataset_dir = _patch_scope(tmp_path, monkeypatch, captions={"a.txt": "keep-a\n", "b.txt": "keep-b\n"})
    snapshot = caption_snapshots.create_snapshot(DATASET_FILE, 0)
    (dataset_dir / "c.png").write_bytes(b"png")
    (dataset_dir / "c.txt").write_text("added later\n", encoding="utf-8")
    return dataset_dir, snapshot


def _assert_restore_refused(dataset_dir, snapshot_id: str, match: str) -> None:
    with pytest.raises(caption_snapshots.CaptionSnapshotError, match=match):
        caption_snapshots.restore_snapshot(snapshot_id)
    # A refused restore must not delete the post-snapshot caption,
    # nor rewrite the captured ones.
    assert (dataset_dir / "c.txt").read_text(encoding="utf-8") == "added later\n"
    assert (dataset_dir / "a.txt").read_text(encoding="utf-8") == "keep-a\n"
    assert (dataset_dir / "b.txt").read_text(encoding="utf-8") == "keep-b\n"


DAMAGED_MANIFESTS = [
    pytest.param(lambda manifest: manifest.update(entries={"a.txt": "caption-a"}), "entries", id="entries-not-a-list"),
    pytest.param(lambda manifest: manifest.update(entries=["not-a-mapping"]), "第 1 条", id="entry-not-an-object"),
    pytest.param(lambda manifest: manifest["entries"][0].update(name="a\x00.txt"), "文件名不合法", id="nul-in-name"),
    pytest.param(lambda manifest: manifest["entries"][0].update(name="../escape.txt"), "文件名不合法", id="path-separator"),
    pytest.param(lambda manifest: manifest["entries"][0].update(name="a.exe"), "不是 caption", id="illegal-suffix"),
    pytest.param(lambda manifest: manifest["entries"][0].update(encoding="utf-16"), "编码不受支持", id="unknown-encoding"),
    pytest.param(
        lambda manifest: manifest["entries"][0].update(encoding="base64", data="!!not-base64!!"),
        "无法解码",
        id="undecodable-payload",
    ),
    pytest.param(lambda manifest: manifest.update(scope_dir=""), "缺少数据集目录", id="empty-scope-dir"),
    pytest.param(lambda manifest: manifest.update(scope_dir="relative/dataset"), "不是绝对路径", id="relative-scope-dir"),
    pytest.param(lambda manifest: manifest.update(scope_dir="/nonexistent-caption-scope"), "不存在", id="missing-scope-dir"),
]


@pytest.mark.parametrize("mutate, match", DAMAGED_MANIFESTS)
def test_damaged_manifest_aborts_the_restore_before_deleting_anything(tmp_path, monkeypatch, mutate, match) -> None:
    dataset_dir, snapshot = _scope_with_post_snapshot_caption(tmp_path, monkeypatch)
    _patch_manifest(snapshot["id"], mutate)

    _assert_restore_refused(dataset_dir, snapshot["id"], match)


def test_uppercase_aggregate_caption_file_is_inside_the_scope(tmp_path, monkeypatch) -> None:
    dataset_dir = _patch_scope(tmp_path, monkeypatch)
    (dataset_dir / "CAPTIONS.JSON").write_text('{"a.png": ["keep"]}\n', encoding="utf-8")

    snapshot = caption_snapshots.create_snapshot(DATASET_FILE, 0)
    assert snapshot["file_count"] == 1

    (dataset_dir / "CAPTIONS.JSON").write_text("{}", encoding="utf-8")
    caption_snapshots.restore_snapshot(snapshot["id"])

    assert "keep" in (dataset_dir / "CAPTIONS.JSON").read_text(encoding="utf-8")


def test_commit_aborts_when_the_restore_point_reports_a_real_fault(tmp_path, monkeypatch) -> None:
    dataset_dir = _patch_scope(tmp_path, monkeypatch, captions={"a.txt": "previous\n"})
    written = []
    monkeypatch.setattr(
        jobs,
        "write_caption",
        lambda *_args, **_kwargs: written.append("write") or {"text": "new", "caption_file": "dataset/a.txt"},
    )
    monkeypatch.setattr(
        jobs,
        "create_caption_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            caption_snapshots.CaptionSnapshotError("还原点存储损坏")
        ),
    )

    async def run() -> None:
        manager = jobs.TaggingJobManager()
        manager.jobs["job-1"] = _job(dataset_dir, ["a.png"])
        await manager.commit("job-1", all_items=True)

    with pytest.raises(caption_snapshots.CaptionSnapshotError):
        asyncio.run(run())
    assert written == []
    assert (dataset_dir / "a.txt").read_text(encoding="utf-8") == "previous\n"


def test_commit_creates_a_restore_point_before_writing(tmp_path, monkeypatch) -> None:
    dataset_dir = _patch_scope(tmp_path, monkeypatch, captions={"a.txt": "previous\n"})

    def fake_write(*_args, **_kwargs):
        (dataset_dir / "a.txt").write_text("proposed caption\n", encoding="utf-8")
        return {"text": "proposed caption", "caption_file": "dataset/a.txt"}

    monkeypatch.setattr(jobs, "write_caption", fake_write)

    async def run() -> dict:
        manager = jobs.TaggingJobManager()
        manager.jobs["job-1"] = _job(dataset_dir, ["a.png"])
        result = await manager.commit("job-1", all_items=True)
        manager.jobs.clear()
        return result

    result = asyncio.run(run())

    assert result["written"] == 1
    assert result["snapshot"]["file_count"] == 1
    assert result["snapshot"]["reason"] == "commit"
    assert result["snapshot_error"] == ""
    assert (dataset_dir / "a.txt").read_text(encoding="utf-8") == "proposed caption\n"

    caption_snapshots.restore_snapshot(result["snapshot"]["id"])
    assert (dataset_dir / "a.txt").read_text(encoding="utf-8") == "previous\n"


def test_commit_still_writes_when_the_scope_exceeds_the_snapshot_budget(tmp_path, monkeypatch) -> None:
    dataset_dir = _patch_scope(tmp_path, monkeypatch, captions={"a.txt": "previous\n"})
    written = []
    monkeypatch.setattr(
        jobs,
        "write_caption",
        lambda *_args, **_kwargs: written.append("write") or {"text": "new", "caption_file": "dataset/a.txt"},
    )
    monkeypatch.setattr(
        jobs,
        "create_caption_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(caption_snapshots.CaptionSnapshotTooLarge("范围过大")),
    )

    async def run() -> dict:
        manager = jobs.TaggingJobManager()
        manager.jobs["job-1"] = _job(dataset_dir, ["a.png"])
        return await manager.commit("job-1", all_items=True)

    result = asyncio.run(run())

    assert written == ["write"]
    assert result["written"] == 1
    assert result["snapshot"] is None
    assert "范围过大" in result["snapshot_error"]


def test_commit_is_cancelled_when_the_restore_point_cannot_be_stored(tmp_path, monkeypatch) -> None:
    dataset_dir = _patch_scope(tmp_path, monkeypatch, captions={"a.txt": "previous\n"})
    written = []
    monkeypatch.setattr(
        jobs,
        "write_caption",
        lambda *_args, **_kwargs: written.append("write") or {"text": "new", "caption_file": "dataset/a.txt"},
    )
    monkeypatch.setattr(
        jobs,
        "create_caption_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk is on fire")),
    )

    async def run() -> None:
        manager = jobs.TaggingJobManager()
        manager.jobs["job-1"] = _job(dataset_dir, ["a.png"])
        await manager.commit("job-1", all_items=True)

    with pytest.raises(caption_snapshots.CaptionSnapshotError):
        asyncio.run(run())
    assert written == []


def test_commit_without_reviewed_candidates_does_not_create_a_snapshot(tmp_path, monkeypatch) -> None:
    dataset_dir = _patch_scope(tmp_path, monkeypatch, captions={"a.txt": "previous\n"})

    async def run() -> dict:
        manager = jobs.TaggingJobManager()
        manager.jobs["job-1"] = _job(dataset_dir, [])
        return await manager.commit("job-1", all_items=True)

    result = asyncio.run(run())

    assert result["written"] == 0
    assert result["snapshot"] is None
    assert caption_snapshots.list_snapshots() == []


class _SnapshotServiceStub:
    def list_caption_snapshots(self, *, dataset_file: str = "", dataset_index: int | None = None):
        return {"ok": True, "snapshots": [{"id": "s1", "dataset_file": dataset_file, "dataset_index": dataset_index}]}

    async def create_caption_snapshot(self, payload):
        return {"ok": True, "snapshot": {"id": "s2", "label": str(payload.get("label") or "")}}

    async def restore_caption_snapshot(self, snapshot_id: str):
        if snapshot_id == "missing":
            raise caption_snapshots.CaptionSnapshotNotFound("找不到还原点：missing")
        return {"ok": True, "id": snapshot_id, "written": 1, "removed": 0, "skipped": []}

    def delete_caption_snapshot(self, snapshot_id: str):
        return {"ok": True, "deleted": snapshot_id}


def test_caption_snapshot_routes_cover_list_create_restore_and_delete() -> None:
    async def run() -> None:
        app = web.Application()
        setup_tagging_routes(app)
        app["tagging_service"] = _SnapshotServiceStub()
        async with TestClient(TestServer(app)) as client:
            listing = await client.get("/api/captioning/caption-snapshots", params={"dataset_file": DATASET_FILE, "dataset_index": "0"})
            assert listing.status == 200
            assert (await listing.json())["snapshots"][0]["dataset_index"] == 0

            created = await client.post("/api/tagging/caption-snapshots", json={"label": "before batch"})
            assert created.status == 201
            assert (await created.json())["snapshot"]["label"] == "before batch"

            restored = await client.post("/api/captioning/caption-snapshots/s1/restore")
            assert restored.status == 200
            assert (await restored.json())["written"] == 1

            missing = await client.post("/api/captioning/caption-snapshots/missing/restore")
            assert missing.status == 404

            deleted = await client.delete("/api/tagging/caption-snapshots/s1")
            assert deleted.status == 200
            assert (await deleted.json())["deleted"] == "s1"

            bad_index = await client.get("/api/captioning/caption-snapshots", params={"dataset_index": "x"})
            assert bad_index.status == 400

    asyncio.run(run())
