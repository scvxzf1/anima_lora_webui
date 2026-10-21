from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from web.services.tagging import TaggingService, caption_edit, caption_snapshots, jobs, storage
from web.services.tagging import settings as tagging_settings

DATASET_FILE = "configs/datasets/example.toml"


@pytest.fixture()
def image(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "dataset" / "sample.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"image")
    resolved = {
        "path": path,
        "file": path.as_posix(),
        "name": path.name,
        "row": {},
        "settings": {"caption_extension": ".txt"},
        "caption": {"path": None, "text": ""},
    }
    monkeypatch.setattr(storage, "resolve_tagging_image", lambda *_args, **_kwargs: resolved)
    return path


def test_overwrite_replaces_plain_text_caption(image) -> None:
    image.with_suffix(".txt").write_text("old tags\n", encoding="utf-8")

    result = storage.write_caption(DATASET_FILE, 0, image.as_posix(), "new, tags")

    assert result["skipped"] is False
    assert result["format"] == "text"
    assert image.with_suffix(".txt").read_text(encoding="utf-8") == "new, tags\n"


def test_append_merges_existing_tags_first(image) -> None:
    image.with_suffix(".txt").write_text("1girl, solo\n", encoding="utf-8")

    result = storage.write_caption(DATASET_FILE, 0, image.as_posix(), "SOLO, smile", policy="append")

    assert result["text"] == "1girl, solo, smile"
    assert image.with_suffix(".txt").read_text(encoding="utf-8") == "1girl, solo, smile\n"


def test_skip_leaves_an_existing_caption_untouched(image) -> None:
    image.with_suffix(".txt").write_text("keep me\n", encoding="utf-8")

    result = storage.write_caption(DATASET_FILE, 0, image.as_posix(), "ignored", policy="skip")

    assert result["skipped"] is True
    assert result["text"] == "keep me"
    assert image.with_suffix(".txt").read_text(encoding="utf-8") == "keep me\n"


def test_skip_still_writes_when_no_effective_caption_exists(image) -> None:
    image.with_suffix(".txt").write_text("   \n", encoding="utf-8")

    result = storage.write_caption(DATASET_FILE, 0, image.as_posix(), "1girl", policy="skip")

    assert result["skipped"] is False
    assert image.with_suffix(".txt").read_text(encoding="utf-8") == "1girl\n"


def test_trigger_word_is_written_once_and_recorded_in_meta(image) -> None:
    image.with_suffix(".json").write_text('{"meta": {"source": "api"}}', encoding="utf-8")

    first = storage.write_caption(DATASET_FILE, 0, image.as_posix(), "1girl", trigger_word="MyStyle")
    second = storage.write_caption(
        DATASET_FILE, 0, image.as_posix(), "MyStyle, 1girl, solo", trigger_word="MyStyle"
    )
    payload = json.loads(image.with_suffix(".json").read_text(encoding="utf-8"))

    assert first["tags"] == ["MyStyle", "1girl"]
    assert second["tags"] == ["MyStyle", "1girl", "solo"]
    assert payload["meta"]["trigger"] == "MyStyle"


def _job(image: Path) -> dict:
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
        "total": 1,
        "completed": 1,
        "failed": 0,
        "canceled": 0,
        "items": [{
            "id": "item-1",
            "file": image.as_posix(),
            "name": image.name,
            "state": "ready",
            "caption": "",
            "proposed_caption": "proposed",
            "error": "",
            "commit_error": "",
        }],
        "error": "",
        "settings": {"model": "vision"},
    }


def test_commit_applies_the_job_policy_and_trigger(image) -> None:
    image.with_suffix(".txt").write_text("1girl\n", encoding="utf-8")

    async def run() -> dict:
        manager = jobs.TaggingJobManager()
        job = _job(image)
        job["on_existing"] = "append"
        job["trigger_word"] = "MyStyle"
        manager.jobs["job-1"] = job
        return await manager.commit("job-1", all_items=True)

    result = asyncio.run(run())

    assert result["written"] == 1
    assert image.with_suffix(".txt").read_text(encoding="utf-8") == "MyStyle, 1girl, proposed\n"


def test_commit_reports_skipped_items_without_counting_them_as_written(image) -> None:
    image.with_suffix(".txt").write_text("keep me\n", encoding="utf-8")

    async def run() -> dict:
        manager = jobs.TaggingJobManager()
        job = _job(image)
        job["on_existing"] = "skip"
        manager.jobs["job-1"] = job
        return await manager.commit("job-1", all_items=True)

    result = asyncio.run(run())

    assert result["written"] == 0
    assert result["skipped"] == 1
    assert image.with_suffix(".txt").read_text(encoding="utf-8") == "keep me\n"


def _dir_job(dataset_dir: Path, names: list[str], proposed: dict[str, str]) -> dict:
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
        "total": len(names),
        "completed": len(names),
        "failed": 0,
        "canceled": 0,
        "items": [
            {
                "id": f"item-{index}",
                "file": (dataset_dir / name).as_posix(),
                "name": name,
                "state": "ready",
                "caption": "",
                "proposed_caption": proposed.get(name, "proposed"),
                "error": "",
                "commit_error": "",
            }
            for index, name in enumerate(names)
        ],
        "error": "",
        "settings": {"model": "vision"},
    }


def test_write_back_freezes_candidates_and_rerun_cannot_clear_them(tmp_path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for name in ("a.png", "b.png"):
        (dataset_dir / name).write_bytes(b"image")
    (dataset_dir / "a.txt").write_text("old-a\n", encoding="utf-8")
    (dataset_dir / "b.txt").write_text("old-b\n", encoding="utf-8")
    started = threading.Event()
    release = threading.Event()
    written_texts: list[str] = []

    def slow_write(_file, _index, image_file, text, **_kwargs):
        written_texts.append(text)
        if len(written_texts) == 1:
            started.set()
            assert release.wait(10)
        Path(image_file).with_suffix(".txt").write_text(f"{text}\n", encoding="utf-8")
        return {"text": text, "caption_file": image_file}

    monkeypatch.setattr(jobs, "write_caption", slow_write)
    monkeypatch.setattr(
        jobs, "create_caption_snapshot", lambda *_args, **_kwargs: {"id": "snap-1", "file_count": 2}
    )
    job = _dir_job(dataset_dir, ["a.png", "b.png"], {"a.png": "reviewed-a", "b.png": "reviewed-b"})

    async def run() -> tuple[str, dict]:
        manager = jobs.TaggingJobManager()
        manager.jobs["job-1"] = job
        task = asyncio.create_task(manager.commit("job-1", all_items=True))
        assert await asyncio.to_thread(started.wait, 10)
        try:
            await manager.rerun("job-1")
            rerun_error = ""
        except RuntimeError as exc:
            rerun_error = str(exc)
        with pytest.raises(RuntimeError, match="写回或重跑"):
            manager.update_items("job-1", [{"id": "item-2", "proposed_caption": ""}])
        # Even an edit that bypasses the manager may not change the frozen text.
        job["items"][1]["proposed_caption"] = ""
        release.set()
        return rerun_error, await task

    rerun_error, result = asyncio.run(run())

    assert "写回" in rerun_error
    assert written_texts == ["reviewed-a", "reviewed-b"]
    assert result["written"] == 2
    assert result["failed"] == 0
    assert (dataset_dir / "b.txt").read_text(encoding="utf-8") == "reviewed-b\n"


def test_candidate_without_tags_never_clears_an_existing_caption(image) -> None:
    image.with_suffix(".txt").write_text("keep me\n", encoding="utf-8")
    job = _job(image)
    job["items"][0]["proposed_caption"] = " , "

    async def run() -> dict:
        manager = jobs.TaggingJobManager()
        manager.jobs["job-1"] = job
        return await manager.commit("job-1", all_items=True)

    result = asyncio.run(run())

    assert result["written"] == 0
    assert result["failed"] == 1
    assert result["skipped"] == 0
    assert "没有有效标签" in result["errors"][0]["error"]
    assert result["job"]["items"][0]["commit_status"] == "failed"
    assert image.with_suffix(".txt").read_text(encoding="utf-8") == "keep me\n"


def test_write_failures_are_not_reported_as_policy_skips(image, monkeypatch) -> None:
    image.with_suffix(".txt").write_text("keep\n", encoding="utf-8")
    monkeypatch.setattr(
        jobs, "create_caption_snapshot", lambda *_args, **_kwargs: {"id": "snap", "file_count": 1}
    )

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(jobs, "write_caption", boom)
    job = _job(image)

    async def run() -> dict:
        manager = jobs.TaggingJobManager()
        manager.jobs["job-1"] = job
        return await manager.commit("job-1", all_items=True)

    result = asyncio.run(run())

    assert result["failed"] == 1
    assert result["skipped"] == 0
    assert result["written"] == 0
    assert "disk full" in result["errors"][0]["error"]
    assert result["job"]["items"][0]["commit_status"] == "failed"
    assert image.with_suffix(".txt").read_text(encoding="utf-8") == "keep\n"


def test_configured_caption_extension_drives_detection_and_the_write_target(image, monkeypatch) -> None:
    resolved = {
        "path": image,
        "file": image.as_posix(),
        "name": image.name,
        "row": {},
        "settings": {"caption_extension": ".tags"},
        "caption": {"path": None, "text": ""},
    }
    monkeypatch.setattr(storage, "resolve_tagging_image", lambda *_args, **_kwargs: resolved)
    image.with_suffix(".tags").write_text("keep, mine\n", encoding="utf-8")

    skipped = storage.write_caption(DATASET_FILE, 0, image.as_posix(), "new", policy="skip")
    assert skipped["skipped"] is True
    assert image.with_suffix(".tags").read_text(encoding="utf-8") == "keep, mine\n"

    appended = storage.write_caption(DATASET_FILE, 0, image.as_posix(), "new", policy="append")
    assert appended["text"] == "keep, mine, new"
    assert image.with_suffix(".tags").read_text(encoding="utf-8") == "keep, mine, new\n"


def test_multi_tag_trigger_word_is_prepended_once(image) -> None:
    trigger = "masterpiece, best quality"
    first = storage.write_caption(DATASET_FILE, 0, image.as_posix(), "1girl", trigger_word=trigger)
    second = storage.write_caption(
        DATASET_FILE, 0, image.as_posix(), "masterpiece, best quality, 1girl", trigger_word=trigger
    )

    assert first["tags"] == ["masterpiece", "best quality", "1girl"]
    assert second["tags"] == ["masterpiece", "best quality", "1girl"]
    assert image.with_suffix(".txt").read_text(encoding="utf-8") == "masterpiece, best quality, 1girl\n"


def test_job_creation_skips_already_captioned_images(monkeypatch, tmp_path) -> None:
    captioned = tmp_path / "dataset" / "done.png"
    captioned.parent.mkdir(parents=True, exist_ok=True)
    captioned.write_bytes(b"image")
    captioned.with_suffix(".txt").write_text("1girl\n", encoding="utf-8")
    fresh = tmp_path / "dataset" / "fresh.png"
    fresh.write_bytes(b"image")

    def resolve(_file, _index, image_file, **_kwargs):
        path = Path(image_file)
        return {
            "path": path,
            "file": path.as_posix(),
            "name": path.name,
            "row": {},
            "settings": {"caption_extension": ".txt"},
            "caption": {"text": ""},
        }

    monkeypatch.setattr(jobs, "resolve_tagging_image", resolve)
    monkeypatch.setattr(
        jobs,
        "get_effective_settings",
        lambda *_args, **_kwargs: {"_profile_id": "p1", "_profile_name": "Fixture"},
    )

    async def run(payload: dict) -> dict:
        manager = jobs.TaggingJobManager()
        created = await manager.create(payload)
        await manager._tasks[created["job"]["id"]]
        return created["job"]

    async def run_all() -> None:
        created = await run({
            "dataset_file": DATASET_FILE,
            "dataset_index": 0,
            "items": [captioned.as_posix(), fresh.as_posix()],
            "on_existing": "skip",
        })
        assert created["on_existing"] == "skip"
        assert [item["name"] for item in created["items"]] == ["fresh.png"]
        manager = jobs.TaggingJobManager()
        with pytest.raises(ValueError, match="跳过"):
            await manager.create({
                "dataset_file": DATASET_FILE,
                "dataset_index": 0,
                "items": [captioned.as_posix()],
                "on_existing": "skip",
            })

    asyncio.run(run_all())


def test_job_prefilter_and_write_back_agree_on_a_custom_caption_extension(image, monkeypatch) -> None:
    image.with_suffix(".tags").write_text("keep, mine\n", encoding="utf-8")
    row_settings = {"caption_extension": ".tags"}
    resolved = {
        "path": image,
        "file": image.as_posix(),
        "name": image.name,
        "row": {},
        "settings": row_settings,
        "caption": {"text": "keep, mine"},
    }
    monkeypatch.setattr(storage, "resolve_tagging_image", lambda *_args, **_kwargs: resolved)

    def resolve(_file, _index, image_file, **_kwargs):
        path = Path(image_file)
        return {**resolved, "path": path, "file": path.as_posix(), "name": path.name}

    monkeypatch.setattr(jobs, "resolve_tagging_image", resolve)
    monkeypatch.setattr(
        jobs,
        "get_effective_settings",
        lambda *_args, **_kwargs: {"_profile_id": "p1", "_profile_name": "Fixture"},
    )

    async def run() -> None:
        manager = jobs.TaggingJobManager()
        with pytest.raises(ValueError, match="跳过"):
            await manager.create(
                {
                    "dataset_file": DATASET_FILE,
                    "dataset_index": 0,
                    "items": [image.as_posix()],
                    "on_existing": "skip",
                }
            )

    asyncio.run(run())
    skipped = storage.write_caption(DATASET_FILE, 0, image.as_posix(), "new", policy="skip")
    assert skipped["skipped"] is True
    assert image.with_suffix(".tags").read_text(encoding="utf-8") == "keep, mine\n"
    assert not image.with_suffix(".txt").exists()


def _aggregate_resolved(image: Path, settings: dict[str, str]) -> dict:
    return {
        "path": image,
        "file": image.as_posix(),
        "name": image.name,
        "row": {},
        "settings": settings,
        "caption": {"text": "existing, tags"},
    }


def test_captions_json_row_refuses_sidecar_writes_before_touching_disk(image, monkeypatch) -> None:
    aggregate = image.parent / "captions.json"
    aggregate.write_text(json.dumps({image.name: ["existing, tags"]}), encoding="utf-8")
    before = aggregate.read_bytes()
    resolved = _aggregate_resolved(
        image, {"caption_extension": ".txt", "caption_source_mode": "captions_json"}
    )
    monkeypatch.setattr(storage, "resolve_tagging_image", lambda *_args, **_kwargs: resolved)

    with pytest.raises(caption_edit.AggregateCaptionWriteError, match="captions.json"):
        storage.write_caption(DATASET_FILE, 0, image.as_posix(), "reviewed, tags")

    assert not image.with_suffix(".txt").exists()
    assert not image.with_suffix(".json").exists()
    assert aggregate.read_bytes() == before


def test_auto_mode_with_an_aggregate_entry_also_fails_closed(image, monkeypatch) -> None:
    aggregate = image.parent / "captions.json"
    aggregate.write_text(json.dumps({image.name: ["existing, tags"]}), encoding="utf-8")
    before = aggregate.read_bytes()
    monkeypatch.setattr(
        storage,
        "resolve_tagging_image",
        lambda *_args, **_kwargs: _aggregate_resolved(image, {"caption_extension": ".txt"}),
    )

    with pytest.raises(caption_edit.AggregateCaptionWriteError, match="captions.json"):
        storage.write_caption(DATASET_FILE, 0, image.as_posix(), "reviewed, tags")

    assert not image.with_suffix(".txt").exists()
    assert aggregate.read_bytes() == before


def test_explicit_sidecar_mode_still_writes_next_to_a_captions_json(image, monkeypatch) -> None:
    # An explicit sidecar mode reads exactly that sidecar, so the aggregate entry
    # for this image cannot shadow the write and must not be blocked.
    aggregate = image.parent / "captions.json"
    aggregate.write_text(json.dumps({image.name: ["existing, tags"]}), encoding="utf-8")
    before = aggregate.read_bytes()
    monkeypatch.setattr(
        storage,
        "resolve_tagging_image",
        lambda *_args, **_kwargs: _aggregate_resolved(
            image, {"caption_extension": ".txt", "caption_source_mode": "txt"}
        ),
    )

    result = storage.write_caption(DATASET_FILE, 0, image.as_posix(), "reviewed, tags")

    assert result["caption_file"].endswith(".txt")
    assert image.with_suffix(".txt").read_text(encoding="utf-8") == "reviewed, tags\n"
    assert aggregate.read_bytes() == before


def test_commit_counts_aggregate_caption_refusals_as_failures(image, monkeypatch) -> None:
    aggregate = image.parent / "captions.json"
    aggregate.write_text(json.dumps({image.name: ["existing, tags"]}), encoding="utf-8")
    before = aggregate.read_bytes()
    monkeypatch.setattr(
        storage,
        "resolve_tagging_image",
        lambda *_args, **_kwargs: _aggregate_resolved(
            image, {"caption_extension": ".txt", "caption_source_mode": "captions_json"}
        ),
    )
    monkeypatch.setattr(
        jobs, "create_caption_snapshot", lambda *_args, **_kwargs: {"id": "snap", "file_count": 1}
    )
    job = _job(image)

    async def run() -> dict:
        manager = jobs.TaggingJobManager()
        manager.jobs["job-1"] = job
        return await manager.commit("job-1", all_items=True)

    result = asyncio.run(run())

    assert result["failed"] == 1
    assert result["skipped"] == 0
    assert result["written"] == 0
    assert "captions.json" in result["errors"][0]["error"]
    assert result["job"]["items"][0]["commit_status"] == "failed"
    assert "captions.json" in result["job"]["items"][0]["commit_error"]
    assert not image.with_suffix(".txt").exists()
    assert aggregate.read_bytes() == before


def _nested_resolved(image: Path, root: Path, settings: dict[str, str]) -> dict:
    return {
        "path": image,
        "file": image.as_posix(),
        "name": image.name,
        "row": {"source_dir": root.as_posix()},
        "settings": settings,
        "caption": {"text": ""},
    }


def _nested_dataset(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "nested-dataset"
    (root / "sub").mkdir(parents=True)
    image = root / "sub" / "img.png"
    image.write_bytes(b"image")
    return root, image


def test_nested_layout_aggregate_caption_fails_closed(tmp_path, monkeypatch) -> None:
    # T9-1: the reader walks up to the dataset root, so the guard must too.
    root, image = _nested_dataset(tmp_path)
    aggregate = root / "captions.json"
    aggregate.write_text(json.dumps({"sub/img.png": ["agg caption"]}), encoding="utf-8")
    before = aggregate.read_bytes()
    monkeypatch.setattr(
        storage,
        "resolve_tagging_image",
        lambda *_args, **_kwargs: _nested_resolved(image, root, {"caption_extension": ".txt"}),
    )

    with pytest.raises(caption_edit.AggregateCaptionWriteError, match="captions.json"):
        storage.write_caption(DATASET_FILE, 0, image.as_posix(), "reviewed, tags")

    assert not image.with_suffix(".txt").exists()
    assert not (root / "img.txt").exists()
    assert aggregate.read_bytes() == before


@pytest.mark.parametrize(
    "aggregate_payload, row_settings",
    [
        ({"sub/other.png": ["other"]}, {"caption_extension": ".txt"}),
        ({"sub/img.png": []}, {"caption_extension": ".txt"}),
        (
            {"sub/img.png": ["agg caption"]},
            {"caption_extension": ".txt", "caption_source_mode": "txt"},
        ),
    ],
    ids=["other-image-only", "empty-aggregate-value", "explicit-txt-mode"],
)
def test_nested_layout_is_not_over_blocked(tmp_path, monkeypatch, aggregate_payload, row_settings) -> None:
    root, image = _nested_dataset(tmp_path)
    aggregate = root / "captions.json"
    aggregate.write_text(json.dumps(aggregate_payload), encoding="utf-8")
    before = aggregate.read_bytes()
    monkeypatch.setattr(
        storage,
        "resolve_tagging_image",
        lambda *_args, **_kwargs: _nested_resolved(image, root, row_settings),
    )

    result = storage.write_caption(DATASET_FILE, 0, image.as_posix(), "reviewed, tags")

    assert result["caption_file"].endswith(".txt")
    assert image.with_suffix(".txt").read_text(encoding="utf-8") == "reviewed, tags\n"
    assert aggregate.read_bytes() == before


def test_nested_explicit_json_mode_still_writes_the_json_sidecar(tmp_path, monkeypatch) -> None:
    root, image = _nested_dataset(tmp_path)
    aggregate = root / "captions.json"
    aggregate.write_text(json.dumps({"sub/img.png": ["agg caption"]}), encoding="utf-8")
    before = aggregate.read_bytes()
    image.with_suffix(".json").write_text(json.dumps({"tags": ["json-old"]}), encoding="utf-8")
    monkeypatch.setattr(
        storage,
        "resolve_tagging_image",
        lambda *_args, **_kwargs: _nested_resolved(
            image, root, {"caption_extension": ".json", "caption_source_mode": "json"}
        ),
    )

    result = storage.write_caption(DATASET_FILE, 0, image.as_posix(), "json-reviewed")

    assert result["format"] == "json"
    assert result["caption_file"].endswith(".json")
    assert aggregate.read_bytes() == before


def _dataset_with_images(dataset_dir: Path, names: list[str]) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (dataset_dir / name).write_bytes(b"image")
        (dataset_dir / name).with_suffix(".txt").write_text("old\n", encoding="utf-8")


def test_concurrent_commits_on_one_dataset_are_serialised(tmp_path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    _dataset_with_images(dataset_dir, ["a.png", "b.png", "c.png", "d.png"])
    guard = threading.Lock()
    first_write_started = threading.Event()
    in_flight = 0
    max_in_flight = 0
    written_texts: list[str] = []

    def slow_write(_file, _index, image_file, text, **_kwargs):
        nonlocal in_flight, max_in_flight
        with guard:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        if text == "A-1":
            first_write_started.set()
        time.sleep(0.05)
        Path(image_file).with_suffix(".txt").write_text(f"{text}\n", encoding="utf-8")
        with guard:
            in_flight -= 1
            written_texts.append(text)
        return {"text": text, "caption_file": image_file}

    monkeypatch.setattr(jobs, "write_caption", slow_write)
    monkeypatch.setattr(
        jobs, "create_caption_snapshot", lambda *_args, **_kwargs: {"id": "snap", "file_count": 4}
    )
    job_a = _dir_job(dataset_dir, ["a.png", "b.png"], {"a.png": "A-1", "b.png": "A-2"})
    job_b = _dir_job(dataset_dir, ["c.png", "d.png"], {"c.png": "B-1", "d.png": "B-2"})
    job_b["id"] = "job-2"

    async def run() -> tuple[dict, dict]:
        manager = jobs.TaggingJobManager()
        manager.jobs["job-1"] = job_a
        manager.jobs["job-2"] = job_b
        first = asyncio.create_task(manager.commit("job-1", all_items=True))
        assert await asyncio.to_thread(first_write_started.wait, 10)
        second = asyncio.create_task(manager.commit("job-2", all_items=True))
        assert not first.done()  # both commits really were in flight together
        return await first, await second

    first, second = asyncio.run(run())

    assert first["written"] == 2 and second["written"] == 2
    assert sorted(written_texts) == ["A-1", "A-2", "B-1", "B-2"]
    # One dataset lock: the two write-backs never overlap.
    assert max_in_flight == 1


def test_commits_on_different_datasets_do_not_block_each_other(tmp_path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    _dataset_with_images(dataset_dir, ["a.png", "b.png"])
    guard = threading.Lock()
    both_started = threading.Event()
    in_flight = 0
    max_in_flight = 0

    def slow_write(_file, _index, image_file, text, **_kwargs):
        nonlocal in_flight, max_in_flight
        with guard:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            if in_flight >= 2:
                both_started.set()
        time.sleep(0.15)
        Path(image_file).with_suffix(".txt").write_text(f"{text}\n", encoding="utf-8")
        with guard:
            in_flight -= 1
        return {"text": text, "caption_file": image_file}

    monkeypatch.setattr(jobs, "write_caption", slow_write)
    monkeypatch.setattr(
        jobs, "create_caption_snapshot", lambda *_args, **_kwargs: {"id": "snap", "file_count": 2}
    )
    job_a = _dir_job(dataset_dir, ["a.png"], {"a.png": "A-1"})
    job_b = _dir_job(dataset_dir, ["b.png"], {"b.png": "B-1"})
    job_b["id"] = "job-2"
    job_b["dataset_index"] = 1

    async def run() -> tuple[dict, dict]:
        manager = jobs.TaggingJobManager()
        manager.jobs["job-1"] = job_a
        manager.jobs["job-2"] = job_b
        return await asyncio.gather(
            manager.commit("job-1", all_items=True),
            manager.commit("job-2", all_items=True),
        )

    first, second = asyncio.run(run())

    assert first["written"] == 1 and second["written"] == 1
    # Different dataset rows must not serialise against each other.
    assert both_started.is_set()
    assert max_in_flight >= 2


@pytest.mark.parametrize("order", ["snapshot-during-commit"])
def test_manual_snapshot_waits_for_the_running_write_back(tmp_path, monkeypatch, order) -> None:
    del order
    dataset_dir = tmp_path / "dataset"
    _dataset_with_images(dataset_dir, ["a.png", "b.png"])
    first_write_started = threading.Event()
    release = threading.Event()

    def slow_write(_file, _index, image_file, text, **_kwargs):
        if text == "new-a":
            first_write_started.set()
            assert release.wait(10)
        Path(image_file).with_suffix(".txt").write_text(f"{text}\n", encoding="utf-8")
        return {"text": text, "caption_file": image_file}

    monkeypatch.setattr(jobs, "write_caption", slow_write)
    monkeypatch.setattr(
        jobs, "create_caption_snapshot", lambda *_args, **_kwargs: {"id": "auto", "file_count": 2}
    )
    monkeypatch.setattr(jobs, "resolve_caption_directory", lambda *_args, **_kwargs: dataset_dir)
    monkeypatch.setattr(
        caption_snapshots, "resolve_caption_directory", lambda *_args, **_kwargs: dataset_dir
    )
    monkeypatch.setattr(
        tagging_settings, "SETTINGS_FILE", tmp_path / "configs" / "captioning" / "settings.toml"
    )
    job = _dir_job(dataset_dir, ["a.png", "b.png"], {"a.png": "new-a", "b.png": "new-b"})

    async def run() -> dict:
        manager = jobs.TaggingJobManager()
        manager.jobs["job-1"] = job
        service = TaggingService.__new__(TaggingService)
        service.jobs = manager
        commit_task = asyncio.create_task(manager.commit("job-1", all_items=True))
        assert await asyncio.to_thread(first_write_started.wait, 10)
        snapshot_task = asyncio.create_task(
            service.create_caption_snapshot({"dataset_file": DATASET_FILE, "dataset_index": 0})
        )
        await asyncio.sleep(0.05)
        # The manual backup is blocked by the dataset lock while the write-back runs.
        assert not snapshot_task.done()
        release.set()
        await commit_task
        return await snapshot_task

    result = asyncio.run(run())

    manifest = caption_snapshots.read_snapshot(result["snapshot"]["id"])
    entries = {entry["name"]: entry["data"] for entry in manifest["entries"]}
    # Taken after the whole write-back, never mid-way (no old-a + new-b mix).
    assert entries == {"a.txt": "new-a\n", "b.txt": "new-b\n"}


def test_dataset_lock_is_released_when_write_back_fails(tmp_path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    _dataset_with_images(dataset_dir, ["a.png"])
    monkeypatch.setattr(
        jobs, "create_caption_snapshot", lambda *_args, **_kwargs: {"id": "snap", "file_count": 1}
    )
    monkeypatch.setattr(
        jobs,
        "write_caption",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    job = _dir_job(dataset_dir, ["a.png"], {"a.png": "A-1"})

    async def run() -> None:
        manager = jobs.TaggingJobManager()
        manager.jobs["job-1"] = job
        result = await manager.commit("job-1", all_items=True)
        assert result["failed"] == 1
        # An exception must not leave the dataset lock (or its entry) behind.
        assert manager._dataset_locks == {}
        monkeypatch.setattr(
            jobs,
            "write_caption",
            lambda _f, _i, image_file, text, **_kwargs: {"text": text, "caption_file": image_file},
        )
        second = await asyncio.wait_for(manager.commit("job-1", all_items=True), timeout=5)
        assert second["written"] == 1
        assert manager._dataset_locks == {}

    asyncio.run(run())


def test_dataset_locks_are_dropped_once_idle(tmp_path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    _dataset_with_images(dataset_dir, ["a.png"])
    monkeypatch.setattr(
        jobs, "write_caption",
        lambda _f, _i, image_file, text, **_kwargs: {"text": text, "caption_file": image_file},
    )
    monkeypatch.setattr(
        jobs, "create_caption_snapshot", lambda *_args, **_kwargs: {"id": "snap", "file_count": 1}
    )
    job = _dir_job(dataset_dir, ["a.png"], {"a.png": "A-1"})

    async def run() -> None:
        manager = jobs.TaggingJobManager()
        manager.jobs["job-1"] = job
        await manager.commit("job-1", all_items=True)
        assert manager._dataset_locks == {}
        manager._dataset_locks["row:stale.toml#0"] = asyncio.Lock()
        manager.clear_finished(["job-1"])
        assert manager._dataset_locks == {}
        manager._dataset_locks["row:stale.toml#1"] = asyncio.Lock()
        manager._prune()
        assert manager._dataset_locks == {}

    asyncio.run(run())


def _shared_dir_jobs(tmp_path: Path, monkeypatch) -> tuple[Path, dict, dict]:
    shared_dir = tmp_path / "shared-dataset"
    _dataset_with_images(shared_dir, ["a.png", "b.png", "c.png", "d.png"])
    # Two different dataset rows (index 0 / 1) that resolve to one physical folder.
    monkeypatch.setattr(jobs, "resolve_caption_directory", lambda *_args, **_kwargs: shared_dir)
    job_a = _dir_job(shared_dir, ["a.png", "b.png"], {"a.png": "A-1", "b.png": "A-2"})
    job_b = _dir_job(shared_dir, ["c.png", "d.png"], {"c.png": "B-1", "d.png": "B-2"})
    job_b["id"] = "job-2"
    job_b["dataset_index"] = 1
    return shared_dir, job_a, job_b


def _run_shared_dir_commits(job_a: dict, job_b: dict, slow_write, started: threading.Event):
    async def run() -> tuple[dict, dict]:
        manager = jobs.TaggingJobManager()
        manager.jobs["job-1"] = job_a
        manager.jobs["job-2"] = job_b
        first = asyncio.create_task(manager.commit("job-1", all_items=True))
        assert await asyncio.to_thread(started.wait, 10)
        second = asyncio.create_task(manager.commit("job-2", all_items=True))
        assert not first.done()
        return await first, await second

    return asyncio.run(run())


def test_rows_sharing_one_caption_directory_share_the_scope_lock(tmp_path, monkeypatch) -> None:
    _, job_a, job_b = _shared_dir_jobs(tmp_path, monkeypatch)
    guard = threading.Lock()
    started = threading.Event()
    in_flight = 0
    max_in_flight = 0

    def slow_write(_f, _i, image_file, text, **_kwargs):
        nonlocal in_flight, max_in_flight
        with guard:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        if text == "A-1":
            started.set()
        time.sleep(0.05)
        Path(image_file).with_suffix(".txt").write_text(f"{text}\n", encoding="utf-8")
        with guard:
            in_flight -= 1
        return {"text": text, "caption_file": image_file}

    monkeypatch.setattr(jobs, "write_caption", slow_write)
    monkeypatch.setattr(
        jobs, "create_caption_snapshot", lambda *_args, **_kwargs: {"id": "snap", "file_count": 4}
    )

    first, second = _run_shared_dir_commits(job_a, job_b, slow_write, started)

    assert first["written"] == 2 and second["written"] == 2
    # t23-F2: rows over one physical directory must not interleave (was 2 before).
    assert max_in_flight == 1


def test_rows_sharing_one_caption_directory_interleave_without_the_lock(tmp_path, monkeypatch) -> None:
    # Negative control: without the scope lock the very same scenario interleaves,
    # which proves the assertion above can detect the bug.
    _, job_a, job_b = _shared_dir_jobs(tmp_path, monkeypatch)
    guard = threading.Lock()
    started = threading.Event()
    in_flight = 0
    max_in_flight = 0

    def slow_write(_f, _i, image_file, text, **_kwargs):
        nonlocal in_flight, max_in_flight
        with guard:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        if text == "A-1":
            started.set()
        time.sleep(0.05)
        Path(image_file).with_suffix(".txt").write_text(f"{text}\n", encoding="utf-8")
        with guard:
            in_flight -= 1
        return {"text": text, "caption_file": image_file}

    @asynccontextmanager
    async def no_lock(_self, _scope_key):
        yield

    monkeypatch.setattr(jobs, "write_caption", slow_write)
    monkeypatch.setattr(
        jobs, "create_caption_snapshot", lambda *_args, **_kwargs: {"id": "snap", "file_count": 4}
    )
    monkeypatch.setattr(jobs.TaggingJobManager, "dataset_write_lock", no_lock)

    first, second = _run_shared_dir_commits(job_a, job_b, slow_write, started)

    assert first["written"] == 2 and second["written"] == 2
    assert max_in_flight >= 2


def test_restore_waits_for_the_running_write_back_and_ends_consistent(tmp_path, monkeypatch) -> None:
    dataset_dir = tmp_path / "restore-dataset"
    _dataset_with_images(dataset_dir, ["a.png", "b.png"])
    (dataset_dir / "a.txt").write_text("old-a\n", encoding="utf-8")
    (dataset_dir / "b.txt").write_text("old-b\n", encoding="utf-8")
    monkeypatch.setattr(jobs, "resolve_caption_directory", lambda *_args, **_kwargs: dataset_dir)
    monkeypatch.setattr(
        caption_snapshots, "resolve_caption_directory", lambda *_args, **_kwargs: dataset_dir
    )
    monkeypatch.setattr(
        tagging_settings, "SETTINGS_FILE", tmp_path / "configs" / "captioning" / "settings.toml"
    )
    snapshot = caption_snapshots.create_snapshot(DATASET_FILE, 0)
    first_write_started = threading.Event()
    release = threading.Event()

    def slow_write(_f, _i, image_file, text, **_kwargs):
        if text == "new-a":
            first_write_started.set()
            assert release.wait(10)
        Path(image_file).with_suffix(".txt").write_text(f"{text}\n", encoding="utf-8")
        return {"text": text, "caption_file": image_file}

    monkeypatch.setattr(jobs, "write_caption", slow_write)
    monkeypatch.setattr(
        jobs, "create_caption_snapshot", lambda *_args, **_kwargs: {"id": "auto", "file_count": 2}
    )
    job = _dir_job(dataset_dir, ["a.png", "b.png"], {"a.png": "new-a", "b.png": "new-b"})

    async def run() -> dict:
        manager = jobs.TaggingJobManager()
        manager.jobs["job-1"] = job
        service = TaggingService.__new__(TaggingService)
        service.jobs = manager
        commit_task = asyncio.create_task(manager.commit("job-1", all_items=True))
        assert await asyncio.to_thread(first_write_started.wait, 10)
        restore_task = asyncio.create_task(service.restore_caption_snapshot(snapshot["id"]))
        await asyncio.sleep(0.05)
        # t23-C7: restore must wait for the running write-back instead of mixing states.
        assert not restore_task.done()
        release.set()
        await commit_task
        return await restore_task

    result = asyncio.run(run())

    assert result["written"] == 2
    state = {
        "a.txt": (dataset_dir / "a.txt").read_text(encoding="utf-8"),
        "b.txt": (dataset_dir / "b.txt").read_text(encoding="utf-8"),
    }
    # Whole write-back or whole restore, never {a: old-a, b: new-b}.
    assert state in (
        {"a.txt": "new-a\n", "b.txt": "new-b\n"},
        {"a.txt": "old-a\n", "b.txt": "old-b\n"},
    )
    # Restore ran after the write-back here, so the snapshot content wins.
    assert state == {"a.txt": "old-a\n", "b.txt": "old-b\n"}


def test_rows_over_different_caption_directories_still_run_in_parallel(tmp_path, monkeypatch) -> None:
    dir_a = tmp_path / "dataset-a"
    dir_b = tmp_path / "dataset-b"
    _dataset_with_images(dir_a, ["a.png"])
    _dataset_with_images(dir_b, ["b.png"])

    def resolve(_file, index, **_kwargs):
        return dir_a if index == 0 else dir_b

    monkeypatch.setattr(jobs, "resolve_caption_directory", resolve)
    guard = threading.Lock()
    both_started = threading.Event()
    in_flight = 0
    max_in_flight = 0

    def slow_write(_f, _i, image_file, text, **_kwargs):
        nonlocal in_flight, max_in_flight
        with guard:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            if in_flight >= 2:
                both_started.set()
        time.sleep(0.15)
        Path(image_file).with_suffix(".txt").write_text(f"{text}\n", encoding="utf-8")
        with guard:
            in_flight -= 1
        return {"text": text, "caption_file": image_file}

    monkeypatch.setattr(jobs, "write_caption", slow_write)
    monkeypatch.setattr(
        jobs, "create_caption_snapshot", lambda *_args, **_kwargs: {"id": "snap", "file_count": 1}
    )
    job_a = _dir_job(dir_a, ["a.png"], {"a.png": "A-1"})
    job_b = _dir_job(dir_b, ["b.png"], {"b.png": "B-1"})
    job_b["id"] = "job-2"
    job_b["dataset_index"] = 1

    async def run() -> tuple[dict, dict]:
        manager = jobs.TaggingJobManager()
        manager.jobs["job-1"] = job_a
        manager.jobs["job-2"] = job_b
        return await asyncio.gather(
            manager.commit("job-1", all_items=True),
            manager.commit("job-2", all_items=True),
        )

    first, second = asyncio.run(run())

    assert first["written"] == 1 and second["written"] == 1
    # The scope lock must not degrade into one global lock.
    assert both_started.is_set()
    assert max_in_flight >= 2


def test_unresolved_caption_directory_falls_back_to_a_row_key(tmp_path, monkeypatch) -> None:
    dataset_dir = tmp_path / "dataset"
    _dataset_with_images(dataset_dir, ["a.png"])

    def missing(*_args, **_kwargs):
        raise FileNotFoundError("数据集图片目录不存在")

    monkeypatch.setattr(jobs, "resolve_caption_directory", missing)
    monkeypatch.setattr(caption_snapshots, "resolve_caption_directory", missing)
    monkeypatch.setattr(
        jobs,
        "write_caption",
        lambda _f, _i, image_file, text, **_kwargs: {"text": text, "caption_file": image_file},
    )
    job = _dir_job(dataset_dir, ["a.png"], {"a.png": "A-1"})

    assert jobs.caption_scope_key(DATASET_FILE, 0) == f"row:{DATASET_FILE}#0"

    async def run() -> tuple[dict, dict]:
        manager = jobs.TaggingJobManager()
        manager.jobs["job-1"] = job
        result = await manager.commit("job-1", all_items=True)
        return result, dict(manager._dataset_locks)

    result, locks = asyncio.run(run())

    # Existing grading: a missing directory is a non-blocking snapshot_error.
    assert result["written"] == 1
    assert result["snapshot"] is None
    assert result["snapshot_error"]
    assert locks == {}
