from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import toml
from aiohttp import web
from aiohttp.test_utils import TestServer

from web.services.tagging import TaggingService, client, jobs, memory_log, prompt_presets, settings, storage


def _patch_settings_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "configs" / "captioning" / "settings.toml")
    monkeypatch.setattr(settings, "SECRETS_FILE", tmp_path / ".anima-captioning-secrets.toml")
    for name in (*settings._ENV_BASE_URL, *settings._ENV_MODEL, *settings._ENV_KEY):
        monkeypatch.delenv(name, raising=False)


def test_tagging_job_retention_prunes_oldest_terminal_jobs_but_keeps_active_jobs() -> None:
    manager = jobs.TaggingJobManager(max_retained_jobs=2)
    manager.jobs["running-oldest"] = {"state": "running"}
    manager.jobs["completed-middle"] = {"state": "completed"}
    manager.jobs["failed-newest"] = {"state": "failed"}

    manager._prune()

    assert list(manager.jobs) == ["running-oldest", "failed-newest"]
    assert manager.set_job_retention(1) == 1
    assert list(manager.jobs) == ["running-oldest"]

    manager.jobs["queued-newest"] = {"state": "queued"}
    manager._prune()
    assert list(manager.jobs) == ["running-oldest", "queued-newest"]


def test_tagging_job_rerun_reuses_source_job_and_resets_only_selected_item(monkeypatch) -> None:
    manager = jobs.TaggingJobManager()
    monkeypatch.setattr(
        jobs,
        "get_effective_settings",
        lambda _profile_id: {
            "provider": "openai_compatible",
            "base_url": "https://vision.example.test/v1",
            "model": "vision-test",
            "system_prompt": "system prompt",
            "_profile_id": "new-profile",
            "_profile_name": "新接入",
        },
    )
    manager.jobs["source-job"] = {
        "id": "source-job",
        "state": "completed",
        "created_at": 1.0,
        "started_at": 1.0,
        "finished_at": 2.0,
        "dataset_file": "configs/datasets/example.toml",
        "dataset_index": 2,
        "source": "training",
        "profile_id": "old-profile",
        "system_prompt": "system prompt",
        "prompt": "user prompt",
        "items": [
            {
                "id": "item-1",
                "file": "nested/sample.png",
                "url": "/api/images/sample.png",
                "thumbnail_url": "/api/images/sample.png?thumbnail=1",
                "caption": "caption on disk",
                "proposed_caption": "old generated result",
                "state": "ready",
                "error": "",
                "commit_error": "",
                "attempts": 1,
                "elapsed_ms": 10,
                "_path": "/data/sample.png",
            }
        ],
        "total": 1,
        "completed": 1,
        "failed": 0,
        "canceled": 0,
        "settings": {"provider": "openai_compatible", "model": "old-model"},
        "_provider_settings": {"provider": "openai_compatible"},
    }

    async def fake_run_job(job):
        for item in manager._run_items(job):
            item["state"] = "ready"
            item["proposed_caption"] = "new generated result"
        manager._finish(job, manager._final_state(job))

    monkeypatch.setattr(manager, "_run_job", fake_run_job)

    async def run() -> None:
        rerun = await manager.rerun("source-job", profile_id="new-profile", item_ids=["item-1"])
        assert rerun["job"]["id"] == "source-job"
        assert len(manager.jobs) == 1
        item = rerun["job"]["items"][0]
        assert item["id"] == "item-1"
        assert item["state"] == "queued"
        assert item["proposed_caption"] == ""
        assert item["caption"] == "caption on disk"
        assert manager.jobs["source-job"]["_run_item_ids"] == ["item-1"]
        await manager._tasks["source-job"]
        assert manager.snapshot("source-job")["job"]["items"][0]["proposed_caption"] == "new generated result"

    asyncio.run(run())


def test_tagging_job_rerun_accepts_a_deduplicated_image_subset_and_rejects_unknown_ids(monkeypatch) -> None:
    manager = jobs.TaggingJobManager()
    monkeypatch.setattr(
        jobs,
        "get_effective_settings",
        lambda _profile_id: {
            "provider": "openai_compatible",
            "base_url": "https://vision.example.test/v1",
            "model": "vision-test",
            "system_prompt": "system",
            "_profile_id": "profile-1",
            "_profile_name": "接入 1",
        },
    )
    manager.jobs["source-job"] = {
        "id": "source-job",
        "state": "completed",
        "created_at": 1.0,
        "started_at": 1.0,
        "finished_at": 2.0,
        "dataset_file": "configs/datasets/example.toml",
        "dataset_index": 0,
        "source": "source",
        "profile_id": "profile-1",
        "system_prompt": "system",
        "prompt": "prompt",
        "items": [
            {"id": "item-1", "file": "one.png", "url": "/one", "thumbnail_url": "/one-thumb", "state": "ready", "proposed_caption": "one old", "_path": "/data/one.png"},
            {"id": "item-2", "file": "two.png", "url": "/two", "thumbnail_url": "/two-thumb", "state": "ready", "proposed_caption": "two old", "_path": "/data/two.png"},
            {"id": "item-3", "file": "three.png", "url": "/three", "thumbnail_url": "/three-thumb", "state": "ready", "proposed_caption": "three old", "_path": "/data/three.png"},
        ],
        "total": 3,
        "completed": 3,
        "failed": 0,
        "canceled": 0,
        "settings": {"provider": "openai_compatible", "model": "vision-test"},
    }

    async def fake_run_job(job):
        for item in manager._run_items(job):
            item["state"] = "ready"
            item["proposed_caption"] = f"new {item['id']}"
        manager._finish(job, manager._final_state(job))

    monkeypatch.setattr(manager, "_run_job", fake_run_job)

    async def run() -> None:
        subset = await manager.rerun("source-job", item_ids=["item-3", "item-1", "item-3"])
        assert subset["job"]["id"] == "source-job"
        assert len(manager.jobs) == 1
        assert manager.jobs["source-job"]["_run_item_ids"] == ["item-3", "item-1"]
        assert [item["state"] for item in subset["job"]["items"]] == ["queued", "ready", "queued"]
        await manager._tasks["source-job"]

        whole = await manager.rerun("source-job", item_ids=[])
        assert whole["job"]["id"] == "source-job"
        assert manager.jobs["source-job"]["_run_item_ids"] == ["item-1", "item-2", "item-3"]
        await manager._tasks["source-job"]

        with pytest.raises(ValueError, match="找不到打标图片"):
            await manager.rerun("source-job", item_ids=["item-1", "missing"])
        assert len(manager.jobs) == 1

    asyncio.run(run())


def test_tagging_job_rerun_rejects_active_source_job() -> None:
    manager = jobs.TaggingJobManager()
    manager.jobs["running-job"] = {"id": "running-job", "state": "running"}

    with pytest.raises(RuntimeError, match="任务仍在运行"):
        asyncio.run(manager.rerun("running-job"))


def test_tagging_settings_keep_api_key_out_of_public_payload(tmp_path, monkeypatch) -> None:
    _patch_settings_paths(tmp_path, monkeypatch)

    saved = settings.save_settings(
        {
            "base_url": "https://vision.example.test/v1",
            "model": "vision-1",
            "api_key": "secret-token-value",
            "timeout_seconds": 42,
        }
    )

    assert saved["ok"] is True
    assert saved["api_key_configured"] is True
    assert "api_key" not in saved
    assert "secret-token-value" not in json.dumps(saved, ensure_ascii=False)
    assert settings.get_api_key() == "secret-token-value"
    assert settings.SECRETS_FILE.stat().st_mode & 0o777 == 0o600

    retained = settings.save_settings({"model": "vision-2", "api_key": ""})
    assert retained["api_key_configured"] is True
    assert settings.get_api_key() == "secret-token-value"

    cleared = settings.save_settings({"clear_api_key": True})
    assert cleared["api_key_configured"] is False
    assert not settings.SECRETS_FILE.exists()
    persisted = toml.loads(settings.SETTINGS_FILE.read_text(encoding="utf-8"))
    assert "api_key" not in persisted["tagging"]


def test_tagging_settings_persist_bounded_log_retention(tmp_path, monkeypatch) -> None:
    _patch_settings_paths(tmp_path, monkeypatch)

    saved = settings.save_settings({"log_retention_lines": 850})
    assert saved["log_retention_lines"] == 850
    assert settings.load_settings()["log_retention_lines"] == 850

    clamped = settings.save_settings({"log_retention_lines": 50_000})
    assert clamped["log_retention_lines"] == memory_log.MAX_LOG_RETENTION_LINES


def test_prompt_preset_crud_uses_separate_captioning_file(tmp_path, monkeypatch) -> None:
    _patch_settings_paths(tmp_path, monkeypatch)

    created = prompt_presets.create_prompt_preset(
        {"name": "基础描述", "system_prompt": "system", "user_prompt": "user"}
    )
    preset_id = created["preset"]["id"]
    assert created["preset"]["user_prompt"] == "user"
    assert prompt_presets._presets_file().name == "prompt-presets.toml"

    updated = prompt_presets.update_prompt_preset(
        preset_id,
        {"name": "细节描述", "system_prompt": "system 2", "user_prompt": "user 2"},
    )
    assert updated["preset"]["name"] == "细节描述"
    assert prompt_presets.list_prompt_presets()["presets"][0]["system_prompt"] == "system 2"

    deleted = prompt_presets.delete_prompt_preset(preset_id)
    assert deleted["presets"] == []


def test_tagging_service_exposes_animalorastudio_builtin_system_prompt_templates(
    tmp_path, monkeypatch
) -> None:
    _patch_settings_paths(tmp_path, monkeypatch)
    service = TaggingService()
    listed = service.list_prompt_presets()
    assert [item["id"] for item in listed["presets"][:6]] == [
        "builtin-detailed",
        "builtin-danbooru",
        "builtin-character-action",
        "builtin-anima-three-format",
        "builtin-anima-style-overfit",
        "builtin-anima-style-trigger-json",
    ]
    assert all(item["builtin"] is True for item in listed["presets"][:6])
    assert all(item["user_prompt"] for item in listed["presets"][:6])
    assert not prompt_presets._presets_file().exists()


def test_tagging_memory_log_is_bounded_filterable_and_resizable() -> None:
    log = memory_log.TaggingMemoryLog()
    assert log.retention_lines == 200
    log.set_retention(50)
    for index in range(60):
        log.append(f"line {index}", job_id="job-a" if index % 2 else "job-b")

    snapshot = log.snapshot()
    assert snapshot["buffered"] == 50
    assert snapshot["lines"][0]["sequence"] == 11
    filtered = log.snapshot(after=50, job_id="job-a")
    assert all(line["sequence"] > 50 and line["job_id"] == "job-a" for line in filtered["lines"])

    assert log.set_retention(100) == 100
    log.append("last")
    assert log.snapshot()["last_sequence"] == 61
    assert log.clear()["lines"] == []


@pytest.mark.parametrize(
    "value",
    [
        "ftp://example.test/v1",
        "https://user:pass@example.test/v1",
        "https://example.test:99999/v1",
        "https://example.test/v1?token=secret",
        "https://example.test/v1#fragment",
    ],
)
def test_tagging_settings_reject_malformed_or_credentialed_urls(value) -> None:
    with pytest.raises(ValueError):
        settings.normalize_settings({"base_url": value})


def test_tagging_client_rejects_private_dns_resolution() -> None:
    class ResolverStub:
        async def resolve(self, *_args, **_kwargs):
            return [{"host": "127.0.0.1", "port": 443}]

        async def close(self):
            return None

    async def run() -> None:
        resolver = client._PublicOnlyResolver(allow_private=False, delegate=ResolverStub())
        with pytest.raises(OSError, match="私有网络"):
            await resolver.resolve("vision.example.test", 443)
        await resolver.close()

    asyncio.run(run())


def test_tagging_client_does_not_follow_redirects(monkeypatch) -> None:
    async def run() -> None:
        app = web.Application()

        async def redirect(_request):
            raise web.HTTPFound("/captured")

        async def captured(_request):
            return web.json_response({"data": []})

        app.router.add_get("/v1/models", redirect)
        app.router.add_get("/captured", captured)
        monkeypatch.setattr(client, "get_api_key", lambda: "secret-value")
        async with TestServer(app) as server:
            base_url = str(server.make_url("/v1")).rstrip("/")
            provider = client.OpenAICompatibleClient(
                {
                    **settings.DEFAULT_SETTINGS,
                    "base_url": base_url,
                    "allow_private_network": True,
                    "timeout_seconds": 5,
                }
            )
            with pytest.raises(client.TaggingApiError, match="重定向"):
                await provider.ping()

    asyncio.run(run())


def test_tagging_client_sends_openai_compatible_vision_payload(tmp_path, monkeypatch) -> None:
    image = tmp_path / "sample.png"
    image.write_bytes(b"small-image")
    captured = {}

    async def completion(request):
        captured["authorization"] = request.headers.get("Authorization")
        captured["payload"] = await request.json()
        return web.json_response(
            {
                "model": "vision-test",
                "choices": [{"message": {"content": "a concise caption"}}],
            }
        )

    async def run() -> None:
        app = web.Application()
        app.router.add_post("/v1/chat/completions", completion)
        monkeypatch.setattr(client, "get_api_key", lambda: "secret-value")
        async with TestServer(app) as server:
            provider = client.OpenAICompatibleClient(
                {
                    **settings.DEFAULT_SETTINGS,
                    "base_url": str(server.make_url("/v1")).rstrip("/"),
                    "model": "vision-test",
                    "allow_private_network": True,
                    "timeout_seconds": 5,
                    "retry_interval_seconds": 0,
                }
            )
            result = await provider.describe_image(image, "describe only visible details")
            assert result["caption"] == "a concise caption"
            assert result["attempts"] == 1

    asyncio.run(run())
    assert captured["authorization"] == "Bearer secret-value"
    payload = captured["payload"]
    assert payload["model"] == "vision-test"
    content = payload["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "describe only visible details"}
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_extract_caption_handles_openai_content_parts() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "first"},
                        {"type": "text", "text": "second"},
                    ]
                }
            }
        ]
    }
    assert client.extract_caption(payload) == "first\nsecond"


def test_tagging_job_cleans_task_after_client_initialization_failure(tmp_path, monkeypatch) -> None:
    image = tmp_path / "source" / "sample.png"
    image.parent.mkdir()
    image.write_bytes(b"not-used")
    monkeypatch.setattr(
        jobs,
        "resolve_tagging_image",
        lambda *_args, **_kwargs: {
            "path": image,
            "file": image.as_posix(),
            "name": image.name,
            "caption": {"text": ""},
        },
    )
    monkeypatch.setattr(jobs, "load_settings", lambda: dict(settings.DEFAULT_SETTINGS))

    class BrokenClient:
        def __init__(self, _settings):
            raise ValueError("provider unavailable")

    monkeypatch.setattr(jobs, "OpenAICompatibleClient", BrokenClient)

    async def run() -> None:
        manager = jobs.TaggingJobManager()
        created = await manager.create(
            {
                "dataset_file": "configs/datasets/example.toml",
                "dataset_index": 0,
                "prompt": "describe",
                "items": [image.as_posix()],
            }
        )
        job_id = created["job"]["id"]
        await manager._tasks[job_id]
        snapshot = manager.snapshot(job_id)["job"]
        assert snapshot["state"] == "failed"
        assert snapshot["items"][0]["error"] == "provider unavailable"
        assert job_id not in manager._tasks

    asyncio.run(run())


def test_tagging_job_cancel_interrupts_in_flight_provider_call(tmp_path, monkeypatch) -> None:
    image = tmp_path / "source" / "sample.png"
    image.parent.mkdir()
    image.write_bytes(b"not-used")
    monkeypatch.setattr(
        jobs,
        "resolve_tagging_image",
        lambda *_args, **_kwargs: {
            "path": image,
            "file": image.as_posix(),
            "name": image.name,
            "caption": {"text": ""},
        },
    )
    monkeypatch.setattr(jobs, "load_settings", lambda: dict(settings.DEFAULT_SETTINGS))
    started = asyncio.Event()
    interrupted = asyncio.Event()

    class BlockingClient:
        def __init__(self, _settings):
            pass

        async def describe_image(self, _path, _prompt):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                interrupted.set()
                raise

    monkeypatch.setattr(jobs, "OpenAICompatibleClient", BlockingClient)

    async def run() -> None:
        manager = jobs.TaggingJobManager()
        created = await manager.create(
            {
                "dataset_file": "configs/datasets/example.toml",
                "dataset_index": 0,
                "prompt": "describe",
                "items": [image.as_posix()],
            }
        )
        job_id = created["job"]["id"]
        await asyncio.wait_for(started.wait(), timeout=1)
        canceled = await asyncio.wait_for(manager.cancel(job_id), timeout=1)
        assert canceled["job"]["state"] == "canceled"
        assert canceled["job"]["items"][0]["state"] == "canceled"
        assert interrupted.is_set()
        assert job_id not in manager._tasks

    asyncio.run(run())


def test_tagging_job_cancel_before_first_task_step_cleans_task_reference(tmp_path, monkeypatch) -> None:
    image = tmp_path / "source" / "sample.png"
    image.parent.mkdir()
    image.write_bytes(b"not-used")
    monkeypatch.setattr(
        jobs,
        "resolve_tagging_image",
        lambda *_args, **_kwargs: {
            "path": image,
            "file": image.as_posix(),
            "name": image.name,
            "caption": {"text": ""},
        },
    )
    monkeypatch.setattr(jobs, "load_settings", lambda: dict(settings.DEFAULT_SETTINGS))

    async def run() -> None:
        manager = jobs.TaggingJobManager()
        created = await manager.create(
            {
                "dataset_file": "configs/datasets/example.toml",
                "dataset_index": 0,
                "prompt": "describe",
                "items": [image.as_posix()],
            }
        )
        job_id = created["job"]["id"]
        canceled = await manager.cancel(job_id)
        assert canceled["job"]["state"] == "canceled"
        assert canceled["job"]["items"][0]["state"] == "canceled"
        assert job_id not in manager._tasks

    asyncio.run(run())


def test_tagging_job_uses_per_job_system_prompt_and_emits_memory_logs(tmp_path, monkeypatch) -> None:
    image = tmp_path / "source" / "sample.png"
    image.parent.mkdir()
    image.write_bytes(b"not-used")
    monkeypatch.setattr(
        jobs,
        "resolve_tagging_image",
        lambda *_args, **_kwargs: {
            "path": image,
            "file": image.as_posix(),
            "name": image.name,
            "caption": {"text": ""},
        },
    )
    monkeypatch.setattr(jobs, "load_settings", lambda: dict(settings.DEFAULT_SETTINGS))
    captured = {}

    class RecordingClient:
        def __init__(self, values):
            captured["system_prompt"] = values["system_prompt"]

        async def describe_image(self, _path, prompt):
            captured["user_prompt"] = prompt
            return {"caption": "generated", "attempts": 1, "elapsed_ms": 2}

    monkeypatch.setattr(jobs, "OpenAICompatibleClient", RecordingClient)

    async def run() -> None:
        manager = jobs.TaggingJobManager()
        created = await manager.create(
            {
                "dataset_file": "configs/datasets/example.toml",
                "dataset_index": 0,
                "system_prompt": "job system",
                "user_prompt": "job user",
                "items": [image.as_posix()],
            }
        )
        job_id = created["job"]["id"]
        await manager._tasks[job_id]
        snapshot = manager.snapshot(job_id)["job"]
        assert snapshot["state"] == "completed"
        assert snapshot["system_prompt"] == "job system"
        assert "_provider_settings" not in snapshot
        assert manager.get_logs(job_id=job_id)["lines"][-1]["event"] == "job_finished"

        emptied = manager.update_item(job_id, snapshot["items"][0]["id"], "")["job"]
        assert emptied["items"][0]["state"] == "empty"
        assert emptied["completed"] == 0

    asyncio.run(run())
    assert captured == {"system_prompt": "job system", "user_prompt": "job user"}


def test_tagging_commit_is_serialized_and_idempotent_for_unchanged_caption(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        jobs,
        "write_caption",
        lambda *_args, **_kwargs: calls.append("write") or {
            "text": "caption",
            "caption_file": "dataset/sample.txt",
        },
    )

    async def run() -> None:
        manager = jobs.TaggingJobManager()
        manager.jobs["job-1"] = {
            "id": "job-1",
            "state": "completed",
            "created_at": 1.0,
            "started_at": 1.0,
            "finished_at": 2.0,
            "dataset_file": "configs/datasets/example.toml",
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
                "file": "dataset/sample.png",
                "name": "sample.png",
                "state": "ready",
                "caption": "",
                "proposed_caption": "caption",
                "error": "",
                "commit_error": "",
            }],
            "error": "",
            "settings": {"model": "vision"},
        }
        results = await asyncio.gather(
            manager.commit("job-1", all_items=True),
            manager.commit("job-1", all_items=True),
        )
        assert sorted(result["written"] for result in results) == [0, 1]
        assert manager.snapshot("job-1")["job"]["items"][0]["caption_file"] == "dataset/sample.txt"

    asyncio.run(run())
    assert calls == ["write"]


def test_tagging_shutdown_has_a_bounded_wait(monkeypatch) -> None:
    monkeypatch.setattr(jobs, "SHUTDOWN_TIMEOUT_SECONDS", 0.01)

    async def run() -> None:
        manager = jobs.TaggingJobManager()
        task = asyncio.create_task(asyncio.sleep(60))
        manager._tasks["slow"] = task
        manager._cancel_events["slow"] = asyncio.Event()
        await manager.shutdown()
        assert task.cancelled()
        assert manager._tasks == {}

    asyncio.run(run())


def test_resolved_tagging_image_exposes_cached_thumbnail_and_original_urls(tmp_path, monkeypatch) -> None:
    image = tmp_path / "dataset" / "sample image.png"
    image.parent.mkdir()
    image.write_bytes(b"image")
    monkeypatch.setattr(
        storage,
        "load_dataset_preset",
        lambda _file: {"datasets": [{"source_dir": str(image.parent)}], "defaults": {}},
    )
    monkeypatch.setattr(storage, "resolve_dataset_preview_image", lambda *_args, **_kwargs: image)
    monkeypatch.setattr(storage, "read_caption_for_image", lambda *_args, **_kwargs: {"text": ""})

    resolved = storage.resolve_tagging_image("configs/datasets/example.toml", 0, image.as_posix())

    assert resolved["url"].startswith("/api/config/dataset-presets/image?")
    assert "sample%20image.png" in resolved["url"]
    assert resolved["thumbnail_url"].startswith("/api/config/dataset-presets/thumbnail?")
    assert "&v=" in resolved["thumbnail_url"]


def test_caption_write_always_creates_one_txt_sidecar_and_preserves_structured_sources(tmp_path, monkeypatch) -> None:
    image = tmp_path / "dataset" / "sample.png"
    image.parent.mkdir()
    image.write_bytes(b"image")
    resolved = {
        "path": image,
        "file": image.as_posix(),
        "name": image.name,
        "row": {},
        "settings": {"caption_extension": ".txt"},
        "caption": {"path": None, "text": ""},
    }
    monkeypatch.setattr(storage, "resolve_tagging_image", lambda *_args, **_kwargs: resolved)

    result = storage.write_caption("preset.toml", 0, image.as_posix(), "new caption")

    assert result["text"] == "new caption"
    assert image.with_suffix(".txt").read_text(encoding="utf-8") == "new caption\n"

    structured = image.with_suffix(".json")
    structured.write_text('{"sample": "old"}', encoding="utf-8")
    resolved["caption"] = {"path": structured, "text": "old"}
    replacement = storage.write_caption("preset.toml", 0, image.as_posix(), "replacement")

    assert replacement["caption_file"].endswith("sample.txt")
    assert image.with_suffix(".txt").read_text(encoding="utf-8") == "replacement\n"
    assert structured.read_text(encoding="utf-8") == '{"sample": "old"}'
