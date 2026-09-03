from __future__ import annotations

import asyncio

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from web.routes.tagging import setup_tagging_routes


class _TaggingServiceStub:
    def __init__(self):
        self.saved = None
        self.reruns = []

    def get_settings(self):
        return {"ok": True, "model": "vision-1", "api_key_configured": True}

    def save_settings(self, value):
        self.saved = value
        return {"ok": True, "model": value.get("model", ""), "api_key_configured": True}

    async def test_provider(self, mode):
        return {"ok": True, "mode": mode, "elapsed_ms": 1}

    def list_prompt_presets(self):
        return {"ok": True, "presets": []}

    def create_prompt_preset(self, data):
        return {"ok": True, "preset": {"id": "preset-1", **data}, "presets": []}

    def update_prompt_preset(self, preset_id, data):
        return {"ok": True, "preset": {"id": preset_id, **data}, "presets": []}

    def delete_prompt_preset(self, preset_id):
        return {"ok": True, "deleted": preset_id, "presets": []}

    def get_logs(self, **_kwargs):
        return {"ok": True, "lines": [], "retention_lines": 200, "last_sequence": 0}

    def clear_logs(self):
        return {"ok": True, "lines": [], "retention_lines": 200, "last_sequence": 0}

    def list_jobs(self):
        return {"ok": True, "jobs": []}

    async def create_job(self, payload):
        return {"ok": True, "job": {"id": "job-1", "state": "queued", "payload": payload}}

    def get_job(self, job_id):
        return {"ok": True, "job": {"id": job_id, "state": "completed", "items": []}}

    async def rerun_job(self, job_id, profile_id=""):
        self.reruns.append((job_id, profile_id))
        return {"ok": True, "job": {"id": "rerun-job", "state": "queued"}}

    async def cancel_job(self, job_id):
        return {"ok": True, "job": {"id": job_id, "state": "canceled"}}

    def update_item(self, job_id, item_id, text):
        return {"ok": True, "job": {"id": job_id}, "item_id": item_id, "text": text}

    async def commit_job(self, job_id, *, all_items=False, item_ids=None):
        return {"ok": True, "job": {"id": job_id}, "all": all_items, "item_ids": item_ids or []}

    def get_tag_dictionary_status(self):
        return {"ok": True, "installed": False, "state": "missing"}

    async def start_tag_dictionary_download(self):
        return {"ok": True, "download": {"id": "dictionary-download", "state": "queued"}}

    def get_tag_dictionary_download(self, download_id):
        return {"ok": True, "download": {"id": download_id, "state": "completed"}}

    async def cancel_tag_dictionary_download(self, download_id):
        return {"ok": True, "download": {"id": download_id, "state": "canceled"}}

    async def translate_tags(self, tags, target_language):
        return {"ok": True, "translations": tags, "target_language": target_language}


def test_captioning_and_tagging_aliases_share_the_same_contract() -> None:
    async def run() -> None:
        app = web.Application()
        service = _TaggingServiceStub()
        app["tagging_service"] = service
        setup_tagging_routes(app)

        async with TestServer(app) as server:
            async with TestClient(server) as client:
                for prefix in ("/api/captioning", "/api/tagging"):
                    settings_response = await client.get(f"{prefix}/settings")
                    settings_payload = await settings_response.json()
                    assert settings_response.status == 200
                    assert settings_payload["api_key_configured"] is True
                    assert "api_key" not in settings_payload

                    jobs_response = await client.get(f"{prefix}/jobs")
                    assert (await jobs_response.json()) == {"ok": True, "jobs": []}

                    dictionary_response = await client.get(f"{prefix}/tag-dictionary")
                    assert (await dictionary_response.json())["state"] == "missing"

                    rerun_response = await client.post(
                        f"{prefix}/jobs/source-job/rerun",
                        json={"profile_id": "local-profile"},
                    )
                    assert rerun_response.status == 202
                    assert (await rerun_response.json())["job"]["id"] == "rerun-job"

                saved_response = await client.put(
                    "/api/captioning/settings",
                    json={"model": "vision-2", "api_key": "secret-value"},
                )
                saved_payload = await saved_response.json()
                assert saved_payload == {"ok": True, "model": "vision-2", "api_key_configured": True}
                assert service.saved["api_key"] == "secret-value"

                test_response = await client.post("/api/captioning/test", json={"mode": "actual"})
                assert (await test_response.json())["mode"] == "actual"

                presets_response = await client.post(
                    "/api/captioning/prompt-presets",
                    json={"name": "preset", "system_prompt": "system", "user_prompt": "user"},
                )
                assert presets_response.status == 201
                assert (await presets_response.json())["preset"]["id"] == "preset-1"

                logs_response = await client.get("/api/tagging/logs?after=0&limit=200")
                assert (await logs_response.json())["retention_lines"] == 200

                created_response = await client.post(
                    "/api/captioning/jobs",
                    json={"dataset_file": "example.toml", "items": ["sample.png"]},
                )
                assert created_response.status == 202
                assert (await created_response.json())["job"]["id"] == "job-1"
                assert service.reruns == [
                    ("source-job", "local-profile"),
                    ("source-job", "local-profile"),
                ]

    asyncio.run(run())


def test_tagging_routes_reject_non_object_json() -> None:
    async def run() -> None:
        app = web.Application()
        app["tagging_service"] = _TaggingServiceStub()
        setup_tagging_routes(app)

        async with TestServer(app) as server:
            async with TestClient(server) as client:
                response = await client.put("/api/captioning/settings", json=["invalid"])
                payload = await response.json()
                assert response.status == 400
                assert payload["ok"] is False
                assert "JSON object" in payload["error"]

                rerun_response = await client.post("/api/captioning/jobs/job-1/rerun", json=["invalid"])
                assert rerun_response.status == 400
                assert "JSON object" in (await rerun_response.json())["error"]

    asyncio.run(run())


def test_tagging_rerun_route_forwards_selected_image_ids_and_rejects_non_array() -> None:
    async def run() -> None:
        app = web.Application()
        service = _TaggingServiceStub()
        calls = []

        async def rerun_subset(job_id, profile_id="", item_ids=None):
            calls.append((job_id, profile_id, item_ids))
            return {"ok": True, "job": {"id": "rerun-subset", "state": "queued"}}

        service.rerun_job = rerun_subset
        app["tagging_service"] = service
        setup_tagging_routes(app)

        async with TestServer(app) as server:
            async with TestClient(server) as client:
                response = await client.post(
                    "/api/captioning/jobs/source-job/rerun",
                    json={"profile_id": "local-profile", "item_ids": ["item-2", "item-1"]},
                )
                assert response.status == 202
                assert calls == [("source-job", "local-profile", ["item-2", "item-1"])]

                invalid = await client.post(
                    "/api/captioning/jobs/source-job/rerun",
                    json={"item_ids": "item-1"},
                )
                assert invalid.status == 400
                assert "item_ids" in (await invalid.json())["error"]

    asyncio.run(run())


def test_prompt_preset_delete_maps_service_validation_error_to_bad_request() -> None:
    async def run() -> None:
        app = web.Application()
        service = _TaggingServiceStub()

        def reject_builtin(_preset_id):
            raise ValueError("内置提示词预设不可删除")

        service.delete_prompt_preset = reject_builtin
        app["tagging_service"] = service
        setup_tagging_routes(app)

        async with TestServer(app) as server:
            async with TestClient(server) as client:
                response = await client.delete("/api/captioning/prompt-presets/builtin-detailed")
                payload = await response.json()
                assert response.status == 400
                assert payload == {"ok": False, "error": "内置提示词预设不可删除"}

    asyncio.run(run())


def test_rerun_and_translation_routes_keep_not_found_and_conflict_statuses() -> None:
    async def run() -> None:
        app = web.Application()
        service = _TaggingServiceStub()

        async def missing_job(_job_id, _profile_id=""):
            raise KeyError("任务不存在")

        async def missing_dictionary(_tags, _target_language):
            raise RuntimeError("请先下载本地中英标签词典")

        service.rerun_job = missing_job
        service.translate_tags = missing_dictionary
        app["tagging_service"] = service
        setup_tagging_routes(app)

        async with TestServer(app) as server:
            async with TestClient(server) as client:
                rerun_response = await client.post(
                    "/api/captioning/jobs/missing/rerun",
                    json={"profile_id": "local-profile"},
                )
                assert rerun_response.status == 404

                translation_response = await client.post(
                    "/api/captioning/translate-tags",
                    json={"tags": ["1girl"], "target_language": "zh"},
                )
                assert translation_response.status == 409
                assert "先下载" in (await translation_response.json())["error"]

    asyncio.run(run())
