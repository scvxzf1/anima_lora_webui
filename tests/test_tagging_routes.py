from __future__ import annotations

import asyncio

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from web.routes.tagging import setup_tagging_routes


class _TaggingServiceStub:
    def __init__(self):
        self.saved = None

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

    async def cancel_job(self, job_id):
        return {"ok": True, "job": {"id": job_id, "state": "canceled"}}

    def update_item(self, job_id, item_id, text):
        return {"ok": True, "job": {"id": job_id}, "item_id": item_id, "text": text}

    async def commit_job(self, job_id, *, all_items=False, item_ids=None):
        return {"ok": True, "job": {"id": job_id}, "all": all_items, "item_ids": item_ids or []}


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

    asyncio.run(run())
