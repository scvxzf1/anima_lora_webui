"""Minimal HTTP contract smoke for backend WebUI routes."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from aiohttp import web

from web.routes import settings as settings_routes
from web.routes import training as training_routes


class _FakeTrainingService:
    def get_status_snapshot(self) -> dict[str, Any]:
        return {"ok": True, "status": "idle", "running": False}

    def get_queue_snapshot(self) -> dict[str, Any]:
        return {
            "ok": True,
            "paused": False,
            "failure_policy": "pause",
            "auto_retry": False,
            "max_attempts": 1,
            "retry_backoff_sec": 0,
            "items": [],
            "summary": {
                "total": 0,
                "queued": 0,
                "running": 0,
                "done": 0,
                "error": 0,
                "canceled": 0,
            },
        }


class _FakeRequest:
    def __init__(
        self,
        *,
        app: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        query: dict[str, str] | None = None,
    ) -> None:
        self.app = app or {}
        self._payload = payload or {}
        self.query = query or {}

    async def json(self) -> dict[str, Any]:
        return self._payload


def _json_payload(response: web.Response) -> dict[str, Any]:
    import json

    return json.loads(response.text or "{}")


def test_http_training_status_contract():
    app = {"training_service": _FakeTrainingService()}
    response = asyncio.run(training_routes.handle_status(_FakeRequest(app=app)))  # type: ignore[arg-type]
    assert response.status == 200
    payload = _json_payload(response)
    assert payload["ok"] is True
    assert payload["status"] == "idle"


def test_http_training_queue_contract():
    app = {"training_service": _FakeTrainingService()}
    response = asyncio.run(training_routes.handle_queue_status(_FakeRequest(app=app)))  # type: ignore[arg-type]
    assert response.status == 200
    payload = _json_payload(response)
    assert payload["ok"] is True
    assert "items" in payload
    assert "failure_policy" in payload


def test_http_preflight_contract(monkeypatch):
    monkeypatch.setattr(
        training_routes,
        "preflight_training_config",
        lambda *args, **kwargs: {
            "ok": True,
            "variant": "lora",
            "preset": "default",
            "methods_subdir": "gui-methods",
            "summary": {"errors": 0, "warnings": 0, "checks": 0},
            "checks": [],
            "errors": [],
            "warnings": [],
        },
    )
    response = asyncio.run(
        training_routes.handle_preflight(
            _FakeRequest(
                payload={
                    "variant": "lora",
                    "preset": "default",
                    "methods_subdir": "gui-methods",
                }
            )  # type: ignore[arg-type]
        )
    )
    assert response.status == 200
    payload = _json_payload(response)
    assert payload["ok"] is True
    assert "checks" in payload


def test_http_global_settings_contract(monkeypatch):
    monkeypatch.setattr(
        settings_routes,
        "get_global_settings",
        lambda: {
            "ok": True,
            "output_root": "output/runs",
            "configs_root": "configs",
            "history_root": "configs/web-training-history",
            "queue_root": "configs/web-training-queue",
        },
    )
    response = asyncio.run(settings_routes.handle_global_settings_get(_FakeRequest()))  # type: ignore[arg-type]
    assert response.status == 200
    payload = _json_payload(response)
    assert payload["ok"] is True
    assert "output_root" in payload
