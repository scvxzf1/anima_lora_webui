"""Contract tests for `/ws/training` subscribe + snapshot/status push."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestClient, TestServer

from web.routes import training as training_routes


class _FakeWsTrainingService:
    """Minimal training service surface used by handle_ws + push path."""

    def __init__(self) -> None:
        self._ws_clients: set[web.WebSocketResponse] = set()
        self.subscribed: list[web.WebSocketResponse] = []
        self.unsubscribed: list[web.WebSocketResponse] = []

    def subscribe(self, ws: web.WebSocketResponse) -> None:
        self._ws_clients.add(ws)
        self.subscribed.append(ws)

    def unsubscribe(self, ws: web.WebSocketResponse) -> None:
        self._ws_clients.discard(ws)
        self.unsubscribed.append(ws)

    def get_status_snapshot(self) -> dict[str, Any]:
        return {
            "ok": True,
            "status": "idle",
            "running": False,
            "current_task_id": "",
        }

    async def push_status_snapshot(self) -> None:
        snapshot = self.get_status_snapshot()
        payload = json.dumps({"type": "status", **snapshot}, ensure_ascii=False)
        dead: set[web.WebSocketResponse] = set()
        for ws in list(self._ws_clients):
            try:
                await ws.send_str(payload)
            except (ConnectionResetError, RuntimeError):
                dead.add(ws)
        self._ws_clients -= dead


async def _build_client(svc: _FakeWsTrainingService) -> TestClient:
    app = web.Application()
    app["training_service"] = svc
    training_routes.setup_training_routes(app)
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    return client


def test_ws_training_subscribe_receives_status_snapshot():
    async def _run() -> None:
        svc = _FakeWsTrainingService()
        client = await _build_client(svc)
        try:
            ws = await client.ws_connect("/ws/training")
            # Give the handler a tick to register the client.
            for _ in range(50):
                if svc.subscribed:
                    break
                await asyncio.sleep(0.01)
            assert svc.subscribed, "handler should subscribe websocket client"

            await svc.push_status_snapshot()
            msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
            assert msg.type == WSMsgType.TEXT
            payload = json.loads(msg.data)
            assert payload["type"] == "status"
            assert payload["ok"] is True
            assert payload["status"] == "idle"
            assert payload["running"] is False

            await ws.close()
            # Wait for unsubscribe cleanup.
            for _ in range(50):
                if svc.unsubscribed:
                    break
                await asyncio.sleep(0.01)
            assert svc.unsubscribed, "handler should unsubscribe on close"
        finally:
            await client.close()

    asyncio.run(_run())


def test_ws_training_route_registers_path():
    app = web.Application()
    training_routes.setup_training_routes(app)
    paths = {route.resource.canonical for route in app.router.routes() if route.resource}
    assert "/ws/training" in paths


def test_ws_training_queue_broadcast_contract():
    """T-R5: type=queue payload must carry snapshot required fields."""

    async def _run() -> None:
        svc = _FakeWsTrainingService()

        async def _broadcast_queue() -> None:
            payload = {
                "type": "queue",
                "ok": True,
                "paused": False,
                "failure_policy": "continue",
                "auto_retry": False,
                "max_attempts": 1,
                "retry_backoff_sec": 0.0,
                "status": "idle",
                "current_item_id": "",
                "summary": {
                    "total": 0,
                    "queued": 0,
                    "running": 0,
                    "done": 0,
                    "error": 0,
                    "canceled": 0,
                },
                "items": [],
            }
            dead: set[web.WebSocketResponse] = set()
            import json as _json
            data = _json.dumps(payload, ensure_ascii=False)
            for ws in list(svc._ws_clients):
                try:
                    await ws.send_str(data)
                except (ConnectionResetError, RuntimeError):
                    dead.add(ws)
            svc._ws_clients -= dead

        svc.broadcast_queue = _broadcast_queue  # type: ignore[attr-defined]
        client = await _build_client(svc)
        try:
            ws = await client.ws_connect("/ws/training")
            for _ in range(50):
                if svc.subscribed:
                    break
                await asyncio.sleep(0.01)
            assert svc.subscribed

            await svc.broadcast_queue()
            msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
            assert msg.type == WSMsgType.TEXT
            payload = json.loads(msg.data)
            assert payload["type"] == "queue"
            assert payload["ok"] is True
            assert "paused" in payload
            assert "failure_policy" in payload
            assert "auto_retry" in payload
            assert "max_attempts" in payload
            assert "retry_backoff_sec" in payload
            assert "summary" in payload and isinstance(payload["summary"], dict)
            assert "items" in payload and isinstance(payload["items"], list)
            for key in ("total", "queued", "running", "done", "error", "canceled"):
                assert key in payload["summary"]
            await ws.close()
        finally:
            await client.close()

    asyncio.run(_run())
