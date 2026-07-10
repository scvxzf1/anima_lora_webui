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
