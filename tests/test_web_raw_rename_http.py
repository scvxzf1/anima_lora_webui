"""Raw rename validates request bodies before touching configuration files."""

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from web.routes import config as routes


@pytest.mark.parametrize("body", [
    "{", "null", "[]", '"file.toml"', "42", "{}",
    '{"source": [], "target": "b.toml"}',
    '{"source": "a.toml", "target": 1}',
    '{"source": "  ", "target": "b.toml"}',
])
def test_raw_rename_rejects_invalid_body(monkeypatch, body):
    def unexpected_rename(*args):
        pytest.fail("invalid requests must not reach the file service")

    monkeypatch.setattr(routes, "rename_raw_file", unexpected_rename)

    async def run():
        app = web.Application()
        app.router.add_post("/rename", routes.handle_raw_rename)
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/rename", data=body, headers={"Content-Type": "application/json"})
            assert response.status == 400
            assert (await response.json())["ok"] is False

    asyncio.run(run())


@pytest.mark.parametrize("success, status", [(True, 200), (False, 400)])
def test_raw_rename_preserves_service_contract(monkeypatch, success, status):
    calls = []

    def rename(source, target):
        calls.append((source, target))
        return success, "result"

    monkeypatch.setattr(routes, "rename_raw_file", rename)

    async def run():
        app = web.Application()
        app.router.add_post("/rename", routes.handle_raw_rename)
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/rename", json={"source": "a.toml", "target": "b.toml"})
            assert response.status == status
            payload = await response.json()
            assert payload["ok"] is success
            if success:
                assert payload["file"] == "b.toml"

    asyncio.run(run())
    assert calls == [("a.toml", "b.toml")]
