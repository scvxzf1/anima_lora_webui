"""In-memory optimistic candidate transactions; no image or TXT fixtures."""
import asyncio
from copy import deepcopy

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from web.routes.tagging import setup_tagging_routes
from web.services.tagging.batch_update import update_candidates


def job():
    return {"state": "completed", "items": [{"id": str(i), "proposed_caption": " red\nblue ", "state": "ready"} for i in range(2)]}


def update(key="0", expected=" red\nblue ", text="green"):
    return {"id": key, "expected_caption": expected, "proposed_caption": text}


@pytest.mark.parametrize("updates,error", [([update(), update("1", "stale")], RuntimeError), ([update(), update()], ValueError), ([update(), update("absent")], ValueError), ([update(text="")], ValueError), ([update(text="x" * 100001)], ValueError), ([], ValueError), ([update()] * 501, ValueError)])
def test_validation_is_all_or_nothing(updates, error):
    value = job()
    before = deepcopy(value)
    with pytest.raises(error):
        update_candidates(value, updates)
    assert value == before


def test_exact_bytes_and_terminal_state():
    value = job()
    assert update_candidates(value, [update(text="  green\nblue  ")]) == 1
    assert value["items"][0]["proposed_caption"] == "  green\nblue  "
    value["state"] = "running"
    with pytest.raises(RuntimeError):
        update_candidates(value, [update("1")])


def test_both_alias_routes_and_conflict_status():
    class Service:
        def __init__(self):
            self.value = job()
        def update_items(self, job_id, updates):
            update_candidates(self.value, updates)
            return {"ok": True, "job": self.value}
    async def run():
        app = web.Application()
        app["tagging_service"] = Service()
        setup_tagging_routes(app)
        async with TestClient(TestServer(app)) as client:
            for prefix in ("captioning", "tagging"):
                response = await client.patch(f"/api/{prefix}/jobs/test/items", json={"items": [update("1", text=" blue ")]})
                assert response.status == (200 if prefix == "captioning" else 409)
            assert app["tagging_service"].value["items"][0]["proposed_caption"] == " red\nblue "
    asyncio.run(run())
