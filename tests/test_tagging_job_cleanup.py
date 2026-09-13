import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from web.routes.tagging import setup_tagging_routes
from web.services.tagging.jobs import TaggingJobManager


def test_cleanup_preserves_busy_jobs_and_files(tmp_path):
    async def run():
        manager = TaggingJobManager()
        caption = tmp_path / "image.txt"
        caption.write_text("original")
        for key, state in [("done", "completed"), ("failed", "failed"),
                           ("active", "running"), ("queued", "queued"),
                           ("commit", "completed"), ("rerun", "completed"),
                           ("tail", "completed")]:
            manager.jobs[key] = {"state": state, "_path": caption}
            manager._cancel_events[key] = asyncio.Event()
            manager._commit_locks[key] = asyncio.Lock()
            manager._rerun_locks[key] = asyncio.Lock()
        await manager._commit_locks["commit"].acquire()
        await manager._rerun_locks["rerun"].acquire()
        tail = asyncio.create_task(asyncio.sleep(0))
        manager._tasks["tail"] = tail
        result = manager.clear_finished(list(manager.jobs) + ["done", "missing"])
        assert result["removed"] == ["done", "failed"]
        assert result["skipped"] == ["active", "queued", "commit", "rerun", "tail"]
        for mapping in (manager.jobs, manager._cancel_events, manager._commit_locks, manager._rerun_locks):
            assert "done" not in mapping
        assert caption.read_text() == "original"
        assert manager.clear_finished(["done"])["removed"] == []
        await tail
    asyncio.run(run())


@pytest.mark.parametrize("value", [None, "all", [1], {}])
def test_cleanup_requires_explicit_ids(value):
    with pytest.raises(ValueError):
        TaggingJobManager().clear_finished(value)


def test_cleanup_routes():
    class Service:
        def clear_finished_jobs(self, ids):
            return TaggingJobManager().clear_finished(ids)

    async def run():
        app = web.Application()
        app["tagging_service"] = Service()
        setup_tagging_routes(app)
        async with TestClient(TestServer(app)) as client:
            for prefix in ("/api/captioning", "/api/tagging"):
                response = await client.post(prefix + "/jobs/cleanup", json={"job_ids": ["missing"]})
                assert response.status == 200
                assert (await response.json())["removed"] == []
                response = await client.post(prefix + "/jobs/cleanup", json={})
                assert response.status == 400
    asyncio.run(run())
