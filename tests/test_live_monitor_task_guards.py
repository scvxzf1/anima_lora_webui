from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from web.routes.training import handle_logs, handle_metrics, handle_stop
from web.services.training import launcher_job


def test_stop_checks_task_identity_after_acquiring_launch_lock(monkeypatch):
    stopped = AsyncMock()
    monkeypatch.setattr(launcher_job, "_stop_unlocked", stopped)

    async def run():
        lock = asyncio.Lock()
        service = SimpleNamespace(_launch_lock=lock, current_task_id="old")
        await lock.acquire()
        pending = asyncio.create_task(launcher_job.stop(service, "old"))
        await asyncio.sleep(0)
        service.current_task_id = "new"
        lock.release()
        with pytest.raises(RuntimeError, match="任务已发生变化"):
            await pending
        stopped.assert_not_awaited()

    asyncio.run(run())


def test_stop_accepts_matching_task_and_legacy_call(monkeypatch):
    stopped = AsyncMock()
    monkeypatch.setattr(launcher_job, "_stop_unlocked", stopped)

    async def run():
        service = SimpleNamespace(_launch_lock=asyncio.Lock(), current_task_id="same")
        await launcher_job.stop(service, "same")
        await launcher_job.stop(service)
        assert stopped.await_count == 2

    asyncio.run(run())


def test_stop_http_body_passes_expected_task_and_returns_conflict():
    stop = AsyncMock(side_effect=RuntimeError("任务已发生变化"))
    request = SimpleNamespace(
        app={"training_service": SimpleNamespace(stop=stop)},
        query={},
        can_read_body=True,
        json=AsyncMock(return_value={"task_id": "old"}),
    )
    with pytest.raises(web.HTTPConflict):
        asyncio.run(handle_stop(request))
    stop.assert_awaited_once_with(expected_task_id="old")


def test_stop_http_legacy_request_keeps_no_argument_call():
    stop = AsyncMock()
    request = SimpleNamespace(
        app={"training_service": SimpleNamespace(stop=stop)},
        query={},
        can_read_body=False,
    )
    response = asyncio.run(handle_stop(request))
    assert response.status == 200
    stop.assert_awaited_once_with()


def test_metrics_rejects_stale_task_query():
    service = SimpleNamespace(
        get_status_snapshot=lambda: {"task_id": "new"},
        get_metrics_history=AsyncMock(),
    )
    request = SimpleNamespace(app={"training_service": service}, query={"task_id": "old"})
    with pytest.raises(web.HTTPConflict):
        asyncio.run(handle_metrics(request))
    service.get_metrics_history.assert_not_awaited()


def test_logs_rejects_invalid_pagination_query():
    service = SimpleNamespace(
        get_status_snapshot=lambda: {"task_id": "same"},
        get_log_records=AsyncMock(),
    )
    request = SimpleNamespace(
        app={"training_service": service},
        query={"task_id": "same", "limit": "bad"},
    )
    with pytest.raises(web.HTTPBadRequest):
        asyncio.run(handle_logs(request))


@pytest.mark.parametrize("payload", [{}, [], None, {"task_id": ""}, {"task_id": "  "},
                                      {"task_id": 3}, {"task_id": ["old"]}])
def test_stop_rejects_body_without_valid_identity(payload):
    stop = AsyncMock()
    request = SimpleNamespace(
        app={"training_service": SimpleNamespace(stop=stop)}, query={},
        can_read_body=True, json=AsyncMock(return_value=payload),
    )
    with pytest.raises(web.HTTPBadRequest):
        asyncio.run(handle_stop(request))
    stop.assert_not_awaited()


def test_stop_rejects_invalid_json_instead_of_legacy_stop():
    stop = AsyncMock()
    request = SimpleNamespace(
        app={"training_service": SimpleNamespace(stop=stop)}, query={},
        can_read_body=True, json=AsyncMock(side_effect=ValueError("invalid json")),
    )
    with pytest.raises(web.HTTPBadRequest):
        asyncio.run(handle_stop(request))
    stop.assert_not_awaited()


def test_stop_rejects_conflicting_query_and_body():
    stop = AsyncMock()
    request = SimpleNamespace(
        app={"training_service": SimpleNamespace(stop=stop)}, query={"task_id": "new"},
        can_read_body=True, json=AsyncMock(return_value={"task_id": "old"}),
    )
    with pytest.raises(web.HTTPBadRequest):
        asyncio.run(handle_stop(request))
    stop.assert_not_awaited()


@pytest.mark.parametrize("body,query,status,expected", [
    (None, "", 200, None),
    ('{"task_id":"run-A"}', "", 200, "run-A"),
    (None, "?task_id=run-A", 200, "run-A"),
    ("{}", "", 400, None),
    ("[]", "", 400, None),
    ("{", "", 400, None),
    ('{"task_id":"run-A"}', "?task_id=run-B", 400, None),
    (None, "?task_id=", 400, None),
])
def test_stop_real_http_body_contract(body, query, status, expected):
    async def run():
        stop = AsyncMock()
        app = web.Application()
        app["training_service"] = SimpleNamespace(stop=stop)
        app.router.add_post("/stop", handle_stop)
        async with TestClient(TestServer(app)) as client:
            response = await client.post(f"/stop{query}", data=body)
            assert response.status == status
            if status == 400:
                stop.assert_not_awaited()
            elif expected:
                stop.assert_awaited_once_with(expected_task_id=expected)
            else:
                stop.assert_awaited_once_with()
    asyncio.run(run())
