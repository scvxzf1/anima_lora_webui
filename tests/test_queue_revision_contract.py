"""Bulk queue commands act only on the confirmed snapshot, with legacy support."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

from tests.training_resume_test_support import _patch_queue_storage
from web.routes.training import handle_queue_cancel_waiting
from web.services.training.queue_revision import QueueRevisionConflict
from web.services.training_service import TrainingService


@pytest.fixture
def service(tmp_path, monkeypatch):
    _patch_queue_storage(monkeypatch, tmp_path)
    svc = TrainingService(web.Application())
    svc._queue = {"paused": True, "items": [
        {"id": "queued", "state": "queued"}, {"id": "done", "state": "done"},
        {"id": "canceled", "state": "canceled"}, {"id": "error", "state": "error"},
    ]}
    monkeypatch.setattr(svc, "_save_queue", Mock())
    monkeypatch.setattr(svc, "_broadcast_queue", AsyncMock())
    monkeypatch.setattr(svc, "_stop_unlocked", AsyncMock())
    return svc


@pytest.mark.parametrize("method", ["cancel_waiting_queue_items", "cancel_all_queue_items",
    "abort_queue_after_current", "force_abort_queue", "clear_finished_queue_items",
    "clear_completed_queue_items", "clear_canceled_queue_items"])
def test_bulk_command_rejects_a_changed_snapshot_without_side_effects(service, method):
    revision = service.get_queue_snapshot()["revision"]
    service._queue_items().append({"id": "new", "state": "queued"})
    before = service.get_queue_snapshot()
    with pytest.raises(QueueRevisionConflict):
        asyncio.run(getattr(service, method)(expected_revision=revision))
    assert service.get_queue_snapshot() == before
    service._save_queue.assert_not_called()
    service._broadcast_queue.assert_not_awaited()
    service._stop_unlocked.assert_not_awaited()


def test_revision_is_stable_until_scope_identity_or_policy_changes(service):
    first = service.get_queue_snapshot()["revision"]
    assert service.get_queue_snapshot()["revision"] == first
    service.current_task_id = "new-task"
    second = service.get_queue_snapshot()["revision"]
    assert first != second
    service._queue["auto_retry"] = True
    assert service.get_queue_snapshot()["revision"] != second


@pytest.mark.parametrize("method", ["abort_queue_after_current", "force_abort_queue", "cancel_all_queue_items"])
def test_locked_command_rechecks_after_waiting_for_launch_lock(service, method):
    async def run():
        revision = service.get_queue_snapshot()["revision"]
        await service._launch_lock.acquire()
        pending = asyncio.create_task(getattr(service, method)(expected_revision=revision))
        await asyncio.sleep(0)
        service.current_task_id = "replacement-task"
        service._launch_lock.release()
        with pytest.raises(QueueRevisionConflict):
            await pending
        service._stop_unlocked.assert_not_awaited()
        service._save_queue.assert_not_called()
    asyncio.run(run())


def test_matching_revision_preserves_unrelated_records(service):
    revision = service.get_queue_snapshot()["revision"]
    result = asyncio.run(service.cancel_waiting_queue_items(expected_revision=revision))
    assert result["canceled"] == 1
    assert result["revision"] != revision
    assert [(item["id"], item["state"]) for item in result["items"]] == [
        ("queued", "canceled"), ("done", "done"), ("canceled", "canceled"), ("error", "error"),
    ]


@pytest.mark.parametrize("body,status,kwargs", [
    (None, 200, {}), ('{"expected_revision":"v1"}', 200, {"expected_revision": "v1"}),
    ("{}", 400, None), ("null", 400, None), ("[]", 400, None), ("{", 400, None),
    ('{"expected_revision":false}', 400, None), ('{"expected_revision":" "}', 400, None),
])
def test_http_revision_body_and_legacy_contract(body, status, kwargs):
    async def run():
        command = AsyncMock(return_value={"ok": True})
        app = web.Application()
        app["training_service"] = SimpleNamespace(cancel_waiting_queue_items=command)
        app.router.add_post("/command", handle_queue_cancel_waiting)
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/command", data=body)
            assert response.status == status
            if kwargs is None:
                command.assert_not_awaited()
            else:
                command.assert_awaited_once_with(**kwargs)
    asyncio.run(run())


def test_http_conflict_does_not_report_success():
    async def run():
        command = AsyncMock(side_effect=QueueRevisionConflict("队列已发生变化"))
        app = web.Application()
        app["training_service"] = SimpleNamespace(cancel_waiting_queue_items=command)
        app.router.add_post("/command", handle_queue_cancel_waiting)
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/command", json={"expected_revision": "old"})
            assert response.status == 409
            assert await response.json() == {"ok": False, "error": "队列已发生变化"}
    asyncio.run(run())
