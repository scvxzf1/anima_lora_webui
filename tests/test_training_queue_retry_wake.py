"""Queue dispatch wake scheduling for next_run_at backoff."""

from __future__ import annotations

import asyncio
import time

from aiohttp import web

from tests.test_training_queue import _patch_queue_paths
from web.services.training_service import TrainingService


def test_queue_dispatch_wakes_after_next_run_at(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    future = time.time() + 0.05
    svc._queue = {
        "paused": False,
        "failure_policy": "continue",
        "auto_retry": True,
        "max_attempts": 3,
        "retry_backoff_sec": 30,
        "items": [
            {
                "id": "later",
                "state": "queued",
                "next_run_at": future,
                "variant": "demo",
                "preset": "default",
                "methods_subdir": "imported",
            },
        ],
    }
    svc._queue_paused = False
    started: list[str] = []

    async def fake_start(item):
        started.append(str(item.get("id")))
        item["state"] = "running"

    async def fake_broadcast():
        return None

    monkeypatch.setattr(svc, "_start_queue_item", fake_start)
    monkeypatch.setattr(svc, "_broadcast_queue", fake_broadcast)

    async def run():
        await svc._dispatch_queue()
        assert started == []
        deadline = time.time() + 1.0
        while not started and time.time() < deadline:
            await asyncio.sleep(0.01)
        assert started == ["later"]

    asyncio.run(run())


def test_queue_dispatch_reschedules_single_wake_timer(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    future = time.time() + 30.0
    svc._queue = {
        "paused": False,
        "items": [
            {
                "id": "later",
                "state": "queued",
                "next_run_at": future,
                "variant": "demo",
                "preset": "default",
                "methods_subdir": "imported",
            },
        ],
    }
    svc._queue_paused = False

    async def fake_start(item):
        raise AssertionError("future item must not start yet")

    async def fake_broadcast():
        return None

    monkeypatch.setattr(svc, "_start_queue_item", fake_start)
    monkeypatch.setattr(svc, "_broadcast_queue", fake_broadcast)

    async def run():
        await svc._dispatch_queue()
        first = getattr(svc, "_queue_dispatch_wake_handle", None)
        assert first is not None
        await svc._dispatch_queue()
        second = getattr(svc, "_queue_dispatch_wake_handle", None)
        assert second is not None
        assert first.cancelled() or first is second
        if first is not second:
            assert not second.cancelled()
        second.cancel()
        svc._queue_dispatch_wake_handle = None

    asyncio.run(run())
