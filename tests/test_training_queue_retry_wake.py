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


def test_queue_launch_failure_can_auto_retry(tmp_path, monkeypatch):
    """Launch failures should share process-fail auto_retry path."""
    from tests.test_training_queue import _runtime_payload
    from web.services import training_service

    _patch_queue_paths(tmp_path, monkeypatch)
    retry_root = tmp_path / "retry-runs"
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: retry_root)
    runtime = _runtime_payload(tmp_path, "launch-fail-run")

    svc = TrainingService(web.Application())
    svc._queue = {
        "paused": False,
        "failure_policy": "continue",
        "auto_retry": True,
        "max_attempts": 2,
        "retry_backoff_sec": 0,
        "items": [
            {
                "id": "q-launch",
                "state": "queued",
                "kind": "training",
                "requires_preprocess": False,
                "variant": "demo",
                "preset": "default",
                "methods_subdir": "imported",
                "runtime_config_file": runtime["runtime_config_file"],
                "source_config_file": "configs/imported/source.toml",
                "extra_args": [],
                "gpu_whitelist": [],
                "continue_info": {},
                "resume_info": {},
                "attempt": 1,
            }
        ],
    }
    svc._queue_paused = False
    svc._queue_failure_policy = "continue"
    svc._queue_auto_retry = True
    svc._queue_max_attempts = 2
    svc._queue_retry_backoff_sec = 0

    async def boom(item):
        raise RuntimeError("simulated launch failure")

    async def fake_broadcast():
        return None

    monkeypatch.setattr(svc, "_start_queue_item", boom)
    monkeypatch.setattr(svc, "_broadcast_queue", fake_broadcast)
    # Prevent immediate re-dispatch so this test locks clone semantics only.
    schedule_calls: list[bool] = []
    monkeypatch.setattr(svc, "_schedule_queue_dispatch", lambda: schedule_calls.append(True))

    asyncio.run(svc._dispatch_queue())
    items = svc._queue_items()
    assert any(item.get("id") == "q-launch" and item.get("state") == "error" for item in items)
    retries = [item for item in items if item.get("retry_of") == "q-launch"]
    assert len(retries) == 1
    assert retries[0]["attempt"] == 2
    assert retries[0]["state"] == "queued"
    assert schedule_calls  # launch fail path still asks for another dispatch


def test_queue_launch_failure_skips_retry_when_auto_retry_disabled(tmp_path, monkeypatch):
    from tests.test_training_queue import _runtime_payload
    from web.services import training_service

    _patch_queue_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: tmp_path / "retry-runs")
    runtime = _runtime_payload(tmp_path, "launch-fail-off")
    svc = TrainingService(web.Application())
    svc._queue = {
        "paused": False,
        "failure_policy": "continue",
        "auto_retry": False,
        "max_attempts": 3,
        "retry_backoff_sec": 0,
        "items": [
            {
                "id": "q-off",
                "state": "queued",
                "kind": "training",
                "requires_preprocess": False,
                "variant": "demo",
                "preset": "default",
                "methods_subdir": "imported",
                "runtime_config_file": runtime["runtime_config_file"],
                "source_config_file": "configs/imported/source.toml",
                "extra_args": [],
                "attempt": 1,
            }
        ],
    }
    svc._queue_paused = False
    svc._queue_failure_policy = "continue"
    svc._queue_auto_retry = False
    svc._queue_max_attempts = 3

    async def boom(item):
        raise RuntimeError("no retry")

    async def fake_broadcast():
        return None

    monkeypatch.setattr(svc, "_start_queue_item", boom)
    monkeypatch.setattr(svc, "_broadcast_queue", fake_broadcast)
    asyncio.run(svc._dispatch_queue())
    assert not any(item.get("retry_of") == "q-off" for item in svc._queue_items())

