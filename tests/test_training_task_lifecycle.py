from __future__ import annotations

import asyncio
from pathlib import Path

from aiohttp import web

from web.services import training_service
from web.services.training.task_lifecycle import track_run_task
from web.services.training_service import TrainingService


def _service(tmp_path: Path, monkeypatch) -> TrainingService:
    queue_dir = tmp_path / "queue"
    monkeypatch.setattr(training_service, "QUEUE_DIR", queue_dir)
    monkeypatch.setattr(training_service, "QUEUE_FILE", queue_dir / "queue.json")
    monkeypatch.setattr(training_service, "HISTORY_DIR", tmp_path / "history")
    return TrainingService(web.Application())


def test_stop_holds_launch_lock_until_output_reader_finishes(
    tmp_path, monkeypatch
) -> None:
    svc = _service(tmp_path, monkeypatch)

    async def run() -> None:
        terminated = asyncio.Event()
        finish_reader = asyncio.Event()
        order: list[str] = []

        class FakeProcess:
            pid = 123
            returncode = None

        class FakePsutilProcess:
            def children(self, recursive=True):
                return []

            def terminate(self):
                terminated.set()

        monkeypatch.setattr(
            training_service.psutil, "Process", lambda pid: FakePsutilProcess()
        )
        monkeypatch.setattr(
            training_service.psutil, "wait_procs", lambda family, timeout: (family, [])
        )

        async def output_reader() -> None:
            await finish_reader.wait()
            order.append("reader_finished")

        async def fake_start_unlocked(*args, **kwargs) -> None:
            order.append("start_entered")
            svc.status = "running"

        svc.process = FakeProcess()
        svc.status = "running"
        svc.current_job = "training"
        svc._run_generation = 1
        output_task = asyncio.create_task(output_reader())
        svc._output_task = output_task
        svc._output_task_generation = 1
        svc._job_tasks = {1: {output_task}}
        monkeypatch.setattr(svc, "_start_unlocked", fake_start_unlocked)

        stop_task = asyncio.create_task(svc.stop())
        await terminated.wait()
        start_task = asyncio.create_task(svc.start("manual", "default"))
        await asyncio.sleep(0)

        assert start_task.done() is False
        assert order == []

        finish_reader.set()
        await stop_task
        await start_task

        assert order == ["reader_finished", "start_entered"]

    asyncio.run(run())


def test_stale_output_reader_cannot_clear_new_run_state(tmp_path, monkeypatch) -> None:
    svc = _service(tmp_path, monkeypatch)

    async def run() -> None:
        wait_started = asyncio.Event()
        release_wait = asyncio.Event()

        class FakeStdout:
            async def read(self, _size):
                return b""

        class OldProcess:
            stdout = FakeStdout()

            async def wait(self):
                wait_started.set()
                await release_wait.wait()
                return 0

        old_process = OldProcess()
        new_process = object()
        svc.process = old_process
        svc.status = "running"
        svc.current_job = "old-training"
        svc._current_queue_item_id = "old-queue"
        svc._run_generation = 1

        reader = asyncio.create_task(
            svc._read_output(process=old_process, generation=1)
        )
        await wait_started.wait()

        svc._run_generation = 2
        svc.process = new_process
        svc.status = "running"
        svc.current_job = "new-training"
        svc._current_queue_item_id = "new-queue"
        release_wait.set()
        await reader

        assert svc.status == "running"
        assert svc.current_job == "new-training"
        assert svc._current_queue_item_id == "new-queue"

    asyncio.run(run())


def test_shutdown_cancels_tracked_idle_background_tasks(tmp_path, monkeypatch) -> None:
    svc = _service(tmp_path, monkeypatch)

    async def run() -> None:
        started = asyncio.Event()

        async def background() -> None:
            started.set()
            await asyncio.Event().wait()

        task = track_run_task(svc, background(), generation=1)
        await started.wait()
        await svc.shutdown()

        assert task.cancelled()
        assert svc._job_tasks == {}

    asyncio.run(run())


def test_stop_does_not_deadlock_with_pending_training_transition(
    tmp_path, monkeypatch
) -> None:
    svc = _service(tmp_path, monkeypatch)

    async def run() -> None:
        reader_reached_cleanup = asyncio.Event()
        release_reader = asyncio.Event()
        pending_starts: list[dict] = []

        class FakeStdout:
            async def read(self, _size):
                return b""

        class FakeProcess:
            stdout = FakeStdout()
            returncode = None

            async def wait(self):
                self.returncode = 0
                return 0

        async def pause_ingest(*, final=False) -> None:
            assert final is True
            reader_reached_cleanup.set()
            await release_reader.wait()

        async def no_broadcast(_payload) -> None:
            return None

        async def record_pending_start(pending: dict) -> None:
            pending_starts.append(pending)

        process = FakeProcess()
        svc.process = process
        svc.status = "running"
        svc.current_job = "preprocess"
        svc._run_generation = 1
        svc._pending_train_after_preprocess = {"variant": "manual"}
        monkeypatch.setattr(svc, "_ingest_progress_jsonl", pause_ingest)
        monkeypatch.setattr(svc, "_broadcast", no_broadcast)
        monkeypatch.setattr(svc, "_finish_history_task", lambda **kwargs: None)
        monkeypatch.setattr(svc, "_start_pending_training", record_pending_start)
        monkeypatch.setattr(svc, "_schedule_queue_dispatch", lambda: None)

        output_task = track_run_task(
            svc,
            svc._read_output(process=process, generation=1),
            generation=1,
            output=True,
        )
        await reader_reached_cleanup.wait()

        stop_task = asyncio.create_task(svc.stop())
        while not svc._stopping:
            await asyncio.sleep(0)
        release_reader.set()
        await asyncio.wait_for(stop_task, timeout=1.0)

        assert output_task.done()
        assert pending_starts == []
        assert svc.status == "idle"
        assert svc._stopping is False

    asyncio.run(run())


def test_shutdown_cancels_queue_dispatcher_and_wake_timer(
    tmp_path, monkeypatch
) -> None:
    svc = _service(tmp_path, monkeypatch)

    async def run() -> None:
        dispatcher_started = asyncio.Event()

        async def dispatcher() -> None:
            dispatcher_started.set()
            await asyncio.Event().wait()

        dispatch_task = asyncio.create_task(dispatcher())
        svc._queue_dispatch_task = dispatch_task
        wake_handle = asyncio.get_running_loop().call_later(60, lambda: None)
        svc._queue_dispatch_wake_handle = wake_handle
        await dispatcher_started.wait()

        await svc.shutdown()

        assert dispatch_task.cancelled()
        assert wake_handle.cancelled()
        assert svc._queue_dispatch_task is None
        assert svc._queue_dispatch_wake_handle is None
        assert svc._shutting_down is True

        svc._queue_paused = False
        svc._queue["items"] = [{"id": "queued", "state": "queued"}]
        svc._schedule_queue_dispatch()
        assert svc._queue_dispatch_task is None

    asyncio.run(run())
