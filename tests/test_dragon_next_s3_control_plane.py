"""S3 control-plane evidence: isolated queue/history storage and launcher identity."""

from __future__ import annotations

import asyncio
import json

from aiohttp import web
import pytest

from tests.training_resume_test_support import _patch_queue_storage, _write_group_task
from web.services import training_service
from web.services.training.queue_revision import QueueRevisionConflict
from web.services.training_service import TrainingService


def test_s3_queue_snapshot_survives_service_restart(tmp_path, monkeypatch):
    queue_dir = _patch_queue_storage(monkeypatch, tmp_path)
    queue_dir.mkdir(parents=True, exist_ok=True)
    (queue_dir / "queue.json").write_text(
        json.dumps({
            "paused": False,
            "failure_policy": "continue",
            "items": [{"id": "s3-queued", "state": "queued", "variant": "queued"}],
        }),
        encoding="utf-8",
    )

    service = TrainingService(web.Application())
    before_revision = service.get_queue_snapshot()["revision"]
    updated = asyncio.run(service.set_queue_settings(paused=True, failure_policy="pause"))
    assert updated["paused"] is True
    assert updated["failure_policy"] == "pause"
    revision_after = updated["revision"]
    assert revision_after != before_revision

    queue_file = queue_dir / "queue.json"
    assert queue_file.exists()
    persisted = json.loads(queue_file.read_text(encoding="utf-8"))
    assert persisted["paused"] is True
    assert persisted["failure_policy"] == "pause"
    assert [item["id"] for item in persisted["items"]] == ["s3-queued"]

    reloaded = TrainingService(web.Application())
    snapshot = reloaded.get_queue_snapshot()
    assert snapshot["paused"] is True
    assert snapshot["failure_policy"] == "pause"
    assert [item["id"] for item in snapshot["items"]] == ["s3-queued"]
    assert snapshot["summary"] == {"total": 1, "queued": 1, "running": 0, "done": 0, "error": 0, "canceled": 0}
    assert snapshot["revision"] == revision_after

    with pytest.raises(QueueRevisionConflict):
        asyncio.run(reloaded.cancel_waiting_queue_items(expected_revision=before_revision))


def test_s3_stop_rejects_stale_launcher_identity(tmp_path, monkeypatch):
    _patch_queue_storage(monkeypatch, tmp_path)
    service = TrainingService(web.Application())
    service._queue = {"paused": False, "items": [{"id": "s3-running", "state": "running"}]}
    service._queue_paused = False
    service._current_queue_item_id = "s3-running"
    service.status = "running"
    service.current_job = "training"
    service.current_task_id = "task-current"

    terminated = []

    class FakeProcess:
        pid = 123
        returncode = None

    class FakePsutilProcess:
        def children(self, recursive=True):
            return []

        def terminate(self):
            terminated.append("terminate")

    monkeypatch.setattr(training_service.psutil, "Process", lambda pid: FakePsutilProcess())
    monkeypatch.setattr(training_service.psutil, "wait_procs", lambda family, timeout: (family, []))
    service.process = FakeProcess()

    with pytest.raises(RuntimeError, match="任务已发生变化"):
        asyncio.run(service.stop(expected_task_id="task-stale"))

    assert service.status == "running"
    assert service.get_queue_snapshot()["paused"] is False
    assert service.get_queue_snapshot()["items"][0]["state"] == "running"
    assert terminated == []

    asyncio.run(service.stop(expected_task_id="task-current"))
    assert service.status == "idle"
    assert service.get_queue_snapshot()["paused"] is True
    assert service.get_queue_snapshot()["items"][0]["state"] == "canceled"
    assert service._stopping is False
    assert terminated == ["terminate"]


def test_s3_history_archive_state_persists_and_can_be_recovered(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    first = "20260925-100000-training-imported-s3-first"
    second = "20260925-100001-training-imported-s3-second"
    _write_group_task(history_dir, first, started_at=1000.0)
    _write_group_task(history_dir, second, started_at=900.0)
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)

    service = TrainingService(web.Application())
    archived = service.batch_update_history_tasks({"action": "archive", "task_ids": [first]})
    assert archived["updated"] == 1
    first_meta = json.loads((history_dir / first / "meta.json").read_text(encoding="utf-8"))
    assert first_meta["archived"] is True

    restarted = TrainingService(web.Application())
    tasks = {task["id"]: task for task in restarted.list_history_tasks(include_archived=True)}
    assert tasks[first]["archived"] is True
    assert tasks[second]["archived"] is False

    restored = restarted.batch_update_history_tasks({"action": "unarchive", "task_ids": [first]})
    assert restored["updated"] == 1
    restored_service = TrainingService(web.Application())
    visible = {task["id"]: task for task in restored_service.list_history_tasks()}
    assert visible[first]["archived"] is False
