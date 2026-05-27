from __future__ import annotations

import asyncio
import json
from pathlib import Path

from aiohttp import web

from web.routes import training as training_routes
from web.services import training_service
from web.services.training_service import TrainingService


class _FakeJsonRequest:
    def __init__(self, payload: dict, app: dict | None = None, match_info: dict | None = None):
        self._payload = payload
        self.app = app or {}
        self.match_info = match_info or {}
        self.query = {}

    async def json(self):
        return self._payload


def _patch_queue_paths(tmp_path: Path, monkeypatch):
    queue_dir = tmp_path / "queue"
    monkeypatch.setattr(training_service, "QUEUE_DIR", queue_dir)
    monkeypatch.setattr(training_service, "QUEUE_FILE", queue_dir / "queue.json")
    monkeypatch.setattr(training_service, "HISTORY_DIR", tmp_path / "history")
    return queue_dir


def _runtime_payload(tmp_path: Path, name: str = "demo") -> dict:
    run_dir = tmp_path / "runs" / name
    runtime_config = run_dir / "config.runtime.toml"
    for path in (
        run_dir / "model_cache",
        run_dir / "dataset_cache",
        run_dir / "training_output",
        run_dir / "training_output" / "sample",
        run_dir / "model_cache" / "logs",
    ):
        path.mkdir(parents=True, exist_ok=True)
    runtime_config.write_text(
        "\n".join([
            f'output_dir = "{(run_dir / "training_output").as_posix()}"',
            f'logging_dir = "{(run_dir / "model_cache" / "logs").as_posix()}"',
            'source_image_dir = "image_dataset/a"',
            'resized_image_dir = "resized/a"',
            'lora_cache_dir = "cache/a"',
        ]),
        encoding="utf-8",
    )
    return {
        "run_dir": str(run_dir),
        "runtime_config_file": str(runtime_config),
        "original_config_file": str(run_dir / "config.original.toml"),
        "dataset_config_file": str(run_dir / "dataset.runtime.toml"),
        "output_dir": str(run_dir / "training_output"),
        "sample_dir": str(run_dir / "training_output" / "sample"),
        "model_cache_dir": str(run_dir / "model_cache"),
        "dataset_cache_dir": str(run_dir / "dataset_cache"),
        "training_output_dir": str(run_dir / "training_output"),
        "logs_dir": str(run_dir / "model_cache" / "logs"),
        "history_source_config_file": "configs/imported/source.toml",
        "sample_config": {},
        "data_dirs": {},
    }


def test_enqueue_training_freezes_runtime_config_while_running(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    runtime = _runtime_payload(tmp_path)
    monkeypatch.setattr(training_service, "_prepare_web_runtime_config", lambda *args, **kwargs: runtime)
    svc = TrainingService(web.Application())
    svc.status = "running"

    payload = asyncio.run(svc.enqueue_training(
        "demo",
        "default",
        "imported",
        config_file="configs/imported/source.toml",
        requires_preprocess=True,
        gpu_whitelist=["1", "bad", 1],
    ))

    assert payload["ok"] is True
    item = payload["item"]
    assert item["state"] == "queued"
    assert item["runtime_config_file"] == runtime["runtime_config_file"]
    assert item["source_config_file"] == "configs/imported/source.toml"
    assert item["gpu_whitelist"] == [1]


def test_queue_move_and_cancel_waiting_items(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue = {
        "paused": True,
        "items": [
            {"id": "a", "state": "queued"},
            {"id": "b", "state": "queued"},
            {"id": "c", "state": "done"},
        ],
    }
    svc._queue_paused = True

    asyncio.run(svc.move_queue_item("b", "up"))
    assert [item["id"] for item in svc.get_queue_snapshot()["items"][:2]] == ["b", "a"]

    asyncio.run(svc.cancel_queue_item("a"))
    item = next(item for item in svc.get_queue_snapshot()["items"] if item["id"] == "a")
    assert item["state"] == "canceled"


def test_queue_startup_repairs_stale_running_item(tmp_path, monkeypatch):
    queue_dir = _patch_queue_paths(tmp_path, monkeypatch)
    queue_dir.mkdir(parents=True)
    (queue_dir / "queue.json").write_text(
        json.dumps({
            "paused": False,
            "items": [
                {"id": "old", "state": "running"},
                {"id": "next", "state": "queued"},
            ],
        }),
        encoding="utf-8",
    )

    svc = TrainingService(web.Application())
    items = svc.get_queue_snapshot()["items"]
    assert items[0]["state"] == "error"
    assert items[1]["state"] == "queued"


def test_stop_running_queue_item_cancels_and_pauses(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue = {"paused": False, "items": [{"id": "q1", "state": "running"}]}
    svc._queue_paused = False
    svc._current_queue_item_id = "q1"
    svc.status = "running"
    svc.current_job = "training"

    class FakeProcess:
        pid = 123
        returncode = None

    class FakePsutilProcess:
        def children(self, recursive=True):
            return []

        def terminate(self):
            return None

    monkeypatch.setattr(training_service.psutil, "Process", lambda pid: FakePsutilProcess())
    monkeypatch.setattr(training_service.psutil, "wait_procs", lambda family, timeout: (family, []))
    svc.process = FakeProcess()

    asyncio.run(svc.stop())

    item = svc.get_queue_snapshot()["items"][0]
    assert svc.get_queue_snapshot()["paused"] is True
    assert item["state"] == "canceled"


def test_handle_queue_start_uses_enqueue_service(monkeypatch):
    class FakeService:
        def __init__(self):
            self.calls = []

        async def enqueue_training(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {"ok": True, "items": [{"id": "q"}], "paused": False}

    svc = FakeService()
    monkeypatch.setattr(training_routes, "preflight_training_config", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(training_routes, "is_web_runtime_config", lambda value: value.endswith("config.runtime.toml"))
    req = _FakeJsonRequest(
        {
            "variant": "demo",
            "preset": "default",
            "methods_subdir": "imported",
            "config_file": "output/runs/demo/config.runtime.toml",
            "confirmed": True,
            "gpu_whitelist": [0],
        },
        {"training_service": svc},
    )

    response = asyncio.run(training_routes.handle_queue_start(req))

    assert response.status == 200
    assert len(svc.calls) == 1
    args, kwargs = svc.calls[0]
    assert args[:3] == ("demo", "default", "imported")
    assert kwargs["requires_preprocess"] is False
    assert kwargs["gpu_whitelist"] == [0]


def test_handle_queue_resume_uses_history_checkpoint_service():
    class FakeService:
        async def enqueue_resume_from_history_task(self, task_id, checkpoint=None, *, gpu_whitelist=None):
            return {
                "ok": True,
                "task_id": task_id,
                "checkpoint": checkpoint,
                "gpu_whitelist": gpu_whitelist,
                "items": [],
                "paused": False,
            }

    req = _FakeJsonRequest(
        {"task_id": "task-a", "checkpoint": "state-dir", "gpu_whitelist": [1]},
        {"training_service": FakeService()},
    )

    response = asyncio.run(training_routes.handle_queue_resume(req))

    assert response.status == 200
    payload = json.loads(response.text)
    assert payload["task_id"] == "task-a"
    assert payload["checkpoint"] == "state-dir"
    assert payload["gpu_whitelist"] == [1]
