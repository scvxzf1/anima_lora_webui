"""Auto-retry failure classification and queue integration contracts."""

from __future__ import annotations

import asyncio
from pathlib import Path

from aiohttp import web

from web.services import training_service
from web.services.training_service import TrainingService
from web.services.training.anomalies import classify_training_failure, should_auto_retry_failure


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
        "\n".join(
            [
                f'output_dir = "{(run_dir / "training_output").as_posix()}"',
                f'logging_dir = "{(run_dir / "model_cache" / "logs").as_posix()}"',
                'source_image_dir = "image_dataset/a"',
            ]
        ),
        encoding="utf-8",
    )
    return {
        "runtime_config_file": runtime_config.as_posix(),
        "run_dir": run_dir.as_posix(),
    }


def test_user_stop_does_not_auto_retry_clone(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    runtime = _runtime_payload(tmp_path, "stop-run")
    svc = TrainingService(web.Application())
    svc._queue = {
        "paused": False,
        "failure_policy": "continue",
        "auto_retry": True,
        "max_attempts": 3,
        "retry_backoff_sec": 0,
        "items": [
            {
                "id": "q1",
                "state": "running",
                "kind": "training",
                "requires_preprocess": False,
                "variant": "demo",
                "preset": "default",
                "methods_subdir": "imported",
                "runtime_config_file": runtime["runtime_config_file"],
                "source_config_file": "configs/imported/source.toml",
                "extra_args": [],
                "gpu_whitelist": [0],
                "continue_info": {},
                "resume_info": {},
                "history_task_ids": ["h1"],
                "attempt": 1,
            }
        ],
    }
    svc._queue_paused = False
    svc._queue_failure_policy = "continue"
    svc._queue_auto_retry = True
    svc._queue_max_attempts = 3
    svc._queue_retry_backoff_sec = 0
    svc._current_queue_item_id = "q1"
    svc.status = "running"
    svc.current_job = "training"
    svc._stop_requested = True

    class FakeStdout:
        async def read(self, _size):
            return b""

    class FakeProcess:
        stdout = FakeStdout()

        async def wait(self):
            return 1

    svc.process = FakeProcess()
    asyncio.run(svc._read_output())
    items = svc.get_queue_snapshot()["items"]
    assert not any(item.get("retry_of") == "q1" for item in items)
    q1 = next(item for item in items if item["id"] == "q1")
    assert q1["state"] in {"canceled", "error", "done"} or q1["state"] != "queued"


def test_checkpoint_missing_launch_failure_does_not_retry(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue_auto_retry = True
    svc._queue_max_attempts = 3
    svc._queue_retry_backoff_sec = 0
    item = {
        "id": "q-ckpt",
        "state": "error",
        "kind": "training",
        "attempt": 1,
        "variant": "demo",
        "preset": "default",
        "methods_subdir": "imported",
        "runtime_config_file": "x.toml",
        "source_config_file": "y.toml",
        "extra_args": [],
        "gpu_whitelist": [],
        "continue_info": {},
        "resume_info": {},
        "history_task_ids": [],
    }
    svc._queue = {"paused": False, "items": [item], "auto_retry": True, "max_attempts": 3, "retry_backoff_sec": 0}
    kind = classify_training_failure(
        reason="launch_failure",
        message="续训检查点状态已不存在，请重新选择包含 train_state.json 的状态目录",
    )
    assert kind == "checkpoint_missing"
    assert should_auto_retry_failure(kind) is False
    result = svc._maybe_auto_retry(
        item,
        reason="launch_failure",
        message="续训检查点状态已不存在，请重新选择包含 train_state.json 的状态目录",
    )
    assert result is None
    assert not any(i.get("retry_of") == "q-ckpt" for i in svc._queue_items())


def test_pause_failure_policy_clones_but_stays_paused(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    runtime = _runtime_payload(tmp_path, "pause-run")
    svc = TrainingService(web.Application())
    svc._queue = {
        "paused": False,
        "failure_policy": "pause",
        "auto_retry": True,
        "max_attempts": 3,
        "retry_backoff_sec": 0,
        "items": [
            {
                "id": "q1",
                "state": "running",
                "kind": "training",
                "requires_preprocess": False,
                "variant": "demo",
                "preset": "default",
                "methods_subdir": "imported",
                "runtime_config_file": runtime["runtime_config_file"],
                "source_config_file": "configs/imported/source.toml",
                "extra_args": [],
                "gpu_whitelist": [0],
                "continue_info": {},
                "resume_info": {},
                "history_task_ids": ["h1"],
                "attempt": 1,
            }
        ],
    }
    svc._queue_paused = False
    svc._queue_failure_policy = "pause"
    svc._queue_auto_retry = True
    svc._queue_max_attempts = 3
    svc._queue_retry_backoff_sec = 0
    svc._current_queue_item_id = "q1"
    svc.status = "running"
    svc.current_job = "training"
    start_calls: list[str] = []
    monkeypatch.setattr(svc, "start", lambda *a, **k: start_calls.append("start") or {"ok": True})
    dispatch_calls: list[bool] = []
    monkeypatch.setattr(svc, "_schedule_queue_dispatch", lambda: dispatch_calls.append(True))

    class FakeStdout:
        async def read(self, _size):
            return b""

    class FakeProcess:
        stdout = FakeStdout()

        async def wait(self):
            return 7

    svc.process = FakeProcess()
    asyncio.run(svc._read_output())
    snapshot = svc.get_queue_snapshot()
    retries = [item for item in snapshot["items"] if item.get("retry_of") == "q1"]
    assert len(retries) == 1
    assert retries[0]["state"] == "queued"
    assert snapshot["paused"] is True
    # With pause policy, auto-dispatch may be scheduled but start should not run immediately
    # unless unpaused; at least queue remains paused.
    assert start_calls == []


def test_classify_user_stop_never_retries():
    kind = classify_training_failure(reason="user_stop", stop_requested=True)
    assert kind == "user_stop"
    assert should_auto_retry_failure(kind) is False


def test_classify_checkpoint_missing_never_retries():
    kind = classify_training_failure(
        reason="launch_failure",
        message="续训检查点状态已不存在，请重新选择包含 train_state.json 的状态目录",
    )
    assert kind == "checkpoint_missing"
    assert should_auto_retry_failure(kind) is False


def test_classify_oom_can_retry():
    kind = classify_training_failure(message="CUDA out of memory at step 12")
    assert kind == "oom"
    assert should_auto_retry_failure(kind) is True


def test_classify_process_exit_can_retry():
    kind = classify_training_failure(reason="process_exit", returncode=7)
    assert kind == "process_exit"
    assert should_auto_retry_failure(kind) is True


def test_classify_english_checkpoint_missing():
    kind = classify_training_failure(
        reason="error",
        message="checkpoint missing for resume",
    )
    assert kind == "checkpoint_missing"
    assert should_auto_retry_failure(kind) is False
