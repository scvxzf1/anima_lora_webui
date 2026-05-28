from __future__ import annotations

import asyncio
import json
from pathlib import Path

from aiohttp import web
import pytest
import toml

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
    assert svc.get_queue_snapshot()["paused"] is True


def test_queue_startup_dispatches_when_unpaused_and_clean(tmp_path, monkeypatch):
    queue_dir = _patch_queue_paths(tmp_path, monkeypatch)
    queue_dir.mkdir(parents=True)
    (queue_dir / "queue.json").write_text(
        json.dumps({"paused": False, "items": [{"id": "next", "state": "queued"}]}),
        encoding="utf-8",
    )
    svc = TrainingService(web.Application())
    called = {"dispatch": False}

    async def fake_dispatch():
        called["dispatch"] = True

    monkeypatch.setattr(svc, "_dispatch_queue", fake_dispatch)

    async def run():
        await svc.start_queue_on_startup()
        await asyncio.sleep(0)

    asyncio.run(run())

    assert called["dispatch"] is True


def test_queue_launch_guard_blocks_manual_start_during_startup_window(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue = {"paused": False, "items": [{"id": "q1", "state": "queued"}]}
    svc._queue_paused = False
    checked = {"manual_start_rejected": False}
    launched = {"item_id": ""}

    async def fake_broadcast_queue():
        if checked["manual_start_rejected"]:
            return
        with pytest.raises(RuntimeError, match="已有任务在运行中"):
            await svc.start("manual", "default")
        checked["manual_start_rejected"] = True

    async def fake_start_queue_item(item):
        launched["item_id"] = item["id"]

    monkeypatch.setattr(svc, "_broadcast_queue", fake_broadcast_queue)
    monkeypatch.setattr(svc, "_start_queue_item", fake_start_queue_item)

    asyncio.run(svc._dispatch_queue())

    assert checked["manual_start_rejected"] is True
    assert launched["item_id"] == "q1"
    assert svc._queue_launching_item_id == ""


def test_queue_state_recovers_from_backup_when_main_file_is_corrupt(tmp_path, monkeypatch):
    queue_dir = _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue = {"paused": False, "items": [{"id": "q1", "state": "queued"}]}

    svc._save_queue()

    backup_file = queue_dir / "queue.json.bak"
    assert backup_file.is_file()
    (queue_dir / "queue.json").write_text("{broken", encoding="utf-8")

    recovered = training_service._load_training_queue_state()

    assert recovered["items"][0]["id"] == "q1"
    restored = json.loads((queue_dir / "queue.json").read_text(encoding="utf-8"))
    assert restored["items"][0]["id"] == "q1"


def test_queue_history_metadata_is_written_on_launch(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue = {
        "paused": True,
        "items": [{
            "id": "q1",
            "state": "running",
            "kind": "training",
            "retry_of": "old-q",
            "attempt": 3,
            "created_at": 123.0,
            "created_at_text": "2026-05-27 10:00:00",
        }],
    }

    async def fake_create_subprocess_exec(*args, **kwargs):
        return object()

    async def fake_background_task():
        return None

    monkeypatch.setattr(training_service.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(svc, "_read_output", fake_background_task)
    monkeypatch.setattr(svc, "_monitor_system", fake_background_task)

    asyncio.run(svc._launch_job(
        ["python", "-c", "pass"],
        {},
        variant="demo",
        preset="default",
        methods_subdir="imported",
        output_dir=str(tmp_path / "out"),
        sample_dir=str(tmp_path / "out" / "sample"),
        data_dirs={},
        sample_config={},
        job="preprocess",
        start_message="queued",
        command_label="queued",
        queue_item_id="q1",
    ))

    meta = json.loads((Path(svc.current_task_dir) / "meta.json").read_text(encoding="utf-8"))
    assert meta["from_queue"] is True
    assert meta["queue_item_id"] == "q1"
    assert meta["queue_retry_of"] == "old-q"
    assert meta["queue_attempt"] == 3


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


def test_queue_process_error_pauses_and_keeps_next_waiting(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue = {
        "paused": False,
        "failure_policy": "pause",
        "items": [
            {"id": "q1", "state": "running"},
            {"id": "q2", "state": "queued"},
        ],
    }
    svc._queue_paused = False
    svc._queue_failure_policy = "pause"
    svc._current_queue_item_id = "q1"
    svc.status = "running"
    svc.current_job = "training"

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
    assert snapshot["paused"] is True
    items = {item["id"]: item for item in snapshot["items"]}
    assert items["q1"]["state"] == "error"
    assert items["q2"]["state"] == "queued"


def test_queue_retry_clones_frozen_runtime_config(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    retry_root = tmp_path / "retry-runs"
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: retry_root)
    runtime = _runtime_payload(tmp_path, "old-run")
    svc = TrainingService(web.Application())
    svc._queue_paused = True
    svc._queue = {
        "paused": True,
        "items": [{
            "id": "q1",
            "state": "error",
            "kind": "training",
            "requires_preprocess": True,
            "variant": "demo",
            "preset": "default",
            "methods_subdir": "imported",
            "runtime_config_file": runtime["runtime_config_file"],
            "source_config_file": "configs/imported/source.toml",
            "extra_args": [],
            "gpu_whitelist": [0],
            "continue_info": {},
            "resume_info": {},
            "history_task_ids": ["old-history"],
            "attempt": 1,
        }],
    }

    payload = asyncio.run(svc.retry_queue_item("q1"))

    retry = payload["item"]
    assert retry["state"] == "queued"
    assert retry["retry_of"] == "q1"
    assert retry["attempt"] == 2
    assert retry["history_task_ids"] == []
    assert retry["runtime_config_file"] != runtime["runtime_config_file"]
    retry_cfg = toml.loads(Path(retry["runtime_config_file"]).read_text(encoding="utf-8"))
    old_cfg = toml.loads(Path(runtime["runtime_config_file"]).read_text(encoding="utf-8"))
    assert retry_cfg["output_dir"] != old_cfg["output_dir"]
    assert retry_cfg["source_image_dir"] == old_cfg["source_image_dir"]


def test_queue_retry_training_clones_dataset_cache_before_old_runtime_delete(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    output_root = tmp_path / "runs"
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: output_root)
    runtime = _runtime_payload(tmp_path, "old-run")
    old_run_dir = Path(runtime["run_dir"])
    old_resized = old_run_dir / "dataset_cache" / "dataset-01" / "resized"
    old_lora = old_run_dir / "dataset_cache" / "dataset-01" / "lora"
    old_resized.mkdir(parents=True, exist_ok=True)
    old_lora.mkdir(parents=True, exist_ok=True)
    (old_resized / "sample.png").write_text("image", encoding="utf-8")
    (old_lora / "sample.npz").write_text("cache", encoding="utf-8")
    dataset_config = old_run_dir / "dataset.runtime.toml"
    dataset_config.write_text(
        toml.dumps({
            "general": {"caption_extension": ".txt", "keep_tokens": 3},
            "datasets": [{
                "batch_size": 1,
                "subsets": [{
                    "image_dir": str(old_resized),
                    "cache_dir": str(old_lora),
                    "num_repeats": 1,
                    "custom_attributes": {"source_dir": "image_dataset/a"},
                }],
            }],
        }),
        encoding="utf-8",
    )
    runtime_config = Path(runtime["runtime_config_file"])
    cfg = toml.loads(runtime_config.read_text(encoding="utf-8"))
    cfg.update({
        "dataset_config": str(dataset_config),
        "source_image_dir": "image_dataset/a",
        "resized_image_dir": str(old_resized),
        "lora_cache_dir": str(old_lora),
    })
    runtime_config.write_text(toml.dumps(cfg), encoding="utf-8")
    runtime["dataset_config_file"] = str(dataset_config)
    runtime["data_dirs"] = {
        "source_image_dir": "image_dataset/a",
        "resized_image_dir": str(old_resized),
        "lora_cache_dir": str(old_lora),
    }

    svc = TrainingService(web.Application())
    svc._queue_paused = True
    svc._queue = {
        "paused": True,
        "items": [{
            "id": "q1",
            "state": "error",
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
            "history_task_ids": ["old-history"],
            "runtime_info": runtime,
            "attempt": 1,
        }],
    }

    payload = asyncio.run(svc.retry_queue_item("q1"))
    retry = payload["item"]
    retry_cfg = toml.loads(Path(retry["runtime_config_file"]).read_text(encoding="utf-8"))
    retry_dataset = toml.loads(Path(retry_cfg["dataset_config"]).read_text(encoding="utf-8"))
    retry_subset = retry_dataset["datasets"][0]["subsets"][0]

    assert retry_cfg["resized_image_dir"] != str(old_resized)
    assert retry_cfg["lora_cache_dir"] != str(old_lora)
    assert retry_subset["image_dir"] == retry_cfg["resized_image_dir"]
    assert retry_subset["cache_dir"] == retry_cfg["lora_cache_dir"]
    assert Path(retry_subset["image_dir"], "sample.png").read_text(encoding="utf-8") == "image"
    assert Path(retry_subset["cache_dir"], "sample.npz").read_text(encoding="utf-8") == "cache"

    deleted = asyncio.run(svc.cancel_queue_item("q1", delete_runtime=True))

    assert deleted["deleted_runtime"] is True
    assert not old_run_dir.exists()
    assert Path(retry_subset["image_dir"], "sample.png").exists()
    assert Path(retry_subset["cache_dir"], "sample.npz").exists()


def test_queue_top_bottom_cancel_waiting_and_clear_finished_keeps_error_records(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue_paused = True
    svc._queue = {
        "paused": True,
        "items": [
            {"id": "a", "state": "queued"},
            {"id": "b", "state": "queued"},
            {"id": "c", "state": "queued"},
            {"id": "d", "state": "done"},
            {"id": "e", "state": "error"},
        ],
    }

    asyncio.run(svc.move_queue_item("c", "top"))
    assert [item["id"] for item in svc.get_queue_snapshot()["items"][:3]] == ["c", "a", "b"]
    asyncio.run(svc.move_queue_item("c", "bottom"))
    assert [item["id"] for item in svc.get_queue_snapshot()["items"][:3]] == ["a", "b", "c"]

    canceled = asyncio.run(svc.cancel_waiting_queue_items())
    assert canceled["canceled"] == 3
    assert all(item["state"] != "queued" for item in svc.get_queue_snapshot()["items"])

    cleared = asyncio.run(svc.clear_finished_queue_items())
    assert cleared["removed"] == 4
    remaining = svc.get_queue_snapshot()["items"]
    assert [(item["id"], item["state"]) for item in remaining] == [("e", "error")]


def test_queue_launch_lock_serializes_manual_and_queue_start(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue = {"paused": False, "items": [{"id": "q1", "state": "queued"}]}
    svc._queue_paused = False
    ready = asyncio.Event()
    release = asyncio.Event()
    launched = []

    async def fake_start_queue_item(item):
        launched.append(item["id"])
        ready.set()
        await release.wait()
        svc.status = "running"

    monkeypatch.setattr(svc, "_start_queue_item", fake_start_queue_item)

    async def run():
        dispatch_task = asyncio.create_task(svc._dispatch_queue())
        await ready.wait()
        manual_task = asyncio.create_task(svc.start("manual", "default"))
        await asyncio.sleep(0)
        assert manual_task.done() is False
        release.set()
        await dispatch_task
        with pytest.raises(RuntimeError, match="已有任务在运行中"):
            await manual_task

    asyncio.run(run())

    assert launched == ["q1"]
    assert svc._queue_launching_item_id == ""


def test_delete_terminal_queue_item_only_removes_that_record(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue_paused = True
    svc._queue = {
        "paused": True,
        "items": [
            {"id": "waiting", "state": "queued"},
            {"id": "failed", "state": "error", "history_task_ids": ["hist-a"]},
            {"id": "done", "state": "done"},
        ],
    }

    deleted = asyncio.run(svc.cancel_queue_item("failed"))

    assert deleted["deleted"] == 1
    assert deleted["message"] == "已删除队列记录"
    assert [item["id"] for item in svc.get_queue_snapshot()["items"]] == ["waiting", "done"]


def test_delete_terminal_queue_item_can_remove_runtime_dir(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    runtime = _runtime_payload(tmp_path)
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: tmp_path / "runs")
    svc = TrainingService(web.Application())
    svc._queue_paused = True
    svc._queue = {
        "paused": True,
        "items": [
            {
                "id": "failed",
                "state": "error",
                "runtime_config_file": runtime["runtime_config_file"],
                "runtime_info": runtime,
            },
        ],
    }

    deleted = asyncio.run(svc.cancel_queue_item("failed", delete_runtime=True))

    assert deleted["deleted"] == 1
    assert deleted["deleted_runtime"] is True
    assert not Path(runtime["run_dir"]).exists()
    assert svc.get_queue_snapshot()["items"] == []


def test_delete_terminal_queue_item_marks_cleanup_before_runtime_delete_failure(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    runtime = _runtime_payload(tmp_path)
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: tmp_path / "runs")
    svc = TrainingService(web.Application())
    svc._queue_paused = True
    svc._queue = {
        "paused": True,
        "items": [
            {
                "id": "failed",
                "state": "error",
                "runtime_config_file": runtime["runtime_config_file"],
                "runtime_info": runtime,
            },
        ],
    }
    saves: list[dict] = []
    original_save = svc._save_queue

    def record_save():
        saves.append(json.loads(json.dumps(svc._queue)))
        original_save()

    def fail_rmtree(path):
        raise OSError("boom")

    monkeypatch.setattr(svc, "_save_queue", record_save)
    monkeypatch.setattr(training_service.shutil, "rmtree", fail_rmtree)

    with pytest.raises(OSError, match="boom"):
        asyncio.run(svc.cancel_queue_item("failed", delete_runtime=True))

    item = svc.get_queue_snapshot()["items"][0]
    assert item["cleanup_state"] == "error"
    assert item["cleanup_error"] == "boom"
    assert Path(runtime["run_dir"]).exists()
    assert any(
        saved["items"][0].get("cleanup_state") == "deleting_runtime"
        for saved in saves
    )


def test_delete_terminal_queue_item_rejects_incomplete_runtime_marker(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    runtime = _runtime_payload(tmp_path)
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: tmp_path / "runs")
    training_service.shutil.rmtree(Path(runtime["run_dir"]) / "dataset_cache")
    svc = TrainingService(web.Application())
    svc._queue_paused = True
    svc._queue = {
        "paused": True,
        "items": [
            {
                "id": "failed",
                "state": "error",
                "runtime_config_file": runtime["runtime_config_file"],
                "runtime_info": runtime,
            },
        ],
    }

    with pytest.raises(ValueError, match="runtime 标记"):
        asyncio.run(svc.cancel_queue_item("failed", delete_runtime=True))

    assert Path(runtime["run_dir"]).exists()
    assert svc.get_queue_snapshot()["items"][0]["id"] == "failed"


def test_delete_terminal_queue_item_rejects_runtime_config_mismatch(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    runtime = _runtime_payload(tmp_path, "old-run")
    other = _runtime_payload(tmp_path, "other-run")
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: tmp_path / "runs")
    svc = TrainingService(web.Application())
    svc._queue_paused = True
    svc._queue = {
        "paused": True,
        "items": [
            {
                "id": "failed",
                "state": "error",
                "runtime_config_file": runtime["runtime_config_file"],
                "runtime_info": {
                    **runtime,
                    "run_dir": other["run_dir"],
                    "runtime_config_file": runtime["runtime_config_file"],
                },
            },
        ],
    }

    with pytest.raises(ValueError, match="runtime 配置不匹配"):
        asyncio.run(svc.cancel_queue_item("failed", delete_runtime=True))

    assert Path(other["run_dir"]).exists()
    assert svc.get_queue_snapshot()["items"][0]["id"] == "failed"


def test_delete_terminal_queue_item_rejects_runtime_outside_output_root(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    runtime = _runtime_payload(tmp_path, "outside")
    monkeypatch.setattr(training_service, "resolve_output_root", lambda: tmp_path / "other-runs")
    svc = TrainingService(web.Application())
    svc._queue_paused = True
    svc._queue = {
        "paused": True,
        "items": [
            {
                "id": "failed",
                "state": "error",
                "runtime_config_file": runtime["runtime_config_file"],
                "runtime_info": runtime,
            },
        ],
    }

    with pytest.raises(ValueError, match="输出根目录"):
        asyncio.run(svc.cancel_queue_item("failed", delete_runtime=True))

    assert Path(runtime["run_dir"]).exists()
    assert svc.get_queue_snapshot()["items"][0]["id"] == "failed"


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


def test_queue_management_routes_call_service():
    class FakeService:
        def __init__(self):
            self.calls = []

        async def set_queue_settings(self, **kwargs):
            self.calls.append(("settings", kwargs))
            return {"ok": True, "paused": kwargs.get("paused"), "failure_policy": kwargs.get("failure_policy")}

        async def retry_queue_item(self, item_id):
            self.calls.append(("retry", item_id))
            return {"ok": True, "item_id": item_id}

        async def cancel_waiting_queue_items(self):
            self.calls.append(("cancel-waiting", None))
            return {"ok": True, "canceled": 2}

        async def clear_finished_queue_items(self):
            self.calls.append(("clear", None))
            return {"ok": True, "removed": 3}

        async def cancel_queue_item(self, item_id, *, delete_runtime=False):
            self.calls.append(("cancel", item_id, delete_runtime))
            return {"ok": True, "item_id": item_id, "deleted_runtime": delete_runtime}

    svc = FakeService()
    app = {"training_service": svc}

    settings = asyncio.run(training_routes.handle_queue_settings(
        _FakeJsonRequest({"paused": True, "failure_policy": "pause"}, app)
    ))
    retry = asyncio.run(training_routes.handle_queue_retry(
        _FakeJsonRequest({}, app, {"item_id": "q1"})
    ))
    cancel = asyncio.run(training_routes.handle_queue_cancel_waiting(_FakeJsonRequest({}, app)))
    clear = asyncio.run(training_routes.handle_queue_clear(_FakeJsonRequest({}, app)))
    delete = asyncio.run(training_routes.handle_queue_cancel(
        _FakeJsonRequest({"delete_runtime": True}, app, {"item_id": "q2"})
    ))

    assert settings.status == retry.status == cancel.status == clear.status == delete.status == 200
    assert svc.calls == [
        ("settings", {"paused": True, "failure_policy": "pause"}),
        ("retry", "q1"),
        ("cancel-waiting", None),
        ("clear", None),
        ("cancel", "q2", True),
    ]
