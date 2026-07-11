"""T-R10 light: manual retry marks clone and resets attempt."""

from __future__ import annotations

import asyncio
from pathlib import Path

from aiohttp import web

from web.services import training_service
from web.services.training_service import TrainingService


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
    return {"runtime_config_file": runtime_config.as_posix(), "run_dir": run_dir.as_posix()}


def test_manual_retry_marks_item_and_resets_attempt(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    runtime = _runtime_payload(tmp_path)
    svc = TrainingService(web.Application())
    svc._queue = {
        "paused": True,
        "items": [
            {
                "id": "q-failed",
                "state": "error",
                "kind": "training",
                "attempt": 3,
                "variant": "demo",
                "preset": "default",
                "methods_subdir": "imported",
                "runtime_config_file": runtime["runtime_config_file"],
                "source_config_file": "configs/imported/source.toml",
                "extra_args": [],
                "gpu_whitelist": [],
                "continue_info": {},
                "resume_info": {},
                "requires_preprocess": False,
                "message": "boom",
            }
        ],
    }
    svc._queue_paused = True
    monkeypatch.setattr(svc, "_schedule_queue_dispatch", lambda: None)

    payload = asyncio.run(svc.retry_queue_item("q-failed"))
    assert payload["ok"] is True
    retry = payload["item"]
    assert retry["retry_of"] == "q-failed"
    assert retry.get("manual_retry") is True
    assert retry.get("retry_source") == "manual"
    assert int(retry.get("attempt") or 0) == 1
    assert retry.get("state") == "queued"
    assert "next_run_at" not in retry or retry.get("next_run_at") in (None, 0, "")
