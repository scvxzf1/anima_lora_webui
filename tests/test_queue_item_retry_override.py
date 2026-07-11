"""T-R2: per-item retry override above queue runtime policy."""

from __future__ import annotations

from pathlib import Path

from aiohttp import web

from web.services import training_service
from web.services.training.service_state import resolve_item_retry_policy
from web.services.training_service import TrainingService


def test_resolve_item_retry_policy_prefers_item_keys():
    out = resolve_item_retry_policy(
        {"auto_retry": False, "max_attempts": 5, "retry_backoff_sec": 12},
        queue_auto_retry=True,
        queue_max_attempts=2,
        queue_retry_backoff_sec=1.0,
    )
    assert out["auto_retry"] is False
    assert out["max_attempts"] == 5
    assert out["retry_backoff_sec"] == 12.0


def test_resolve_item_retry_policy_partial_override():
    out = resolve_item_retry_policy(
        {"max_attempts": 4},
        queue_auto_retry=True,
        queue_max_attempts=2,
        queue_retry_backoff_sec=3.0,
    )
    assert out["auto_retry"] is True
    assert out["max_attempts"] == 4
    assert out["retry_backoff_sec"] == 3.0


def test_resolve_item_retry_policy_missing_item_uses_queue():
    out = resolve_item_retry_policy(
        {},
        queue_auto_retry=True,
        queue_max_attempts=3,
        queue_retry_backoff_sec=9.0,
    )
    assert out == {
        "auto_retry": True,
        "max_attempts": 3,
        "retry_backoff_sec": 9.0,
    }


def test_resolve_item_retry_policy_clamps_like_queue():
    out = resolve_item_retry_policy(
        {"max_attempts": 99, "retry_backoff_sec": 99999},
        queue_auto_retry=False,
        queue_max_attempts=1,
        queue_retry_backoff_sec=0.0,
    )
    assert out["max_attempts"] == 10
    assert out["retry_backoff_sec"] == 3600.0


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


def test_item_auto_retry_false_blocks_clone_even_if_queue_enabled(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    runtime = _runtime_payload(tmp_path, "no-retry")
    svc = TrainingService(web.Application())
    svc._queue_auto_retry = True
    svc._queue_max_attempts = 5
    svc._queue_retry_backoff_sec = 0
    item = {
        "id": "q-item-off",
        "state": "error",
        "kind": "training",
        "attempt": 1,
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
        "auto_retry": False,
    }
    result = svc._maybe_auto_retry(
        item,
        reason="process_failed",
        message="CUDA OOM",
        stop_requested=False,
    )
    assert result is None


def test_item_max_attempts_override_allows_clone_when_queue_max_is_one(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    runtime = _runtime_payload(tmp_path, "more-attempts")
    svc = TrainingService(web.Application())
    svc._queue = {"paused": False, "items": []}
    svc._queue_auto_retry = True
    svc._queue_max_attempts = 1
    svc._queue_retry_backoff_sec = 0
    item = {
        "id": "q-item-more",
        "state": "error",
        "kind": "training",
        "attempt": 1,
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
        "max_attempts": 3,
        "retry_backoff_sec": 7,
    }
    result = svc._maybe_auto_retry(
        item,
        reason="process_failed",
        message="CUDA OOM",
        stop_requested=False,
    )
    assert result is not None
    assert result["attempt"] == 2
    assert result.get("max_attempts") == 3
    assert float(result.get("retry_backoff_sec") or 0) == 7.0
    assert result.get("next_run_at") is not None


def test_enqueue_training_persists_item_retry_fields(tmp_path, monkeypatch):
    _patch_queue_paths(tmp_path, monkeypatch)
    svc = TrainingService(web.Application())
    svc._queue = {"paused": True, "items": []}
    svc._queue_paused = True

    async def _fake_prepare(*_args, **_kwargs):
        raise AssertionError("should not prepare runtime when requires_preprocess=False")

    monkeypatch.setattr(
        "web.services.training.queue_enqueue._prepare_web_runtime_config",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no prepare")),
    )

    import asyncio

    payload = asyncio.run(
        svc.enqueue_training(
            "demo",
            "default",
            "imported",
            requires_preprocess=False,
            start_paused=True,
            auto_retry=False,
            max_attempts=4,
            retry_backoff_sec=15,
        )
    )
    assert payload["ok"] is True
    item = payload["item"]
    assert item["auto_retry"] is False
    assert item["max_attempts"] == 4
    assert float(item["retry_backoff_sec"]) == 15.0
    # also present in snapshot items
    snap_item = next(i for i in payload["items"] if i["id"] == item["id"])
    assert snap_item["max_attempts"] == 4

