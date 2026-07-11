"""Lock training_policy defaults vs queue.json runtime overrides + clamp."""

from __future__ import annotations

import json
from pathlib import Path

import toml
from aiohttp import web

from web.services import settings_service, training_service
from web.services.training_service import TrainingService


def _patch_queue_and_settings(tmp_path: Path, monkeypatch):
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir(parents=True)
    history_dir = tmp_path / "history"
    history_dir.mkdir(parents=True)
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    settings_file.parent.mkdir(parents=True)
    settings_file.write_text("[global]\noutput_root = \"output/runs\"\n", encoding="utf-8")

    monkeypatch.setattr(training_service, "QUEUE_DIR", queue_dir)
    monkeypatch.setattr(training_service, "QUEUE_FILE", queue_dir / "queue.json")
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    monkeypatch.setattr(settings_service, "ROOT", tmp_path)
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", settings_file)
    return queue_dir / "queue.json", settings_file


def test_queue_seeds_retry_policy_when_keys_missing(tmp_path, monkeypatch):
    queue_file, _settings = _patch_queue_and_settings(tmp_path, monkeypatch)
    # queue.json without retry keys
    queue_file.write_text(json.dumps({"paused": False, "items": []}), encoding="utf-8")
    settings_service.save_training_policy(
        {
            "auto_retry": True,
            "max_attempts": 4,
            "retry_backoff_sec": 12.5,
        }
    )

    svc = TrainingService(web.Application())
    assert svc._queue_auto_retry is True
    assert svc._queue_max_attempts == 4
    assert svc._queue_retry_backoff_sec == 12.5
    # File should still not invent keys on load alone
    raw = json.loads(queue_file.read_text(encoding="utf-8"))
    assert "auto_retry" not in raw
    assert "max_attempts" not in raw
    assert "retry_backoff_sec" not in raw


def test_queue_runtime_keys_not_overwritten_by_policy_change(tmp_path, monkeypatch):
    queue_file, _settings = _patch_queue_and_settings(tmp_path, monkeypatch)
    queue_file.write_text(
        json.dumps(
            {
                "paused": False,
                "auto_retry": False,
                "max_attempts": 2,
                "retry_backoff_sec": 1.0,
                "items": [],
            }
        ),
        encoding="utf-8",
    )
    settings_service.save_training_policy(
        {
            "auto_retry": True,
            "max_attempts": 9,
            "retry_backoff_sec": 99.0,
        }
    )

    svc = TrainingService(web.Application())
    assert svc._queue_auto_retry is False
    assert svc._queue_max_attempts == 2
    assert svc._queue_retry_backoff_sec == 1.0


def test_queue_settings_clamp_match_policy_bounds(tmp_path, monkeypatch):
    queue_file, _settings = _patch_queue_and_settings(tmp_path, monkeypatch)
    queue_file.write_text(json.dumps({"paused": False, "items": []}), encoding="utf-8")
    svc = TrainingService(web.Application())
    import asyncio
    snapshot = asyncio.run(svc.set_queue_settings(max_attempts=999, retry_backoff_sec=99999))
    assert snapshot["max_attempts"] == 10
    assert snapshot["retry_backoff_sec"] == 3600.0
    raw = json.loads(queue_file.read_text(encoding="utf-8"))
    assert raw["max_attempts"] == 10
    assert float(raw["retry_backoff_sec"]) == 3600.0
