"""Duration overrides must never waive checkpoint integrity checks."""

import asyncio
from unittest.mock import AsyncMock, Mock

from aiohttp import web
import pytest

from tests.training_resume_test_support import _patch_queue_storage, _write_resume_history
from web.services import config_service, training_service
from web.services.training import history_resume, queue_enqueue
from web.services.training_service import TrainingService
from web.services.training.runtime_resume import _apply_resume_duration_overrides


@pytest.mark.parametrize("entrypoint", ["resume_from_history_task", "enqueue_resume_from_history_task"])
def test_incomplete_completed_checkpoint_rejected_before_launch(tmp_path, monkeypatch, entrypoint):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    _patch_queue_storage(monkeypatch, tmp_path)
    (state_dir / "optimizer.bin").unlink()
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    monkeypatch.setattr(config_service, "estimate_training_steps", lambda *args, **kwargs: {"total_steps": 42})
    clone = Mock(side_effect=AssertionError("must not clone incomplete state"))
    monkeypatch.setattr(history_resume, "_clone_resume_runtime", clone)
    monkeypatch.setattr(queue_enqueue, "_clone_frozen_runtime_config", clone)
    svc = TrainingService(web.Application())
    start = AsyncMock(side_effect=AssertionError("must not launch"))
    monkeypatch.setattr(svc, "start", start)

    with pytest.raises(ValueError, match="optimizer.bin"):
        asyncio.run(getattr(svc, entrypoint)(task_id, str(state_dir), duration_overrides={"max_train_steps": 100}))

    clone.assert_not_called()
    start.assert_not_called()
    assert not svc._queue.get("items")


@pytest.mark.parametrize("integrity", [None, {}, {"complete": True}, {"ok": False}])
def test_target_override_requires_real_integrity_contract(integrity):
    checkpoint = {"step": 42, "target_total_steps": 42, "unavailable_reason": "unavailable", "state_integrity": integrity}
    with pytest.raises(ValueError, match="unavailable"):
        history_resume._ensure_resume_checkpoint_available(checkpoint, allow_completed_by_duration_override=True)


def test_complete_schedule_free_checkpoint_can_override_target(tmp_path, monkeypatch):
    history_dir, task_id, state_dir = _write_resume_history(tmp_path)
    _patch_queue_storage(monkeypatch, tmp_path)
    snapshot = history_dir / task_id / "config.snapshot.toml"
    snapshot.write_text(snapshot.read_text() + '\noptimizer_type = "AdamWScheduleFree"\n')
    (state_dir / "scheduler.bin").unlink()
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    monkeypatch.setattr(config_service, "estimate_training_steps", lambda *args, **kwargs: {"total_steps": 42})
    svc = TrainingService(web.Application())
    _, selected, _, _ = svc._build_resume_payload(task_id, str(state_dir), duration_overrides={"max_train_steps": 100})
    assert selected["state_integrity"]["ok"] is True
    assert selected["state_integrity"]["scheduler"] is False


def test_frontend_absolute_target_conversion_matches_backend_append_contract():
    config = {"max_train_steps": 42}
    result = _apply_resume_duration_overrides(config, [], resume_step=42,
        duration_overrides={"max_train_steps": 200 - 42})
    assert config["max_train_steps"] == 200
    assert result["append_steps"] == 158
    assert result["target_total_steps"] == 200
