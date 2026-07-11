"""Failure classification for queue auto-retry decisions."""

from __future__ import annotations

from web.services.training.anomalies import (
    classify_training_failure,
    should_auto_retry_failure,
)


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
    kind = classify_training_failure(reason="error", message="checkpoint missing for resume")
    assert kind == "checkpoint_missing"
    assert should_auto_retry_failure(kind) is False
