"""Legacy learning-rate fields normalize to the canonical history API field."""

import json

import pytest

from web.services import training_service
from web.services.training.progress_parser import metric_from_progress_jsonl_event, normalize_metric_record


@pytest.mark.parametrize(
    "record,expected",
    [
        ({"learningRate": 0.001}, 0.001),
        ({"learning_rate": 0.002}, 0.002),
        ({"lr": 0, "learningRate": 0.001}, 0.0),
        ({"lr": "bad", "learningRate": 0.001, "learning_rate": 0.002}, 0.001),
        ({"lr": float("nan"), "learning_rate": 0.002}, 0.002),
        ({"lr": True, "learning_rate": 0.002}, 0.002),
    ],
)
def test_normalize_metric_record_uses_legacy_lr_aliases(record, expected):
    assert normalize_metric_record(record) == {"lr": expected}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True])
def test_normalize_metric_record_rejects_invalid_lr_without_alias(value):
    assert normalize_metric_record({"lr": value}) is None


def test_progress_event_lr_alias_does_not_change_existing_precedence():
    assert metric_from_progress_jsonl_event(
        {"ev": "step", "global_step": 1, "lr": 0.0, "learningRate": 0.1, "lr/unet": 0.2}, 1.0
    )["lr"] == 0.0
    assert metric_from_progress_jsonl_event(
        {"ev": "step", "global_step": 1, "learningRate": 0.1, "lr/unet": 0.2}, 1.0
    )["lr"] == 0.1
    assert metric_from_progress_jsonl_event(
        {"ev": "step", "global_step": 1, "lr": float("nan"), "learning_rate": 0.15}, 1.0
    )["lr"] == 0.15
    assert metric_from_progress_jsonl_event(
        {"ev": "step", "global_step": 1, "lr/unet": 0.2, "lr/group0": 0.3}, 1.0
    )["lr"] == 0.2


def test_history_detail_normalizes_metrics_jsonl_lr_aliases(tmp_path, monkeypatch):
    history_dir = tmp_path / "history"
    task_dir = history_dir / "task"
    task_dir.mkdir(parents=True)
    monkeypatch.setattr(training_service, "HISTORY_DIR", history_dir)
    (task_dir / "meta.json").write_text(
        json.dumps({"id": "task", "job": "training", "state": "idle", "started_at": 1000.0}),
        encoding="utf-8",
    )
    (task_dir / "metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"step": 1, "loss": 0.5, "learningRate": 0.001}),
                json.dumps({"step": 2, "loss": 0.4, "learning_rate": 0.0005}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = training_service._load_history_task("task")

    assert [item["lr"] for item in payload["metrics"]] == [0.001, 0.0005]
    assert all("learningRate" not in item and "learning_rate" not in item for item in payload["metrics"])
