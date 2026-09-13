"""Training summaries must not consume validation CMMD as loss."""

import json

import pytest

from web.services.training.history_store import _history_metric_summary, _history_summary


@pytest.mark.parametrize("filename,marker", [("metrics.jsonl", "kind"), ("progress.jsonl", "ev")])
def test_training_summary_ignores_validation(tmp_path, filename, marker):
    events = [
        {marker: "step", "step": 10, "loss": 0.2},
        {marker: "val", "step": 11, "loss": 0.91, "cmmd": 0.91},
    ]
    (tmp_path / filename).write_text("\n".join(map(json.dumps, events)), encoding="utf-8")
    assert _history_metric_summary(tmp_path, 2) == {
        "final_loss": 0.2, "last_step": 10, "loss_preview": [0.2],
    }


def test_validation_only_metrics_can_fall_back_to_training_progress(tmp_path):
    (tmp_path / "metrics.jsonl").write_text(json.dumps({"kind": "val", "loss": 0.91}), encoding="utf-8")
    assert _history_metric_summary(tmp_path, 1) == {"final_loss": None, "loss_preview": []}
    (tmp_path / "progress.jsonl").write_text(json.dumps({"ev": "step", "global_step": 4, "loss": 0}), encoding="utf-8")
    assert _history_metric_summary(tmp_path, 1) == {"last_step": 4, "final_loss": 0, "loss_preview": [0]}


@pytest.mark.parametrize("count", [0, 1])
def test_validation_only_replaces_stale_metadata_loss_without_writing_files(tmp_path, count):
    path = tmp_path / "metrics.jsonl"
    content = json.dumps({"kind": "val", "step": 10, "loss": 0.91})
    path.write_text(content, encoding="utf-8")
    meta = {"job": "training", "final_loss": 0.91, "loss_preview": [0.91], "metric_count": count}
    summary = _history_summary(meta, tmp_path)
    assert summary["final_loss"] is None
    assert summary["loss_preview"] == []
    assert meta["final_loss"] == 0.91
    assert path.read_text(encoding="utf-8") == content


def test_summary_only_legacy_record_preserves_values_when_event_files_are_missing(tmp_path):
    summary = _history_summary({"job": "training", "final_loss": 0.2, "last_step": 10}, tmp_path)
    assert summary["final_loss"] == 0.2
    assert summary["last_step"] == 10
