"""Training error hints / anomaly payload tests split from test_training_resume.py."""

from __future__ import annotations

from tests import training_resume_test_support as _training_resume_support

globals().update(
    {
        name: value
        for name, value in vars(_training_resume_support).items()
        if name == "Path" or not name.startswith("__")
    }
)

# Split: anomaly

def test_training_error_classifier_detects_cuda_oom():
    text = (
        "torch.OutOfMemoryError: CUDA out of memory. "
        "Tried to allocate 64.00 MiB."
    )

    assert training_service.classify_training_error(text) == "大概率爆显存"

def test_training_error_hint_is_added_once():
    assert (
        training_service._message_with_error_hint("训练异常退出 (code=1)", "大概率爆显存")
        == "训练异常退出 (code=1)：大概率爆显存"
    )
    assert (
        training_service._message_with_error_hint(
            "训练异常退出 (code=1)：大概率爆显存",
            "大概率爆显存",
        )
        == "训练异常退出 (code=1)：大概率爆显存"
    )

def test_status_snapshot_anomaly_payload_is_json_safe():
    svc = TrainingService(web.Application())
    svc.status = "running"
    svc._metrics_history = [{"step": 0, "loss": float("nan"), "lr": float("inf"), "rate": "1.00s/step"}]
    svc._latest_system_stats = {"vram_used_gb": 7.2, "vram_total_gb": 8.0}

    snapshot = svc.get_status_snapshot()

    assert snapshot["latest_metric"]["loss"] == "NaN"
    assert snapshot["latest_metric"]["lr"] == "Infinity"
    assert "损失值变为 NaN" in snapshot["anomaly_message"]
    assert "第 0 步" in snapshot["anomaly_message"]
    json.dumps(snapshot, allow_nan=False)

