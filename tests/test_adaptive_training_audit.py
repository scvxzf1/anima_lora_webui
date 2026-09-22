import json

import pytest
from safetensors.torch import save_file
import torch

from bench.adaptive_runtime.audit_training import audit_training


def fixture(directory):
    state = directory / ".adaptive-recovery/attempt-000/state-00000003"
    state.mkdir(parents=True)
    save_file({"adapter": torch.ones(2)}, str(state / "model.safetensors"))
    save_file({"adapter": torch.ones(2)}, str(directory / "adaptive-smoke.safetensors"))
    torch.save({"state": {0: {"step": torch.tensor(3), "exp_avg": torch.ones(2),
                             "exp_avg_sq": torch.ones(2)}}}, state / "optimizer.bin")
    torch.save({"_growth_tracker": 3, "growth_interval": 2000, "scale": 1024}, state / "scaler.pt")
    torch.save({"last_epoch": 3}, state / "scheduler.bin")
    (state / "adaptive_precision.json").write_text(json.dumps({"assignments": {"linear": "fp16"}}))
    (state / "snapshot.json").write_text(json.dumps({"global_step": 3,
        "data_cursor_resume_supported": False, "files": {p.name: p.stat().st_size for p in state.iterdir()}}))
    (directory / ".adaptive-recovery/summary.json").write_text(json.dumps({"status": "ok", "attempts": [{
        "result": {"status": "ok", "completed_steps": 3, "saved_state": str(state), "elapsed_seconds": 12}}]}))
    (directory / "memory.jsonl").write_text(json.dumps({"cuda_max_allocated_gb": 4.1}) + "\n")
    return state


def test_training_audit_is_not_quality_or_resume_proof(tmp_path):
    fixture(tmp_path)
    report = audit_training(tmp_path)
    assert report["status"] == "ok" and report["adapter_snapshot_exact"]
    assert report["optimizer_states"] == 1 and report["completed_steps"] == 3
    assert not report["precision_calibrated"] and not report["data_cursor_resume_supported"]


def test_training_audit_rejects_adapter_drift(tmp_path):
    fixture(tmp_path)
    save_file({"adapter": torch.zeros(2)}, str(tmp_path / "adaptive-smoke.safetensors"))
    with pytest.raises(ValueError, match="differs"):
        audit_training(tmp_path)
