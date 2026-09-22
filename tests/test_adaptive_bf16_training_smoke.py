import json

import pytest
from safetensors.torch import save_file
import toml
import torch

from bench.adaptive_runtime.bf16_training_smoke import audit, baseline_config


def test_bf16_baseline_preserves_comparable_settings(tmp_path):
    source = tmp_path / "source.toml"
    source.write_text(toml.dumps({
        "model_family": "krea2_raw", "max_train_steps": 3, "optimizer_type": "AdamW",
        "save_precision": "float", "mixed_precision": "fp16", "network_dim": 4,
        "blocks_to_swap": 26, "seed": 20260922, "dataset_config": "shared.toml",
        "adaptive_precision": "fp16_fp32", "adaptive_oom_retry": True,
    }))
    config = baseline_config(source, tmp_path / "fresh")
    assert config["mixed_precision"] == "bf16"
    assert config["adaptive_precision"] == "off"
    assert not config["adaptive_oom_retry"] and not config["full_bf16"]
    assert config["blocks_to_swap"] == 26 and config["network_dim"] == 4
    assert config["seed"] == 20260922 and config["dataset_config"] == "shared.toml"
    assert config["save_state_on_train_end"]
    assert config["progress_jsonl"] == str(tmp_path / "fresh/progress.jsonl")
    assert toml.load(source)["mixed_precision"] == "fp16"
    with pytest.raises(FileExistsError):
        baseline_config(source, tmp_path)


def fixture(directory):
    state = directory / "bf16-smoke-state"
    state.mkdir()
    for path in (state / "model.safetensors", directory / "bf16-smoke.safetensors"):
        save_file({"adapter": torch.ones(2)}, str(path))
    torch.save({"state": {0: {"step": torch.tensor(3), "exp_avg": torch.ones(2),
                             "exp_avg_sq": torch.ones(2)}}}, state / "optimizer.bin")
    torch.save({"last_epoch": 3}, state / "scheduler.bin")
    (state / "random_states_0.pkl").touch()
    events = [{"ev": "step", "ts": step * 5.0, "global_step": step, "avr_loss": 0.1}
              for step in range(1, 4)]
    events.append({"ev": "run_end", "status": "ok", "final_step": 3})
    (directory / "progress.jsonl").write_text("\n".join(map(json.dumps, events)))
    (directory / "memory.jsonl").write_text(json.dumps({
        "cuda_max_allocated_gb": 4.1, "cuda_max_reserved_gb": 4.3,
        "label": "accelerator_prepared", "unet": {"dtypes": {"bfloat16": 10}},
        "network": {"all_parameters": {"dtypes": {"float32": 2}}},
    }) + "\n")
    return state


def test_bf16_audit_accepts_no_scaler(tmp_path):
    fixture(tmp_path)
    report = audit(tmp_path)
    assert report["status"] == "ok" and report["completed_steps"] == 3
    assert report["adapter_snapshot_exact"] and not report["scaler_expected"]
    assert report["step_event_intervals_seconds"] == [5, 5]
    assert report["loss_metric"] == "avr_loss"
    assert not report["precision_calibrated"] and not report["resume_tested"]


@pytest.mark.parametrize("fault", ["scaler", "weights", "optimizer", "scheduler", "loss", "dtype"])
def test_bf16_audit_rejects_invalid_evidence(tmp_path, fault):
    state = fixture(tmp_path)
    if fault == "scaler":
        (state / "scaler.pt").touch()
    elif fault == "weights":
        save_file({"adapter": torch.zeros(2)}, str(tmp_path / "bf16-smoke.safetensors"))
    elif fault == "optimizer":
        torch.save({"state": {}}, state / "optimizer.bin")
    elif fault == "scheduler":
        torch.save({"last_epoch": 2}, state / "scheduler.bin")
    elif fault == "loss":
        path = tmp_path / "progress.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        events[0]["avr_loss"] = float("nan")
        path.write_text("\n".join(map(json.dumps, events)))
    else:
        path = tmp_path / "memory.jsonl"
        memory = json.loads(path.read_text())
        memory["unet"]["dtypes"] = {"float16": 10}
        path.write_text(json.dumps(memory))
    with pytest.raises(ValueError):
        audit(tmp_path)


def test_bf16_audit_names_tracked_current_loss(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "progress.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    for event in events:
        if event["ev"] == "step":
            event["loss/current"] = 0.2
    path.write_text("\n".join(map(json.dumps, events)))
    report = audit(tmp_path)
    assert report["loss_metric"] == "loss/current"
    assert report["losses"] == [0.2] * 3
