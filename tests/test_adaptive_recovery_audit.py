import json
import shutil

import pytest
from safetensors.torch import save_file
import torch

from bench.adaptive_runtime.audit_recovery import audit
from bench.adaptive_runtime.checkpoint import restore_checkpoint, save_checkpoint
from bench.adaptive_runtime.scaling import make_scaler


def artifacts(tmp_path):
    folder = tmp_path / "attempt-000"
    folder.mkdir()
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    initial = [p.detach().clone() for p in model.parameters()]
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    signature = {"precision": "bf16", "fp32_modules": [], "inputs_sha256": "fixture"}
    updates = [{"step": 1, "swap": 24, "peak_allocated": 100}]
    path = save_checkpoint(folder / "step.pt", network=model, optimizer=optimizer, initial=initial,
                           updates=updates, signature=signature, device="cpu")
    save_file(model.state_dict(), str(folder / "result.safetensors"))
    result = {"status": "ok", "precision": "bf16", "fp32_modules": [], "swap": 24,
              "optimizer_started": True, "inputs_sha256": "fixture",
              "updates": updates, "committed_checkpoint": path}
    supervisor = {**result, "elapsed_seconds": 1}
    summary = {"status": "ok", "precision": "bf16",
               "attempts": [{"attempt": 0, "plan": {"blocks_to_swap": 24}, "result": supervisor}]}
    for file, value in [(folder / "result.json", result), (folder / "supervisor.json", supervisor),
                        (tmp_path / "summary.json", summary)]:
        file.write_text(json.dumps(value))
    return folder


def test_successful_artifact_audit_is_not_quality_or_resume_claim(tmp_path):
    artifacts(tmp_path)
    result = audit(tmp_path)
    assert result["status"] == "audited"
    assert result["checkpoint_output_exact"] is True
    assert result["checkpoint_resume_observed"] is False
    assert result["production_ready"] is False
    assert result["optimizer_states"] == 2


@pytest.mark.parametrize("corrupt", ["worker", "output", "optimizer", "signature", "summary"])
def test_audit_rejects_inconsistent_artifacts(tmp_path, corrupt):
    folder = artifacts(tmp_path)
    if corrupt in {"optimizer", "signature"}:
        path = folder / "step.pt"
        payload = torch.load(path, weights_only=True)
        if corrupt == "optimizer":
            next(iter(payload["optimizer"]["state"].values()))["step"].fill_(0)
        else:
            payload["signature"]["inputs_sha256"] = "other"
        torch.save(payload, path)
    elif corrupt == "output":
        save_file({"wrong": torch.ones(1)}, str(folder / "result.safetensors"))
    else:
        path = folder / "result.json" if corrupt == "worker" else tmp_path / "summary.json"
        value = json.loads(path.read_text())
        value["status"] = "error"
        path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        audit(tmp_path)


def test_resume_path_without_new_progress_is_not_recovery_evidence(tmp_path):
    folder = artifacts(tmp_path)
    second = tmp_path / "attempt-001"
    shutil.copytree(folder, second)
    first = json.loads((folder / "result.json").read_text())
    first["status"] = "cuda_oom"
    final = json.loads((second / "result.json").read_text())
    final["resumed_from"] = first["committed_checkpoint"]
    final["committed_checkpoint"] = str(second / "step.pt")
    attempts = []
    for index, (directory, result) in enumerate([(folder, first), (second, final)]):
        supervisor = {**result, "elapsed_seconds": 1}
        (directory / "result.json").write_text(json.dumps(result))
        (directory / "supervisor.json").write_text(json.dumps(supervisor))
        attempts.append({"attempt": index, "plan": {"blocks_to_swap": 24}, "result": supervisor})
    (tmp_path / "summary.json").write_text(json.dumps({"status": "ok", "precision": "bf16",
                                                     "attempts": attempts}))
    with pytest.raises(ValueError, match="no new committed progress"):
        audit(tmp_path)


@pytest.mark.parametrize("corrupt_source", [False, True])
def test_scaled_checkpoint_resume_audit(tmp_path, corrupt_source):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = make_scaler(1024, device="cpu")
    initial = [p.detach().clone() for p in model.parameters()]
    signature = {"precision": "fp16-islands", "fp32_modules": [],
                 "inputs_sha256": "fixture", "loss_scale": 1024}
    updates, attempts = [], []
    first_checkpoint = None
    for index, swap in enumerate([24, 26]):
        folder = tmp_path / f"attempt-{index:03d}"
        folder.mkdir()
        if index:
            payload = torch.load(first_checkpoint, weights_only=True)
            scaler = make_scaler(1024, device="cpu")
            initial, updates = restore_checkpoint(
                payload, network=model, optimizer=optimizer, params=list(model.parameters()),
                device="cpu", scaler=scaler)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(model(torch.ones(1, 2)).sum()).backward()
        scaler.step(optimizer)
        scaler.update()
        updates.append({"step": index + 1, "swap": swap, "peak_allocated": 100})
        path = save_checkpoint(folder / "step.pt", network=model, optimizer=optimizer,
                               initial=initial, updates=updates, signature=signature,
                               device="cpu", scaler=scaler)
        result = {**signature, "status": "ok" if index else "cuda_oom", "swap": swap,
                  "optimizer_started": True, "updates": list(updates),
                  "committed_checkpoint": path}
        if index:
            result["resumed_from"] = first_checkpoint
            save_file(model.state_dict(), str(folder / "result.safetensors"))
        else:
            first_checkpoint = path
        supervisor = {**result, "elapsed_seconds": 1}
        (folder / "result.json").write_text(json.dumps(result))
        (folder / "supervisor.json").write_text(json.dumps(supervisor))
        attempts.append({"attempt": index, "plan": {"blocks_to_swap": swap}, "result": supervisor})
    (tmp_path / "summary.json").write_text(json.dumps(
        {**signature, "status": "ok", "attempts": attempts}))
    if corrupt_source:
        payload = torch.load(first_checkpoint, weights_only=True)
        payload["signature"]["inputs_sha256"] = "changed"
        torch.save(payload, first_checkpoint)
        with pytest.raises(ValueError, match="training signature changed"):
            audit(tmp_path)
    else:
        result = audit(tmp_path)
        assert result["checkpoint_resume_observed"] is True
        assert result["scaler"]["_growth_tracker"] == 2
        assert result["checkpoint_output_exact"] is True
