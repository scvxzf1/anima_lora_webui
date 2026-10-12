import json

import pytest
from safetensors.torch import save_file
import torch

from bench.adaptive_runtime.anima_update_comparison import sha256_file
from bench.adaptive_runtime.anima_validation_state import StateBundle, StateBundleWriter
from bench.adaptive_runtime.anima_validation_updates import (
    CASES, ValidationUpdateCapture, case_id, compare_validation_updates,
)


IDENTITY = {"family": "anima", "weights_sha256": "a" * 64,
            "source_code_sha256": "b" * 64, "backend": "torch", "rank": 2,
            "alpha": 2, "optimizer": "torch.optim.AdamW", "scheduler": "none"}


def make_bundle(tmp_path):
    torch.manual_seed(3)
    network = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.01)
    writer = StateBundleWriter(tmp_path / "bundle", identity=IDENTITY, uuid="uuid")
    writer.record("initial", network, optimizer, "cpu")
    network(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    writer.record("nonzero", network, optimizer, "cpu")
    writer.finish({"warmup_inputs_sha256": "c" * 64})
    return StateBundle(tmp_path / "bundle", identity=IDENTITY, uuid="uuid"), network, optimizer


def make_worker(tmp_path, bundle, network, optimizer, precision, replay_mode):
    worker = tmp_path / precision
    worker.mkdir()
    training_dir = worker / "training-capture"
    training_dir.mkdir()
    signature = {"precision": precision, "steps": 6, "sigmas": [0.2, 0.5, 0.8], "weights_sha256": "a" * 64}
    capture = ValidationUpdateCapture(worker / "update-capture", network=network,
                                      signature=signature, uuid="uuid")
    training = {"status": "captured", "signature": signature,
                "expected_gradients": [f"gradient.{n}" for n, _ in network.named_parameters()], "cases": []}
    for step, (label, sigma) in enumerate(CASES, 1):
        evidence = bundle.restore(label, network, optimizer, "cpu")
        replay = {key: "b" * 64 for key in ("snapshot_sha256", "state_sha256", "inputs_sha256", "rng_sha256")}
        path = training_dir / f"step-{step:06d}.safetensors"
        save_file({"gradient.weight": torch.ones_like(network.weight),
                   "gradient.bias": torch.ones_like(network.bias)}, str(path))
        training_case = {"step": step, "sigma": sigma, "file": path.name,
                         "sha256": sha256_file(path), "replay": replay}
        training["cases"].append(training_case)
        optimizer.zero_grad()
        network(torch.full((1, 3), sigma)).sum().backward()
        capture.before(network=network, optimizer=optimizer, step=step,
                       case_id=case_id(label, sigma), state_label=label, sigma=sigma,
                       replay=replay, training_case=training_case,
                       state_evidence=evidence, clip_max_norm=1.0)
        optimizer.step()
        capture.after(network=network, optimizer=optimizer, gradient_norm=1.0)
        assert all(s["step"].item() == (1 if label == "initial" else 2)
                   for s in optimizer.state.values())
    capture.close()
    (training_dir / "manifest.json").write_text(json.dumps(training))
    summaries = []
    from safetensors.torch import load_file
    for case in capture.manifest["cases"]:
        values = load_file(str(worker / "update-capture" / case["file"]))
        summaries.append({"step": case["step"], "case_id": case["case_id"],
                          "state_label": case["state_label"], "prestate_evidence": case["state_evidence"],
                          "update_sha256": case["sha256"], "gradient_norm": case["gradient_norm"],
                          "effective_update": True, "adapter_max_delta": max(float(v.abs().max()) for v in values.values()),
                          "loss_scale_before": 1.0, "loss_scale_after": 1.0})
    (worker / "result.json").write_text(json.dumps({"status": "ok", "signature": signature,
        "uuid": "uuid", "replay_mode": replay_mode,
        "updates": summaries}))
    capture.bind_result(worker)
    return worker


def test_six_independent_updates_and_binding(tmp_path):
    bundle, network, optimizer = make_bundle(tmp_path)
    reference = make_worker(tmp_path, bundle, network, optimizer, "fp32-reference", "record")
    candidate = make_worker(tmp_path, bundle, network, optimizer, "candidate", "replay")
    result = compare_validation_updates(reference, candidate)
    assert result["status"] == "compared"
    assert [c["case_id"] for c in result["cases"]] == [case_id(*c) for c in CASES]
    result_path = candidate / "result.json"
    worker = json.loads(result_path.read_text())
    worker["uuid"] = "wrong"
    result_path.write_text(json.dumps(worker))
    with pytest.raises(ValueError):
        compare_validation_updates(reference, candidate)


@pytest.mark.parametrize("change", [{"effective_update": False}, {"adapter_max_delta": -999},
                                    {"state_label": "nonzero"}, {"update_sha256": "0" * 64}])
def test_rebound_result_cannot_misstate_actual_update(tmp_path, change):
    bundle, network, optimizer = make_bundle(tmp_path)
    reference = make_worker(tmp_path, bundle, network, optimizer, "fp32-reference", "record")
    candidate = make_worker(tmp_path, bundle, network, optimizer, "candidate", "replay")
    result_path = candidate / "result.json"
    result = json.loads(result_path.read_text())
    result["updates"][0].update(change)
    result_path.write_text(json.dumps(result))
    manifest_path = candidate / "update-capture" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["result_sha256"] = sha256_file(result_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="summary differs"):
        compare_validation_updates(reference, candidate)
