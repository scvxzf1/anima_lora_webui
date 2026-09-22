import json

import pytest
import torch

from bench.adaptive_runtime.compare_training import compare, compare_case
from bench.adaptive_runtime.training_capture import TrainingCapture


def capture(tmp_path, name, *, precision, delta=0, max_bytes=1024**3):
    directory = tmp_path / name / "training-capture"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        network = torch.nn.Linear(4, 3)
        recorder = TrainingCapture(directory, network=network,
                                   signature={"precision": precision, "inputs_sha256": "same"},
                                   steps=1, max_bytes=max_bytes)
        noisy = torch.randn(2, 4, requires_grad=True)
        prediction = network(noisy)
        prediction.square().mean().backward()
        recorder.capture(network=network, prediction=prediction + delta, noisy=noisy, step=1, sigma=0.2)
        recorder.close()
        (directory.parent / "result.json").write_text(json.dumps({
            "status": "ok", "precision": precision, "inputs_sha256": "same", "updates": [{"step": 1}],
            "uuid": "test-device",
        }))
    return directory


def test_identical_prediction_and_parameter_gradients_pass(tmp_path):
    reference = capture(tmp_path, "reference", precision="fp32-reference")
    candidate = capture(tmp_path, "candidate", precision="fp16-islands")
    report = compare(reference, candidate)
    assert report["within_experimental_tolerances"] is True
    assert report["full_model_calibrated"] is False
    assert report["cases"][0]["adapter_gradient"]["relative_l2"] == 0
    assert report["cases"][0]["input_gradient"]["cosine"] == pytest.approx(1)


def test_output_drift_fails_gate_without_claiming_quality(tmp_path):
    reference = capture(tmp_path, "reference", precision="fp32-reference")
    candidate = capture(tmp_path, "candidate", precision="fp16-islands", delta=1)
    report = compare(reference, candidate)
    assert report["within_experimental_tolerances"] is False
    assert report["production_ready"] is False


def test_absolute_error_ranking_does_not_follow_tiny_gradient_relative_error():
    reference = {"prediction": torch.ones(1), "input_gradient": torch.ones(1),
                 "gradient.tiny": torch.tensor([1e-8]), "gradient.large": torch.ones(1)}
    candidate = {**reference, "gradient.tiny": torch.tensor([1e-6]),
                 "gradient.large": torch.tensor([1.1])}
    report = compare_case(reference, candidate)
    assert report["worst_adapter_parameters"][0]["name"] == "gradient.tiny"
    assert report["largest_absolute_adapter_errors"][0]["name"] == "gradient.large"
    assert report["largest_absolute_adapter_errors"][0]["absolute_l2"] == pytest.approx(0.1)


@pytest.mark.parametrize("uuid", [None, "", "different-device"])
def test_comparison_requires_same_identified_gpu(tmp_path, uuid):
    reference = capture(tmp_path, "reference", precision="fp32-reference")
    candidate = capture(tmp_path, "candidate", precision="fp16-islands")
    path = candidate.parent / "result.json"
    worker = json.loads(path.read_text())
    worker["uuid"] = uuid
    path.write_text(json.dumps(worker))
    with pytest.raises(ValueError, match="same GPU UUID"):
        compare(reference, candidate)


def test_snapshot_byte_budget(tmp_path):
    with pytest.raises(ValueError, match="budget"):
        capture(tmp_path, "small", precision="fp32-reference", max_bytes=1)
    assert not list((tmp_path / "small" / "training-capture").glob("*.safetensors"))


@pytest.mark.parametrize("field", ["initial_adapter_sha256", "signature", "worker", "digest",
                                   "precision", "expected_gradients"])
def test_comparison_rejects_incompatible_or_corrupt_capture(tmp_path, field):
    reference = capture(tmp_path, "reference", precision="fp32-reference")
    candidate = capture(tmp_path, "candidate", precision="fp16-islands")
    path = candidate / "manifest.json"
    manifest = json.loads(path.read_text())
    if field == "signature":
        manifest[field]["inputs_sha256"] = "different"
    elif field == "digest":
        manifest["cases"][0]["sha256"] = "wrong"
    elif field == "worker":
        (candidate.parent / "result.json").write_text('{"status":"error"}')
    elif field == "precision":
        manifest["signature"]["precision"] = "unknown"
    elif field == "expected_gradients":
        manifest[field] = ["gradient.missing"]
    else:
        manifest[field] = "different"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        compare(reference, candidate)


def test_missing_parameter_gradient_is_not_silently_omitted(tmp_path):
    network = torch.nn.Linear(4, 3)
    recorder = TrainingCapture(tmp_path / "capture", network=network, signature={}, steps=1)
    noisy = torch.randn(2, 4, requires_grad=True)
    prediction = network(noisy)
    prediction.square().mean().backward()
    network.bias.grad = None
    with pytest.raises(ValueError, match="Missing or changed"):
        recorder.capture(network=network, prediction=prediction, noisy=noisy, step=1, sigma=0.2)
