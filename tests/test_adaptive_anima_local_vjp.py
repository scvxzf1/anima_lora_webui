"""CPU contracts for sampled Anima base-Linear VJP capture."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from bench.adaptive_runtime import anima_local_vjp as vjp
from bench.adaptive_runtime import probe_anima_train as probe
from bench.adaptive_runtime.compare_training import compare
from library.training.adaptive_runtime.islands import install_precision_islands
from networks.lora_modules.lora import LoRAModule


def _model(monkeypatch):
    monkeypatch.setattr(vjp, "NAMES", ("blocks.0.mlp.layer1",))
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([torch.nn.Module()])
    model.blocks[0].mlp = torch.nn.Module()
    linear = torch.nn.Linear(4, 3, bias=False)
    model.blocks[0].mlp.layer1 = linear
    adapter = LoRAModule("test", linear, lora_dim=2, alpha=2)
    adapter.apply_to()
    with torch.no_grad():
        adapter.lora_up.weight.fill_(0.25)
    return model, SimpleNamespace(unet_loras=[adapter]), linear, adapter


def _signature():
    return {"steps": 1, "sigmas": [0.5], "precision": "fp32-reference"}


@pytest.mark.parametrize("layout", [(1, 72, 4), (1, 1, 8, 9, 4)])
def test_real_lora_checkpoint_captures_base_not_adapter(tmp_path, monkeypatch, layout):
    model, network, linear, adapter = _model(monkeypatch)
    original = adapter.org_forward
    capture = vjp.LocalVJPCapture(tmp_path / "local", model, network,
                                  signature=_signature(), uuid="GPU-test", scale=64, replay={})
    x = torch.randn(*layout, requires_grad=True)
    total = checkpoint(lambda value: linear(value), x, use_reentrant=False)
    total.sum().mul(64).backward()
    capture.close()
    assert adapter.org_forward == original
    manifest = vjp.read_manifest(tmp_path / "local")
    item = vjp.load_layer(tmp_path / "local", manifest["layers"][0])
    assert torch.allclose(item["y"], torch.nn.functional.linear(item["x"], linear.weight))
    assert not torch.allclose(total.detach().flatten(1, -2)[:, vjp.sampled_indices(72)], item["y"])
    assert manifest["layers"][0]["original_input_shape"] == list(layout)
    assert torch.equal(item["g"], torch.ones_like(item["g"]))


def test_preflight_rejects_budget_before_weight_copy(tmp_path, monkeypatch):
    model, network, _, _ = _model(monkeypatch)
    monkeypatch.setattr(vjp, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="preflight"):
        vjp.LocalVJPCapture(tmp_path / "local", model, network,
                            signature=_signature(), uuid="GPU-test", scale=1, replay={})
    assert not (tmp_path / "local").exists()


def test_sampling_does_not_advance_rng():
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    indices = vjp.sampled_indices(256)
    assert indices.shape == (64,) and indices[0] == 0 and indices[-1] == 255
    assert torch.equal(before, torch.random.get_rng_state())


def test_local_metrics_zero_and_fp16_overflow():
    item = {"x": torch.zeros(1, 64, 2), "y": torch.zeros(1, 64, 1),
            "g": torch.full((1, 64, 1), 100.0), "weight": torch.ones(1, 2)}
    result = vjp.local_analysis(item, device=torch.device("cpu"))
    assert result["1024"]["finite"] is False
    assert result["1024"]["g_scaled_overflow"] == 64
    assert result["1"]["finite"] is True
    assert vjp.metrics(torch.zeros(2), torch.ones(2))["relative_l2"] is None
    assert vjp.metrics(torch.zeros(2), torch.zeros(2))["cosine"] == 1


def test_manifest_rejects_path_and_shape_before_load(tmp_path, monkeypatch):
    model, network, linear, _ = _model(monkeypatch)
    capture = vjp.LocalVJPCapture(tmp_path / "local", model, network,
                                  signature=_signature(), uuid="GPU-test", scale=1, replay={})
    linear(torch.ones(1, 64, 4, requires_grad=True)).sum().backward()
    capture.close()
    path = tmp_path / "local" / "manifest.json"
    source = json.loads(path.read_text())
    for change in ({"file": "../outside.safetensors"}, {"sample_shape": [2, 64, 4]}):
        bad = json.loads(json.dumps(source))
        bad["layers"][0].update(change)
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            vjp.read_manifest(tmp_path / "local")


def _training_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(vjp, "NAMES", ("blocks.0.mlp.layer1",))

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Module()])
            self.blocks[0].mlp = torch.nn.Module()
            self.blocks[0].mlp.layer1 = torch.nn.Linear(4, 4, bias=False)
            self.requires_grad_(False)
            self.blocks[0].mlp.layer1.weight.copy_(torch.eye(4))

        def forward(self, noisy, sigma, prompt, padding_mask=None):
            x = noisy.flatten(2).transpose(1, 2)
            y = checkpoint(self.blocks[0].mlp.layer1, x, use_reentrant=False)
            return y.transpose(1, 2).reshape_as(noisy)

    inputs = {"latents": torch.ones(1, 4, 1, 8, 8), "noise": torch.zeros(1, 4, 1, 8, 8),
              "prompt": torch.zeros(1, 64, 4)}
    reports = []
    for name, precision, scale in (("reference", "fp32-reference", 1),
                                    ("candidate", "fp16-islands", 1024)):
        torch.manual_seed(123)
        model = ToyModel()
        linear = model.blocks[0].mlp.layer1
        if name == "candidate":
            install_precision_islands(model, {"blocks.0.mlp.layer1": "fp16"})
        adapter = LoRAModule("lora", linear, lora_dim=2, alpha=2)
        adapter.apply_to()
        network = torch.nn.Module()
        network.add_module("lora", adapter)
        network.unet_loras = [adapter]
        args = SimpleNamespace(output=tmp_path / name / "result.json", steps=1, single_sigma=0.5,
                               capture_local_vjp=True, precision=precision, loss_scale=scale,
                               record_replay=name == "reference", replay_reference=
                               None if name == "reference" else tmp_path / "reference" / "training-replay")
        args.output.parent.mkdir()
        report = {"uuid": "GPU-test", "weights_sha256": "weights", "inputs_sha256": "inputs",
                  "fp32_modules": [], "updates": [], "precision": precision,
                  "loss_scale": scale, "replay_mode": "record" if name == "reference" else "replay",
                  "precision_manifest": {"assignments": {"blocks.0.mlp.layer1": "fp16"}}}
        resources = []
        probe.train_steps(args, model, network, inputs, report, resources)
        probe.close_resources(resources)
        report["status"] = "ok"
        args.output.write_text(json.dumps(report))
        reports.append(report)
    return (tmp_path / "reference" / "local-vjp", tmp_path / "candidate" / "local-vjp", reports)


def test_real_training_replay_analyze_and_forgery_rejection(tmp_path, monkeypatch):
    reference, candidate, reports = _training_pair(tmp_path, monkeypatch)
    comparison = compare(reference.parent / "training-capture", candidate.parent / "training-capture")
    report = vjp.analyze(reference, candidate, comparison, device=torch.device("cpu"))
    assert report["status"] == "analyzed" and len(report["layers"]) == 1
    assert report["layers"][0]["local_fixed_cotangent"]["1024"]["actual_fp16_dx"]["relative_l2"] == 0
    forged = {"status": "compared", "scope": "fixed_pre_step_forged",
              "reference": {"uuid": "GPU-test"}, "candidate": {"uuid": "GPU-test"}}
    with pytest.raises(ValueError, match="bound"):
        vjp.analyze(reference, candidate, forged, device=torch.device("cpu"))
    reports[1]["precision_manifest"]["assignments"]["blocks.0.mlp.layer1"] = "fp32"
    (candidate.parent / "result.json").write_text(json.dumps(reports[1]))
    with pytest.raises(ValueError, match="MLP-only"):
        vjp.analyze(reference, candidate, comparison, device=torch.device("cpu"))


def test_local_capture_hash_and_replay_tampering_rejected(tmp_path, monkeypatch):
    reference, candidate, _ = _training_pair(tmp_path, monkeypatch)
    comparison = compare(reference.parent / "training-capture", candidate.parent / "training-capture")
    path = candidate / "manifest.json"
    original = json.loads(path.read_text())
    changed = json.loads(json.dumps(original))
    changed["replay"]["rng_sha256"] = "a" * 64
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="identity"):
        vjp.analyze(reference, candidate, comparison, device=torch.device("cpu"))
    path.write_text(json.dumps(original))
    entry = original["layers"][0]
    tensor_path = candidate / entry["file"]
    tensor_path.write_bytes(tensor_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash"):
        vjp.load_layer(candidate, entry)
