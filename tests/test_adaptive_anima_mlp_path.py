"""CPU contracts for opt-in sampled Anima MLP/GELU path diagnosis."""

import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from bench.adaptive_runtime import anima_local_vjp as vjp
from bench.adaptive_runtime import anima_mlp_path as path
from networks.lora_modules.lora import LoRAModule


def _fixture(monkeypatch):
    monkeypatch.setattr(path, "NAMES", ("blocks.0.mlp.layer1", "blocks.0.mlp.layer2"))
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([torch.nn.Module()])
    mlp = model.blocks[0].mlp = torch.nn.Module()
    mlp.layer1 = torch.nn.Linear(4, 8)
    mlp.activation = torch.nn.GELU()
    mlp.layer2 = torch.nn.Linear(8, 4)
    adapters = []
    for name, linear in (("first", mlp.layer1), ("second", mlp.layer2)):
        adapter = LoRAModule(name, linear, lora_dim=2, alpha=2)
        adapter.apply_to()
        adapters.append(adapter)
    return model, SimpleNamespace(unet_loras=adapters)


def _capture(tmp_path, monkeypatch, *, scale=64):
    model, network = _fixture(monkeypatch)
    signature = {"steps": 1, "sigmas": [0.5], "local_vjp_profile": "mlp-path"}
    capture = vjp.LocalVJPCapture(tmp_path / "local", model, network,
                                  signature=signature, uuid="GPU-test", scale=scale, replay={})
    mlp = model.blocks[0].mlp
    x = torch.randn(1, 1, 8, 9, 4, requires_grad=True)

    def forward(value):
        return mlp.layer2(mlp.activation(mlp.layer1(value)))

    output = checkpoint(forward, x, use_reentrant=False)
    output.square().sum().mul(scale).backward()
    capture.close()
    manifest = vjp.read_manifest(tmp_path / "local")
    items = [vjp.load_layer(tmp_path / "local", entry) for entry in manifest["layers"]]
    return model, network, manifest, items


def test_checkpoint_path_captures_gelu_boundary_and_unscaled_input_vjp(tmp_path, monkeypatch):
    model, _, manifest, (first, second) = _capture(tmp_path, monkeypatch)
    assert all(entry["zero_up"] and entry["input_cotangent"] for entry in manifest["layers"])
    assert torch.allclose(torch.nn.functional.gelu(first["y"]), second["x"], atol=1e-6)
    expected = path.gelu_vjp(first["y"], second["gx"], "none")
    assert torch.allclose(first["g"], expected, atol=1e-5)
    direct = second["g"] @ model.blocks[0].mlp.layer2.weight.detach()
    assert torch.allclose(second["gx"], direct, atol=1e-5)


def test_nonzero_up_and_preflight_fail_before_capture(tmp_path, monkeypatch):
    model, network = _fixture(monkeypatch)
    with torch.no_grad():
        network.unet_loras[0].lora_up.weight.fill_(0.1)
    signature = {"steps": 1, "sigmas": [0.5], "local_vjp_profile": "mlp-path"}
    with pytest.raises(ValueError, match="zero-initialized"):
        vjp.LocalVJPCapture(tmp_path / "rejected", model, network, signature=signature,
                            uuid="GPU-test", scale=1, replay={})
    assert not (tmp_path / "rejected").exists()
    network.unet_loras[0].lora_up.weight.data.zero_()
    monkeypatch.setattr(vjp, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="preflight"):
        vjp.LocalVJPCapture(tmp_path / "over-budget", model, network, signature=signature,
                            uuid="GPU-test", scale=1, replay={})


def test_path_analysis_and_manifest_tampering(tmp_path, monkeypatch):
    _, _, manifest, items = _capture(tmp_path, monkeypatch)
    rows = [{"name": entry["name"], "approximate": entry["approximate"], "data": item}
            for entry, item in zip(manifest["layers"], items, strict=True)]
    candidate = [{**row, "data": {key: tensor.clone() for key, tensor in row["data"].items()}}
                 for row in rows]
    candidate[0]["data"]["y"].add_(0.01)
    candidate[1]["data"]["gx"].add_(0.02)
    result = path.analyze_path(rows, candidate, device=torch.device("cpu"))[0]
    assert result["fixed_gy_z_only"]["absolute_l2"] > 0
    assert result["fixed_z_gy_only"]["absolute_l2"] > 0
    assert result["reference_vjp_fidelity"]["relative_l2"] < 1e-5
    manifest_path = tmp_path / "local" / "manifest.json"
    source = json.loads(manifest_path.read_text())
    for change in ({"name": "blocks.0.mlp.layer3"}, {"input_cotangent": False},
                   {"file": "../outside.safetensors"}):
        bad = json.loads(json.dumps(source))
        bad["layers"][0].update(change)
        manifest_path.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            vjp.read_manifest(tmp_path / "local")


def test_path_rows_retain_only_required_samples():
    data = {key: torch.zeros(1) for key in ("x", "y", "g", "gx", "weight", "bias")}
    first = path.sample_row({"name": "blocks.0.mlp.layer1", "approximate": "none"}, data)
    second = path.sample_row({"name": "blocks.0.mlp.layer2", "approximate": "none"}, data)
    assert set(first["data"]) == {"y", "g"}
    assert set(second["data"]) == {"x", "gx"}
    assert first["data"]["y"] is data["y"]
