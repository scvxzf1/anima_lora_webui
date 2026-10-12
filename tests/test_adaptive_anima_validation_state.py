import json

import pytest
import torch

from bench.adaptive_runtime.anima_validation_state import StateBundle, StateBundleWriter


IDENTITY = {"family": "anima", "weights_sha256": "a" * 64,
            "source_code_sha256": "b" * 64, "backend": "torch", "rank": 2,
            "alpha": 2, "optimizer": "torch.optim.AdamW", "scheduler": "none"}


def fixture_bundle(tmp_path):
    torch.manual_seed(4)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    path = tmp_path / "bundle"
    writer = StateBundleWriter(path, identity=IDENTITY, uuid="test-uuid")
    writer.record("initial", model, optimizer, "cpu")
    optimizer.zero_grad()
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    writer.record("nonzero", model, optimizer, "cpu")
    writer.finish({"warmup_inputs_sha256": "f" * 64})
    return path, model, optimizer


def test_roundtrip_independent_states_and_real_step(tmp_path):
    path, model, optimizer = fixture_bundle(tmp_path)
    bundle = StateBundle(path, identity=IDENTITY, uuid="test-uuid")
    initial = bundle.restore("initial", model, optimizer, "cpu")
    assert not optimizer.state
    for sigma in (0.8, 0.2, 0.5):
        evidence = bundle.restore("nonzero", model, optimizer, "cpu")
        assert evidence == bundle.restore("nonzero", model, optimizer, "cpu")
        optimizer.zero_grad()
        model(torch.full((1, 3), sigma)).sum().backward()
        optimizer.step()
        assert all(state["step"].item() == 2 for state in optimizer.state.values())
        assert bundle.restore("nonzero", model, optimizer, "cpu") == evidence
    assert bundle.restore("initial", model, optimizer, "cpu") == initial


def test_frozen_parameter_and_optimizer_options(tmp_path):
    torch.manual_seed(4)
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    model[0].requires_grad_(False)
    optimizer = torch.optim.AdamW(model[1].parameters(), lr=0.01)
    writer = StateBundleWriter(tmp_path / "bundle", identity=IDENTITY, uuid="test-uuid")
    writer.record("initial", model, optimizer, "cpu")
    model(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    writer.record("nonzero", model, optimizer, "cpu")
    writer.finish({"warmup_inputs_sha256": "f" * 64})
    bundle = StateBundle(tmp_path / "bundle", identity=IDENTITY, uuid="test-uuid")
    bundle.restore("nonzero", model, optimizer, "cpu")
    optimizer.param_groups[0]["lr"] = 0.02
    with pytest.raises(ValueError, match="options"):
        bundle.restore("nonzero", model, optimizer, "cpu")


@pytest.mark.parametrize("change", ["moment", "label", "parameter", "step", "manifest", "budget"])
def test_corruption_rejected(tmp_path, change):
    path, _, _ = fixture_bundle(tmp_path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if change == "moment":
        target = path / "nonzero.safetensors"
        data = bytearray(target.read_bytes())
        data[-1] ^= 1
        target.write_bytes(data)
    elif change == "label":
        manifest["states"]["nonzero"]["file"] = "initial.safetensors"
    elif change == "parameter":
        manifest["states"]["nonzero"]["inventory"]["parameter.weight"] = ["torch.float32", [99]]
    elif change == "step":
        manifest["states"]["nonzero"]["step"] = 2
    elif change == "manifest":
        manifest["identity"]["rank"] = 99
    else:
        manifest["states"]["nonzero"]["tensor_bytes"] = 256 * 1024**2 + 1
    if change != "moment":
        manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        StateBundle(path, identity=IDENTITY, uuid="test-uuid")
