"""CPU gates for the disposable full block VJP matrix."""

import copy
import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from bench.adaptive_runtime import anima_block_vjp as vjp
from bench.adaptive_runtime import diagnose_anima_block_vjp as driver
from bench.adaptive_runtime.replay import network_tensors, tensor_digest


def _matrix():
    captured = {}
    outputs = {}
    for policy, weight in (("reference", 2.0), ("candidate", 3.0)):
        captured[policy] = {}
        for x_side in ("reference", "candidate"):
            for g_side, cotangent in (("reference", 4.0), ("candidate", 5.0)):
                actual_x = torch.tensor([1.0 if x_side == "reference" else 1.5],
                                        requires_grad=True)
                y = weight * actual_x.square()
                (gradient,) = torch.autograd.grad(y, actual_x,
                                                  grad_outputs=torch.tensor([cotangent]))
                outputs[(policy, x_side, g_side)] = {
                    "gx": gradient.detach(),
                    "forward_fidelity": {"relative_l2": 0.0} if x_side == policy else None,
                }
        captured[policy]["gx"] = outputs[(policy, policy, policy)]["gx"].clone()
    return outputs, captured


def test_eight_cell_matrix_uses_real_autograd_and_separate_axes():
    outputs, captured = _matrix()
    fidelity, contrasts = vjp.matrix_metrics(outputs, captured)
    assert set(fidelity) == {"reference", "candidate"}
    assert all(len(axis) == 4 for axis in contrasts.values())
    assert all(metric["relative_l2"] == 0 for metric in
               (fidelity[side]["input_cotangent"] for side in fidelity))
    assert all(metric["relative_l2"] > 0 for axis in contrasts.values()
               for metric in axis.values())
    assert all(metric["relative_l2"] == pytest.approx(0.5) for metric in
               contrasts["activation_fixed_policy_and_cotangent"].values())
    assert all(metric["relative_l2"] == pytest.approx(0.25) for metric in
               contrasts["cotangent_fixed_policy_and_activation"].values())


def test_same_side_vjp_fidelity_fails_closed():
    outputs, captured = _matrix()
    bad = copy.deepcopy(captured)
    bad["candidate"]["gx"] += 0.01
    with pytest.raises(ValueError, match="capture fidelity failed"):
        vjp.matrix_metrics(outputs, bad)


def test_missing_matrix_cell_fails_closed():
    outputs, captured = _matrix()
    outputs.pop(("candidate", "reference", "candidate"))
    with pytest.raises(ValueError, match="all eight"):
        vjp.matrix_metrics(outputs, captured)


def test_conditioning_metadata_mismatch_precedes_gpu_work(monkeypatch, tmp_path):
    signature = {"weights_sha256": "a" * 64}
    ref = {"signature": signature, "conditioning": {"use_fp32": False}}
    candidate = {"signature": signature, "conditioning": {"use_fp32": True}}
    tensors = {"conditioning.emb_B_T_D": torch.ones(1), "x": torch.ones(1)}
    monkeypatch.setattr(vjp, "validate_pair", lambda *_, **_kw: ([(ref, tensors), (candidate, tensors)], []))
    monkeypatch.setattr(vjp, "sha256_file", lambda *_: "a" * 64)
    monkeypatch.setattr(vjp, "_build_block", lambda *_args, **_kwargs: pytest.fail("GPU rebuild"))
    with pytest.raises(ValueError, match="Conditioning metadata differs"):
        vjp.analyze(tmp_path / "ref", tmp_path / "candidate", {}, weights=tmp_path / "weights",
                    device=torch.device("cuda"))


def test_replay_restores_exact_parameter_and_buffer_inventory(tmp_path):
    network = nn.Linear(2, 1)
    network.register_buffer("mask", torch.tensor([3.0]), persistent=False)
    original = {name: value.detach().clone() for name, value in network_tensors(network).items()}
    evidence = {"state_sha256": tensor_digest(original)}
    directory = tmp_path / "training-replay"
    directory.mkdir()
    path = directory / "step-000001.safetensors"
    save_file(original, str(path))
    evidence["snapshot_sha256"] = vjp.sha256_file(path)
    (directory / "manifest.json").write_text(json.dumps({
        "status": "captured", "steps": 1,
        "cases": [{"evidence": evidence, "file": path.name}],
    }))
    with torch.no_grad():
        for value in network_tensors(network).values():
            value.zero_()
    vjp._replay_state(tmp_path, evidence, network)
    assert tensor_digest(network_tensors(network)) == evidence["state_sha256"]
    changed = dict(original)
    changed["parameter.extra"] = torch.ones(1)
    save_file(changed, str(path))
    evidence["snapshot_sha256"] = vjp.sha256_file(path)
    evidence["state_sha256"] = tensor_digest(changed)
    (directory / "manifest.json").write_text(json.dumps({
        "status": "captured", "steps": 1,
        "cases": [{"evidence": evidence, "file": path.name}],
    }))
    with pytest.raises(ValueError, match="inventory/digest mismatch"):
        vjp._replay_state(tmp_path, evidence, network)


def test_matrix_evidence_preserves_all_cells_and_axis_order(monkeypatch, tmp_path):
    from safetensors.torch import load_file

    outputs, _ = _matrix()
    monkeypatch.setattr(vjp.capture, "SHAPE", (1,))
    manifests = {side: {"sha256": side, "signature": {"suite": "toy"},
                        "replay": {"state": "fixed"}}
                 for side in ("reference", "candidate")}
    directory = tmp_path / "evidence"
    result = vjp.save_matrix_evidence(directory, outputs, manifests)
    values = load_file(str(directory / result["file"]))
    assert result["axis_order"] == ["policy", "activation_source", "cotangent_source"]
    assert result["sha256"] == vjp.sha256_file(directory / result["file"])
    assert len(values) == 8
    for key, item in outputs.items():
        assert torch.equal(values["/".join(key)], item["gx"])
    with pytest.raises(FileExistsError):
        vjp.save_matrix_evidence(directory, outputs, manifests)
    outputs.pop(("candidate", "reference", "candidate"))
    with pytest.raises(ValueError, match="Invalid full-shape"):
        vjp.save_matrix_evidence(tmp_path / "missing", outputs, manifests)


def test_parent_validates_persisted_matrix_and_capture_binding(monkeypatch, tmp_path):
    outputs, _ = _matrix()
    monkeypatch.setattr(vjp.capture, "SHAPE", (1,))
    source = {"weights_sha256": "w", "inputs_sha256": "i",
              "probe_sha256": "p", "block_vjp_helper_sha256": "h"}
    replay = {"state_sha256": "s"}
    manifests = {}
    state = {}
    for side in ("reference", "candidate"):
        capture_dir = tmp_path / side / "block-vjp"
        capture_dir.mkdir(parents=True)
        capture_file = capture_dir / "capture.safetensors"
        capture_file.write_bytes(side.encode())
        state[side] = {"base_state_sha256": side, "adapter_state_sha256": "adapter",
                       "loss_scale": 1 if side == "reference" else 1024,
                       "fp32_residual": side == "candidate"}
        manifest = {"file": capture_file.name, "sha256": vjp.sha256_file(capture_file),
                    "signature": {**source, "precision": side}, "replay": replay,
                    **state[side]}
        (capture_dir / "manifest.json").write_text(json.dumps(manifest))
        manifests[side] = manifest
    directory = tmp_path / "analysis" / "matrix-evidence"
    evidence = vjp.save_matrix_evidence(directory, outputs, manifests)
    analysis = {"matrix_evidence": evidence, "source": source, "replay": replay,
                "state": state}
    inventory = driver.validate_matrix_evidence(directory, analysis)
    assert len(inventory) == 8
    assert all(item == {"shape": [1], "dtype": "torch.float32", "bytes": 4}
               for item in inventory.values())

    matrix_file = directory / evidence["file"]
    matrix_file.unlink()
    with pytest.raises(FileNotFoundError):
        driver.validate_matrix_evidence(directory, analysis)
    save_file({key: value["gx"] for key, value in
               (("/".join(cell), output) for cell, output in outputs.items())},
              str(matrix_file))
    with matrix_file.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="hash/budget"):
        driver.validate_matrix_evidence(directory, analysis)


def test_parent_rejects_missing_or_wrong_capture_binding(monkeypatch, tmp_path):
    outputs, _ = _matrix()
    monkeypatch.setattr(vjp.capture, "SHAPE", (1,))
    source = {"weights_sha256": "w"}
    replay = {"state_sha256": "s"}
    manifests = {}
    for side in ("reference", "candidate"):
        capture_dir = tmp_path / side / "block-vjp"
        capture_dir.mkdir(parents=True)
        capture_file = capture_dir / "capture.safetensors"
        capture_file.write_bytes(side.encode())
        manifest = {"file": capture_file.name, "sha256": vjp.sha256_file(capture_file),
                    "signature": source, "replay": replay}
        (capture_dir / "manifest.json").write_text(json.dumps(manifest))
        manifests[side] = manifest
    directory = tmp_path / "analysis" / "matrix-evidence"
    evidence = vjp.save_matrix_evidence(directory, outputs, manifests)
    analysis = {"matrix_evidence": evidence, "source": source, "replay": replay,
                "state": {side: {} for side in manifests}}
    assert len(driver.validate_matrix_evidence(directory, analysis)) == 8
    (tmp_path / "candidate" / "block-vjp" / "capture.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="capture binding mismatch"):
        driver.validate_matrix_evidence(directory, analysis)
