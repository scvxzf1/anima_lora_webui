"""Small CPU checks for K-interpolation gradient distribution diagnostics."""

import json

import numpy as np
import pytest
from safetensors.torch import save_file
import torch

from bench.adaptive_runtime import anima_k_distribution as distribution
from bench.adaptive_runtime import anima_k_interpolation as interpolation
from bench.adaptive_runtime.anima_inputs import sha256_file


def test_head_token_distribution_and_partition():
    reference = np.ones((20, 3, 16, 128), dtype=np.float32)
    candidate = reference.copy()
    candidate[3, 1, 7, :] += 2
    candidate[4, 1, 7, :] += 1
    candidate[3, 0, 2, :] += 1
    report = distribution.pair_distribution(reference, candidate, chunk_tokens=3)
    assert report["k"]["error_square"] == 640
    assert report["q"]["error_square"] == 128
    assert report["v"]["zero_error"] is True
    assert report["v"]["head_error_share"] is None
    assert report["fused"]["error_square"] == 768
    assert report["fused"]["reference_square"] == 20 * 3 * 16 * 128
    assert report["fused"]["top1_heads"] == [7]
    assert report["fused"]["top4_heads"][:2] == [7, 2]
    assert report["fused"]["top_token_index"] == 3
    assert report["fused"]["token_concentration"]["top5pct"] == 640 / 768
    assert sum(report["fused"]["head_error_share"]) == pytest.approx(1)
    assert sum(report["fused"]["token_error_share"]) == pytest.approx(1)
    assert len(list(distribution.comparisons())) == 21


def test_zero_reference_and_zero_error_are_explicit():
    zeros = np.zeros((2, 3, 16, 128), dtype=np.float32)
    same = distribution.pair_distribution(zeros, zeros)["fused"]
    assert same["relative_l2"] is None
    assert same["top_token_index"] is None
    candidate = zeros.copy()
    candidate[0, 0, 0, 0] = 1
    changed = distribution.pair_distribution(zeros, candidate)["fused"]
    assert changed["reference_square"] == 0
    assert changed["relative_l2"] is None
    assert changed["relative_denominator_reliable"] is False
    assert changed["top1_heads"] == [0]
    tiny = np.full_like(zeros, 1e-8)
    assert (
        distribution.pair_distribution(tiny, candidate)["fused"]["relative_l2"] is None
    )


def test_endpoint_metrics_must_be_self_consistent():
    metric = {
        "finite": True,
        "relative_l2": 0.0,
        "absolute_l2": 1.0,
        "reference_l2": 1.0,
        "cosine": -1.0,
    }
    with pytest.raises(ValueError, match="ratio"):
        distribution.validate_endpoint(metric)
    metric.update(absolute_l2=0.0)
    with pytest.raises(ValueError, match="cosine"):
        distribution.validate_endpoint(metric)


def test_corrupt_source_binding_rejected(tmp_path):
    summary = {
        "status": "completed",
        "production_ready": False,
        "scope": interpolation.SCOPE,
        "analysis": {},
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "result.json").write_text(json.dumps({"status": "ok", "analysis": {}}))
    (tmp_path / "environment.json").write_text(json.dumps({}))
    (tmp_path / "audit-cpu.json").write_text(
        json.dumps(
            {"status": "audited", "production_ready": False, "summary_sha256": "0" * 64}
        )
    )
    with pytest.raises(ValueError, match="binding"):
        distribution.validate_source(tmp_path)


def test_valid_attention_shape_binding_and_corrupt_audit_values(tmp_path):
    metric = {
        "finite": True,
        "relative_l2": 1e-8,
        "absolute_l2": 1e-10,
        "reference_l2": 0.01,
        "cosine": 1.0,
    }
    source = interpolation.code_hashes()
    analysis = {
        "uuid": "toy",
        "scope": interpolation.SCOPE,
        "production_ready": False,
        "shape": list(interpolation.matrix.SHAPE),
        "source": source,
        "files": dict.fromkeys(interpolation.CELLS, {}),
        "replay": dict.fromkeys(interpolation.CELLS, {}),
        "forward_bitwise_equal": {"reference_core": True},
        "statistics": {"test": {"fused": metric}},
    }
    driver = sha256_file(
        distribution.Path(interpolation.__file__).with_name(
            "diagnose_anima_k_interpolation.py"
        )
    )
    summary = {
        "status": "completed",
        "scope": interpolation.SCOPE,
        "production_ready": False,
        "analysis": analysis,
    }
    result = {
        "status": "ok",
        "production_ready": False,
        "analysis": analysis,
        "uuid": "toy",
        "driver_sha256": driver,
    }
    environment = {
        **source,
        "gpu_uuid": "toy",
        "training_steps": 0,
        "driver_sha256": driver,
    }
    for name, payload in (
        ("summary.json", summary),
        ("result.json", result),
        ("environment.json", environment),
    ):
        (tmp_path / name).write_text(json.dumps(payload))
    audit = {
        "status": "audited",
        "production_ready": False,
        "summary_sha256": sha256_file(tmp_path / "summary.json"),
        "statistics": analysis["statistics"],
        "reference_historical_c": metric,
        "source_endpoints": {
            interpolation.label(value, side): metric
            for value in (0, 1)
            for side in interpolation.SIDES
        },
    }
    path = tmp_path / "audit-cpu.json"
    path.write_text(json.dumps(audit))
    assert distribution.validate_source(tmp_path)[0]["shape"][-1] == 2048
    bad = json.loads(json.dumps(audit))
    bad["statistics"]["test"]["fused"]["relative_l2"] = 0.1
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="metric mismatch"):
        distribution.validate_source(tmp_path)
    for key in ("source_endpoints", "reference_historical_c"):
        bad = json.loads(json.dumps(audit))
        target = bad[key]["k000_vR"] if key == "source_endpoints" else bad[key]
        target["relative_l2"] = 0.1
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="fidelity"):
            distribution.validate_source(tmp_path)


def test_manifest_sha_and_nonfinite_rejected(tmp_path, monkeypatch):
    shape = (1, 1, 1, 5, 6144)
    monkeypatch.setattr(distribution, "SHAPE", shape)
    monkeypatch.setattr(distribution, "TOKENS", 5)
    source = {"toy": True}
    name = "k000_vR"
    manifest = interpolation.matrix.norm.core.parent.save_evidence(
        tmp_path,
        f"cell-{name}",
        {"c": torch.ones(shape)},
        source,
        scope=interpolation.SCOPE,
    )
    analysis = {"files": {name: manifest}, "source": source}
    assert distribution.load_cell(tmp_path, analysis, name).shape == (5, 3, 16, 128)
    bad = {"files": {name: {**manifest, "sha256": "0" * 64}}, "source": source}
    with pytest.raises(ValueError, match="SHA"):
        distribution.load_cell(tmp_path, bad, name)
    # Re-signing a nonfinite tensor must still fail tensor validation.
    payload = tmp_path / f"cell-{name}" / "evidence.safetensors"
    save_file({"c": torch.full(shape, float("nan"))}, str(payload))
    manifest.update(sha256=sha256_file(payload), disk_bytes=payload.stat().st_size)
    (payload.parent / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="finite"):
        distribution.load_cell(tmp_path, analysis, name)


def test_fresh_disjoint_output_required(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    root = tmp_path / "source"
    root.mkdir()
    with pytest.raises(ValueError, match="disjoint"):
        distribution.run(root, root / "child")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="Fresh"):
        distribution.run(root, existing)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        distribution.run(root, tmp_path / "fresh")
