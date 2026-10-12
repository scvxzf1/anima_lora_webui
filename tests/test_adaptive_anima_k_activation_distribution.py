"""Small CPU contracts for K activation/gradient spatial comparisons."""

import json

import numpy as np
import pytest

from bench.adaptive_runtime import anima_k_activation_distribution as activation


def test_fp64_energy_ranking_and_cross_denominators():
    reference = np.ones((20, 16, 128), dtype=np.float32)
    candidate = reference.copy()
    candidate[3, 7] += 2
    candidate[4, 7] += 1
    energy, baseline = activation.axis_energy(reference, candidate, chunk_tokens=3)
    report = activation.distribution(energy, baseline)
    assert report["error_square"] == 640
    assert report["reference_square"] == 20 * 16 * 128
    assert report["head_rank"][0] == 7
    assert report["token_rank"][:2] == [3, 4]
    grad = np.zeros_like(energy)
    grad[4, 7] = 2
    grad[3, 2] = 8
    overlap = activation.overlap(energy, grad)
    # Top 5% token sets differ; each cross-energy share has a different denominator.
    assert overlap["token"]["top5pct"]["activation_indices"] == [3]
    assert overlap["token"]["top5pct"]["gradient_indices"] == [3]
    assert overlap["head"]["top5pct"]["intersection_count"] == 0
    assert (
        overlap["head"]["top5pct"]["gradient_energy_in_activation_top_share"] == 2 / 10
    )
    assert overlap["head"]["top5pct"]["activation_energy_in_gradient_top_share"] == 0


def test_zero_and_nonfinite_energy_contract():
    zeros = np.zeros((2, 16), dtype=np.float64)
    report = activation.distribution(zeros, zeros)
    assert report["relative_l2"] is None
    assert report["head_error_share"] is None
    assert (
        activation.overlap(zeros, zeros)["token"]["top1pct"][
            "gradient_energy_in_activation_top_share"
        ]
        is None
    )
    assert (
        activation.overlap(zeros, zeros)["token"]["top1pct"]["intersection_count"]
        is None
    )
    assert (
        activation.overlap(zeros, np.ones_like(zeros))["head"]["top1pct"][
            "intersection"
        ]
        is None
    )
    with pytest.raises(ValueError, match="Invalid K energy"):
        activation.distribution(np.full_like(zeros, np.nan), zeros)
    with pytest.raises(ValueError, match="Nonfinite K"):
        activation.axis_energy(np.full((2, 16, 128), np.nan), np.zeros((2, 16, 128)))


def test_saved_report_energy_and_spatial_corruption_rejected():
    energy = np.zeros((2, 16), dtype=np.float64)
    energy[0, 3] = 2
    baseline = np.ones_like(energy)
    recorded = activation.gradients._part_report(
        energy.sum(axis=0), energy.sum(axis=1), float(baseline.sum())
    )
    activation.validate_saved_k_report(energy, baseline, recorded)
    bad = {**recorded, "token_error_share": [0.0, 1.0]}
    with pytest.raises(ValueError, match="inconsistent"):
        activation.validate_saved_k_report(energy, baseline, bad)
    bad = {**recorded, "reference_square": recorded["reference_square"] + 1}
    with pytest.raises(ValueError, match="inconsistent"):
        activation.validate_saved_k_report(energy, baseline, bad)
    for key, value in (
        ("top_head_index", 7),
        ("zero_error", True),
        ("token_concentration", {"top1pct": 0, "top5pct": 0, "top10pct": 0}),
    ):
        with pytest.raises(ValueError, match="inconsistent"):
            activation.validate_saved_k_report(
                energy, baseline, {**recorded, key: value}
            )
    zeros = np.zeros_like(energy)
    zero_report = activation.gradients._part_report(
        zeros.sum(axis=0), zeros.sum(axis=1), 0.0
    )
    activation.validate_saved_k_report(zeros, zeros, zero_report)


def test_k_and_fused_energy_are_both_preserved(monkeypatch):
    left = np.ones((3, 3, 16, 128), dtype=np.float32)
    right = left.copy()
    right[0, 0, 2, 0] += 3
    right[1, 1, 7, 0] += 2
    right[2, 2, 1, 0] += 1
    monkeypatch.setattr(
        activation.gradients,
        "load_cell",
        lambda run, analysis, name: left if name.startswith("k000") else right,
    )
    parts = activation.gradient_energy(None, None, "R")
    assert parts["k"][0].sum() == 4
    assert parts["fused"][0].sum() == 14
    assert parts["fused"][1].sum() == left.size


def test_corrupt_saved_distribution_binding_rejected(tmp_path, monkeypatch):
    run = tmp_path / "interpolation"
    dist = tmp_path / "distribution"
    run.mkdir()
    dist.mkdir()
    monkeypatch.setattr(
        activation.gradients,
        "validate_source",
        lambda _: ({"files": {}}, {"summary.json": "correct"}),
    )
    (dist / "distribution.json").write_text(
        json.dumps(
            {
                "status": "analyzed",
                "production_ready": False,
                "scope": "saved_k_interpolation_gradient_head_token_distribution",
                "analysis_sha256": activation.sha256_file(
                    activation.Path(activation.gradients.__file__)
                ),
                "source_hashes": {"summary.json": "stale"},
                "cell_sha256": {},
                "shape": list(activation.gradients.SHAPE),
                "comparisons": {"baseline": {}},
            }
        )
    )
    with pytest.raises(ValueError, match="binding"):
        activation.validate_distribution(run, dist)


def test_fresh_disjoint_and_cpu_required(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    weights = tmp_path / "weights.safetensors"
    weights.touch()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="disjoint"):
        activation.run(source, source, weights, source / "child")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="Fresh"):
        activation.run(source, source, weights, existing)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES"):
        activation.run(source, source, weights, tmp_path / "fresh")
