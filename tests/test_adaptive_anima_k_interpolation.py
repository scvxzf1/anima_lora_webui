"""CPU contracts for bench-only K interpolation and V-background evidence."""

import copy

import pytest
import torch

from bench.adaptive_runtime import anima_k_interpolation as interpolation
from bench.adaptive_runtime.audit_anima_k_interpolation import audit_statistics
from networks.attention_dispatch import AttentionParams


def _toy():
    torch.manual_seed(91)
    q = torch.randn(1, 5, 2, 4)
    k = torch.randn(1, 5, 2, 4)
    v = torch.randn(1, 5, 2, 4)
    inputs = {"reference": (q, k, v),
              "candidate": (q + 0.04, k + 0.08 * torch.randn_like(k),
                            v + 0.06 * torch.randn_like(v))}
    return inputs, torch.randn(1, 5, 8), AttentionParams.create_attention_params("torch")


def test_ten_cells_real_dispatch_endpoints_and_fixed_q():
    inputs, du, params = _toy()
    assert len(interpolation.CELLS) == 10
    assert len(set(interpolation.CELLS)) == 10
    assert interpolation.select_inputs(0, "R", inputs)[1] is inputs["reference"][1]
    assert interpolation.select_inputs(1, "C", inputs)[1] is inputs["candidate"][1]
    assert interpolation.select_inputs(0.5, "C", inputs)[0] is inputs["reference"][0]
    assert interpolation.select_inputs(0.5, "C", inputs)[2] is inputs["candidate"][2]
    cells = {}
    for side in interpolation.SIDES:
        for value in interpolation.LAMBDAS:
            name = interpolation.label(value, side)
            u, c = interpolation.replay_cell(value, side, inputs, du, params)
            repeated_u, repeated_c = interpolation.replay_cell(value, side, inputs, du, params)
            assert torch.equal(u, repeated_u)
            torch.testing.assert_close(c, repeated_c, atol=1e-6, rtol=1e-5)
            assert c.shape == (1, 5, 24)
            cells[name] = c
    assert not torch.equal(cells["k000_vR"], cells["k100_vR"])
    assert not torch.equal(cells["k050_vR"], cells["k050_vC"])
    for bad in (-0.1, 0.3, 1.1):
        with pytest.raises(ValueError):
            interpolation.select_inputs(bad, "R", inputs)


def test_statistics_axes_midpoint_and_tamper(monkeypatch):
    monkeypatch.setattr(interpolation.matrix, "SHAPE", (1, 1, 1, 5, 8))
    cells = {name: torch.full((1, 1, 1, 5, 24), float(index + 1))
             for index, name in enumerate(interpolation.CELLS)}
    report = interpolation.statistics(cells)
    audit_statistics({key: value.numpy() for key, value in cells.items()}, report)
    assert all(len(report["baseline"][side]) == 4 for side in interpolation.SIDES)
    assert all(len(report["adjacent"][side]) == 4 for side in interpolation.SIDES)
    assert len(report["v_contrast"]) == 5
    assert len(report["midpoint_nonlinearity"]) == 2
    bad = copy.deepcopy(report)
    bad["v_contrast"]["k050_vR__k050_vC"]["k"]["absolute_l2"] += 1
    with pytest.raises(ValueError, match="metric"):
        audit_statistics({key: value.numpy() for key, value in cells.items()}, bad)
    bad = copy.deepcopy(report)
    del bad["adjacent"]["R"]["k000_vR__k025_vR"]
    with pytest.raises(ValueError, match="labels"):
        audit_statistics({key: value.numpy() for key, value in cells.items()}, bad)


def test_evidence_budget_sha_and_statistics_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(interpolation.matrix, "SHAPE", (1, 1, 1, 5, 8))
    monkeypatch.setattr(interpolation.matrix, "WIDTH", 24)
    source = {"toy": True}
    cells = {name: torch.ones(1, 1, 1, 5, 24) for name in interpolation.CELLS}
    files = {name: interpolation.matrix.norm.core.parent.save_evidence(
        tmp_path, f"cell-{name}", {"c": value}, source, scope=interpolation.SCOPE)
        for name, value in cells.items()}
    metric = interpolation.components(next(iter(cells.values())), next(iter(cells.values())))["fused"]
    analysis = {"status": "analyzed", "scope": interpolation.SCOPE,
                "production_ready": False, "shape": list(interpolation.matrix.SHAPE),
                "source": source, "files": files,
                "disk_bytes": sum(item["disk_bytes"] for item in files.values()),
                "replay": dict.fromkeys(interpolation.CELLS, metric),
                "forward_bitwise_equal": {"reference_core": True},
                "statistics": interpolation.statistics(cells)}
    assert set(interpolation.validate_files(tmp_path, analysis)) == set(interpolation.CELLS)
    bad = copy.deepcopy(analysis)
    bad["files"]["k000_vR"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="manifest/digest"):
        interpolation.validate_files(tmp_path, bad)
    bad = copy.deepcopy(analysis)
    bad["disk_bytes"] = 1024**3 + 1
    with pytest.raises(ValueError, match="budget"):
        interpolation.validate_files(tmp_path, bad)
    bad = copy.deepcopy(analysis)
    bad["statistics"]["baseline"]["R"]["k025_vR"]["fused"]["absolute_l2"] += 1
    matrix_cells = {("RR" if value == 0 else "RC") + side: cells[interpolation.label(value, side)]
                    for side in interpolation.SIDES for value in (0, 1)}
    with pytest.raises(ValueError, match="statistics"):
        interpolation.validate_statistics(
            bad, cells, matrix_cells, {"reference": {"c": cells["k000_vR"]}})
