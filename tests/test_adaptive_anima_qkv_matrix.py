"""CPU contracts for actual post-RoPE Q/K/V substitution evidence."""

import copy

import numpy as np
import pytest
import torch

from bench.adaptive_runtime import anima_qkv_matrix as matrix
from bench.adaptive_runtime.audit_anima_qkv_matrix import _audit_axes
from library.anima.models import RMSNorm
from networks.attention_dispatch import AttentionParams


def _toy():
    torch.manual_seed(88)
    q_norm, k_norm = RMSNorm(4, eps=1e-6), RMSNorm(4, eps=1e-6)
    with torch.no_grad():
        q_norm.weight.copy_(torch.tensor([1.1, 0.8, 1.2, 0.9]))
        k_norm.weight.copy_(torch.tensor([0.7, 1.3, 0.9, 1.1]))
    angle = torch.randn(5, 1, 1, 4)
    rope = (angle.cos(), angle.sin())
    p = {"reference": torch.randn(1, 5, 24)}
    p["candidate"] = p["reference"] + 0.08 * torch.randn(1, 5, 24)
    inputs = {side: matrix.postrope(value, q_norm, k_norm, rope, 2, 4)
              for side, value in p.items()}
    du = torch.randn(1, 5, 8)
    return inputs, du, AttentionParams.create_attention_params("torch")


def test_actual_eight_cells_endpoints_and_repeated_vjp():
    inputs, du, params = _toy()
    assert len(matrix.CELLS) == 8
    assert len(matrix.axis_pairs()) == 3
    assert all(len(pairs) == 4 for pairs in matrix.axis_pairs().values())
    cells = {}
    for cell in matrix.CELLS:
        u, c = matrix.replay_cell(cell, inputs, du, params)
        repeated_u, repeated_c = matrix.replay_cell(cell, inputs, du, params)
        assert torch.equal(u, repeated_u)
        torch.testing.assert_close(c, repeated_c, atol=1e-6, rtol=1e-5)
        assert c.shape == (1, 5, 24)
        cells[cell] = c
    assert not torch.equal(cells["RRC"], cells["RRR"])
    assert not torch.equal(cells["RCR"], cells["RRR"])
    assert not torch.equal(cells["CRR"], cells["RRR"])
    u, _ = matrix.replay_cell("RRR", inputs, du, params)
    selected = [value.detach().requires_grad_(True) for value in inputs["reference"]]
    from networks.attention_dispatch import dispatch_attention
    baseline = dispatch_attention(list(selected), attn_params=params.for_attention_kind(is_selfattn=True))
    assert torch.equal(u, baseline)
    expected = torch.autograd.grad(baseline, selected, du)
    torch.testing.assert_close(cells["RRR"], matrix.norm._pack(*expected, (1, 5, 24)))


def test_axes_have_fixed_other_factors_and_tampering_fails(monkeypatch):
    monkeypatch.setattr(matrix, "SHAPE", (1, 1, 1, 5, 8))
    cells = {cell: torch.full((1, 1, 1, 5, 24), float(index + 1))
             for index, cell in enumerate(matrix.CELLS)}
    recorded = matrix.statistics(cells)
    audited = _audit_axes({key: value.numpy() for key, value in cells.items()}, recorded)
    assert set(audited) == {"q", "k", "v"}
    bad = copy.deepcopy(recorded)
    bad["q"]["RRR"]["k"]["absolute_l2"] += 1
    with pytest.raises(ValueError, match="metric"):
        _audit_axes({key: value.numpy() for key, value in cells.items()}, bad)
    assert all(sum(a != b for a, b in zip(*pair)) == 1
               for pairs in matrix.axis_pairs().values() for pair in pairs.values())


def test_evidence_budget_and_sha_rejection(tmp_path, monkeypatch):
    monkeypatch.setattr(matrix, "SHAPE", (1, 1, 1, 5, 8))
    monkeypatch.setattr(matrix, "WIDTH", 24)
    source = {"toy": True}
    cells = {cell: torch.ones(1, 1, 1, 5, 24) for cell in matrix.CELLS}
    files = {cell: matrix.norm.core.parent.save_evidence(
        tmp_path, f"cell-{cell}", {"c": value}, source, scope=matrix.SCOPE)
        for cell, value in cells.items()}
    metric = matrix.norm.core.stages.vjp._metric(next(iter(cells.values())), next(iter(cells.values())))
    analysis = {"status": "analyzed", "scope": matrix.SCOPE, "production_ready": False,
                "shape": list(matrix.SHAPE), "source": source, "files": files,
                "disk_bytes": sum(item["disk_bytes"] for item in files.values()),
                "replay": {cell: metric for cell in matrix.CELLS},
                "endpoint": {side: metric for side in matrix.SIDES},
                "forward_bitwise_equal": dict.fromkeys(matrix.SIDES, True),
                "axes": matrix.statistics(cells)}
    assert set(matrix.validate_files(tmp_path, analysis)) == set(matrix.CELLS)
    wrong = copy.deepcopy(analysis)
    wrong["files"]["RRR"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="manifest/digest"):
        matrix.validate_files(tmp_path, wrong)
    wrong = copy.deepcopy(analysis)
    wrong["disk_bytes"] = 1024**3 + 1
    with pytest.raises(ValueError, match="budget"):
        matrix.validate_files(tmp_path, wrong)
    wrong = copy.deepcopy(analysis)
    wrong["axes"]["q"]["RRR"]["fused"]["cosine"] = np.nan
    with pytest.raises(ValueError, match="axis statistics"):
        matrix.validate_statistics(wrong, cells, {}, {side: {"c": cells[cell]} for side, cell in
                                                     (("reference", "RRR"), ("candidate", "CCC"))})
