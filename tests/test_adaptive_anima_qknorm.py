"""CPU contracts for the isolated production-layout Q/K norm VJP."""

import copy

import numpy as np
import pytest
import torch

from bench.adaptive_runtime import anima_qknorm_paths as paths
from bench.adaptive_runtime import anima_qknorm_math as norm_math
from bench.adaptive_runtime.audit_anima_qknorm import rope_vjp_numpy
from library.anima.models import RMSNorm, apply_rotary_pos_emb_qk
from networks.attention_dispatch import AttentionParams, dispatch_attention


def _toy():
    torch.manual_seed(71)
    q_norm, k_norm = RMSNorm(4, eps=1e-6), RMSNorm(4, eps=1e-6)
    with torch.no_grad():
        q_norm.weight.copy_(torch.tensor([1.2, 0.9, 1.1, 0.7]))
        k_norm.weight.copy_(torch.tensor([0.8, 1.3, 0.7, 1.1]))
    length = 5
    angle = torch.randn(length, 1, 1, 4)
    rope = (angle.cos(), angle.sin())
    params = AttentionParams.create_attention_params("torch")
    p = torch.randn(1, length, 24)
    du = torch.randn(1, length, 8)
    return q_norm, k_norm, rope, params, p, du


def _production_like(p, q_norm, k_norm, rope, params):
    q, k, v = p.unflatten(-1, (3, 2, 4)).unbind(-3)
    q, k = q_norm(q), k_norm(k)
    q, k = apply_rotary_pos_emb_qk(q, k, rope, tensor_format="bshd")
    return dispatch_attention([q, k, v], attn_params=params.for_attention_kind(is_selfattn=True))


def test_actual_layout_rope_dispatch_and_chain_vjp():
    q_norm, k_norm, rope, params, p, du = _toy()
    original = p.clone().requires_grad_(True)
    expected_u = _production_like(original, q_norm, k_norm, rope, params)
    expected_dp = torch.autograd.grad(expected_u, original, du)[0]
    u, c, a, dp, n, r = paths.replay_core(p, du, q_norm, k_norm, rope, params, 2, 4)
    assert torch.equal(u, expected_u)
    torch.testing.assert_close(dp, expected_dp, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(paths.norm_vjp(p, a, q_norm, k_norm, 2, 4), dp, rtol=1e-5, atol=1e-6)
    assert c.shape == a.shape == dp.shape == p.shape
    assert n.shape == r.shape == (1, 5, 16)
    assert not torch.equal(c, a)
    assert torch.equal(paths._split(c, 2, 4)[2], paths._split(a, 2, 4)[2])
    c_q = paths._split(c, 2, 4)[0].numpy()
    a_q = paths._split(a, 2, 4)[0].numpy()
    cos, sin = (entry.transpose(0, 1).numpy() for entry in rope)
    np.testing.assert_allclose(rope_vjp_numpy(c_q, cos, sin), a_q, rtol=1e-6, atol=1e-6)


def test_cross_cotangent_matrix_is_not_diagonal_substitution():
    q_norm, k_norm, rope, params, p_ref, du = _toy()
    p_cand = p_ref + torch.randn_like(p_ref) * 0.07
    rows = {}
    for side, p in (("reference", p_ref), ("candidate", p_cand)):
        _, _, a, dp, _, _ = paths.replay_core(p, du, q_norm, k_norm, rope, params, 2, 4)
        rows[side] = (p, a, dp)
    cells = {(i, j): paths.norm_vjp(rows[i][0], rows[j][1], q_norm, k_norm, 2, 4)
             for i in rows for j in rows}
    for side in rows:
        torch.testing.assert_close(cells[side, side], rows[side][2], rtol=1e-5, atol=1e-6)
    assert not torch.equal(cells["reference", "candidate"], cells["reference", "reference"])
    assert not torch.equal(cells["candidate", "reference"], cells["reference", "reference"])
    # V is Identity: its local VJP does not depend on p at fixed cotangent.
    assert torch.equal(paths._split(cells["reference", "reference"], 2, 4)[2],
                       paths._split(cells["candidate", "reference"], 2, 4)[2])


def test_wrong_fused_layout_and_missing_rope_rejected():
    q_norm, k_norm, rope, params, p, du = _toy()
    with pytest.raises(ValueError, match="layout"):
        paths.replay_core(p[..., :-1], du, q_norm, k_norm, rope, params, 2, 4)
    with pytest.raises(ValueError, match="RoPE"):
        paths.replay_core(p, du, q_norm, k_norm, None, params, 2, 4)


def test_evidence_digest_and_statistic_tampering_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SHAPE", (1, 1, 1, 5, 8))
    monkeypatch.setattr(paths, "WIDTH", 24)
    source = {"test": "binding"}
    value = torch.ones(1, 1, 1, 5, 24)
    files = {}
    for side in paths.SIDES:
        files[side] = paths.core.parent.save_evidence(
            tmp_path, f"trace-{side}", {"a": value.clone(), "c": value.clone()}, source, scope=paths.SCOPE)
    cells = {f"norm.{i}.{j}": value.clone() for i in paths.SIDES for j in paths.SIDES}
    files["matrix"] = paths.core.parent.save_evidence(tmp_path, "norm-matrix", cells, source, scope=paths.SCOPE)
    metric = paths.core.stages.vjp._metric(value, value)
    analysis = {"status": "analyzed", "scope": paths.SCOPE, "production_ready": False,
                "shape": list(paths.SHAPE), "source": source, "files": files,
                "disk_bytes": sum(v["disk_bytes"] for v in files.values()), "heads": 2, "head_dim": 4,
                "fidelity": {side: metric for side in paths.SIDES},
                "boundaries": {key: metric for key in ("a", "c")},
                "activation_boundaries": {key: metric for key in ("n", "r")},
                "norm_formula_checks": {key: {"q": 0.0, "k": 0.0, "v_bitwise_equal": True}
                                        for key in paths.MATRIX_KEYS},
                "contrasts": {axis: {side: metric for side in paths.SIDES}
                              for axis in ("activation_fixed_cotangent", "cotangent_fixed_activation")},
                "components": {axis: {side: {name: metric for name in ("q", "k", "v")}
                                      for side in paths.SIDES}
                               for axis in ("activation_fixed_cotangent", "cotangent_fixed_activation")}}
    verified = paths.validate_files(tmp_path, analysis)
    old = {side: {"dp": value} for side in paths.SIDES}
    with pytest.raises(ValueError, match="QKV metric"):
        paths.validate_statistics(analysis, verified, old)
    bad = copy.deepcopy(analysis)
    bad["files"]["reference"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="manifest/hash"):
        paths.validate_files(tmp_path, bad)


def test_analytic_cross_cell_rejects_resigned_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SHAPE", (1, 1, 1, 5, 8))
    monkeypatch.setattr(paths, "WIDTH", 24)
    q_norm, k_norm, _, _, _, _ = _toy()
    generator = torch.Generator().manual_seed(112)
    old, traces, cells = {}, {}, {}
    for side in paths.SIDES:
        p = torch.randn(1, 1, 1, 5, 24, generator=generator)
        a = torch.randn(1, 1, 1, 5, 24, generator=generator)
        old[side] = {"p": p, "dp": paths.norm_vjp(p.reshape(1, 5, 24), a.reshape(1, 5, 24),
                                                  q_norm, k_norm, 2, 4).reshape_as(p)}
        traces[side] = {"a": a, "c": a.clone()}
    for i in paths.SIDES:
        for j in paths.SIDES:
            cells[f"norm.{i}.{j}"] = paths.norm_vjp(old[i]["p"].reshape(1, 5, 24),
                                                      traces[j]["a"].reshape(1, 5, 24),
                                                      q_norm, k_norm, 2, 4).reshape(1, 1, 1, 5, 24)
    weights = {"q": q_norm.weight.detach().numpy(), "k": k_norm.weight.detach().numpy()}
    old["norm_parameters"] = weights
    source = {"test": "signed source"}
    metric = paths.core.stages.vjp._metric

    def write_analysis(directory, formula_checks=None):
        files = {side: paths.core.parent.save_evidence(
            directory, f"trace-{side}", traces[side], source, scope=paths.SCOPE)
            for side in paths.SIDES}
        files["matrix"] = paths.core.parent.save_evidence(
            directory, "norm-matrix", cells, source, scope=paths.SCOPE)
        contrasts = {
            "activation_fixed_cotangent": {j: metric(cells[f"norm.reference.{j}"],
                                                      cells[f"norm.candidate.{j}"]) for j in paths.SIDES},
            "cotangent_fixed_activation": {i: metric(cells[f"norm.{i}.reference"],
                                                      cells[f"norm.{i}.candidate"]) for i in paths.SIDES}}
        pairs = {"activation_fixed_cotangent": {j: (f"norm.reference.{j}", f"norm.candidate.{j}")
                                                   for j in paths.SIDES},
                 "cotangent_fixed_activation": {i: (f"norm.{i}.reference", f"norm.{i}.candidate")
                                                  for i in paths.SIDES}}
        components = {axis: {side: {name: metric(cells[left][..., part * 8:(part + 1) * 8],
                                                 cells[right][..., part * 8:(part + 1) * 8])
                                     for part, name in enumerate(("q", "k", "v"))}
                             for side, (left, right) in rows.items()} for axis, rows in pairs.items()}
        return {"status": "analyzed", "scope": paths.SCOPE, "production_ready": False,
                "shape": list(paths.SHAPE), "source": source, "files": files,
                "disk_bytes": sum(item["disk_bytes"] for item in files.values()),
                "heads": 2, "head_dim": 4,
                "fidelity": {side: metric(old[side]["dp"], old[side]["dp"]) for side in paths.SIDES},
                "activation_boundaries": {key: metric(old["reference"]["dp"], old["candidate"]["dp"])
                                          for key in ("n", "r")},
                "boundaries": {key: metric(traces["reference"][key], traces["candidate"][key])
                               for key in ("a", "c")},
                "contrasts": contrasts, "components": components,
                "norm_formula_checks": formula_checks if formula_checks is not None else
                norm_math.check_norm_matrix(old, traces, cells, weights, 2, 4)}

    analysis = write_analysis(tmp_path / "original")
    paths.validate_statistics(analysis, paths.validate_files(tmp_path / "original", analysis), old)
    cells["norm.reference.candidate"] = cells["norm.reference.candidate"].clone()
    cells["norm.reference.candidate"][..., 0] += 0.25
    signed = write_analysis(tmp_path / "resigned", analysis["norm_formula_checks"])
    with pytest.raises(ValueError, match="analytic VJP"):
        paths.validate_statistics(signed, paths.validate_files(tmp_path / "resigned", signed), old)
    cells["norm.reference.candidate"][..., 0] -= 0.25
    cells["norm.reference.candidate"][..., 16] += 0.25
    with pytest.raises(ValueError, match="Identity V"):
        norm_math.check_norm_matrix(old, traces, cells, weights, 2, 4)
