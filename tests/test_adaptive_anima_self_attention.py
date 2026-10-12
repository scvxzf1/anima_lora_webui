"""CPU graph fidelity and cross-cotangent matrix checks before real-model analysis."""

import copy
import json

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from bench.adaptive_runtime import anima_self_attention_paths as paths
from bench.adaptive_runtime.anima_self_attention_trace import SelfAttentionTrace
from bench.adaptive_runtime.audit_anima_self_attention import audit


class ToyAttention(nn.Module):
    def forward(self, x):
        # Token coupling and nonlinearity make the boundary cotangent input-dependent.
        return torch.sin(x) + x.softmax(dim=1)


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradient_checkpointing = True
        self.fp32_residual = False
        self.norm = nn.LayerNorm(4, elementwise_affine=False)
        self.self_attn = ToyAttention()

    def _gated_residual_add(self, x, gate, branch):
        return x + gate * branch

    def _forward(self, x):
        shape = x.shape
        z = (self.norm(x) * 1.3 + 0.2).flatten(1, 3)
        x = self._gated_residual_add(x, torch.full((1, 1, 1, 1, 4), 0.3), self.self_attn(z).reshape(shape))
        for _ in range(2):
            x = self._gated_residual_add(x, torch.ones(1, 1, 1, 1, 4), x.tanh())
        return x

    def forward(self, x):
        return checkpoint(self._forward, x, use_reentrant=False)


def trace(block, raw, terminal):
    tracer = SelfAttentionTrace(block)
    try:
        x = raw.detach().requires_grad_(True)
        y = block(x)
        tracer.validate()
        h = tracer.attention_vjp(terminal)
        dx = tracer.branch_vjp(terminal)
        chain = tracer.normalization_vjp(h.detach())
        assert torch.allclose(dx, chain, atol=1e-6, rtol=1e-5)
        assert len(tracer.stages) == 3
        return tracer, x, y, h.detach(), dx.detach()
    except BaseException:
        tracer.restore()
        raise


def test_cross_cotangent_matrix_matches_uncheckpointed_chain():
    torch.manual_seed(17)
    block = ToyBlock()
    raw = torch.randn(1, 1, 2, 2, 4)
    terminal = torch.randn_like(raw)
    r, x, y, h_r, dx_r = trace(block, raw, terminal)
    try:
        # The recorder introduces no extra forward operations or altered whole gradients.
        plain_x = raw.clone().requires_grad_(True)
        plain_y = block._forward(plain_x)
        assert torch.equal(y, plain_y)
        assert torch.allclose(torch.autograd.grad(y, x, torch.ones_like(y), retain_graph=True)[0],
                              torch.autograd.grad(plain_y, plain_x, torch.ones_like(y))[0])
    finally:
        r.restore()
    changed = raw + torch.randn_like(raw) * 0.03
    c, _, _, h_c, dx_c = trace(block, changed, terminal)
    try:
        cross = c.normalization_vjp(h_r)
        direct_x = changed.clone().requires_grad_(True)
        direct_z = (block.norm(direct_x) * 1.3 + 0.2).flatten(1, 3)
        expected = torch.autograd.grad(direct_z, direct_x, h_r)[0]
        assert torch.allclose(cross, expected)
        assert not torch.equal(h_r, h_c) and not torch.equal(dx_r, dx_c)
        for _ in range(3):
            assert torch.allclose(c.normalization_vjp(h_c), dx_c, atol=1e-6)
    finally:
        c.restore()
    assert not block.self_attn._forward_hooks and not block.self_attn._forward_pre_hooks
    assert block.forward.__func__ is ToyBlock.forward


def test_trace_one_restores_on_failed_fidelity(monkeypatch):
    monkeypatch.setattr(paths, "SHAPE", (1, 1, 2, 2, 4))
    block = ToyBlock()
    raw = torch.randn(paths.SHAPE)
    with pytest.raises(ValueError, match="differs from baseline"):
        paths.trace_one(block, raw, torch.ones_like(raw), torch.ones_like(raw), {},
                        (torch.full_like(raw, 999), torch.ones_like(raw)), torch.ones_like(raw), {}, torch.device("cpu"))
    assert block.forward.__func__ is ToyBlock.forward
    assert not block.self_attn._forward_hooks and not block.self_attn._forward_pre_hooks


def test_trace_one_preserves_whole_gradient_and_cross_cell(monkeypatch):
    monkeypatch.setattr(paths, "SHAPE", (1, 1, 2, 2, 4))
    torch.manual_seed(11)
    block = ToyBlock()
    raw, terminal, g = (torch.randn(paths.SHAPE) for _ in range(3))
    baseline = paths.stages._baseline(block, raw, g, {}, torch.device("cpu"))
    plain_x = raw.clone().requires_grad_(True)
    plain_branch = block.self_attn((block.norm(plain_x) * 1.3 + 0.2).flatten(1, 3)).reshape(paths.SHAPE)
    expected = torch.autograd.grad(plain_branch, plain_x, terminal)[0]
    evidence, own, cells, fidelity = paths.trace_one(
        block, raw, terminal, g, {}, baseline, expected, {}, torch.device("cpu"))
    assert not cells and fidelity["forward_bitwise_equal"]
    assert torch.allclose(own, evidence["branch_dx"], atol=1e-6)
    _, _, cross, _ = paths.trace_one(block, raw, terminal, g, {}, baseline, expected,
                                    {"reference": evidence["h"]}, torch.device("cpu"))
    assert torch.equal(cross["reference"], own)


def test_parent_hash_inventory_and_fidelity_gates(tmp_path, monkeypatch):
    shape = (1, 1, 2, 2, 4)
    monkeypatch.setattr(paths, "SHAPE", shape)
    source = {"state": "toy"}
    saved = {}
    for side in (*paths.SIDES, "reference_replay"):
        values = {key: torch.ones((1, 1, 1, 1, 4) if key == "gate" else shape) for key in paths.TRACE_KEYS}
        saved[side] = paths.save_evidence(tmp_path, f"trace-{side}", values, source)
    saved["matrix"] = paths.save_evidence(tmp_path, "norm-matrix", {key: torch.ones(shape) for key in paths.MATRIX_KEYS}, source)
    metric = {"finite": True, "relative_l2": 0}
    row = {"forward_bitwise_equal": True, **dict.fromkeys(("whole_block", "source_branch", "chain_identity"), metric)}
    analysis = {"status": "analyzed", "scope": paths.SCOPE, "production_ready": False, "shape": list(shape),
                "source": source, "files": saved, "disk_bytes": sum(m["disk_bytes"] for m in saved.values()),
                "fidelity": {side: {**row, "source_whole_block": metric} for side in paths.SIDES}}
    analysis["fidelity"]["reference_replay"] = {**row, "repeated_gradient": dict.fromkeys(("h", "branch_dx"), metric)}
    paths.validate_files(tmp_path, analysis)
    broken = copy.deepcopy(analysis)
    broken["fidelity"]["candidate"]["chain_identity"]["relative_l2"] = 0.1
    with pytest.raises(ValueError, match="fidelity failed"):
        paths.validate_files(tmp_path, broken)
    (tmp_path / "norm-matrix/evidence.safetensors").write_bytes(b"bad")
    with pytest.raises(ValueError, match="manifest/hash/budget"):
        paths.validate_files(tmp_path, analysis)


def test_evidence_budget_is_enforced_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="512MiB"):
        paths.save_evidence(tmp_path, "too-large", {"x": torch.ones(4)}, {})
    assert not (tmp_path / "too-large").exists()


def test_statistics_rejects_changed_axis_and_broken_chain():
    values = {side: {key: torch.ones(2) for key in paths.TRACE_KEYS} for side in (*paths.SIDES, "reference_replay")}
    values["matrix"] = {key: torch.ones(2) for key in paths.MATRIX_KEYS}
    metric = paths.stages.vjp._metric(torch.ones(2), torch.ones(2))
    report = {"normalization_contrasts": {axis: dict.fromkeys(paths.SIDES, metric)
                                        for axis in ("activation_fixed_cotangent", "cotangent_fixed_activation")},
              "boundaries": dict.fromkeys(("x", "z", "b", "h", "branch_dx"), metric)}
    paths.validate_statistics(report, values)
    values["reference_replay"]["h"] = torch.full((2,), 1.001)
    with pytest.raises(ValueError, match="saved repeated reference h"):
        paths.validate_statistics(report, values)
    values["reference_replay"]["h"] = torch.ones(2)
    wrong = copy.deepcopy(report)
    wrong["normalization_contrasts"]["activation_fixed_cotangent"]["reference"]["relative_l2"] = 0.1
    with pytest.raises(ValueError, match="statistics not bound"):
        paths.validate_statistics(wrong, values)
    for side in paths.SIDES:
        values[side]["branch_dx"] = torch.full((2,), 2.0)
    report["boundaries"]["branch_dx"] = paths.stages.vjp._metric(values["reference"]["branch_dx"], values["candidate"]["branch_dx"])
    with pytest.raises(ValueError, match="diagonal chain identity"):
        paths.validate_statistics(report, values)


def test_numpy_audit_reads_actual_files_and_rejects_wrong_statistics(tmp_path):
    files = {}
    for side in (*paths.SIDES, "reference_replay"):
        files[side] = paths.save_evidence(tmp_path, f"trace-{side}",
                                         {key: torch.ones(2) for key in paths.TRACE_KEYS}, {})
    files["matrix"] = paths.save_evidence(tmp_path, "norm-matrix", {key: torch.ones(2) for key in paths.MATRIX_KEYS}, {})
    metric = paths.stages.vjp._metric(torch.ones(2), torch.ones(2))
    analysis = {"files": files, "normalization_contrasts": {axis: dict.fromkeys(paths.SIDES, metric)
                                                          for axis in ("activation_fixed_cotangent", "cotangent_fixed_activation")},
                "boundaries": dict.fromkeys(("x", "z", "b", "h", "branch_dx"), metric)}
    summary = {"status": "completed", "production_ready": False, "analysis": analysis}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert audit(tmp_path)["status"] == "audited"
    corrupted = copy.deepcopy(summary)
    corrupted["analysis"]["normalization_contrasts"]["activation_fixed_cotangent"]["reference"]["relative_l2"] = 0.02
    (tmp_path / "summary.json").write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="metric mismatch"):
        audit(tmp_path)
