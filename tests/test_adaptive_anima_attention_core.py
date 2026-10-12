"""Independent graph/affine checks for fused QKV core boundaries."""

import copy
import json

import pytest
import torch
from torch import nn

from bench.adaptive_runtime import anima_attention_core_paths as paths
from bench.adaptive_runtime.anima_attention_core_trace import AttentionCoreTrace
from bench.adaptive_runtime.audit_anima_attention_core import audit
from tests.test_adaptive_anima_self_attention import ToyBlock


class ToyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(4, 12, bias=False)
        self.output_proj = nn.Linear(4, 4, bias=False)
        original = self.qkv_proj.forward
        # Mimic an applied low-rank adapter without replacing the module object.
        self.adapter = nn.Parameter(torch.randn(4, 12) * 0.01)
        self.qkv_proj.forward = lambda x: original(x) + x @ self.adapter

    def forward(self, x):
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        u = (q @ k.transpose(-1, -2) / 2).softmax(-1) @ v
        return self.output_proj(u)


def block():
    b = ToyBlock()
    b.self_attn = ToyAttention()
    return b


def test_actual_applied_projection_chain_and_fixed_cotangents(monkeypatch):
    monkeypatch.setattr(paths, "SHAPE", (1, 1, 2, 2, 4))
    torch.manual_seed(31)
    b = block()
    raw, terminal, g = (torch.randn(paths.SHAPE) for _ in range(3))
    evidence, fixed = {}, {}
    for side, x in (("reference", raw), ("candidate", raw + torch.randn_like(raw) * 0.03)):
        baseline = paths.stages._baseline(b, x, g, {}, torch.device("cpu"))
        xx = x.clone().requires_grad_(True)
        branch = b.self_attn((b.norm(xx) * 1.3 + 0.2).flatten(1, 3)).reshape(paths.SHAPE)
        dx = torch.autograd.grad(branch, xx, terminal)[0]
        ref = evidence.get("reference")
        row, qkv, fidelity = paths.trace_one(b, x, terminal, g, {}, baseline, dx,
                                             None if ref is None else ref["du"],
                                             None if ref is None else ref["dp"], torch.device("cpu"))
        assert fidelity["qkv_chain"]["relative_l2"] < 1e-5
        evidence[side], fixed[side] = row, qkv
    assert torch.equal(evidence["reference"]["du"], evidence["candidate"]["du"])
    assert torch.equal(fixed["reference"], fixed["candidate"])
    assert not torch.equal(evidence["reference"]["dp"], evidence["candidate"]["dp"])
    assert b.forward.__func__ is ToyBlock.forward
    assert not b.self_attn.qkv_proj._forward_hooks and not b.self_attn.output_proj._forward_pre_hooks


def test_hooks_restore_if_core_boundary_missing():
    b = block()
    trace = AttentionCoreTrace(b)
    try:
        # Bypassing output_proj deliberately violates the trace contract.
        b.self_attn.forward = lambda x: b.self_attn.qkv_proj(x)[..., :4]
        b(torch.randn(1, 1, 2, 2, 4, requires_grad=True))
        with pytest.raises(ValueError, match="boundary layout"):
            trace.validate()
    finally:
        trace.restore()
    assert not b.self_attn.qkv_proj._forward_hooks


def test_core_parent_validates_scope_shapes_hash_and_statistics(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SHAPE", (1, 1, 2, 2, 4))
    saved, verified = {}, {}
    for side in paths.SIDES:
        row = {k: torch.ones((1, 1, 1, 1, 4) if k == "gate" else
                             (1, 1, 2, 2, 12) if k in ("p", "dp") else paths.SHAPE) for k in paths.KEYS}
        saved[side] = paths.parent.save_evidence(tmp_path, f"trace-{side}", row, {}, scope=paths.SCOPE)
        verified[side] = row
    matrix = {k: torch.ones(paths.SHAPE) for k in ("terminal", "fixed_qkv.reference", "fixed_qkv.candidate")}
    saved["matrix"] = paths.parent.save_evidence(tmp_path, "core-matrix", matrix, {}, scope=paths.SCOPE)
    verified["matrix"] = matrix
    metric = {"finite": True, "relative_l2": 0}
    fidelity = {"forward_bitwise_equal": True, **dict.fromkeys(("whole_block", "source_branch", "qkv_chain", "source_whole_block"), metric)}
    analysis = {"status": "analyzed", "scope": paths.SCOPE, "production_ready": False, "source": {}, "shape": list(paths.SHAPE),
                "files": saved, "disk_bytes": sum(m["disk_bytes"] for m in saved.values()),
                "fidelity": {"reference": fidelity, "candidate": {**fidelity, "fixed_output_projection": metric}}}
    paths.validate_files(tmp_path, analysis)
    bad = copy.deepcopy(analysis)
    bad["fidelity"]["candidate"]["qkv_chain"]["relative_l2"] = 0.02
    with pytest.raises(ValueError, match="fidelity failed"):
        paths.validate_files(tmp_path, bad)
    with pytest.raises(ValueError, match="statistics not bound"):
        paths.validate_statistics(analysis, verified)
    analysis["boundaries"] = {key: paths.stages.vjp._metric(verified["reference"][key], verified["candidate"][key])
                              for key in ("x", "z", "p", "u", "du", "dp", "h")}
    analysis["core_gradient_components"] = {key: paths.stages.vjp._metric(verified["reference"]["dp"][..., :4], verified["candidate"]["dp"][..., :4])
                                           for key in ("q", "k", "v")}
    analysis["fixed_qkv_cotangent"] = paths.stages.vjp._metric(matrix["fixed_qkv.reference"], matrix["fixed_qkv.candidate"])
    paths.validate_statistics(analysis, verified)
    summary = {"status": "completed", "production_ready": False, "analysis": analysis}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    assert audit(tmp_path)["status"] == "audited"
    corrupted = copy.deepcopy(summary)
    corrupted["analysis"]["core_gradient_components"]["q"]["relative_l2"] = 0.3
    (tmp_path / "summary.json").write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="metric mismatch"):
        audit(tmp_path)
    (tmp_path / "core-matrix/evidence.safetensors").write_bytes(b"bad")
    with pytest.raises(ValueError, match="digest/budget"):
        paths.validate_files(tmp_path, analysis)
