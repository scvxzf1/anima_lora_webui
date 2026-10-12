"""Meaningful CPU checks for stage tracing under non-reentrant checkpoint."""

import torch
import copy
from torch import nn
from torch.utils.checkpoint import checkpoint
import pytest

from bench.adaptive_runtime.anima_block_stages import ResidualStageTrace, STAGES
from bench.adaptive_runtime import anima_block17_paths as paths
from bench.adaptive_runtime import diagnose_anima_block17_paths as driver


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradient_checkpointing = True
        self.fp32_residual = False

    def _gated_residual_add(self, residual, gate, branch):
        return residual + gate * branch

    def _forward(self, x, gain):
        for i in range(3):
            normalized = nn.functional.layer_norm(x, (x.shape[-1],))
            branch = torch.sin(normalized * gain * (i + 1))
            x = self._gated_residual_add(x, torch.full_like(gain, 0.3 + i * 0.1), branch)
        return x

    def forward(self, x, *, gain):
        return checkpoint(self._forward, x, gain, use_reentrant=False)


def test_checkpoint_trace_matches_original_and_each_local_residual_vjp():
    torch.manual_seed(7)
    block = ToyBlock()
    raw = torch.randn(1, 1, 2, 2, 4)
    g = torch.randn_like(raw)
    kwargs = {"gain": torch.ones(1, 1, 1, 1, 4)}
    baseline = paths._baseline(block, raw, g, kwargs, torch.device("cpu"))
    evidence, matrices, fixed, checks = paths._trace(
        block, raw, g, kwargs, None, baseline, torch.device("cpu"))
    assert len(fixed) == 3
    assert checks["whole_block"]["relative_l2"] == 0
    assert torch.equal(evidence["activation.3"], baseline[0])
    assert torch.equal(evidence["cotangent.0"], baseline[1])
    for i, name in enumerate(STAGES):
        assert torch.allclose(matrices[f"complete.{name}"], evidence[f"cotangent.{i}"])
        assert torch.allclose(matrices[f"complete.{name}"],
                              fixed[i] + matrices[f"branch.{name}"], atol=1e-6)
    # A second trace uses the same stage cotangents, even though its own chain differs.
    changed = raw + torch.randn_like(raw) * 0.01
    trial_base = paths._baseline(block, changed, g, kwargs, torch.device("cpu"))
    _, trial, _, _ = paths._trace(block, changed, g, kwargs, fixed, trial_base, torch.device("cpu"))
    assert any(not torch.equal(matrices[key], trial[key]) for key in matrices)
    assert block.forward.__func__ is ToyBlock.forward
    assert block._gated_residual_add.__func__ is ToyBlock._gated_residual_add


def test_checkpoint_recomputation_does_not_duplicate_stages():
    block = ToyBlock()
    trace = ResidualStageTrace(block)
    try:
        x = torch.randn(1, 1, 2, 2, 4, requires_grad=True)
        y = block(x, gain=torch.ones(1, 1, 1, 1, 4))
        assert len(trace.stages) == 3
        gradients = trace.cotangents(y, torch.randn_like(y))
        for i in range(3):
            trace.local_vjp(i, gradients[i + 1].detach())
        assert len(trace.stages) == 3 and not trace.recording
    finally:
        trace.restore()


def test_trace_rejects_unsupported_residual_policy():
    block = ToyBlock()
    block.fp32_residual = True
    with pytest.raises(ValueError, match="reference FP32"):
        ResidualStageTrace(block)


def test_stage_evidence_hash_and_budget(tmp_path, monkeypatch):
    values = {"complete.self_attn.reference": torch.ones(1, 1, 2, 2, 4)}
    result = paths.save_evidence(tmp_path, "matrix", values, {"source": "toy"})
    assert result["sha256"] == paths.sha256_file(tmp_path / "matrix/evidence.safetensors")
    with pytest.raises(FileExistsError):
        paths.save_evidence(tmp_path, "matrix", values, {})
    monkeypatch.setattr(paths.capture, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="512MiB"):
        paths.save_evidence(tmp_path, "too-large", values, {})
    assert not (tmp_path / "too-large").exists()


def test_parent_requires_complete_files_hashes_and_fidelity(tmp_path, monkeypatch):
    shape = (1, 1, 2, 2, 4)
    monkeypatch.setattr(paths.capture, "SHAPE", shape)
    source = {"state": "toy"}
    saved = {}
    for side in ("reference", "candidate"):
        tensors = {f"{part}.{i}": torch.ones(shape)
                   for part in ("activation", "cotangent") for i in range(4)}
        tensors.update({f"gate.{stage}": torch.ones(1, 1, 1, 1, 4) for stage in STAGES})
        saved[side] = paths.save_evidence(tmp_path, f"trace-{side}", tensors, source)
    matrix = {f"{part}.{stage}.{side}": torch.ones(shape)
              for part in ("complete", "branch") for stage in STAGES
              for side in ("reference", "candidate")}
    saved["matrix"] = paths.save_evidence(tmp_path, "stage-matrix", matrix, source)
    metric = {"finite": True, "relative_l2": 0}
    analysis = {"status": "analyzed", "scope": paths.SCOPE, "production_ready": False,
                "shape": list(shape), "source": source, "files": saved,
                "disk_bytes": sum(item["disk_bytes"] for item in saved.values()),
                "stages": dict.fromkeys(STAGES), "fidelity": {
                    side: {"forward_bitwise_equal": True, "whole_block": metric, "source_matrix": metric,
                           "residual_identity": dict.fromkeys(STAGES, metric)}
                    for side in ("reference", "candidate")}}
    driver.validate_files(tmp_path, analysis)
    broken = copy.deepcopy(analysis)
    broken["fidelity"]["candidate"]["whole_block"]["relative_l2"] = 0.02
    with pytest.raises(ValueError, match="fidelity failed"):
        driver.validate_files(tmp_path, broken)
    (tmp_path / "stage-matrix/evidence.safetensors").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="hash or budget"):
        driver.validate_files(tmp_path, analysis)
