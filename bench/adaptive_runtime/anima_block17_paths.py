"""Full-shape block17 stage VJPs at fixed reference conditioning and cotangent."""

from __future__ import annotations

import gc
import json
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch

from bench.adaptive_runtime import anima_block_vjp as vjp
from bench.adaptive_runtime import anima_block_vjp_capture as capture
from bench.adaptive_runtime.anima_block_stages import ResidualStageTrace, STAGES
from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime.diagnose_anima_block_vjp import validate_matrix_evidence
from library.training.auto_block_swap.process import write_result


SCOPE = "block17_reference_policy_stage_vjp_fixed_terminal_and_stage_cotangents_not_production"
TOTAL_MAX_BYTES = 1024**3


def source_evidence(root, weights):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    if summary.get("status") != "completed" or summary.get("production_ready") is not False:
        raise ValueError("Stage analysis requires completed upstream diagnostic evidence")
    reference = root / "reference/block-vjp/block-17"
    candidate = root / "candidate/block-vjp/block-17"
    comparison = json.loads((root / "comparison.json").read_text())
    pairs, _ = vjp.validate_pair(reference, candidate, comparison, block_index=17)
    manifests = {side: value[0] for side, value in zip(("reference", "candidate"), pairs, strict=True)}
    tensors = {side: value[1] for side, value in zip(("reference", "candidate"), pairs, strict=True)}
    old_analysis = json.loads((root / "analysis-block-17/result.json").read_text())["analysis"]
    validate_matrix_evidence(root / "analysis-block-17/matrix-evidence", old_analysis, capture_root=root)
    if old_analysis != summary["analyses"]["17"]:
        raise ValueError("Source matrix and upstream summary mismatch")
    ref, trial = tensors["reference"], tensors["candidate"]
    if (manifests["reference"]["conditioning"] != manifests["candidate"]["conditioning"]
            or {key for key in ref if key.startswith("conditioning.")} !=
            {key for key in trial if key.startswith("conditioning.")}
            or any(not torch.equal(ref[key], trial[key]) for key in ref if key.startswith("conditioning."))):
        raise ValueError("Stage comparison requires bitwise-identical conditioning")
    if sha256_file(weights) != manifests["reference"]["signature"]["weights_sha256"]:
        raise ValueError("Stage checkpoint source mismatch")
    matrix = load_file(str(root / "analysis-block-17/matrix-evidence/matrix.safetensors"))
    return manifests, tensors, matrix


def _cpu(value):
    value = value.detach().float().cpu().contiguous().clone()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("Nonfinite stage evidence")
    return value


def _baseline(block, x, cotangent, kwargs, device):
    x = x.to(device).detach().requires_grad_(True)
    y = block(x, **kwargs)
    (gx,) = torch.autograd.grad(y, x, grad_outputs=cotangent.to(device))
    return _cpu(y), _cpu(gx)


def _trace(block, x, cotangent, kwargs, reference_cotangents, baseline, device):
    trace = ResidualStageTrace(block)
    matrices, checks = {}, {}
    try:
        x = x.to(device).detach().requires_grad_(True)
        output = block(x, **kwargs)
        boundaries = trace.boundaries()
        gradients = trace.cotangents(output, cotangent.to(device))
        if not torch.equal(_cpu(output), baseline[0]):
            raise ValueError("Instrumented stage forward differs from unwrapped checkpoint")
        fidelity = vjp._fidelity("instrumented full block", _cpu(gradients[0]), baseline[1],
                                 vjp.GRADIENT_RELATIVE_L2_MAX)
        evidence = {f"activation.{i}": _cpu(value) for i, value in enumerate(boundaries)}
        evidence.update({f"cotangent.{i}": _cpu(value) for i, value in enumerate(gradients)})
        fixed = ([_cpu(value) for value in gradients[1:]] if reference_cotangents is None
                 else reference_cotangents)
        for index, stage in enumerate(trace.stages):
            g = fixed[index].to(device)
            complete, branch = trace.local_vjp(index, g)
            reconstructed = g + branch
            check = vjp._fidelity(f"{stage['name']} residual identity", _cpu(complete),
                                  _cpu(reconstructed), vjp.GRADIENT_RELATIVE_L2_MAX)
            if reference_cotangents is None:
                vjp._fidelity(f"{stage['name']} reference stage", _cpu(complete),
                              _cpu(gradients[index]), vjp.GRADIENT_RELATIVE_L2_MAX)
            evidence[f"gate.{stage['name']}"] = _cpu(stage["gate"])
            matrices[f"complete.{stage['name']}"] = _cpu(complete)
            matrices[f"branch.{stage['name']}"] = _cpu(branch)
            checks[stage["name"]] = check
        return evidence, matrices, fixed, {"forward_bitwise_equal": True,
                                          "whole_block": fidelity, "residual_identity": checks}
    finally:
        trace.restore()


def save_evidence(directory, name, tensors, source):
    if (not tensors or any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
                           for value in tensors.values())):
        raise ValueError("Invalid stage tensor evidence")
    total = sum(value.numel() * value.element_size() for value in tensors.values())
    if total > capture.MAX_BYTES:
        raise ValueError("Stage evidence exceeds independent 512MiB budget")
    directory = Path(directory) / name
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "evidence.safetensors"
    save_file(tensors, str(path))
    if path.stat().st_size > capture.MAX_BYTES:
        raise ValueError("Stage file exceeds independent 512MiB budget")
    manifest = {"schema": "anima_block17_stages_v1", "scope": SCOPE, "production_ready": False,
                "name": name, "file": path.name, "sha256": sha256_file(path), "tensor_bytes": total,
                "disk_bytes": path.stat().st_size,
                "inventory": {key: list(value.shape) for key, value in tensors.items()},
                "source": source}
    write_result(directory / "manifest.json", manifest)
    return manifest


def analyze(root, weights, directory, device):
    if device.type != "cuda":
        raise ValueError("Stage analysis requires monitored CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    manifests, tensors, old_matrix = source_evidence(root, weights)
    ref = manifests["reference"]
    source = {"capture_sha256": {side: value["sha256"] for side, value in manifests.items()},
              "signature": ref["signature"], "replay": ref["replay"],
              "block_index": 17, "policy": "reference", "scale": 1,
              "helper_sha256": sha256_file(Path(__file__).with_name("anima_block_stages.py")),
              "analysis_sha256": sha256_file(Path(__file__)),
              "block_source_sha256": sha256_file(Path(__file__).resolve().parents[2] / "library/anima/models.py")}
    block, adapters = vjp._build_block(weights, Path(root) / "reference", ref,
                                       policy="reference", device=device)
    saved, trace_values, matrices, checks = {}, {}, {}, {}
    fixed = None
    try:
        kwargs = capture.conditioning(ref, tensors["reference"], device)
        g = tensors["reference"]["g"]
        for side in ("reference", "candidate"):
            baseline = _baseline(block, tensors[side]["x"], g, kwargs, device)
            target = old_matrix[f"reference/{side}/reference"]
            source_fidelity = vjp._fidelity(f"{side} source matrix", baseline[1], target,
                                            vjp.GRADIENT_RELATIVE_L2_MAX)
            if side == "reference" and not torch.equal(baseline[0], tensors[side]["y"]):
                raise ValueError("Reference block forward no longer matches capture")
            evidence, local, reference_g, check = _trace(
                block, tensors[side]["x"], g, kwargs, fixed, baseline, device)
            if side == "reference":
                fixed = reference_g
            check["source_matrix"] = source_fidelity
            trace_values[side] = evidence
            checks[side] = check
            matrices.update({f"{key}.{side}": value for key, value in local.items()})
            saved[side] = save_evidence(directory, f"trace-{side}", evidence, source)
            gc.collect()
    finally:
        del block, adapters
        gc.collect()
        torch.cuda.empty_cache()
    saved["matrix"] = save_evidence(directory, "stage-matrix", matrices, source)
    total_disk = sum(value["disk_bytes"] for value in saved.values())
    if total_disk > TOTAL_MAX_BYTES:
        raise ValueError("Stage traces and matrices exceed aggregate 1GiB budget")
    rows = {}
    for index, stage in enumerate(STAGES):
        rows[stage] = {"input_activation": vjp._metric(trace_values["reference"][f"activation.{index}"],
                                                     trace_values["candidate"][f"activation.{index}"]),
                       "complete_fixed_cotangent": vjp._metric(matrices[f"complete.{stage}.reference"],
                                                               matrices[f"complete.{stage}.candidate"]),
                       "branch_fixed_cotangent": vjp._metric(matrices[f"branch.{stage}.reference"],
                                                             matrices[f"branch.{stage}.candidate"])}
    return {"status": "analyzed", "scope": SCOPE, "production_ready": False, "uuid": ref["uuid"],
            "source": source, "shape": list(capture.SHAPE), "stages": rows, "fidelity": checks,
            "files": saved, "disk_bytes": total_disk,
            "whole_block_fixed_terminal_cotangent": vjp._metric(
                trace_values["reference"]["cotangent.0"], trace_values["candidate"]["cotangent.0"]),
            "interpretation": "Stages include normalization, modulation and projection; contrasts are not additive causes."}
