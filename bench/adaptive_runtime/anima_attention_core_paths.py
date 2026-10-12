"""Fixed-terminal projection/core decomposition at actual block17 boundaries."""

import gc
import json
from pathlib import Path

from safetensors.torch import load_file
import torch

from bench.adaptive_runtime import anima_self_attention_paths as parent
from bench.adaptive_runtime.anima_attention_core_trace import AttentionCoreTrace
from bench.adaptive_runtime.anima_inputs import sha256_file


stages = parent.stages
SIDES, SHAPE = parent.SIDES, parent.SHAPE
SCOPE = "block17_reference_policy_qkv_core_output_projection_not_production"
KEYS = {"x", "z", "p", "u", "du", "dp", "h", "gate"}


def code_hashes():
    return {**parent.code_hashes(), "analysis_sha256": sha256_file(Path(__file__)),
            "helper_sha256": sha256_file(Path(__file__).with_name("anima_attention_core_trace.py")),
            "parent_analysis_sha256": sha256_file(Path(parent.__file__)),
            "boundary_helper_sha256": sha256_file(Path(parent.__file__).with_name("anima_self_attention_trace.py"))}


def source_evidence(upstream, stage_run, weights):
    source, manifests, tensors, prior = parent.source_evidence(upstream, stage_run, weights)
    return {**source, **code_hashes()}, manifests, tensors, prior


def _cpu(value):
    shape = (*SHAPE[:-1], 3 * SHAPE[-1]) if value.shape[-1] == 3 * SHAPE[-1] else SHAPE
    return stages._cpu(value).reshape(shape)


def trace_one(block, raw, terminal, block_g, kwargs, baseline, previous_branch, fixed_du, fixed_dp, device):
    trace = AttentionCoreTrace(block)
    try:
        x = raw.to(device).detach().requires_grad_(True)
        output = block(x, **kwargs)
        trace.validate()
        if not torch.equal(_cpu(output), baseline[0]):
            raise ValueError("Core trace forward differs from baseline")
        whole = trace.cotangents(output, block_g.to(device))[0]
        h = trace.attention_vjp(terminal.to(device))
        du = trace.projection_vjp(terminal.to(device))
        dp = trace.core_vjp(du.detach() if fixed_du is None else fixed_du.to(device))
        dz = trace.qkv_vjp(dp.detach())
        dx = trace.normalization_vjp(h.detach())
        fidelity = {"forward_bitwise_equal": True,
                    "whole_block": stages.vjp._fidelity("core whole block", _cpu(whole), baseline[1], stages.vjp.GRADIENT_RELATIVE_L2_MAX),
                    "source_branch": stages.vjp._fidelity("core source branch", _cpu(dx), previous_branch, stages.vjp.GRADIENT_RELATIVE_L2_MAX),
                    "qkv_chain": stages.vjp._fidelity("QKV-core chain", _cpu(dz), _cpu(h), stages.vjp.GRADIENT_RELATIVE_L2_MAX)}
        if fixed_du is not None:
            fidelity["fixed_output_projection"] = stages.vjp._fidelity("fixed output projection VJP", _cpu(du), fixed_du, stages.vjp.GRADIENT_RELATIVE_L2_MAX)
        if fixed_dp is not None:
            fixed = trace.qkv_vjp(fixed_dp.to(device))
        else:
            fixed = dz
        evidence = {"x": _cpu(x), "z": _cpu(trace.attention_input), "p": _cpu(trace.projected),
                    "u": _cpu(trace.core_output), "du": _cpu(du), "dp": _cpu(dp), "h": _cpu(h),
                    "gate": stages._cpu(trace.stages[0]["gate"])}
        return evidence, _cpu(fixed), fidelity
    finally:
        trace.restore()


def analyze(upstream, stage_run, weights, directory, device):
    if device.type != "cuda":
        raise ValueError("Monitored CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source, manifests, tensors, prior = source_evidence(upstream, stage_run, weights)
    block, adapters = stages.vjp._build_block(weights, Path(upstream) / "reference", manifests["reference"], policy="reference", device=device)
    evidence, checks, files, matrix = {}, {}, {}, {}
    terminal = prior["reference"]["cotangent.1"] * prior["reference"]["gate.self_attn"]
    matrix["terminal"] = terminal
    fixed_du = fixed_dp = None
    try:
        kwargs = stages.capture.conditioning(manifests["reference"], tensors["reference"], device)
        for side in SIDES:
            baseline = stages._baseline(block, tensors[side]["x"], tensors["reference"]["g"], kwargs, device)
            if not torch.equal(baseline[0], prior[side]["activation.3"]):
                raise ValueError("Core baseline no longer matches stage forward")
            row, fixed, fidelity = trace_one(block, tensors[side]["x"], terminal, tensors["reference"]["g"], kwargs,
                                             baseline, prior[side]["branch_dx"], fixed_du, fixed_dp, device)
            if not torch.equal(row["gate"], prior["reference"]["gate.self_attn"]):
                raise ValueError("Core gate differs from stage source")
            fidelity["source_whole_block"] = stages.vjp._fidelity("core source stage gradient", baseline[1], prior[side]["cotangent.0"], stages.vjp.GRADIENT_RELATIVE_L2_MAX)
            evidence[side], checks[side] = row, fidelity
            matrix[f"fixed_qkv.{side}"] = fixed
            if side == "reference":
                fixed_du, fixed_dp = row["du"], row["dp"]
            files[side] = parent.save_evidence(directory, f"trace-{side}", row, source, scope=SCOPE)
            gc.collect()
    finally:
        del block, adapters
        gc.collect()
        torch.cuda.empty_cache()
    files["matrix"] = parent.save_evidence(directory, "core-matrix", matrix, source, scope=SCOPE)
    total = sum(m["disk_bytes"] for m in files.values())
    if total > parent.TOTAL_MAX_BYTES:
        raise ValueError("Core aggregate 1GiB budget exceeded")
    metric = stages.vjp._metric
    return {"status": "analyzed", "scope": SCOPE, "production_ready": False, "uuid": manifests["reference"]["uuid"],
            "shape": list(SHAPE), "source": source, "files": files, "disk_bytes": total, "fidelity": checks,
            "boundaries": {key: metric(evidence["reference"][key], evidence["candidate"][key]) for key in ("x", "z", "p", "u", "du", "dp", "h")},
            "core_gradient_components": {name: metric(evidence["reference"]["dp"][..., i * SHAPE[-1]:(i + 1) * SHAPE[-1]],
                                                        evidence["candidate"]["dp"][..., i * SHAPE[-1]:(i + 1) * SHAPE[-1]]) for i, name in enumerate(("q", "k", "v"))},
            "fixed_qkv_cotangent": metric(matrix["fixed_qkv.reference"], matrix["fixed_qkv.candidate"]),
            "interpretation": "Core includes QK-norm, RoPE and attention. Fixed du removes output-projection cotangent changes; fixed dp tests QKV input dependence. Not additive attribution."}


def validate_files(directory, analysis):
    if (analysis.get("status") != "analyzed" or analysis.get("scope") != SCOPE
            or analysis.get("production_ready") is not False or analysis.get("shape") != list(SHAPE)
            or set(analysis.get("files", {})) != {*SIDES, "matrix"} or set(analysis.get("fidelity", {})) != set(SIDES)):
        raise ValueError("Incomplete core analysis")
    verified, total = {}, 0
    for side in (*SIDES, "matrix"):
        name = "core-matrix" if side == "matrix" else f"trace-{side}"
        folder = Path(directory) / name
        m = json.loads((folder / "manifest.json").read_text())
        path = folder / "evidence.safetensors"
        if (m != analysis["files"][side] or m.get("source") != analysis["source"]
                or m.get("schema") != "anima_self_attention_paths_v1" or m.get("scope") != SCOPE
                or m.get("production_ready") is not False or m.get("name") != name or m.get("file") != path.name
                or path.stat().st_size != m.get("disk_bytes") or path.stat().st_size > parent.MAX_BYTES
                or sha256_file(path) != m.get("sha256")):
            raise ValueError("Core evidence manifest/digest/budget mismatch")
        t = load_file(str(path))
        expected = {"terminal", "fixed_qkv.reference", "fixed_qkv.candidate"} if side == "matrix" else KEYS
        if (set(t) != expected or m["inventory"] != {k: list(v.shape) for k, v in t.items()}
                or m["tensor_bytes"] != sum(v.numel() * v.element_size() for v in t.values())
                or any(v.dtype != torch.float32 or not bool(torch.isfinite(v).all())
                       or tuple(v.shape) != ((1, 1, 1, 1, SHAPE[-1]) if k == "gate" else
                                            (*SHAPE[:-1], 3 * SHAPE[-1]) if k in ("p", "dp") else SHAPE) for k, v in t.items())):
            raise ValueError("Core evidence inventory/shape/dtype/finite mismatch")
        verified[side] = t
        total += path.stat().st_size
    if total != analysis["disk_bytes"] or total > parent.TOTAL_MAX_BYTES:
        raise ValueError("Core aggregate budget mismatch")
    for side, row in analysis["fidelity"].items():
        expected = {"forward_bitwise_equal", "whole_block", "source_branch", "qkv_chain", "source_whole_block"}
        if side == "candidate":
            expected.add("fixed_output_projection")
        if set(row) != expected or row["forward_bitwise_equal"] is not True:
            raise ValueError("Incomplete core fidelity")
        if any(m.get("finite") is not True or m.get("relative_l2") is None
               or not 0 <= m["relative_l2"] <= stages.vjp.GRADIENT_RELATIVE_L2_MAX for key, m in row.items() if key != "forward_bitwise_equal"):
            raise ValueError("Core fidelity failed")
    return verified


def validate_statistics(analysis, verified):
    metric = stages.vjp._metric
    boundaries = {key: metric(verified["reference"][key], verified["candidate"][key]) for key in ("x", "z", "p", "u", "du", "dp", "h")}
    components = {name: metric(verified["reference"]["dp"][..., i * SHAPE[-1]:(i + 1) * SHAPE[-1]],
                               verified["candidate"]["dp"][..., i * SHAPE[-1]:(i + 1) * SHAPE[-1]]) for i, name in enumerate(("q", "k", "v"))}
    fixed = metric(verified["matrix"]["fixed_qkv.reference"], verified["matrix"]["fixed_qkv.candidate"])
    if boundaries != analysis.get("boundaries") or components != analysis.get("core_gradient_components") or fixed != analysis.get("fixed_qkv_cotangent"):
        raise ValueError("Core statistics not bound to tensors")
    stages.vjp._fidelity("saved fixed output projection", verified["candidate"]["du"], verified["reference"]["du"], stages.vjp.GRADIENT_RELATIVE_L2_MAX)
