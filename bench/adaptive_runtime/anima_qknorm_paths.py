"""Isolated block17 Q/K normalization VJPs at fixed upstream core cotangent."""

import gc
import json
from pathlib import Path

from safetensors.torch import load_file
import torch

from bench.adaptive_runtime import anima_attention_core_paths as core
from bench.adaptive_runtime import anima_qknorm_math as norm_math
from bench.adaptive_runtime.anima_inputs import sha256_file
from library.anima.models import RMSNorm, apply_rotary_pos_emb_qk
from networks.attention_dispatch import dispatch_attention


SIDES = core.SIDES
SHAPE = core.SHAPE
WIDTH = 3 * SHAPE[-1]
SCOPE = "block17_reference_policy_isolated_qknorm_fixed_du_not_production"
TRACE_KEYS = {"c", "a"}
MATRIX_KEYS = {f"norm.{i}.{j}" for i in SIDES for j in SIDES}


def code_hashes():
    return {"analysis_sha256": sha256_file(Path(__file__)),
            "math_sha256": sha256_file(Path(norm_math.__file__)),
            "core_analysis_sha256": sha256_file(Path(core.__file__)),
            "block_source_sha256": sha256_file(Path(__file__).resolve().parents[2] / "library/anima/models.py"),
            "dispatch_source_sha256": sha256_file(Path(__file__).resolve().parents[2] / "networks/attention_dispatch.py")}


def source_evidence(core_run, upstream, stage_run, weights):
    core_run = Path(core_run)
    summary = json.loads((core_run / "summary.json").read_text())
    environment = json.loads((core_run / "environment.json").read_text())
    result = json.loads((core_run / "result.json").read_text())
    if (summary.get("status") != "completed" or summary.get("production_ready") is not False
            or Path(environment["source_run"]).resolve() != Path(upstream).resolve()
            or Path(environment["stage_run"]).resolve() != Path(stage_run).resolve()
            or result.get("status") != "ok" or result.get("analysis") != summary.get("analysis")):
        raise ValueError("Completed core result/source binding required")
    analysis = summary["analysis"]
    verified = core.validate_files(core_run, analysis)
    core.validate_statistics(analysis, verified)
    old_source, manifests, captured, prior = core.source_evidence(upstream, stage_run, weights)
    if old_source != analysis.get("source") or sha256_file(weights) != environment["weights_sha256"]:
        raise ValueError("Core result does not match actual upstream/stage/weights/code")
    if not torch.equal(verified["reference"]["du"], verified["candidate"]["du"]):
        raise ValueError("Core output-projection cotangent is not bitwise fixed")
    if (result.get("uuid") != analysis.get("uuid") or environment.get("gpu_uuid") != analysis.get("uuid")
            or result.get("driver_sha256") != environment.get("driver_sha256")):
        raise ValueError("Core GPU/driver binding mismatch")
    verified["norm_parameters"] = norm_math.read_norm_weights(
        weights, analysis["source"]["upstream_source"]["signature"]["weights_sha256"], 128)
    source = {"core_summary_sha256": sha256_file(core_run / "summary.json"),
              "core_evidence_sha256": {side: analysis["files"][side]["sha256"] for side in (*SIDES, "matrix")},
              "core_source": old_source, **code_hashes()}
    return source, manifests, captured, prior, verified


def _split(p, n_heads, head_dim):
    if p.ndim != 3 or p.shape[-1] != 3 * n_heads * head_dim:
        raise ValueError("Invalid fused QKV layout")
    return p.unflatten(-1, (3, n_heads, head_dim)).unbind(dim=-3)


def _pack(q, k, v, shape):
    return torch.stack((q, k, v), dim=-3).reshape(shape)


def replay_core(p, du, q_norm, k_norm, rope, params, n_heads, head_dim):
    """Return actual core output, cotangents and unsaved Q/K boundary activations."""
    p = p.detach().requires_grad_(True)
    q, k, v = _split(p, n_heads, head_dim)
    nq, nk = q_norm(q), k_norm(k)
    if rope is None:
        raise ValueError("Anima self-attention RoPE missing")
    rq, rk = apply_rotary_pos_emb_qk(nq, nk, rope, tensor_format="bshd")
    u = dispatch_attention([rq, rk, v], attn_params=params.for_attention_kind(is_selfattn=True))
    if u.shape != du.shape or u.dtype != torch.float32:
        raise ValueError("Isolated attention output contract changed")
    c_q, c_k, c_v = torch.autograd.grad(u, (rq, rk, v), du, retain_graph=True)
    a_q, a_k = torch.autograd.grad((rq, rk), (nq, nk), (c_q, c_k), retain_graph=True)
    dp = torch.autograd.grad(u, p, du)[0]
    c, a = _pack(c_q, c_k, c_v, p.shape), _pack(a_q, a_k, c_v, p.shape)
    n = torch.stack((nq, nk), dim=-3).flatten(-3)
    r = torch.stack((rq, rk), dim=-3).flatten(-3)
    return u.detach(), c.detach(), a.detach(), dp.detach(), n.detach(), r.detach()


def norm_vjp(p, a, q_norm, k_norm, n_heads, head_dim):
    p = p.detach().requires_grad_(True)
    q, k, v = _split(p, n_heads, head_dim)
    aq, ak, av = _split(a, n_heads, head_dim)
    nq, nk = q_norm(q), k_norm(k)
    gq, gk = torch.autograd.grad((nq, nk), (q, k), (aq, ak))
    return _pack(gq, gk, av, p.shape).detach()


def _cpu(value):
    return core.stages._cpu(value).reshape(*SHAPE[:-1], WIDTH)


def analyze(core_run, upstream, stage_run, weights, directory, device):
    if device.type != "cuda":
        raise ValueError("Monitored CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source, manifests, captured, _, old = source_evidence(core_run, upstream, stage_run, weights)
    kwargs = core.stages.capture.conditioning(manifests["reference"], captured["reference"], device)
    if kwargs["attn_params"].attn_mode != "torch":
        raise ValueError("FP32 torch attention required")
    block, adapters = core.stages.vjp._build_block(
        weights, Path(upstream) / "reference", manifests["reference"], policy="reference", device=device)
    attn = block.self_attn
    if (not attn.is_selfattn or attn.qkv_format != "bshd" or not isinstance(attn.v_norm, torch.nn.Identity)
            or attn.n_heads * attn.head_dim != SHAPE[-1]):
        raise ValueError("Actual self-attention configuration changed")
    n_heads, head_dim = attn.n_heads, attn.head_dim
    q_norm, k_norm = attn.q_norm, attn.k_norm
    if (not isinstance(q_norm, RMSNorm) or not isinstance(k_norm, RMSNorm)
            or q_norm.eps != 1e-6 or k_norm.eps != 1e-6 or head_dim != 128):
        raise ValueError("Actual Q/K RMSNorm contract changed")
    attn.q_norm = attn.k_norm = torch.nn.Identity()
    del attn, block, adapters
    gc.collect()
    torch.cuda.empty_cache()
    files, saved, matrix, checks, activations = {}, {}, {}, {}, {}
    fixed_du = old["reference"]["du"].reshape(1, 4200, SHAPE[-1]).to(device)
    try:
        for side in SIDES:
            p = old[side]["p"].reshape(1, 4200, WIDTH).to(device)
            u, c, a, dp, n, r = replay_core(p, fixed_du, q_norm, k_norm, kwargs["rope_cos_sin"],
                                            kwargs["attn_params"], n_heads, head_dim)
            if not torch.equal(core.stages._cpu(u).reshape(SHAPE), old[side]["u"]):
                raise ValueError("Isolated core output differs bitwise from source")
            checks[side] = core.stages.vjp._fidelity(
                "isolated core dp", _cpu(dp), old[side]["dp"], core.stages.vjp.GRADIENT_RELATIVE_L2_MAX)
            saved[side] = {"c": _cpu(c), "a": _cpu(a)}
            activations[side] = {"n": core.stages._cpu(n), "r": core.stages._cpu(r)}
            matrix[f"norm.{side}.{side}"] = _cpu(norm_vjp(p, a, q_norm, k_norm, n_heads, head_dim))
            core.stages.vjp._fidelity("isolated norm chain", matrix[f"norm.{side}.{side}"], _cpu(dp),
                                      core.stages.vjp.GRADIENT_RELATIVE_L2_MAX)
            files[side] = core.parent.save_evidence(directory, f"trace-{side}", saved[side], source, scope=SCOPE)
            del p, u, c, a, dp, n, r
            gc.collect()
            torch.cuda.empty_cache()
        for i in SIDES:
            p = old[i]["p"].reshape(1, 4200, WIDTH).to(device)
            for j in SIDES:
                if i != j:
                    a = saved[j]["a"].reshape(1, 4200, WIDTH).to(device)
                    matrix[f"norm.{i}.{j}"] = _cpu(norm_vjp(p, a, q_norm, k_norm, n_heads, head_dim))
                    del a
            del p
        files["matrix"] = core.parent.save_evidence(directory, "norm-matrix", matrix, source, scope=SCOPE)
    finally:
        del q_norm, k_norm
        gc.collect()
        torch.cuda.empty_cache()
    total = sum(item["disk_bytes"] for item in files.values())
    if total > core.parent.TOTAL_MAX_BYTES:
        raise ValueError("QK norm evidence exceeds aggregate 1GiB budget")
    metric = core.stages.vjp._metric
    return {"status": "analyzed", "scope": SCOPE, "production_ready": False,
            "uuid": manifests["reference"]["uuid"], "shape": list(SHAPE), "source": source,
            "files": files, "disk_bytes": total, "heads": n_heads, "head_dim": head_dim,
            "fidelity": checks,
            "norm_formula_checks": norm_math.check_norm_matrix(old, saved, matrix, old["norm_parameters"],
                                                                n_heads, head_dim),
            "boundaries": {key: metric(saved["reference"][key], saved["candidate"][key]) for key in TRACE_KEYS},
            "activation_boundaries": {key: metric(activations["reference"][key], activations["candidate"][key])
                                      for key in ("n", "r")},
            "contrasts": {"activation_fixed_cotangent": {j: metric(matrix[f"norm.reference.{j}"], matrix[f"norm.candidate.{j}"]) for j in SIDES},
                          "cotangent_fixed_activation": {i: metric(matrix[f"norm.{i}.reference"], matrix[f"norm.{i}.candidate"]) for i in SIDES}},
            "components": {axis: {side: {name: metric(matrix[f"norm.{pair[0]}.{pair[1]}"][..., t * SHAPE[-1]:(t + 1) * SHAPE[-1]],
                                                    matrix[f"norm.{other[0]}.{other[1]}"][..., t * SHAPE[-1]:(t + 1) * SHAPE[-1]])
                                             for t, name in enumerate(("q", "k", "v"))}
                                     for side, (pair, other) in rows.items()}
                           for axis, rows in {
                               "activation_fixed_cotangent": {j: (("reference", j), ("candidate", j)) for j in SIDES},
                               "cotangent_fixed_activation": {i: ((i, "reference"), (i, "candidate")) for i in SIDES}}.items()},
            "interpretation": "Controlled Q/K norm VJP axes at fixed du; contrasts are non-additive and do not identify initial rounding."}


def validate_files(directory, analysis):
    if (analysis.get("status") != "analyzed" or analysis.get("scope") != SCOPE
            or analysis.get("production_ready") is not False or analysis.get("shape") != list(SHAPE)
            or set(analysis.get("files", {})) != {*SIDES, "matrix"}
            or set(analysis.get("fidelity", {})) != set(SIDES)
            or set(analysis.get("activation_boundaries", {})) != {"n", "r"}
            or set(analysis.get("norm_formula_checks", {})) != MATRIX_KEYS
            or type(analysis.get("heads")) is not int or type(analysis.get("head_dim")) is not int
            or analysis["heads"] <= 0 or analysis["head_dim"] <= 0
            or analysis.get("heads") * analysis.get("head_dim") != SHAPE[-1]):
        raise ValueError("Incomplete QK norm analysis")
    verified, total = {}, 0
    for side in (*SIDES, "matrix"):
        name = "norm-matrix" if side == "matrix" else f"trace-{side}"
        folder = Path(directory) / name
        m = json.loads((folder / "manifest.json").read_text())
        path = folder / "evidence.safetensors"
        if (m != analysis["files"][side] or m.get("schema") != "anima_self_attention_paths_v1"
                or m.get("scope") != SCOPE or m.get("production_ready") is not False
                or m.get("source") != analysis["source"] or m.get("name") != name
                or m.get("file") != path.name or path.stat().st_size != m.get("disk_bytes")
                or path.stat().st_size > core.parent.MAX_BYTES or sha256_file(path) != m.get("sha256")):
            raise ValueError("QK norm evidence manifest/hash/budget mismatch")
        t = load_file(str(path))
        if (set(t) != (MATRIX_KEYS if side == "matrix" else TRACE_KEYS)
                or m.get("inventory") != {key: list(value.shape) for key, value in t.items()}
                or m.get("tensor_bytes") != sum(value.numel() * value.element_size() for value in t.values())
                or any(value.dtype != torch.float32 or tuple(value.shape) != (*SHAPE[:-1], WIDTH)
                       or not bool(torch.isfinite(value).all()) for value in t.values())):
            raise ValueError("QK norm evidence inventory/shape/dtype/finite mismatch")
        verified[side] = t
        total += path.stat().st_size
    if total != analysis.get("disk_bytes") or total > core.parent.TOTAL_MAX_BYTES:
        raise ValueError("QK norm aggregate budget mismatch")
    for value in analysis["fidelity"].values():
        if value.get("finite") is not True or value.get("relative_l2") is None or not 0 <= value["relative_l2"] <= core.stages.vjp.GRADIENT_RELATIVE_L2_MAX:
            raise ValueError("QK norm core fidelity failed")
    for value in analysis["activation_boundaries"].values():
        if value.get("finite") is not True or value.get("relative_l2") is None or value["relative_l2"] < 0:
            raise ValueError("QK norm activation summary invalid")
    for value in analysis["norm_formula_checks"].values():
        if (set(value) != {"q", "k", "v_bitwise_equal"} or value["v_bitwise_equal"] is not True
                or any(type(value[part]) not in (int, float) or not 0 <= value[part] <= norm_math.NORM_RELATIVE_L2_MAX
                       for part in ("q", "k"))):
            raise ValueError("QK norm formula summary invalid")
    return verified


def validate_statistics(analysis, verified, old):
    metric = core.stages.vjp._metric
    matrix = verified["matrix"]
    boundaries = {key: metric(verified["reference"][key], verified["candidate"][key]) for key in TRACE_KEYS}
    contrasts = {"activation_fixed_cotangent": {j: metric(matrix[f"norm.reference.{j}"], matrix[f"norm.candidate.{j}"]) for j in SIDES},
                 "cotangent_fixed_activation": {i: metric(matrix[f"norm.{i}.reference"], matrix[f"norm.{i}.candidate"]) for i in SIDES}}
    if boundaries != analysis.get("boundaries") or contrasts != analysis.get("contrasts"):
        raise ValueError("QK norm metrics differ from tensor evidence")
    for axis in contrasts:
        for side in SIDES:
            i, j = (("reference", side), ("candidate", side)) if axis == "activation_fixed_cotangent" else ((side, "reference"), (side, "candidate"))
            for t, name in enumerate(("q", "k", "v")):
                start, end = t * SHAPE[-1], (t + 1) * SHAPE[-1]
                actual = metric(matrix[f"norm.{i[0]}.{i[1]}"][..., start:end], matrix[f"norm.{j[0]}.{j[1]}"][..., start:end])
                if actual != analysis.get("components", {}).get(axis, {}).get(side, {}).get(name):
                    raise ValueError("QK norm QKV metric differs from tensor evidence")
    for side in SIDES:
        core.stages.vjp._fidelity("saved QK norm diagonal", matrix[f"norm.{side}.{side}"], old[side]["dp"],
                                  core.stages.vjp.GRADIENT_RELATIVE_L2_MAX)
    checks = norm_math.check_norm_matrix(old, verified, matrix, old["norm_parameters"],
                                         analysis["heads"], analysis["head_dim"])
    if checks != analysis.get("norm_formula_checks"):
        raise ValueError("QK norm formula checks differ from tensor evidence")
