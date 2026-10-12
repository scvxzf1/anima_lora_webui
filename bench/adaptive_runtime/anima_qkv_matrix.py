"""Actual post-RoPE Q/K/V activation substitutions at fixed block17 du."""

import gc
import itertools
import json
from pathlib import Path

from safetensors.torch import load_file
import torch

from bench.adaptive_runtime import anima_qknorm_paths as norm
from bench.adaptive_runtime.anima_inputs import sha256_file
from library.anima.models import RMSNorm, apply_rotary_pos_emb_qk
from networks.attention_dispatch import dispatch_attention


SIDES = norm.SIDES
CELLS = tuple("".join(bits) for bits in itertools.product("RC", repeat=3))
SHAPE, WIDTH = norm.SHAPE, norm.WIDTH
SCOPE = "block17_reference_policy_actual_postrope_qkv_fixed_du_not_production"


def code_hashes():
    return {"analysis_sha256": sha256_file(Path(__file__)),
            "qknorm_analysis_sha256": sha256_file(Path(norm.__file__)),
            "model_source_sha256": sha256_file(Path(__file__).resolve().parents[2] / "library/anima/models.py"),
            "dispatch_source_sha256": sha256_file(Path(__file__).resolve().parents[2] / "networks/attention_dispatch.py")}


def source_evidence(qknorm_run, core_run, upstream, stage_run, weights):
    root = Path(qknorm_run)
    summary = json.loads((root / "summary.json").read_text())
    result = json.loads((root / "result.json").read_text())
    environment = json.loads((root / "environment.json").read_text())
    if (summary.get("status") != "completed" or summary.get("production_ready") is not False
            or result.get("status") != "ok" or result.get("analysis") != summary.get("analysis")
            or result.get("uuid") != summary["analysis"].get("uuid")
            or result.get("driver_sha256") != environment.get("driver_sha256")
            or environment.get("driver_sha256") != sha256_file(Path(norm.__file__).with_name("diagnose_anima_qknorm.py"))
            or any(Path(environment[key]).resolve() != Path(path).resolve() for key, path in
                   (("core_run", core_run), ("source_run", upstream), ("stage_run", stage_run)))
            or environment.get("weights_sha256") != sha256_file(weights)
            or any(environment.get(key) != value for key, value in norm.code_hashes().items())):
        raise ValueError("Completed QK norm result/environment/source binding required")
    source, manifests, captured, prior, old = norm.source_evidence(core_run, upstream, stage_run, weights)
    if summary["analysis"].get("source") != source:
        raise ValueError("QK norm source evidence changed")
    verified = norm.validate_files(root, summary["analysis"])
    norm.validate_statistics(summary["analysis"], verified, old)
    binding = {"qknorm_summary_sha256": sha256_file(root / "summary.json"),
               "qknorm_evidence_sha256": {name: item["sha256"] for name, item in summary["analysis"]["files"].items()},
               "qknorm_source": source, **code_hashes()}
    return binding, manifests, captured, old, verified


def make_norms(weights, device, head_dim=128):
    q, k = (RMSNorm(head_dim, eps=1e-6).to(device) for _ in range(2))
    with torch.no_grad():
        for module, part in ((q, "q"), (k, "k")):
            module.weight.copy_(torch.from_numpy(weights[part]).to(device))
    return q, k


def postrope(p, q_norm, k_norm, rope, heads, head_dim):
    q, k, v = norm._split(p, heads, head_dim)
    rq, rk = apply_rotary_pos_emb_qk(q_norm(q), k_norm(k), rope, tensor_format="bshd")
    return tuple(value.detach() for value in (rq, rk, v))


def replay_cell(cell, inputs, du, params):
    selected = tuple(inputs["reference" if bit == "R" else "candidate"][index]
                     .detach().requires_grad_(True) for index, bit in enumerate(cell))
    u = dispatch_attention(list(selected), attn_params=params.for_attention_kind(is_selfattn=True))
    if u.shape != du.shape or u.dtype != torch.float32:
        raise ValueError("Attention output contract changed")
    gradients = torch.autograd.grad(u, selected, du)
    return u.detach(), norm._pack(*gradients, (du.shape[0], du.shape[1], 3 * du.shape[2])).detach()


def axis_pairs():
    return {axis: {"".join(fixed): ("".join((*fixed[:index], "R", *fixed[index + 1:])),
                                      "".join((*fixed[:index], "C", *fixed[index + 1:])))
                   for fixed in itertools.product("RC", repeat=3) if fixed[index] == "R"}
            for index, axis in enumerate("qkv")}


def statistics(cells):
    metric = norm.core.stages.vjp._metric
    result = {}
    for axis, pairs in axis_pairs().items():
        result[axis] = {}
        for fixed, (left, right) in pairs.items():
            a, b = cells[left], cells[right]
            result[axis][fixed] = {"fused": metric(a, b)}
            for index, part in enumerate("qkv"):
                sl = slice(index * SHAPE[-1], (index + 1) * SHAPE[-1])
                result[axis][fixed][part] = metric(a[..., sl], b[..., sl])
    return result


def _endpoint_chain(side, c, old, q_norm, k_norm, rope, device):
    p = old[side]["p"].reshape(1, 4200, WIDTH).to(device).detach().requires_grad_(True)
    q, k, v = norm._split(p, 16, 128)
    nq, nk = q_norm(q), k_norm(k)
    rq, rk = apply_rotary_pos_emb_qk(nq, nk, rope, tensor_format="bshd")
    cq, ck, cv = norm._split(c.reshape(1, 4200, WIDTH).to(device), 16, 128)
    a = torch.autograd.grad((rq, rk), (nq, nk), (cq, ck), retain_graph=True)
    dp = torch.autograd.grad((nq, nk), (q, k), a)
    packed = norm._cpu(norm._pack(dp[0], dp[1], cv, p.shape))
    return norm.core.stages.vjp._fidelity("post-RoPE endpoint dp", packed, old[side]["dp"], 1e-4)


def analyze(qknorm_run, core_run, upstream, stage_run, weights, directory, device):
    if device.type != "cuda":
        raise ValueError("Monitored CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source, manifests, captured, old, historical = source_evidence(
        qknorm_run, core_run, upstream, stage_run, weights)
    kwargs = norm.core.stages.capture.conditioning(manifests["reference"], captured["reference"], device)
    if kwargs["attn_params"].attn_mode != "torch" or kwargs["rope_cos_sin"] is None:
        raise ValueError("FP32 torch attention with RoPE required")
    q_norm, k_norm = make_norms(old["norm_parameters"], device)
    inputs = {side: postrope(old[side]["p"].reshape(1, 4200, WIDTH).to(device), q_norm, k_norm,
                             kwargs["rope_cos_sin"], 16, 128) for side in SIDES}
    du = old["reference"]["du"].reshape(1, 4200, SHAPE[-1]).to(device)
    cells, replay, endpoint, forward = {}, {}, {}, {}
    try:
        for cell in CELLS:
            first_u, first_c = replay_cell(cell, inputs, du, kwargs["attn_params"])
            second_u, second_c = replay_cell(cell, inputs, du, kwargs["attn_params"])
            if not torch.equal(first_u, second_u):
                raise ValueError(f"{cell} repeated forward differs bitwise")
            replay[cell] = norm.core.stages.vjp._fidelity(
                f"{cell} repeated VJP", norm._cpu(second_c), norm._cpu(first_c), 1e-4)
            if cell in ("RRR", "CCC"):
                side = "reference" if cell == "RRR" else "candidate"
                if not torch.equal(norm.core.stages._cpu(first_u).reshape(SHAPE), old[side]["u"]):
                    raise ValueError(f"{cell} core output not bitwise equal")
                forward[side] = True
                norm.core.stages.vjp._fidelity(
                    f"{cell} historical c", norm._cpu(first_c), historical[side]["c"], 1e-4)
                endpoint[side] = _endpoint_chain(side, first_c, old, q_norm, k_norm,
                                                 kwargs["rope_cos_sin"], device)
            cells[cell] = norm._cpu(first_c)
            del first_u, first_c, second_u, second_c
        files = {cell: norm.core.parent.save_evidence(directory, f"cell-{cell}", {"c": cells[cell]}, source,
                                                       scope=SCOPE) for cell in CELLS}
    finally:
        del inputs, q_norm, k_norm
        gc.collect()
        torch.cuda.empty_cache()
    total = sum(item["disk_bytes"] for item in files.values())
    if total > norm.core.parent.TOTAL_MAX_BYTES:
        raise ValueError("QKV matrix aggregate 1GiB budget exceeded")
    return {"status": "analyzed", "scope": SCOPE, "production_ready": False,
            "uuid": manifests["reference"]["uuid"], "shape": list(SHAPE), "source": source,
            "files": files, "disk_bytes": total, "replay": replay, "endpoint": endpoint,
            "forward_bitwise_equal": forward,
            "axes": statistics(cells)}


def validate_files(directory, analysis):
    if (analysis.get("status") != "analyzed" or analysis.get("scope") != SCOPE
            or analysis.get("production_ready") is not False or analysis.get("shape") != list(SHAPE)
            or set(analysis.get("files", {})) != set(CELLS)
            or set(analysis.get("replay", {})) != set(CELLS)
            or set(analysis.get("endpoint", {})) != set(SIDES)
            or analysis.get("forward_bitwise_equal") != dict.fromkeys(SIDES, True)):
        raise ValueError("Incomplete QKV matrix analysis")
    verified, total = {}, 0
    for cell in CELLS:
        folder = Path(directory) / f"cell-{cell}"
        manifest = json.loads((folder / "manifest.json").read_text())
        path = folder / "evidence.safetensors"
        if (manifest != analysis["files"][cell] or manifest.get("scope") != SCOPE
                or manifest.get("name") != f"cell-{cell}" or manifest.get("source") != analysis["source"]
                or manifest.get("schema") != "anima_self_attention_paths_v1"
                or manifest.get("production_ready") is not False or manifest.get("file") != path.name
                or path.stat().st_size != manifest.get("disk_bytes")
                or path.stat().st_size > norm.core.parent.MAX_BYTES or sha256_file(path) != manifest.get("sha256")):
            raise ValueError("QKV matrix manifest/digest/budget mismatch")
        tensors = load_file(str(path))
        if (set(tensors) != {"c"} or manifest.get("inventory") != {"c": [*SHAPE[:-1], WIDTH]}
                or manifest.get("tensor_bytes") != tensors["c"].numel() * tensors["c"].element_size()
                or tensors["c"].dtype != torch.float32 or tuple(tensors["c"].shape) != (*SHAPE[:-1], WIDTH)
                or not bool(torch.isfinite(tensors["c"]).all())):
            raise ValueError("QKV matrix tensor inventory/finite mismatch")
        verified[cell] = tensors["c"]
        total += path.stat().st_size
    if total != analysis.get("disk_bytes") or total > norm.core.parent.TOTAL_MAX_BYTES:
        raise ValueError("QKV matrix aggregate budget mismatch")
    return verified


def validate_statistics(analysis, cells, old, historical):
    if statistics(cells) != analysis.get("axes"):
        raise ValueError("QKV matrix axis statistics mismatch")
    for cell, metric in analysis["replay"].items():
        if metric.get("finite") is not True or metric.get("relative_l2") is None or not 0 <= metric["relative_l2"] <= 1e-4:
            raise ValueError(f"{cell} repeat VJP fidelity invalid")
    for side, cell in (("reference", "RRR"), ("candidate", "CCC")):
        norm.core.stages.vjp._fidelity(f"{cell} historical c", cells[cell], historical[side]["c"], 1e-4)
        metric = analysis["endpoint"][side]
        if metric.get("finite") is not True or metric.get("relative_l2") is None or not 0 <= metric["relative_l2"] <= 1e-4:
            raise ValueError(f"{cell} endpoint dp fidelity invalid")
