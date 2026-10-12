"""Bench-only K difference interpolation with fixed reference Q and cotangent."""

import gc
import json
from pathlib import Path

from safetensors.torch import load_file
import torch

from bench.adaptive_runtime import anima_qkv_matrix as matrix
from bench.adaptive_runtime.anima_inputs import sha256_file


LAMBDAS = (0, 0.25, 0.5, 0.75, 1)
SIDES = ("R", "C")
CELLS = tuple(f"k{int(value * 100):03d}_v{side}" for side in SIDES for value in LAMBDAS)
SCOPE = "block17_reference_q_k_difference_interpolation_v_background_fixed_du_not_production"


def label(value, side):
    if value not in LAMBDAS or side not in SIDES:
        raise ValueError("Unknown K interpolation cell")
    return f"k{int(value * 100):03d}_v{side}"


def code_hashes():
    return {"analysis_sha256": sha256_file(Path(__file__)),
            "matrix_analysis_sha256": sha256_file(Path(matrix.__file__)),
            "model_source_sha256": matrix.code_hashes()["model_source_sha256"],
            "dispatch_source_sha256": matrix.code_hashes()["dispatch_source_sha256"]}


def source_evidence(matrix_run, qknorm_run, core_run, upstream, stage_run, weights):
    root = Path(matrix_run)
    summary = json.loads((root / "summary.json").read_text())
    result = json.loads((root / "result.json").read_text())
    environment = json.loads((root / "environment.json").read_text())
    analysis = summary.get("analysis", {})
    if (summary.get("status") != "completed" or summary.get("production_ready") is not False
            or result.get("status") != "ok" or result.get("analysis") != analysis
            or result.get("uuid") != analysis.get("uuid")
            or result.get("driver_sha256") != environment.get("driver_sha256")
            or environment.get("driver_sha256") != sha256_file(Path(matrix.__file__).with_name("diagnose_anima_qkv_matrix.py"))
            or environment.get("gpu_uuid") != analysis.get("uuid")
            or environment.get("weights_sha256") != sha256_file(weights)
            or any(Path(environment[key]).resolve() != Path(path).resolve() for key, path in
                   (("qknorm_run", qknorm_run), ("core_run", core_run),
                    ("source_run", upstream), ("stage_run", stage_run)))
            or any(environment.get(key) != value for key, value in matrix.code_hashes().items())):
        raise ValueError("Completed QKV matrix result/environment/code/weights binding required")
    source, manifests, captured, old, historical = matrix.source_evidence(
        qknorm_run, core_run, upstream, stage_run, weights)
    if source != analysis.get("source"):
        raise ValueError("QKV matrix source changed")
    verified = matrix.validate_files(root, analysis)
    matrix.validate_statistics(analysis, verified, old, historical)
    binding = {"matrix_summary_sha256": sha256_file(root / "summary.json"),
               "matrix_evidence_sha256": {key: value["sha256"] for key, value in analysis["files"].items()},
               "matrix_source": source, **code_hashes()}
    return binding, manifests, captured, old, historical, verified


def select_inputs(value, side, inputs):
    reference, candidate = inputs["reference"], inputs["candidate"]
    if value == 0:
        k = reference[1]
    elif value == 1:
        k = candidate[1]
    elif value in LAMBDAS:
        k = reference[1] + torch.tensor(value, dtype=torch.float32, device=reference[1].device) * (candidate[1] - reference[1])
    else:
        raise ValueError("Unsupported K interpolation value")
    if side not in SIDES:
        raise ValueError("Unsupported V background")
    return reference[0], k, inputs["reference" if side == "R" else "candidate"][2]


def replay_cell(value, side, inputs, du, params):
    selected = tuple(item.detach().requires_grad_(True) for item in select_inputs(value, side, inputs))
    from networks.attention_dispatch import dispatch_attention
    u = dispatch_attention(list(selected), attn_params=params.for_attention_kind(is_selfattn=True))
    if u.shape != du.shape or u.dtype != torch.float32:
        raise ValueError("Attention output contract changed")
    gradients = torch.autograd.grad(u, selected, du)
    return u.detach(), matrix.norm._pack(*gradients, (du.shape[0], du.shape[1], 3 * du.shape[2])).detach()


def components(a, b):
    metric = matrix.norm.core.stages.vjp._metric
    result = {"fused": metric(a, b)}
    for index, part in enumerate("qkv"):
        sl = slice(index * matrix.SHAPE[-1], (index + 1) * matrix.SHAPE[-1])
        result[part] = metric(a[..., sl], b[..., sl])
    return result


def statistics(cells):
    baseline = {side: {label(value, side): components(cells[label(0, side)], cells[label(value, side)])
                       for value in LAMBDAS[1:]} for side in SIDES}
    adjacent = {side: {f"{label(left, side)}__{label(right, side)}": components(
        cells[label(left, side)], cells[label(right, side)])
        for left, right in zip(LAMBDAS, LAMBDAS[1:])} for side in SIDES}
    v_contrast = {label(value, "R") + "__" + label(value, "C"): components(
        cells[label(value, "R")], cells[label(value, "C")]) for value in LAMBDAS}
    midpoint = {side: components(
        0.5 * (cells[label(0, side)].double() + cells[label(1, side)].double()),
        cells[label(0.5, side)].double())
        for side in SIDES}
    return {"baseline": baseline, "adjacent": adjacent, "v_contrast": v_contrast,
            "midpoint_nonlinearity": midpoint}


def analyze(matrix_run, qknorm_run, core_run, upstream, stage_run, weights, directory, device):
    if device.type != "cuda":
        raise ValueError("Monitored CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source, manifests, captured, old, historical, matrix_cells = source_evidence(
        matrix_run, qknorm_run, core_run, upstream, stage_run, weights)
    kwargs = matrix.norm.core.stages.capture.conditioning(manifests["reference"], captured["reference"], device)
    if kwargs["attn_params"].attn_mode != "torch" or kwargs["rope_cos_sin"] is None:
        raise ValueError("FP32 torch attention with RoPE required")
    q_norm, k_norm = matrix.make_norms(old["norm_parameters"], device)
    inputs = {side: matrix.postrope(old[side]["p"].reshape(1, 4200, matrix.WIDTH).to(device),
                                   q_norm, k_norm, kwargs["rope_cos_sin"], 16, 128)
              for side in matrix.SIDES}
    du = old["reference"]["du"].reshape(1, 4200, matrix.SHAPE[-1]).to(device)
    cells, replay, forward = {}, {}, {}
    try:
        for side in SIDES:
            for value in LAMBDAS:
                name = label(value, side)
                u, c = replay_cell(value, side, inputs, du, kwargs["attn_params"])
                repeated_u, repeated_c = replay_cell(value, side, inputs, du, kwargs["attn_params"])
                if not torch.equal(u, repeated_u):
                    raise ValueError(f"{name} repeated forward differs bitwise")
                replay[name] = matrix.norm.core.stages.vjp._fidelity(
                    f"{name} repeated VJP", matrix.norm._cpu(repeated_c), matrix.norm._cpu(c), 1e-4)
                if value in (0, 1):
                    matrix_cell = ("RR" if value == 0 else "RC") + side
                    matrix.norm.core.stages.vjp._fidelity(
                        f"{name} matrix source c", matrix.norm._cpu(c), matrix_cells[matrix_cell], 1e-4)
                if value == 0 and side == "R":
                    if not torch.equal(matrix.norm.core.stages._cpu(u).reshape(matrix.SHAPE), old["reference"]["u"]):
                        raise ValueError("Reference core output not bitwise equal")
                    forward["reference_core"] = True
                    matrix.norm.core.stages.vjp._fidelity(
                        "reference historical c", matrix.norm._cpu(c), historical["reference"]["c"], 1e-4)
                cells[name] = matrix.norm._cpu(c)
                del u, c, repeated_u, repeated_c
        files = {name: matrix.norm.core.parent.save_evidence(
            directory, f"cell-{name}", {"c": cells[name]}, source, scope=SCOPE) for name in CELLS}
    finally:
        del inputs, q_norm, k_norm
        gc.collect()
        torch.cuda.empty_cache()
    total = sum(item["disk_bytes"] for item in files.values())
    if total > matrix.norm.core.parent.TOTAL_MAX_BYTES:
        raise ValueError("K interpolation aggregate 1GiB budget exceeded")
    return {"status": "analyzed", "scope": SCOPE, "production_ready": False,
            "uuid": manifests["reference"]["uuid"], "shape": list(matrix.SHAPE),
            "source": source, "files": files, "disk_bytes": total,
            "replay": replay, "forward_bitwise_equal": forward, "statistics": statistics(cells)}


def validate_files(directory, analysis):
    if (analysis.get("status") != "analyzed" or analysis.get("scope") != SCOPE
            or analysis.get("production_ready") is not False or analysis.get("shape") != list(matrix.SHAPE)
            or set(analysis.get("files", {})) != set(CELLS)
            or set(analysis.get("replay", {})) != set(CELLS)
            or analysis.get("forward_bitwise_equal") != {"reference_core": True}):
        raise ValueError("Incomplete K interpolation analysis")
    verified, total = {}, 0
    for name in CELLS:
        folder = Path(directory) / f"cell-{name}"
        manifest = json.loads((folder / "manifest.json").read_text())
        path = folder / "evidence.safetensors"
        if (manifest != analysis["files"][name] or manifest.get("scope") != SCOPE
                or manifest.get("name") != f"cell-{name}" or manifest.get("source") != analysis["source"]
                or manifest.get("schema") != "anima_self_attention_paths_v1"
                or manifest.get("production_ready") is not False or manifest.get("file") != path.name
                or path.stat().st_size != manifest.get("disk_bytes")
                or path.stat().st_size > matrix.norm.core.parent.MAX_BYTES
                or sha256_file(path) != manifest.get("sha256")):
            raise ValueError("K interpolation manifest/digest/budget mismatch")
        tensors = load_file(str(path))
        shape = (*matrix.SHAPE[:-1], matrix.WIDTH)
        if (set(tensors) != {"c"} or manifest.get("inventory") != {"c": list(shape)}
                or manifest.get("tensor_bytes") != tensors["c"].numel() * tensors["c"].element_size()
                or tensors["c"].dtype != torch.float32 or tuple(tensors["c"].shape) != shape
                or not bool(torch.isfinite(tensors["c"]).all())):
            raise ValueError("K interpolation tensor inventory/finite mismatch")
        verified[name] = tensors["c"]
        total += path.stat().st_size
    if total != analysis.get("disk_bytes") or total > matrix.norm.core.parent.TOTAL_MAX_BYTES:
        raise ValueError("K interpolation aggregate budget mismatch")
    return verified


def validate_statistics(analysis, cells, matrix_cells, historical):
    if statistics(cells) != analysis.get("statistics"):
        raise ValueError("K interpolation statistics mismatch")
    for name, value in analysis["replay"].items():
        if value.get("finite") is not True or value.get("relative_l2") is None or not 0 <= value["relative_l2"] <= 1e-4:
            raise ValueError(f"{name} replay fidelity invalid")
    for side in SIDES:
        for value in (0, 1):
            name = label(value, side)
            matrix_cell = ("RR" if value == 0 else "RC") + side
            matrix.norm.core.stages.vjp._fidelity(
                f"{name} source c", cells[name], matrix_cells[matrix_cell], 1e-4)
    matrix.norm.core.stages.vjp._fidelity(
        "reference historical c", cells[label(0, "R")], historical["reference"]["c"], 1e-4)
