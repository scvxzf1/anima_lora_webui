"""Independent NumPy CPU audit of isolated Q/K normalization evidence."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file

from bench.adaptive_runtime import anima_qknorm_math as norm_math
from bench.adaptive_runtime import anima_qknorm_paths as paths
from bench.adaptive_runtime.audit_anima_self_attention import agrees, metrics


SIDES = ("reference", "candidate")


def rope_vjp_numpy(c, cos, sin):
    """Transpose of production split-half RoPE, in fused Q/K head coordinates."""
    if c.shape[-1] % 2 or cos.shape != sin.shape or cos.shape[-1] != c.shape[-1]:
        raise ValueError("Invalid RoPE audit layout")
    weighted = c * sin
    half = c.shape[-1] // 2
    rotated = np.concatenate((-weighted[..., half:], weighted[..., :half]), axis=-1)
    return c * cos - rotated


def audit(root, weights):
    # Source comparisons bind exact historical FP64 reductions made with four threads.
    torch.set_num_threads(4)
    root = Path(root)
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "completed" or summary.get("production_ready") is not False:
        raise ValueError("Completed QK norm diagnostic required")
    analysis = summary["analysis"]
    if (analysis.get("status") != "analyzed" or analysis.get("production_ready") is not False
            or set(analysis.get("files", {})) != {*SIDES, "matrix"}):
        raise ValueError("Incomplete QK norm analysis")
    evidence = {}
    total = 0
    shape = tuple(analysis["shape"][:-1]) + (3 * analysis["shape"][-1],)
    environment = json.loads((root / "environment.json").read_text())
    result = json.loads((root / "result.json").read_text())
    core_run = Path(environment["core_run"])
    if (result.get("status") != "ok" or result.get("analysis") != analysis
            or result.get("uuid") != analysis.get("uuid")
            or environment.get("gpu_uuid") != analysis.get("uuid")
            or result.get("driver_sha256") != environment.get("driver_sha256")
            or environment.get("driver_sha256") != paths.sha256_file(Path(paths.__file__).with_name("diagnose_anima_qknorm.py"))
            or any(environment.get(key) != value or analysis["source"].get(key) != value
                   for key, value in paths.code_hashes().items())
            or paths.sha256_file(weights) != environment.get("weights_sha256")):
        raise ValueError("QK norm CPU audit result/GPU/driver/code/weights binding mismatch")
    expected_source, _, _, _, _ = paths.source_evidence(
        core_run, environment["source_run"], environment["stage_run"], weights)
    if expected_source != analysis["source"]:
        raise ValueError("QK norm CPU audit current upstream source mismatch")
    core_summary = core_run / "summary.json"
    if hashlib.sha256(core_summary.read_bytes()).hexdigest() != analysis["source"]["core_summary_sha256"]:
        raise ValueError("QK norm CPU audit core summary binding mismatch")
    old = {}
    for side in SIDES:
        path = core_run / f"trace-{side}" / "evidence.safetensors"
        expected = analysis["source"]["core_evidence_sha256"][side]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("QK norm CPU audit core tensor binding mismatch")
        core_tensor = load_file(path)
        old[side] = {key: core_tensor[key] for key in ("p", "dp")}
    capture_path = Path(environment["source_run"]) / "reference/block-vjp/block-17/capture.safetensors"
    capture_sha = analysis["source"]["core_source"]["upstream_source"]["capture_sha256"]["reference"]
    if hashlib.sha256(capture_path.read_bytes()).hexdigest() != capture_sha:
        raise ValueError("QK norm CPU audit RoPE source binding mismatch")
    capture = load_file(capture_path)
    cos, sin = capture["conditioning.rope_0"], capture["conditioning.rope_1"]
    for side in (*SIDES, "matrix"):
        name = "norm-matrix" if side == "matrix" else f"trace-{side}"
        folder = root / name
        manifest = json.loads((folder / "manifest.json").read_text())
        path = folder / "evidence.safetensors"
        size = path.stat().st_size
        if (manifest != analysis["files"][side] or manifest.get("source") != analysis["source"]
                or manifest.get("scope") != analysis["scope"] or manifest.get("name") != name
                or manifest.get("production_ready") is not False or size != manifest.get("disk_bytes")
                or size > 512 * 1024**2 or hashlib.sha256(path.read_bytes()).hexdigest() != manifest.get("sha256")):
            raise ValueError("QK norm CPU audit manifest/digest mismatch")
        t = load_file(path)
        expected = {f"norm.{i}.{j}" for i in SIDES for j in SIDES} if side == "matrix" else {"c", "a"}
        if (set(t) != expected or manifest.get("inventory") != {key: list(value.shape) for key, value in t.items()}
                or manifest.get("tensor_bytes") != sum(value.nbytes for value in t.values())
                or any(value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all()
                       for value in t.values())):
            raise ValueError("QK norm CPU audit tensor contract mismatch")
        evidence[side] = t
        total += size
    if total != analysis.get("disk_bytes") or total > 1024**3:
        raise ValueError("QK norm CPU audit budget mismatch")
    matrix = evidence["matrix"]
    boundaries = {key: metrics(evidence["reference"][key], evidence["candidate"][key]) for key in ("c", "a")}
    for key, value in boundaries.items():
        agrees(value, analysis["boundaries"][key])
    axes = {"activation_fixed_cotangent": {j: (f"norm.reference.{j}", f"norm.candidate.{j}") for j in SIDES},
            "cotangent_fixed_activation": {i: (f"norm.{i}.reference", f"norm.{i}.candidate") for i in SIDES}}
    contrasts, components = {}, {}
    width = analysis["shape"][-1]
    heads, head_dim = analysis["heads"], analysis["head_dim"]
    if (heads * head_dim != width or head_dim % 2 or cos.shape != sin.shape
            or cos.shape != (shape[-2] * shape[-3], 1, 1, head_dim)):
        raise ValueError("QK norm CPU audit RoPE/head layout mismatch")
    rope_checks, diagonal_checks = {}, {}
    cos = cos.transpose(1, 0, 2, 3)
    sin = sin.transpose(1, 0, 2, 3)
    for side in SIDES:
        c = evidence[side]["c"].reshape(1, -1, 3, heads, head_dim)
        a = evidence[side]["a"].reshape(1, -1, 3, heads, head_dim)
        rope_checks[side] = {}
        for index, name in enumerate(("q", "k")):
            predicted = rope_vjp_numpy(c[:, :, index], cos, sin)
            value = metrics(predicted, a[:, :, index])
            if value["relative_l2"] is None or value["relative_l2"] > 1e-6:
                raise ValueError("QK norm CPU audit RoPE transpose mismatch")
            rope_checks[side][name] = value
        if not np.array_equal(c[:, :, 2], a[:, :, 2]):
            raise ValueError("QK norm CPU audit Identity V mismatch")
        diagonal = metrics(old[side]["dp"], matrix[f"norm.{side}.{side}"])
        if diagonal["relative_l2"] is None or diagonal["relative_l2"] > 1e-4:
            raise ValueError("QK norm CPU audit diagonal core chain mismatch")
        diagonal_checks[side] = diagonal
    weights_qk = norm_math.read_norm_weights(weights, environment["weights_sha256"], head_dim)
    formula_checks = norm_math.check_norm_matrix(old, evidence, matrix, weights_qk, heads, head_dim)
    if formula_checks != analysis.get("norm_formula_checks"):
        raise ValueError("QK norm CPU audit formula checks differ from summary")
    for axis, rows in axes.items():
        contrasts[axis], components[axis] = {}, {}
        for side, (left, right) in rows.items():
            contrasts[axis][side] = metrics(matrix[left], matrix[right])
            agrees(contrasts[axis][side], analysis["contrasts"][axis][side])
            components[axis][side] = {}
            for part, name in enumerate(("q", "k", "v")):
                sl = slice(part * width, (part + 1) * width)
                value = metrics(matrix[left][..., sl], matrix[right][..., sl])
                agrees(value, analysis["components"][axis][side][name])
                components[axis][side][name] = value
            squares = sum(value["absolute_l2"] ** 2 for value in components[axis][side].values())
            if not np.isclose(squares, contrasts[axis][side]["absolute_l2"] ** 2, rtol=1e-12, atol=0):
                raise ValueError("QKV squared norm does not partition fused axis")
    return {"status": "audited", "method": "independent NumPy FP64 CPU reductions",
            "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            "boundaries": boundaries, "contrasts": contrasts, "components": components,
            "rope_transpose": rope_checks, "core_diagonal": diagonal_checks,
            "norm_formula_checks": formula_checks,
            "production_ready": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    output = args.run / "audit-cpu.json"
    if output.exists():
        raise FileExistsError(output)
    result = audit(args.run, args.weights)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
