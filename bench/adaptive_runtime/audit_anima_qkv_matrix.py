"""NumPy FP64 audit of saved QKV matrix; mixed VJPs are not recomputed."""

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file
import torch

from bench.adaptive_runtime import anima_qkv_matrix as matrix
from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime.anima_qknorm_math import rmsnorm_vjp
from bench.adaptive_runtime.audit_anima_qknorm import rope_vjp_numpy
from bench.adaptive_runtime.audit_anima_self_attention import agrees, metrics


def _audit_axes(cells, recorded):
    result = {}
    for axis, pairs in matrix.axis_pairs().items():
        if set(recorded.get(axis, {})) != set(pairs):
            raise ValueError("QKV matrix axis pairs incomplete")
        result[axis] = {}
        for fixed, (left, right) in pairs.items():
            a, b = cells[left], cells[right]
            values = {"fused": metrics(a, b)}
            for index, part in enumerate("qkv"):
                sl = slice(index * matrix.SHAPE[-1], (index + 1) * matrix.SHAPE[-1])
                values[part] = metrics(a[..., sl], b[..., sl])
            if set(recorded[axis][fixed]) != set(values):
                raise ValueError("QKV matrix component inventory mismatch")
            for part, value in values.items():
                agrees(value, recorded[axis][fixed][part])
            squares = sum(values[part]["absolute_l2"] ** 2 for part in "qkv")
            if not np.isclose(squares, values["fused"]["absolute_l2"] ** 2, rtol=1e-12, atol=0):
                raise ValueError("QKV squared norm does not partition fused axis")
            result[axis][fixed] = values
    if set(recorded) != set(result):
        raise ValueError("QKV matrix axis inventory mismatch")
    return result


def _endpoint_dp(c, p, weights, cos, sin):
    c = c.reshape(-1, 3, 16, 128).astype(np.float64)
    p = p.reshape(-1, 3, 16, 128).astype(np.float64)
    a_q = rope_vjp_numpy(c[:, 0], cos, sin)
    a_k = rope_vjp_numpy(c[:, 1], cos, sin)
    pieces = [rmsnorm_vjp(p[:, 0], a_q, weights["q"], 1e-6),
              rmsnorm_vjp(p[:, 1], a_k, weights["k"], 1e-6), c[:, 2]]
    return np.stack(pieces, axis=1).reshape(matrix.SHAPE[:-1] + (matrix.WIDTH,))


def audit(root, weights):
    torch.set_num_threads(4)
    root = Path(root)
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text())
    environment = json.loads((root / "environment.json").read_text())
    result = json.loads((root / "result.json").read_text())
    analysis = summary.get("analysis", {})
    if (summary.get("status") != "completed" or summary.get("production_ready") is not False
            or result.get("status") != "ok" or result.get("analysis") != analysis
            or result.get("uuid") != analysis.get("uuid")
            or result.get("driver_sha256") != environment.get("driver_sha256")
            or environment.get("driver_sha256") != sha256_file(Path(matrix.__file__).with_name("diagnose_anima_qkv_matrix.py"))
            or environment.get("gpu_uuid") != analysis.get("uuid")
            or environment.get("weights_sha256") != sha256_file(weights)
            or any(environment.get(key) != value for key, value in matrix.code_hashes().items())):
        raise ValueError("QKV matrix result/environment/code/weights binding mismatch")
    source, _, _, old, historical = matrix.source_evidence(
        environment["qknorm_run"], environment["core_run"], environment["source_run"],
        environment["stage_run"], weights)
    if source != analysis.get("source"):
        raise ValueError("QKV matrix current source mismatch")
    verified = matrix.validate_files(root, analysis)
    matrix.validate_statistics(analysis, verified, old, historical)
    cells = {cell: verified[cell].numpy() for cell in matrix.CELLS}
    axes = _audit_axes(cells, analysis["axes"])
    capture_path = Path(environment["source_run"]) / "reference/block-vjp/block-17/capture.safetensors"
    expected_sha = source["qknorm_source"]["core_source"]["upstream_source"]["capture_sha256"]["reference"]
    if sha256_file(capture_path) != expected_sha:
        raise ValueError("QKV matrix RoPE source SHA mismatch")
    capture = load_file(capture_path)
    cos = capture["conditioning.rope_0"].astype(np.float64).reshape(-1, 1, 128)
    sin = capture["conditioning.rope_1"].astype(np.float64).reshape(-1, 1, 128)
    if cos.shape != (4200, 1, 128) or sin.shape != cos.shape:
        raise ValueError("QKV matrix RoPE layout mismatch")
    endpoint = {}
    for side, cell in (("reference", "RRR"), ("candidate", "CCC")):
        historical_c = metrics(historical[side]["c"].numpy(), cells[cell])
        if historical_c["relative_l2"] is None or historical_c["relative_l2"] > 1e-4:
            raise ValueError(f"{cell} historical c mismatch")
        dp = _endpoint_dp(cells[cell], old[side]["p"].numpy(), old["norm_parameters"], cos, sin)
        value = metrics(old[side]["dp"].numpy(), dp)
        if value["relative_l2"] is None or value["relative_l2"] > 1e-4:
            raise ValueError(f"{cell} independent endpoint dp mismatch")
        endpoint[side] = {"historical_c": historical_c, "core_dp": value}
    return {"status": "audited", "method": "NumPy FP64 saved-gradient statistics and endpoint chain",
            "mixed_backward_independently_recomputed": False,
            "summary_sha256": sha256_file(summary_path), "axes": axes, "endpoint": endpoint,
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
