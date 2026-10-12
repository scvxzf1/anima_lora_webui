"""Independent NumPy CPU reduction of saved self-attention diagnostic evidence."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file


SIDES = ("reference", "candidate")


def metrics(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError("Audit shape mismatch")
    x, y = reference.astype(np.float64).ravel(), candidate.astype(np.float64).ravel()
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite audit tensor")
    rr, yy = np.sum(x * x), np.sum(y * y)
    delta = float(np.sqrt(np.sum((y - x) ** 2)))
    norm = float(np.sqrt(rr))
    relative = delta / norm if norm else (0.0 if yy == 0 else None)
    cosine = float(np.clip(np.sum(x * y) / np.sqrt(rr * yy), -1, 1)) if rr and yy else (1.0 if rr == yy == 0 else 0.0)
    return {"finite": True, "relative_l2": relative, "absolute_l2": delta,
            "reference_l2": norm, "cosine": cosine}


def agrees(actual, recorded):
    if recorded.get("finite") is not True:
        raise ValueError("Recorded metric is not finite")
    for key in ("relative_l2", "absolute_l2", "reference_l2", "cosine"):
        a, b = actual[key], recorded.get(key)
        if (a is None) != (b is None) or (a is not None and (not np.isfinite(b) or abs(a - b) > 1e-12)):
            raise ValueError(f"CPU audit metric mismatch: {key}")


def audit(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    if summary.get("status") != "completed" or summary.get("production_ready") is not False:
        raise ValueError("Completed diagnostic summary required")
    analysis = summary["analysis"]
    evidence = {}
    for side in (*SIDES, "matrix", "reference_replay"):
        name = "norm-matrix" if side == "matrix" else f"trace-{side}"
        folder = root / name
        manifest = json.loads((folder / "manifest.json").read_text())
        path = folder / "evidence.safetensors"
        if (manifest != analysis["files"][side] or path.stat().st_size != manifest["disk_bytes"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]):
            raise ValueError("CPU audit manifest/digest mismatch")
        evidence[side] = load_file(path)
        if manifest["inventory"] != {key: list(v.shape) for key, v in evidence[side].items()}:
            raise ValueError("CPU audit inventory mismatch")
    matrix = evidence["matrix"]
    contrasts = {
        "activation_fixed_cotangent": {h: metrics(matrix[f"norm.reference.{h}"], matrix[f"norm.candidate.{h}"]) for h in SIDES},
        "cotangent_fixed_activation": {x: metrics(matrix[f"norm.{x}.reference"], matrix[f"norm.{x}.candidate"]) for x in SIDES}}
    for axis, rows in contrasts.items():
        for side, actual in rows.items():
            agrees(actual, analysis["normalization_contrasts"][axis][side])
    boundaries = {key: metrics(evidence["reference"][key], evidence["candidate"][key])
                  for key in ("x", "z", "b", "h", "branch_dx")}
    for key, actual in boundaries.items():
        agrees(actual, analysis["boundaries"][key])
    chains = {side: metrics(evidence[side]["branch_dx"], matrix[f"norm.{side}.{side}"]) for side in SIDES}
    if any(m["relative_l2"] is None or m["relative_l2"] > 1e-4 for m in chains.values()):
        raise ValueError("CPU audit diagonal chain identity failed")
    replay = evidence["reference_replay"]
    if any(not np.array_equal(replay[key], evidence["reference"][key]) for key in ("x", "z", "b", "gate")):
        raise ValueError("CPU audit repeated forward mismatch")
    repeated = {key: metrics(evidence["reference"][key], replay[key]) for key in ("h", "branch_dx")}
    if any(m["relative_l2"] is None or m["relative_l2"] > 1e-4 for m in repeated.values()):
        raise ValueError("CPU audit repeated gradient fidelity failed")
    return {"status": "audited", "method": "independent NumPy FP64 CPU reductions",
            "summary_sha256": hashlib.sha256((root / "summary.json").read_bytes()).hexdigest(),
            "normalization_contrasts": contrasts, "boundaries": boundaries,
            "chain_identities": chains, "repeated_gradient": repeated, "production_ready": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    output = args.run / "audit-cpu.json"
    if output.exists():
        raise FileExistsError(output)
    result = audit(args.run)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
