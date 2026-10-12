"""Independent CPU audit of projection/core diagnostic files and Q/K/V axes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from bench.adaptive_runtime.audit_anima_self_attention import metrics, agrees


def audit(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    if summary.get("status") != "completed" or summary.get("production_ready") is not False:
        raise ValueError("Completed core diagnostic required")
    a = summary["analysis"]
    evidence = {}
    for side in ("reference", "candidate", "matrix"):
        folder = root / ("core-matrix" if side == "matrix" else f"trace-{side}")
        path = folder / "evidence.safetensors"
        m = json.loads((folder / "manifest.json").read_text())
        if (m != a["files"][side] or path.stat().st_size != m["disk_bytes"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != m["sha256"]):
            raise ValueError("Core audit file digest mismatch")
        evidence[side] = load_file(path)
        if m["inventory"] != {k: list(v.shape) for k, v in evidence[side].items()}:
            raise ValueError("Core audit inventory mismatch")
    r, c = evidence["reference"], evidence["candidate"]
    rows = {key: metrics(r[key], c[key]) for key in ("x", "z", "p", "u", "du", "dp", "h")}
    for key, m in rows.items():
        agrees(m, a["boundaries"][key])
    width = r["x"].shape[-1]
    if r["dp"].shape[-1] != 3 * width:
        raise ValueError("Core audit fused-QKV split mismatch")
    components = {key: metrics(r["dp"][..., i * width:(i + 1) * width], c["dp"][..., i * width:(i + 1) * width])
                  for i, key in enumerate(("q", "k", "v"))}
    for key, m in components.items():
        agrees(m, a["core_gradient_components"][key])
    # The Q/K/V squared absolute norms exactly partition the fused gradient norm.
    if not np.isclose(sum(m["absolute_l2"] ** 2 for m in components.values()), rows["dp"]["absolute_l2"] ** 2, rtol=1e-12, atol=0):
        raise ValueError("Core audit gradient partition mismatch")
    matrix = evidence["matrix"]
    fixed = metrics(matrix["fixed_qkv.reference"], matrix["fixed_qkv.candidate"])
    agrees(fixed, a["fixed_qkv_cotangent"])
    if rows["du"]["relative_l2"] is None or rows["du"]["relative_l2"] > 1e-4:
        raise ValueError("Core audit fixed output projection fidelity failed")
    return {"status": "audited", "method": "independent NumPy FP64 CPU reductions",
            "summary_sha256": hashlib.sha256((root / "summary.json").read_bytes()).hexdigest(),
            "boundaries": rows, "core_gradient_components": components,
            "fixed_qkv_cotangent": fixed, "production_ready": False}


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
