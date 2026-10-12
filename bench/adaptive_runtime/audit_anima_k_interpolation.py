"""Independent NumPy FP64 audit of saved K-interpolation gradient statistics."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bench.adaptive_runtime import anima_k_interpolation as interpolation
from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime.audit_anima_self_attention import agrees, metrics


def components(a, b):
    result = {"fused": metrics(a, b)}
    for index, part in enumerate("qkv"):
        sl = slice(index * interpolation.matrix.SHAPE[-1], (index + 1) * interpolation.matrix.SHAPE[-1])
        result[part] = metrics(a[..., sl], b[..., sl])
    return result


def audit_statistics(cells, recorded):
    baseline = {side: {interpolation.label(value, side): components(
        cells[interpolation.label(0, side)], cells[interpolation.label(value, side)])
        for value in interpolation.LAMBDAS[1:]} for side in interpolation.SIDES}
    adjacent = {side: {f"{interpolation.label(left, side)}__{interpolation.label(right, side)}": components(
        cells[interpolation.label(left, side)], cells[interpolation.label(right, side)])
        for left, right in zip(interpolation.LAMBDAS, interpolation.LAMBDAS[1:])}
        for side in interpolation.SIDES}
    v_contrast = {interpolation.label(value, "R") + "__" + interpolation.label(value, "C"): components(
        cells[interpolation.label(value, "R")], cells[interpolation.label(value, "C")])
        for value in interpolation.LAMBDAS}
    midpoint = {side: components(
        0.5 * (cells[interpolation.label(0, side)].astype(np.float64)
               + cells[interpolation.label(1, side)].astype(np.float64)),
        cells[interpolation.label(0.5, side)]) for side in interpolation.SIDES}
    actual = {"baseline": baseline, "adjacent": adjacent, "v_contrast": v_contrast,
              "midpoint_nonlinearity": midpoint}
    if set(actual) != set(recorded):
        raise ValueError("K interpolation statistics inventory mismatch")
    for group in actual:
        if set(actual[group]) != set(recorded[group]):
            raise ValueError(f"K interpolation {group} labels mismatch")
        for key, value in actual[group].items():
            if group in ("baseline", "adjacent"):
                if set(value) != set(recorded[group][key]):
                    raise ValueError("K interpolation pair labels mismatch")
                pairs = value.items()
            else:
                pairs = ((key, value),)
            for pair, parts in pairs:
                report = recorded[group][key][pair] if group in ("baseline", "adjacent") else recorded[group][key]
                if set(parts) != set(report):
                    raise ValueError("K interpolation component inventory mismatch")
                for part, metric in parts.items():
                    agrees(metric, report[part])
                squares = sum(parts[part]["absolute_l2"] ** 2 for part in "qkv")
                if not np.isclose(squares, parts["fused"]["absolute_l2"] ** 2, rtol=1e-12, atol=0):
                    raise ValueError("K interpolation squared norm partition mismatch")
    return actual


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
            or environment.get("driver_sha256") != sha256_file(Path(interpolation.__file__).with_name("diagnose_anima_k_interpolation.py"))
            or environment.get("gpu_uuid") != analysis.get("uuid")
            or environment.get("weights_sha256") != sha256_file(weights)
            or any(environment.get(key) != value for key, value in interpolation.code_hashes().items())):
        raise ValueError("K interpolation result/environment/code/weights binding mismatch")
    source, _, _, _, historical, matrix_cells = interpolation.source_evidence(
        environment["matrix_run"], environment["qknorm_run"], environment["core_run"],
        environment["source_run"], environment["stage_run"], weights)
    if source != analysis.get("source"):
        raise ValueError("K interpolation current source mismatch")
    verified = interpolation.validate_files(root, analysis)
    interpolation.validate_statistics(analysis, verified, matrix_cells, historical)
    cells = {key: tensor.numpy() for key, tensor in verified.items()}
    statistics = audit_statistics(cells, analysis["statistics"])
    endpoints = {}
    for side in interpolation.SIDES:
        for value in (0, 1):
            name = interpolation.label(value, side)
            source_name = ("RR" if value == 0 else "RC") + side
            metric = metrics(matrix_cells[source_name].numpy(), cells[name])
            if metric["relative_l2"] is None or metric["relative_l2"] > 1e-4:
                raise ValueError(f"{name} source endpoint fidelity failed")
            endpoints[name] = metric
    historical_metric = metrics(historical["reference"]["c"].numpy(), cells[interpolation.label(0, "R")])
    if historical_metric["relative_l2"] is None or historical_metric["relative_l2"] > 1e-4:
        raise ValueError("Reference historical c fidelity failed")
    return {"status": "audited", "method": "NumPy FP64 saved-gradient statistics",
            "mixed_backward_independently_recomputed": False,
            "summary_sha256": sha256_file(summary_path), "statistics": statistics,
            "source_endpoints": endpoints, "reference_historical_c": historical_metric,
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
