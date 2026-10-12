"""Frozen conservative gates; gradient norm alone never implies equivalence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file

from .manifest import write_json
from .numerics import compare_named, vector_metrics

GATES = {
    "output_rel_l2": 0.03,
    "loss_rel": 0.05,
    "grad_norm_rel": 0.05,
    "grad_rel_l2": 0.05,
    "grad_cosine": 0.995,
    "update_rel_l2": 0.05,
    "layer_rel_l2": 0.10,
    "layer_cosine": 0.99,
    "zero_norm": 1e-12,
    "zero_absolute_l2": 1e-8,
}
EXPECTED_SAMPLES = [
    f"cache{cache}-seed{seed}-sigma{sigma:g}"
    for cache in (0, 1)
    for seed in (0, 1, 2)
    for sigma in (0.1, 0.5, 0.9)
]


def vector_pass(metrics, relative, cosine=None):
    if not metrics["finite"]:
        return False
    if metrics["reference_zero"]:
        return metrics["absolute_l2"] <= GATES["zero_absolute_l2"]
    return metrics["relative_l2"] <= relative and (
        cosine is None or metrics["cosine"] is not None and metrics["cosine"] >= cosine
    )


def assess_sample(reference, candidate, initial_parameters=None):
    if initial_parameters is None:
        manifests = [
            json.loads((folder.parent / "manifest.json").read_text())
            for folder in (reference, candidate)
        ]
        if (
            manifests[0]["model"]["initial_parameters"]
            != manifests[1]["model"]["initial_parameters"]
        ):
            raise ValueError("initial adapter mismatch")
        initial_parameters = manifests[0]["model"]["initial_parameters"]
    expected_names = set(initial_parameters)
    if not expected_names:
        raise ValueError("no initial adapter parameters")
    a, b = (
        json.loads((reference / "sample.json").read_text()),
        json.loads((candidate / "sample.json").read_text()),
    )
    if a["inputs"]["input_hash"] != b["inputs"]["input_hash"]:
        raise ValueError("comparison input mismatch")
    tensors = {
        kind: [
            load_file(str(folder / f"{kind}.safetensors"))
            for folder in (reference, candidate)
        ]
        for kind in ("grads", "updates")
    }
    gradients = compare_named(*tensors["grads"], GATES["zero_norm"])
    updates = compare_named(*tensors["updates"], GATES["zero_norm"])
    output = vector_metrics(
        load_file(str(reference / "output.safetensors"))["output"],
        load_file(str(candidate / "output.safetensors"))["output"],
        GATES["zero_norm"],
    )
    loss_rel = abs(a["loss"] - b["loss"]) / max(abs(a["loss"]), 1e-12)
    norm = gradients["overall"]
    norm_rel = (
        abs(norm["reference_norm"] - norm["candidate_norm"])
        / max(norm["reference_norm"], 1e-12)
        if norm["finite"]
        else None
    )
    failures = [
        n
        for n, m in gradients["layers"].items()
        if not vector_pass(m, GATES["layer_rel_l2"], GATES["layer_cosine"])
    ]
    update_failures = [
        n
        for n, m in updates["layers"].items()
        if not vector_pass(m, GATES["layer_rel_l2"], GATES["layer_cosine"])
    ]
    name_checks = {
        kind: all(set(values) == expected_names for values in pair)
        for kind, pair in tensors.items()
    }
    checks = {
        "finite": all(a["finite"]) and all(b["finite"]),
        "output": vector_pass(output, GATES["output_rel_l2"]),
        "loss": loss_rel <= GATES["loss_rel"],
        "gradient_vector": vector_pass(
            norm, GATES["grad_rel_l2"], GATES["grad_cosine"]
        ),
        "gradient_norm": norm_rel is not None and norm_rel <= GATES["grad_norm_rel"],
        "updates": vector_pass(updates["overall"], GATES["update_rel_l2"]),
        "missing": a.get("missing_gradients") == []
        and b.get("missing_gradients") == [],
        "gradient_names": name_checks["grads"],
        "update_names": name_checks["updates"],
        "per_layer": not failures,
        "update_per_layer": not update_failures,
    }
    return {
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "output": output,
        "loss_relative": loss_rel,
        "gradient_norm_relative": norm_rel,
        "gradients": gradients,
        "updates": updates,
        "failing_layers": failures,
        "failing_update_layers": update_failures,
    }


def validate_run(folder, manifest):
    expected = set(EXPECTED_SAMPLES)
    declared = manifest.get("expected_samples", [])
    result = json.loads((folder / "result.json").read_text())
    completed = [sample["name"] for sample in result.get("samples", [])]
    actual = {path.name for path in folder.glob("cache*-seed*-sigma*") if path.is_dir()}
    if (
        len(declared) != len(expected)
        or set(declared) != expected
        or result.get("status") != "PASS"
        or result.get("count") != len(expected)
        or len(completed) != len(expected)
        or set(completed) != expected
        or actual != expected
        or (folder / "failure.json").exists()
    ):
        raise ValueError(f"incomplete numerical sample matrix: {folder}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    ref = json.loads((args.reference / "manifest.json").read_text())
    cand = json.loads((args.candidate / "manifest.json").read_text())
    if ref["initial_hash"] != cand["initial_hash"]:
        raise ValueError("initial adapter mismatch")
    if ref["model"]["initial_parameters"] != cand["model"]["initial_parameters"]:
        raise ValueError("initial adapter mismatch")
    validate_run(args.reference, ref)
    validate_run(args.candidate, cand)
    results = {}
    for folder in sorted(args.reference.glob("cache*-seed*-sigma*")):
        results[folder.name] = assess_sample(
            folder, args.candidate / folder.name, ref["model"]["initial_parameters"]
        )
    if not results:
        raise ValueError("no numerical samples")
    write_json(
        args.out,
        {
            "gates": GATES,
            "gates_frozen_before_candidates": True,
            "status": "PASS"
            if all(r["status"] == "PASS" for r in results.values())
            else "FAIL",
            "samples": results,
        },
    )
    print(
        json.dumps(
            {
                "samples": len(results),
                "passed": sum(r["status"] == "PASS" for r in results.values()),
            }
        )
    )


if __name__ == "__main__":
    main()
