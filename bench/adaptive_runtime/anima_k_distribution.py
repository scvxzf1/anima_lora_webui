"""Bench-only CPU head/token distribution of saved K-interpolation gradients."""

import argparse
import json
import math
import os
from pathlib import Path
import resource
import time

import numpy as np
from safetensors.torch import load_file
import torch

from bench.adaptive_runtime import anima_k_interpolation as interpolation
from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime.audit_anima_self_attention import agrees


SHAPE = (1, 1, 75, 56, 6144)
TOKENS = 4200
HEADS = 16
HEAD_WIDTH = 128
PARTS = ("q", "k", "v", "fused")


def validate_audit_values(actual, recorded):
    if (
        not isinstance(actual, dict)
        or not isinstance(recorded, dict)
        or set(actual) != set(recorded)
    ):
        raise ValueError("Audit statistics inventory mismatch")
    if "relative_l2" in actual:
        agrees(actual, recorded)
        agrees(recorded, actual)
        return
    for key in actual:
        validate_audit_values(actual[key], recorded[key])


def validate_endpoint(metric):
    if not isinstance(metric, dict) or metric.get("finite") is not True:
        raise ValueError("Audit endpoint finite contract invalid")
    for key in ("relative_l2", "absolute_l2", "reference_l2", "cosine"):
        value = metric.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Audit endpoint metric invalid")
    if (
        not 0 <= metric["relative_l2"] <= 1e-4
        or metric["absolute_l2"] < 0
        or metric["reference_l2"] <= 0
    ):
        raise ValueError("Audit endpoint fidelity failed")
    relative = metric["absolute_l2"] / metric["reference_l2"]
    if not math.isclose(relative, metric["relative_l2"], rel_tol=1e-10, abs_tol=1e-15):
        raise ValueError("Audit endpoint norm ratio inconsistent")
    if (
        not -1 <= metric["cosine"] <= 1
        or 1 - metric["cosine"] > 2 * relative**2 / (1 - relative) ** 2 + 1e-12
    ):
        raise ValueError("Audit endpoint cosine inconsistent")


def validate_source(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    result = json.loads((root / "result.json").read_text())
    environment = json.loads((root / "environment.json").read_text())
    audit = json.loads((root / "audit-cpu.json").read_text())
    analysis = summary.get("analysis", {})
    if (
        summary.get("status") != "completed"
        or summary.get("production_ready") is not False
        or summary.get("scope") != interpolation.SCOPE
        or result.get("status") != "ok"
        or result.get("production_ready") is not False
        or result.get("analysis") != analysis
        or result.get("uuid") != analysis.get("uuid")
        or result.get("driver_sha256") != environment.get("driver_sha256")
        or environment.get("gpu_uuid") != analysis.get("uuid")
        or environment.get("training_steps") != 0
        or audit.get("status") != "audited"
        or audit.get("production_ready") is not False
        or audit.get("summary_sha256") != sha256_file(root / "summary.json")
        or set(audit.get("statistics", {})) != set(analysis.get("statistics", {}))
        or set(audit.get("source_endpoints", {}))
        != {
            interpolation.label(value, side)
            for value in (0, 1)
            for side in interpolation.SIDES
        }
        or analysis.get("scope") != interpolation.SCOPE
        or analysis.get("production_ready") is not False
        or analysis.get("shape") != list(interpolation.matrix.SHAPE)
        or set(analysis.get("files", {})) != set(interpolation.CELLS)
        or set(analysis.get("replay", {})) != set(interpolation.CELLS)
        or analysis.get("forward_bitwise_equal") != {"reference_core": True}
        or any(
            environment.get(key) != value
            or analysis.get("source", {}).get(key) != value
            for key, value in interpolation.code_hashes().items()
        )
        or environment.get("driver_sha256")
        != sha256_file(
            Path(interpolation.__file__).with_name("diagnose_anima_k_interpolation.py")
        )
    ):
        raise ValueError(
            "K interpolation summary/result/environment/audit binding invalid"
        )
    validate_audit_values(audit["statistics"], analysis["statistics"])
    for metric in (
        *audit["source_endpoints"].values(),
        audit.get("reference_historical_c"),
    ):
        validate_endpoint(metric)
    return analysis, {
        name: sha256_file(root / name)
        for name in (
            "summary.json",
            "result.json",
            "environment.json",
            "audit-cpu.json",
        )
    }


def load_cell(root, analysis, name):
    folder = Path(root) / f"cell-{name}"
    manifest = json.loads((folder / "manifest.json").read_text())
    expected = analysis["files"][name]
    path = folder / "evidence.safetensors"
    if (
        manifest != expected
        or manifest.get("schema") != "anima_self_attention_paths_v1"
        or manifest.get("scope") != interpolation.SCOPE
        or manifest.get("source") != analysis["source"]
        or manifest.get("production_ready") is not False
        or manifest.get("name") != f"cell-{name}"
        or manifest.get("file") != path.name
        or manifest.get("inventory") != {"c": list(SHAPE)}
        or manifest.get("tensor_bytes") != math.prod(SHAPE) * 4
        or path.stat().st_size != manifest.get("disk_bytes")
        or path.stat().st_size > interpolation.matrix.norm.core.parent.MAX_BYTES
        or sha256_file(path) != manifest.get("sha256")
    ):
        raise ValueError(f"{name} manifest/SHA/shape/budget mismatch")
    tensors = load_file(str(path), device="cpu")
    if (
        set(tensors) != {"c"}
        or tensors["c"].dtype != torch.float32
        or tuple(tensors["c"].shape) != SHAPE
        or not bool(torch.isfinite(tensors["c"]).all())
    ):
        raise ValueError(f"{name} tensor dtype/shape/finite mismatch")
    return tensors["c"].numpy().reshape(TOKENS, 3, HEADS, HEAD_WIDTH)


def _rank(scores, count):
    return np.argsort(-scores, kind="stable")[:count].astype(int).tolist()


def _part_report(error_by_head, error_by_token, reference_square):
    error_square = float(error_by_head.sum())
    reliable_denominator = reference_square > 1e-10 * max(1.0, error_square)
    if error_square == 0:
        return {
            "error_square": 0.0,
            "reference_square": reference_square,
            "relative_l2": 0.0 if reliable_denominator else None,
            "relative_denominator_reliable": reliable_denominator,
            "zero_error": True,
            "head_error_share": None,
            "token_error_share": None,
            "top1_heads": [],
            "top4_heads": [],
            "top_token_index": None,
            "top_head_index": None,
            "token_concentration": None,
        }
    head_order = _rank(error_by_head, min(4, len(error_by_head)))
    token_order = _rank(error_by_token, len(error_by_token))
    concentration = {
        f"top{percent}pct": float(
            error_by_token[
                token_order[: math.ceil(len(token_order) * percent / 100)]
            ].sum()
            / error_square
        )
        for percent in (1, 5, 10)
    }
    return {
        "error_square": error_square,
        "reference_square": reference_square,
        "relative_l2": math.sqrt(error_square / reference_square)
        if reliable_denominator
        else None,
        "relative_denominator_reliable": reliable_denominator,
        "zero_error": False,
        "head_error_share": (error_by_head / error_square).tolist(),
        "token_error_share": (error_by_token / error_square).tolist(),
        "top1_heads": head_order[:1],
        "top4_heads": head_order,
        "top_token_index": token_order[0],
        "top_head_index": head_order[0],
        "token_concentration": concentration,
    }


def pair_distribution(reference, candidate, chunk_tokens=128):
    if (
        reference.shape != candidate.shape
        or reference.ndim != 4
        or reference.shape[1:] != (3, HEADS, HEAD_WIDTH)
    ):
        raise ValueError("Expected matching [tokens,3,16,128] gradients")
    tokens = reference.shape[0]
    errors = np.zeros((3, tokens, HEADS), dtype=np.float64)
    norms = np.zeros(3, dtype=np.float64)
    for start in range(0, tokens, chunk_tokens):
        stop = min(tokens, start + chunk_tokens)
        a = reference[start:stop].astype(np.float64)
        delta = candidate[start:stop].astype(np.float64) - a
        errors[:, start:stop] = np.sum(delta * delta, axis=3).transpose(1, 0, 2)
        norms += np.sum(a * a, axis=(0, 2, 3))
    reports = {}
    for index, part in enumerate(PARTS):
        selected = errors[index] if part != "fused" else errors.sum(axis=0)
        reference_square = float(norms[index] if part != "fused" else norms.sum())
        reports[part] = _part_report(
            selected.sum(axis=0), selected.sum(axis=1), reference_square
        )
    return reports


def comparisons():
    for side in interpolation.SIDES:
        for value in interpolation.LAMBDAS[1:]:
            yield (
                "baseline",
                interpolation.label(0, side),
                interpolation.label(value, side),
            )
        for left, right in zip(interpolation.LAMBDAS, interpolation.LAMBDAS[1:]):
            yield (
                "adjacent",
                interpolation.label(left, side),
                interpolation.label(right, side),
            )
    for value in interpolation.LAMBDAS:
        yield (
            "v_contrast",
            interpolation.label(value, "R"),
            interpolation.label(value, "C"),
        )


def analyze(root):
    torch.set_num_threads(4)
    analysis, source_hashes = validate_source(root)
    if sum(item["disk_bytes"] for item in analysis["files"].values()) != analysis.get(
        "disk_bytes"
    ):
        raise ValueError("K interpolation aggregate budget mismatch")
    if analysis["disk_bytes"] > interpolation.matrix.norm.core.parent.TOTAL_MAX_BYTES:
        raise ValueError("K interpolation aggregate budget exceeded")
    reports = {"baseline": {}, "adjacent": {}, "v_contrast": {}}
    # Repeated disk reads cap live gradient tensors at two cells, irrespective of run size.
    for group, left, right in comparisons():
        reference = load_cell(root, analysis, left)
        candidate = load_cell(root, analysis, right)
        reports[group][f"{left}__{right}"] = pair_distribution(reference, candidate)
        del reference, candidate
    return {
        "status": "analyzed",
        "scope": "saved_k_interpolation_gradient_head_token_distribution",
        "production_ready": False,
        "method": "CPU NumPy FP64 chunked saved-gradient comparison",
        "analysis_sha256": sha256_file(Path(__file__)),
        "audit_scope": "Validates saved audit metadata and statistics; does not rerun source endpoint VJP or independent audit.",
        "source_hashes": source_hashes,
        "cell_sha256": {
            name: analysis["files"][name]["sha256"] for name in interpolation.CELLS
        },
        "shape": list(SHAPE),
        "heads": HEADS,
        "head_width": HEAD_WIDTH,
        "tokens": TOKENS,
        "comparisons": reports,
        "interpretation_limit": "Saved gradient differences only; no attention replay or causal attribution.",
    }


def run(root, output_dir):
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    if (
        output_dir.exists()
        or output_dir == root
        or output_dir in root.parents
        or root in output_dir.parents
    ):
        raise ValueError("Fresh output directory disjoint from source required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Set CUDA_VISIBLE_DEVICES='' for CPU-only analysis")
    started = time.perf_counter()
    usage_before = resource.getrusage(resource.RUSAGE_SELF)
    report = analyze(root)
    usage_after = resource.getrusage(resource.RUSAGE_SELF)
    report["resources"] = {
        "elapsed_seconds": time.perf_counter() - started,
        "user_cpu_seconds": usage_after.ru_utime - usage_before.ru_utime,
        "system_cpu_seconds": usage_after.ru_stime - usage_before.ru_stime,
        "max_rss_kib": usage_after.ru_maxrss,
        "torch_threads": torch.get_num_threads(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "distribution.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.run, args.output_dir)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": str(args.output_dir),
                "resources": report["resources"],
            }
        )
    )


if __name__ == "__main__":
    main()
