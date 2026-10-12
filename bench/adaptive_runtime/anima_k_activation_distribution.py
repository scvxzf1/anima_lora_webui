"""CPU-only post-RoPE K activation drift versus saved K-interpolation gradients."""

import argparse
import json
import math
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from bench.adaptive_runtime import anima_k_distribution as gradients
from bench.adaptive_runtime import anima_k_interpolation as interpolation
from bench.adaptive_runtime.anima_inputs import sha256_file


SCOPE = "block17_postrope_k_activation_vs_saved_gradient_distribution_not_production"
PERCENTS = (1, 5, 10)


def axis_energy(reference, candidate, chunk_tokens=128):
    """Return FP64 squared drift and reference energy by head and token."""
    if (
        reference.shape != candidate.shape
        or reference.ndim != 3
        or reference.shape[1:] != (gradients.HEADS, gradients.HEAD_WIDTH)
        or reference.shape[0] == 0
        or chunk_tokens <= 0
    ):
        raise ValueError("Expected matching nonempty [tokens,16,128] K tensors")
    error = np.empty((reference.shape[0], gradients.HEADS), dtype=np.float64)
    baseline = np.empty_like(error)
    for start in range(0, reference.shape[0], chunk_tokens):
        end = min(start + chunk_tokens, reference.shape[0])
        a = np.asarray(reference[start:end], dtype=np.float64)
        b = np.asarray(candidate[start:end], dtype=np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Nonfinite K activation")
        error[start:end] = np.sum((b - a) ** 2, axis=-1)
        baseline[start:end] = np.sum(a**2, axis=-1)
    return error, baseline


def distribution(energy, reference_energy):
    if (
        energy.shape != reference_energy.shape
        or energy.ndim != 2
        or not np.isfinite(energy).all()
        or not np.isfinite(reference_energy).all()
        or np.any(energy < 0)
        or np.any(reference_energy < 0)
    ):
        raise ValueError("Invalid K energy axes")
    total, baseline = float(energy.sum()), float(reference_energy.sum())
    heads, tokens = energy.sum(axis=0), energy.sum(axis=1)
    head_order = np.argsort(-heads, kind="stable").astype(int).tolist()
    token_order = np.argsort(-tokens, kind="stable").astype(int).tolist()
    reliable = baseline > 1e-10 * max(1.0, total)
    return {
        "error_square": total,
        "reference_square": baseline,
        "relative_l2": math.sqrt(total / baseline) if reliable else None,
        "relative_denominator_reliable": reliable,
        "head_error_share": (heads / total).tolist() if total else None,
        "token_error_share": (tokens / total).tolist() if total else None,
        "head_rank": head_order if total else [],
        "token_rank": token_order if total else [],
        "token_concentration": {
            f"top{pct}pct": float(
                tokens[token_order[: math.ceil(len(tokens) * pct / 100)]].sum() / total
            )
            for pct in PERCENTS
        }
        if total
        else None,
    }


def overlap(activation_energy, gradient_energy):
    """Compare independently ranked sets; coverage uses each distribution's own total."""
    if activation_energy.shape != gradient_energy.shape or activation_energy.ndim != 2:
        raise ValueError("K/gradient energy axis mismatch")
    if (
        not np.isfinite(activation_energy).all()
        or not np.isfinite(gradient_energy).all()
        or np.any(activation_energy < 0)
        or np.any(gradient_energy < 0)
    ):
        raise ValueError("Invalid overlap energy")
    axes = {
        "head": (activation_energy.sum(axis=0), gradient_energy.sum(axis=0)),
        "token": (activation_energy.sum(axis=1), gradient_energy.sum(axis=1)),
    }
    result = {}
    for axis, (act, grad) in axes.items():
        ranked_act = np.argsort(-act, kind="stable")
        ranked_grad = np.argsort(-grad, kind="stable")
        result[axis] = {}
        for pct in PERCENTS:
            count = math.ceil(len(act) * pct / 100)
            if act.sum() == 0 or grad.sum() == 0:
                result[axis][f"top{pct}pct"] = {
                    "count": count,
                    "activation_indices": [],
                    "gradient_indices": [],
                    "intersection": None,
                    "intersection_count": None,
                    "gradient_energy_in_activation_top_share": None,
                    "activation_energy_in_gradient_top_share": None,
                }
                continue
            a = ranked_act[:count].astype(int).tolist()
            g = ranked_grad[:count].astype(int).tolist()
            result[axis][f"top{pct}pct"] = {
                "count": count,
                "activation_indices": a,
                "gradient_indices": g,
                "intersection": sorted(set(a) & set(g)),
                "intersection_count": len(set(a) & set(g)),
                "gradient_energy_in_activation_top_share": float(
                    grad[a].sum() / grad.sum()
                )
                if grad.sum()
                else None,
                "activation_energy_in_gradient_top_share": float(
                    act[g].sum() / act.sum()
                )
                if act.sum()
                else None,
            }
    return result


def validate_distribution(run, distribution_run):
    analysis, source_hashes = gradients.validate_source(run)
    path = Path(distribution_run) / "distribution.json"
    report = json.loads(path.read_text())
    if (
        report.get("status") != "analyzed"
        or report.get("production_ready") is not False
        or report.get("scope")
        != "saved_k_interpolation_gradient_head_token_distribution"
        or report.get("analysis_sha256") != sha256_file(Path(gradients.__file__))
        or report.get("source_hashes") != source_hashes
        or report.get("cell_sha256")
        != {name: analysis["files"][name]["sha256"] for name in interpolation.CELLS}
        or report.get("shape") != list(gradients.SHAPE)
    ):
        raise ValueError("Saved gradient distribution source binding invalid")
    for side in interpolation.SIDES:
        pair = f"{interpolation.label(0, side)}__{interpolation.label(1, side)}"
        if pair not in report.get("comparisons", {}).get("baseline", {}):
            raise ValueError("Saved gradient endpoint distribution missing")
    return analysis, report, {**source_hashes, "distribution.json": sha256_file(path)}


def load_k_inputs(run, analysis, weights):
    environment = json.loads((Path(run) / "environment.json").read_text())
    paths = [
        Path(environment[key])
        for key in ("matrix_run", "qknorm_run", "core_run", "source_run", "stage_run")
    ]
    if sha256_file(weights) != environment.get("weights_sha256"):
        raise ValueError("Weight SHA mismatch")
    source, manifests, captured, old, _, _ = interpolation.source_evidence(
        *paths, weights
    )
    if source != analysis["source"]:
        raise ValueError("Rebuilt activation source differs from gradient source")
    kwargs = interpolation.matrix.norm.core.stages.capture.conditioning(
        manifests["reference"], captured["reference"], torch.device("cpu")
    )
    rope = kwargs["rope_cos_sin"]
    if (
        kwargs["attn_params"].attn_mode != "torch"
        or rope is None
        or any(x.dtype != torch.float32 for x in rope)
    ):
        raise ValueError("Expected actual FP32 torch attention conditioning/RoPE")
    q_norm, k_norm = interpolation.matrix.make_norms(
        old["norm_parameters"], torch.device("cpu")
    )
    with torch.no_grad():
        values = {}
        for side in interpolation.matrix.SIDES:
            p = old[side]["p"]
            if p.dtype != torch.float32 or tuple(p.shape) != gradients.SHAPE:
                raise ValueError("Core QKV source shape/dtype changed")
            k = interpolation.matrix.postrope(
                p.reshape(1, gradients.TOKENS, interpolation.matrix.WIDTH),
                q_norm,
                k_norm,
                rope,
                gradients.HEADS,
                gradients.HEAD_WIDTH,
            )[1]
            values[side] = k.squeeze(0).numpy().copy()
    return values, {
        key: sha256_file(path / "summary.json")
        for key, path in zip(("matrix", "qknorm", "core", "upstream", "stage"), paths)
    }


def gradient_energy(run, analysis, side):
    left = gradients.load_cell(run, analysis, interpolation.label(0, side))
    right = gradients.load_cell(run, analysis, interpolation.label(1, side))
    parts = [axis_energy(left[:, index], right[:, index]) for index in range(3)]
    return {
        "k": parts[1],
        "fused": (sum(item[0] for item in parts), sum(item[1] for item in parts)),
    }


def validate_saved_k_report(energy, reference_energy, recorded):
    rebuilt = gradients._part_report(
        energy.sum(axis=0), energy.sum(axis=1), float(reference_energy.sum())
    )
    if set(rebuilt) != set(recorded):
        raise ValueError("Saved gradient distribution inventory inconsistent")
    for key, value in rebuilt.items():
        other = recorded[key]
        if key in ("head_error_share", "token_error_share") and value is not None:
            valid = other is not None and np.allclose(
                value, other, rtol=1e-11, atol=1e-15
            )
        elif key == "token_concentration" and value is not None:
            valid = (
                isinstance(other, dict)
                and set(value) == set(other)
                and all(
                    math.isclose(v, other[k], rel_tol=1e-11, abs_tol=1e-15)
                    for k, v in value.items()
                )
            )
        elif isinstance(value, float):
            valid = isinstance(other, (int, float)) and math.isclose(
                value, other, rel_tol=1e-11, abs_tol=1e-20
            )
        else:
            valid = value == other
        if not valid:
            raise ValueError("Saved gradient distribution inconsistent")


def analyze(run, distribution_run, weights):
    torch.set_num_threads(4)
    analysis, saved, source_hashes = validate_distribution(run, distribution_run)
    values, chain_hashes = load_k_inputs(run, analysis, weights)
    activation_energy, reference_energy = axis_energy(
        values["reference"], values["candidate"]
    )
    activation = distribution(activation_energy, reference_energy)
    comparisons = {}
    for side in interpolation.SIDES:
        pair = f"{interpolation.label(0, side)}__{interpolation.label(1, side)}"
        comparisons[side] = {}
        for part, (grad_energy, grad_reference) in gradient_energy(
            run, analysis, side
        ).items():
            recorded = saved["comparisons"]["baseline"][pair][part]
            validate_saved_k_report(grad_energy, grad_reference, recorded)
            comparisons[side][part] = {
                "saved_gradient_error_square": float(grad_energy.sum()),
                "overlap": overlap(activation_energy, grad_energy),
            }
    return {
        "status": "analyzed",
        "scope": SCOPE,
        "production_ready": False,
        "method": "CPU FP32 actual RMSNorm/RoPE K forward; NumPy FP64 squared-energy statistics",
        "code_sha256": sha256_file(Path(__file__)),
        "source_hashes": source_hashes,
        "source_code_sha256": {
            **interpolation.code_hashes(),
            "qknorm_math_sha256": sha256_file(
                Path(interpolation.matrix.norm.norm_math.__file__)
            ),
        },
        "chain_summary_sha256": chain_hashes,
        "weights_sha256": sha256_file(weights),
        "shape": [gradients.TOKENS, gradients.HEADS, gradients.HEAD_WIDTH],
        "activation": activation,
        "comparisons": comparisons,
        "interpretation_limit": "Spatial association only, not initial rounding source or causal/precision attribution.",
    }


def run(run_dir, distribution_run, weights, output_dir):
    sources = [
        Path(run_dir).resolve(),
        Path(distribution_run).resolve(),
        Path(weights).resolve(),
    ]
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() or any(
        output_dir == source
        or output_dir in source.parents
        or source in output_dir.parents
        for source in sources
    ):
        raise ValueError("Fresh output directory disjoint from sources required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Set CUDA_VISIBLE_DEVICES='' for CPU-only analysis")
    started = time.perf_counter()
    before = resource.getrusage(resource.RUSAGE_SELF)
    report = analyze(*sources)
    after = resource.getrusage(resource.RUSAGE_SELF)
    report["resources"] = {
        "elapsed_seconds": time.perf_counter() - started,
        "user_cpu_seconds": after.ru_utime - before.ru_utime,
        "system_cpu_seconds": after.ru_stime - before.ru_stime,
        "max_rss_kib": after.ru_maxrss,
        "torch_threads": torch.get_num_threads(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "activation-distribution.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--distribution-run", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.run, args.distribution_run, args.weights, args.output_dir)
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
