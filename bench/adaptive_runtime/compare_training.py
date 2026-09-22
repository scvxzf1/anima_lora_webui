"""Compare complete short-probe trajectories against FP32 from the same input cache."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from safetensors.torch import load_file
import torch

from bench.adaptive_runtime.capture import capture_path
from bench.adaptive_runtime.replay import MODE
from library.training.adaptive_runtime.precision import Tolerances
from library.training.auto_block_swap.process import write_result


def read_capture(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    worker = json.loads((directory.parent / "result.json").read_text())
    if (manifest.get("schema") != "adaptive_training_capture_v1"
            or manifest.get("status") != "captured" or worker.get("status") != "ok"
            or type(manifest.get("steps")) is not int or manifest["steps"] < 1
            or len(manifest.get("cases", [])) != manifest["steps"]):
        raise ValueError("Completed training capture and successful worker required")
    if [c.get("step") for c in manifest["cases"]] != list(range(1, manifest["steps"] + 1)):
        raise ValueError("Noncontiguous training capture")
    signature = manifest.get("signature", {})
    scale = signature.get("loss_scale", 1.0)
    if (signature.get("precision") not in {"bf16", "fp16-islands", "fp32-reference"}
            or worker.get("precision") != signature.get("precision")
            or worker.get("inputs_sha256") != signature.get("inputs_sha256")
            or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale < 1
            or worker.get("loss_scale", 1.0) != scale
            or [u.get("step") for u in worker.get("updates", [])]
            != list(range(1, manifest["steps"] + 1))):
        raise ValueError("Worker and capture contract mismatch")
    gradients = manifest.get("expected_gradients")
    if (not isinstance(gradients, list) or not gradients
            or not all(isinstance(n, str) and n.startswith("gradient.") for n in gradients)
            or len(set(gradients)) != len(gradients)):
        raise ValueError("Missing expected trainable parameter inventory")
    return manifest, worker


def load_case(directory, case, expected_gradients):
    path = capture_path(directory, case["file"])
    if path.stat().st_size > 512 * 1024**2:
        raise ValueError("Training comparison case exceeds 512MiB budget")
    if hashlib.sha256(path.read_bytes()).hexdigest() != case["sha256"]:
        raise ValueError("Training capture digest mismatch")
    tensors = load_file(str(path))
    if (not {"prediction", "input_gradient"} <= set(tensors) or len(tensors) <= 2
            or set(tensors) - {"prediction", "input_gradient"} != set(expected_gradients)
            or not all(k in {"prediction", "input_gradient"} or k.startswith("gradient.")
                       for k in tensors)
            or not all(t.dtype == torch.float32 and t.numel() and torch.isfinite(t).all()
                       for t in tensors.values())):
        raise ValueError("Invalid training capture tensors")
    return tensors


def aggregate_metrics(reference, candidate, names):
    ref_square = actual_square = delta_square = dot = 0.0
    for name in names:
        ref, actual = reference[name].double(), candidate[name].double()
        ref_square += float(ref.square().sum())
        actual_square += float(actual.square().sum())
        delta_square += float((actual - ref).square().sum())
        dot += float((ref * actual).sum())
    cosine = (1.0 if ref_square == actual_square == 0 else
              dot / max(math.sqrt(ref_square * actual_square), 1e-30))
    return {"relative_l2": math.sqrt(delta_square) / max(math.sqrt(ref_square), 1e-30),
            "absolute_l2": math.sqrt(delta_square),
            "cosine": max(-1.0, min(1.0, cosine)), "reference_l2": math.sqrt(ref_square)}


def compare_case(reference, candidate):
    if (set(reference) != set(candidate)
            or any(reference[k].shape != candidate[k].shape for k in reference)):
        raise ValueError("Prediction/gradient key or shape contract changed")
    names = [n for n in reference if n.startswith("gradient.")]
    groups = {"prediction": ["prediction"], "input_gradient": ["input_gradient"],
              "adapter_gradient": names}
    metrics = {key: aggregate_metrics(reference, candidate, group) for key, group in groups.items()}
    worst = [{"name": n, **aggregate_metrics(reference, candidate, [n])} for n in names]
    metrics["worst_adapter_parameters"] = sorted(worst, key=lambda r: r["relative_l2"], reverse=True)[:10]
    metrics["largest_absolute_adapter_errors"] = sorted(
        worst, key=lambda r: r["absolute_l2"], reverse=True)[:10]
    limits = Tolerances()
    metrics["within_experimental_tolerances"] = (
        metrics["prediction"]["relative_l2"] <= limits.output_relative_l2
        and all(metrics[key]["relative_l2"] <= limits.gradient_relative_l2
                and metrics[key]["cosine"] >= limits.gradient_cosine
                for key in ("input_gradient", "adapter_gradient"))
    )
    return metrics


def compare(reference, candidate):
    ref, ref_worker = read_capture(reference)
    trial, trial_worker = read_capture(candidate)
    if ref["signature"].get("precision") != "fp32-reference":
        raise ValueError("Reference must compute the nonquantized model in FP32")
    if (not isinstance(ref_worker.get("uuid"), str) or not ref_worker["uuid"]
            or ref_worker["uuid"] != trial_worker.get("uuid")):
        raise ValueError("Reference and candidate must identify the same GPU UUID")
    excluded = {"precision", "fp32_modules", "loss_scale"}
    ref_contract = {k: v for k, v in ref["signature"].items() if k not in excluded}
    trial_contract = {k: v for k, v in trial["signature"].items() if k not in excluded}
    if (ref_contract != trial_contract or ref["steps"] != trial["steps"]
            or ref["initial_adapter_sha256"] != trial["initial_adapter_sha256"]):
        raise ValueError("Training reference/input/base/initial-adapter contract mismatch")
    fixed = ref["signature"].get("comparison_mode") == MODE
    if fixed and (ref_worker.get("replay_mode") != "record" or trial_worker.get("replay_mode") != "replay"):
        raise ValueError("Fixed-state comparison requires recorded reference and replayed candidate")
    cases = []
    for a, b in zip(ref["cases"], trial["cases"], strict=True):
        if a["sigma"] != b["sigma"]:
            raise ValueError("Training sigma sequence mismatch")
        if fixed and (not isinstance(a.get("replay"), dict) or a["replay"] != b.get("replay")
                      or set(a["replay"]) != {"snapshot_sha256", "state_sha256", "inputs_sha256", "rng_sha256"}
                      or not all(isinstance(v, str) and len(v) == 64 for v in a["replay"].values())):
            raise ValueError("Fixed-state comparison requires matching per-step replay evidence")
        cases.append({"step": a["step"], "sigma": a["sigma"],
                      "reference_sha256": a["sha256"], "candidate_sha256": b["sha256"],
                      **compare_case(load_case(reference, a, ref["expected_gradients"]),
                                     load_case(candidate, b, trial["expected_gradients"]))})
    return {"status": "compared", "production_ready": False, "full_model_calibrated": False,
            "scope": ("fixed_pre_step_state_and_inputs_not_quality_certificate" if fixed else
                      "aggregate_short_training_trajectory_not_quality_certificate"),
            "reference": {**{k: ref_worker.get(k) for k in ("precision", "uuid", "swap")},
                          "loss_scale": ref_worker.get("loss_scale", 1.0)},
            "candidate": {**{k: trial_worker.get(k) for k in ("precision", "uuid", "swap")},
                          "loss_scale": trial_worker.get("loss_scale", 1.0)},
            "cases": cases, "tolerances": vars(Tolerances()),
            "within_experimental_tolerances": all(c["within_experimental_tolerances"] for c in cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    report = compare(args.reference, args.candidate)
    write_result(args.output, report)
    print(json.dumps({"status": report["status"],
                      "within_experimental_tolerances": report["within_experimental_tolerances"]}, indent=2))


if __name__ == "__main__":
    main()
