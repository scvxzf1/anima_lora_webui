"""Select an experimental Z-Image profile using fixed-state gradient comparisons."""

import argparse
import json
from pathlib import Path
import sys

import torch

from bench.adaptive_runtime.compare_training import compare, read_capture
from bench.adaptive_runtime.replay import MODE
from library.training.adaptive_runtime.precision import preferred_candidate
from library.training.adaptive_runtime.process import IsolatedRunner
from library.training.adaptive_runtime.profile_search import PrecisionProfile, search_profiles
from library.training.adaptive_runtime.retry import MemoryPlan, RetryLimits
from library.training.auto_block_swap.process import write_result


SEED = ("*.attention.to_out.0", "*.feed_forward.w2", "layers.28.attention.to_k",
        "layers.27.attention.to_k", "layers.28.attention.to_q", "layers.28.attention.to_v")
CONDITIONING = ("t_embedder.*", "all_x_embedder.*", "cap_embedder.*",
                "*.adaLN_modulation.*", "all_final_layer.*")


def profiles_for(names, *, candidate, loss_scale):
    if candidate not in {"bf16", "fp16", "fp32"}:
        raise ValueError("Unknown hardware precision candidate")
    choices = {
        "bf16": PrecisionProfile("bf16", "bf16"),
        "fp32": PrecisionProfile("fp32", "fp32-reference"),
        "seed": PrecisionProfile("seed", "fp16-islands", SEED, loss_scale),
        "conditioning": PrecisionProfile("conditioning", "fp16-islands", (*SEED, *CONDITIONING), loss_scale),
        "conditioning-mlp": PrecisionProfile("conditioning-mlp", "fp16-islands",
                                             (*SEED, *CONDITIONING, "*.feed_forward.*"), loss_scale),
        "conditioning-attention": PrecisionProfile("conditioning-attention", "fp16-islands",
                                                   (*SEED, *CONDITIONING, "*.attention.*"), loss_scale),
        "mlp": PrecisionProfile("mlp", "fp16-islands", (*SEED, "*.feed_forward.*"), loss_scale),
        "attention": PrecisionProfile("attention", "fp16-islands",
                                      ("*.attention.*", "*.feed_forward.w2"), loss_scale),
        "fp32-control": PrecisionProfile("fp32-control", "fp16-islands", ("*",), loss_scale),
    }
    if not names:
        names = (["fp32"] if candidate == "fp32" else
                 (["bf16"] if candidate == "bf16" else []) + ["seed", "conditioning", "mlp", "attention"])
    if "bf16" in names and candidate != "bf16":
        raise ValueError("BF16 profile requires native hardware support")
    return [choices[name] for name in names]


def command(args, profile, request):
    data = json.loads(request.read_text())
    plan = data["plan"]
    if data["resume"] is not None or not plan["gradient_checkpointing"]:
        raise ValueError("Precision search requires fresh checkpointed disposable workers")
    argv = [sys.executable, "-m", "bench.adaptive_runtime.probe_z_image_train",
            "--weights", str(args.weights), "--inputs", str(args.inputs),
            "--output", str(request.parent / "result.json"), "--structured-worker",
            "--disposable-probe", "--capture-training", "--replay-reference", str(args.reference),
            "--precision", profile.precision, "--loss-scale", str(profile.loss_scale),
            "--swap", str(plan["blocks_to_swap"]), "--steps", str(args.steps),
            "--resolution", str(args.resolution)]
    for pattern in profile.fp32_patterns:
        argv.extend(["--fp32-pattern", pattern])
    if args.memory_limit_gib is not None:
        argv.extend(["--memory-limit-gib", str(args.memory_limit_gib)])
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "inputs", "reference", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--profile", action="append",
                        choices=("bf16", "seed", "conditioning", "conditioning-mlp",
                                 "conditioning-attention", "mlp", "attention", "fp32-control", "fp32"))
    parser.add_argument("--loss-scale", type=float, default=1024)
    parser.add_argument("--initial-swap", type=int, default=24)
    parser.add_argument("--max-swap", type=int, default=28)
    parser.add_argument("--swap-increment", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=6)
    parser.add_argument("--worker-timeout", type=float, default=420)
    parser.add_argument("--memory-limit-gib", type=float)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 <= args.initial_swap <= args.max_swap <= 28:
        parser.error("Z-Image swap bounds must be within 0..28")
    manifest = json.loads((args.reference / "manifest.json").read_text())
    if (manifest.get("schema") != "adaptive_training_replay_v1"
            or manifest.get("status") != "captured"
            or manifest.get("signature", {}).get("comparison_mode") != MODE
            or manifest.get("signature", {}).get("model_family") != "z_image"):
        parser.error("Completed Z-Image fixed-state replay reference required")
    args.steps, args.resolution = manifest["steps"], manifest["signature"]["resolution"]
    read_capture(args.reference.parent / "training-capture")
    if manifest.get("uuid") != str(torch.cuda.get_device_properties(0).uuid):
        parser.error("Replay reference must belong to the selected GPU")
    profiles = profiles_for(args.profile, candidate=preferred_candidate(
        torch.cuda.get_device_capability()), loss_scale=args.loss_scale)
    limits = RetryLimits(max_blocks=args.max_swap, max_attempts=args.max_attempts,
                         swap_increment=args.swap_increment)

    def runner(profile, plan, *, attempt):
        return IsolatedRunner(args.output, lambda request: command(args, profile, request),
                              timeout=args.worker_timeout,
                              env={"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})(
                                  plan, attempt=attempt, resume=None)

    def comparator(attempt):
        report = compare(args.reference.parent / "training-capture",
                         args.output / f"attempt-{attempt:03d}" / "training-capture")
        write_result(args.output / f"comparison-{attempt:03d}.json", report)
        return report

    def record(report):
        write_result(args.output / "summary.json", {**report, "reference": str(args.reference),
                     "weights": str(args.weights), "inputs": str(args.inputs),
                     "memory_limit_gib": args.memory_limit_gib})

    args.output.mkdir(parents=True, exist_ok=False)
    report = search_profiles(MemoryPlan(args.initial_swap, True), limits, profiles,
                             runner, comparator, record=record)
    print(json.dumps({k: report.get(k) for k in ("status", "selected_profile", "selected_plan")}, indent=2))
    if report["status"] != "probe_validated":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
