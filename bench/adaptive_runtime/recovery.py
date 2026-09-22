"""Bounded real-DiT OOM recovery; precision is fixed across memory trials."""

import argparse
import json
import math
from pathlib import Path
import sys

import torch

from library.training.adaptive_runtime.precision import preferred_probe_precision
from library.training.adaptive_runtime.process import IsolatedRunner
from library.training.adaptive_runtime.retry import MemoryPlan, RetryLimits, run_recovery
from library.training.auto_block_swap.process import write_result


FAMILIES = {"krea2": ("probe_krea_train", 26), "z_image": ("probe_z_image_train", 28)}


def validate_recovery_scaling(args, precision, model_family):
    scale = getattr(args, "loss_scale", 1.0)
    enabled = getattr(args, "scaled_checkpoint", False)
    if not math.isfinite(scale) or scale < 1:
        raise ValueError("Recovery loss scale must be finite and >= 1")
    if enabled:
        if model_family not in FAMILIES or precision != "fp16-islands" or scale == 1:
            raise ValueError("Scaled recovery requires a supported FP16-islands family and loss scale > 1")
    elif scale != 1:
        raise ValueError("Nondefault recovery loss scale requires --scaled-checkpoint")


def worker_command(args, precision, request, *, model_family):
    module, _ = FAMILIES[model_family]
    validate_recovery_scaling(args, precision, model_family)
    data = json.loads(request.read_text())
    argv = [sys.executable, "-m", f"bench.adaptive_runtime.{module}",
            "--weights", str(args.weights), "--inputs", str(args.inputs),
            "--output", str(request.parent / "result.json"), "--structured-worker",
            "--precision", precision, "--swap", str(data["plan"]["blocks_to_swap"]),
            "--checkpoint-every-step"]
    if getattr(args, "scaled_checkpoint", False):
        argv.extend(["--scaled-checkpoint", "--loss-scale", str(args.loss_scale)])
    if data["resume"] is not None:
        argv.extend(["--resume", data["resume"]])
    if args.memory_limit_gib is not None:
        argv.extend(["--memory-limit-gib", str(args.memory_limit_gib)])
    for pattern in args.fp32_pattern:
        argv.extend(["--fp32-pattern", pattern])
    return argv


def main(*, model_family):
    _, maximum = FAMILIES[model_family]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16-islands", "fp32-reference"),
                        default="auto")
    parser.add_argument("--fp32-pattern", action="append", default=[])
    parser.add_argument("--loss-scale", type=float, default=1.0)
    parser.add_argument("--scaled-checkpoint", action="store_true")
    parser.add_argument("--memory-limit-gib", type=float)
    parser.add_argument("--initial-swap", type=int, default=0)
    parser.add_argument("--max-swap", type=int, default=maximum)
    parser.add_argument("--swap-increment", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=7)
    parser.add_argument("--worker-timeout", type=float, default=360)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 <= args.initial_swap <= args.max_swap <= maximum:
        parser.error(f"{model_family} swap bounds must satisfy 0 <= initial <= maximum <= {maximum}")
    precision = args.precision
    if precision == "auto":
        precision = preferred_probe_precision(torch.cuda.get_device_capability())
    if precision != "fp16-islands" and args.fp32_pattern:
        parser.error("FP32 patterns require fp16-islands in this experimental worker")
    try:
        validate_recovery_scaling(args, precision, model_family)
    except ValueError as exc:
        parser.error(str(exc))

    def command(request):
        return worker_command(args, precision, request, model_family=model_family)

    runner = IsolatedRunner(args.output, command, timeout=args.worker_timeout,
                            env={"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})

    def record(report):
        write_result(args.output / "summary.json", {
            **report, "precision": precision, "precision_calibrated": False,
            "model_family": model_family,
            "initial_fp32_patterns": args.fp32_pattern,
            "memory_limit_gib": args.memory_limit_gib,
            "loss_scale": args.loss_scale, "scaled_checkpoint": args.scaled_checkpoint,
        })

    report = run_recovery(
        MemoryPlan(blocks_to_swap=args.initial_swap, gradient_checkpointing=True),
        RetryLimits(max_blocks=args.max_swap, max_attempts=args.max_attempts,
                    swap_increment=args.swap_increment), runner, record=record,
    )
    print(json.dumps({"status": report["status"], "selected": report.get("selected"),
                      "attempts": len(report["attempts"]), "precision": precision}, indent=2))
    if report["status"] != "ok":
        raise SystemExit(1)
