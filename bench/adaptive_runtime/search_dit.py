"""Joint precision-island and OOM discovery with disposable real-DiT updates."""

import argparse
import json
from pathlib import Path
import sys

import torch

from bench.adaptive_runtime.recovery import FAMILIES
from library.training.adaptive_runtime.joint import discover_preflight_plan
from library.training.adaptive_runtime.precision import preferred_probe_precision
from library.training.adaptive_runtime.process import IsolatedRunner
from library.training.adaptive_runtime.retry import MemoryPlan, RetryLimits
from library.training.auto_block_swap.process import write_result


def probe_command(args, precision, promoted, request):
    module, _ = FAMILIES[args.model_family]
    data = json.loads(request.read_text())
    if data["resume"] is not None:
        raise ValueError("Joint preflight must not resume training progress")
    plan = data["plan"]
    if (not plan["gradient_checkpointing"] or plan["micro_batch"] != 1
            or plan["accumulation"] != 1):
        raise ValueError("DiT preflight worker requires checkpointing and fixed batch=1")
    argv = [sys.executable, "-m", f"bench.adaptive_runtime.{module}",
            "--weights", str(args.weights), "--inputs", str(args.inputs),
            "--output", str(request.parent / "result.json"), "--structured-worker",
            "--disposable-probe", "--precision", precision,
            "--swap", str(plan["blocks_to_swap"])]
    if args.model_family == "z_image":
        argv.append("--trace-numerics")
    for pattern in [*args.fp32_pattern, *promoted]:
        argv.extend(["--fp32-pattern", pattern])
    if args.memory_limit_gib is not None:
        argv.extend(["--memory-limit-gib", str(args.memory_limit_gib)])
    return argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", choices=tuple(FAMILIES), required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16-islands", "fp32-reference"),
                        default="auto")
    parser.add_argument("--initial-swap", type=int, default=20)
    parser.add_argument("--max-swap", type=int)
    parser.add_argument("--swap-increment", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--max-promotions", type=int, default=8)
    parser.add_argument("--fp32-pattern", action="append", default=[])
    parser.add_argument("--worker-timeout", type=float, default=420)
    parser.add_argument("--memory-limit-gib", type=float)
    args = parser.parse_args()
    maximum = FAMILIES[args.model_family][1]
    if args.max_swap is None:
        args.max_swap = maximum
    if not 0 <= args.initial_swap <= args.max_swap <= maximum or args.max_promotions < 0:
        parser.error("Invalid model swap bounds or promotion budget")
    limits = RetryLimits(max_blocks=args.max_swap, max_attempts=args.max_attempts,
                         swap_increment=args.swap_increment)
    if args.output.exists():
        raise FileExistsError(args.output)
    precision = args.precision
    if precision == "auto":
        precision = preferred_probe_precision(torch.cuda.get_device_capability())
    if precision != "fp16-islands" and args.fp32_pattern:
        parser.error("FP32 patterns require fp16-islands in this experimental worker")

    def runner(plan, promoted, *, attempt):
        process = IsolatedRunner(
            args.output, lambda request: probe_command(args, precision, promoted, request),
            timeout=args.worker_timeout,
            env={"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
        )
        return process(plan, attempt=attempt, resume=None)

    def record(report):
        write_result(args.output / "summary.json", {
            **report, "model_family": args.model_family, "precision": precision,
            "inputs": str(args.inputs), "initial_fp32_patterns": args.fp32_pattern,
            "memory_limit_gib": args.memory_limit_gib,
        })

    report = discover_preflight_plan(
        MemoryPlan(blocks_to_swap=args.initial_swap, gradient_checkpointing=True),
        limits, runner, record=record, max_promotions=args.max_promotions,
        allow_precision_promotions=precision == "fp16-islands",
    )
    print(json.dumps({k: report.get(k) for k in ("status", "selected", "fp32_modules")}, indent=2))
    if report["status"] != "finite_only":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
