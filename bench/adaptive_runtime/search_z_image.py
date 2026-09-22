"""Isolated FP32 island discovery on the full pretrained Z-Image probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from library.training.adaptive_runtime.numerics import discover_finite_plan
from library.training.adaptive_runtime.process import IsolatedRunner
from library.training.adaptive_runtime.retry import MemoryPlan
from library.training.auto_block_swap.process import write_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--swap", type=int, default=8)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--fp32-pattern", action="append", default=[])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    def runner(promoted, *, attempt):
        def command(request):
            argv = [sys.executable, "-m", "bench.adaptive_runtime.probe_z_image_train",
                    "--weights", str(args.weights), "--output", str(request.parent / "result.json"),
                    "--precision", "fp16-islands", "--trace-numerics", "--structured-worker",
                    "--swap", str(args.swap)]
            if args.inputs:
                argv.extend(["--inputs", str(args.inputs)])
            for pattern in args.fp32_pattern:
                argv.extend(["--fp32-pattern", pattern])
            for name in promoted:
                argv.extend(["--fp32-module", name])
            return argv

        process = IsolatedRunner(args.output, command, timeout=300,
                                 env={"HF_HUB_OFFLINE": "1"})
        return process(MemoryPlan(blocks_to_swap=args.swap), attempt=attempt, resume=None)

    def record(value):
        write_result(args.output / "summary.json", {
            **value, "initial_fp32_patterns": args.fp32_pattern,
            "inputs": str(args.inputs) if args.inputs else None,
        })

    report = discover_finite_plan(
        runner, max_attempts=args.max_attempts,
        record=record,
    )
    print(json.dumps(report, indent=2))
    if report["status"] != "finite_only":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
