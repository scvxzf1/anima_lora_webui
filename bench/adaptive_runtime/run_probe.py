"""Run one structured probe under owned-process timeout and host RAM guards."""

import argparse
import json
from pathlib import Path
import sys

from library.training.adaptive_runtime.process import IsolatedRunner
from library.training.adaptive_runtime.retry import MemoryPlan
from library.training.auto_block_swap.process import write_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=360)
    parser.add_argument("--swap", type=int, default=20)
    parser.add_argument("--module", choices=("probe_krea_train", "probe_z_image_train"), required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if any(value in ("--output", "--swap") or value.startswith(("--output=", "--swap="))
           for value in arguments):
        parser.error("Worker output and swap plan are owned by the supervisor")
    runner = IsolatedRunner(
        args.output,
        lambda request: [sys.executable, "-m", f"bench.adaptive_runtime.{args.module}",
                         "--output", str(request.parent / "result.json"),
                         "--swap", str(args.swap), "--structured-worker", *arguments],
        timeout=args.timeout,
        env={"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
    )
    report = runner(MemoryPlan(blocks_to_swap=args.swap, gradient_checkpointing=True),
                    attempt=0, resume=None)
    write_result(args.output / "summary.json", report)
    print(json.dumps(report, indent=2))
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
