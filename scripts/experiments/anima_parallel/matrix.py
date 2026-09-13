"""Run a bounded serial matrix without touching unrelated GPU processes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from time import monotonic

from .common import write_json


def run_case(case, root, *, timeout):
    name = case["name"]
    output = root / name
    if output.exists():
        return {"name": name, "status": "skipped_existing"}
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = case.get("devices", "0,1")
    env["OMP_NUM_THREADS"] = "2"
    command = [sys.executable, "-m"]
    if case["mode"] != "single":
        command += ["torch.distributed.run", "--standalone", "--nproc_per_node=2", "-m"]
    command += [
        "scripts.experiments.anima_pipeline_bench",
        "--mode",
        case["mode"],
        "--output-dir",
        str(output),
        *case.get("args", []),
    ]
    started = monotonic()
    with (root / f"{name}.log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
            status = "passed" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            code, status = process.returncode, "timeout"
    return {
        "name": name,
        "status": status,
        "exit_code": code,
        "seconds": monotonic() - started,
        "command": command,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case-timeout", type=int, default=480)
    parser.add_argument("--budget-seconds", type=int, default=2400)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = monotonic()
    results = []
    for case in plan:
        remaining = args.budget_seconds - (monotonic() - started)
        if remaining < 60:
            break
        print(f"START {case['name']}", flush=True)
        result = run_case(
            case, args.output_root, timeout=min(args.case_timeout, remaining)
        )
        results.append(result)
        write_json(
            args.output_root / "matrix-status.json",
            {"cases": results, "elapsed_seconds": monotonic() - started},
        )
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
