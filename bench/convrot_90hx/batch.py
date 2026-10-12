"""Serial isolated subprocesses, with durable job status and no automatic retries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .manifest import GPU_UUID, ROOT, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--status", type=Path, required=True)
    args = parser.parse_args()
    jobs = json.loads(args.jobs.read_text())
    status = {
        "pid": os.getpid(),
        "started": time.time(),
        "state": "RUNNING",
        "jobs": [],
    }
    write_json(args.status, status)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=GPU_UUID,
        PYTHONUNBUFFERED="1",
        ANIMA_CONVROT_INT8_GEMM="int_mm",
        ANIMA_CONVROT_STE_TF32="0",
        ANIMA_CONVROT_RHT="dense",
        ANIMA_CONVROT_HADAMARD="sylvester",
    )
    for job in jobs:
        item = {"name": job["name"], "started": time.time(), "state": "RUNNING"}
        status["jobs"].append(item)
        command = [
            sys.executable,
            "-m",
            job.get("module", "bench.convrot_90hx.runner"),
            *job["args"],
        ]
        item["command"] = command
        write_json(args.status, status)
        with Path(job["log"]).open("w") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            item["child_pid"] = process.pid
            write_json(args.status, status)
            item["exit_code"] = process.wait()
        item["ended"] = time.time()
        item["state"] = "PASS" if item["exit_code"] == 0 else "FAIL"
        write_json(args.status, status)
        print(json.dumps(item), flush=True)
        if item["exit_code"] and not job.get("continue_on_failure", False):
            status["state"] = "FAIL"
            write_json(args.status, status)
            raise SystemExit(item["exit_code"])
    status.update(state="COMPLETE", ended=time.time())
    write_json(args.status, status)


if __name__ == "__main__":
    main()
