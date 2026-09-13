"""Read-only, bounded NVML CLI sampling for the AUTO hardware experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import time

import psutil


def sample():
    fields = [
        "memory.used",
        "utilization.gpu",
        "temperature.gpu",
        "clocks.sm",
        "power.draw",
    ]
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={','.join(fields)}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    )
    devices = [
        dict(zip(fields, row))
        for row in csv.reader(raw.splitlines(), skipinitialspace=True)
    ]
    host = psutil.virtual_memory()
    return {
        "time": time.time(),
        "devices": devices,
        "host_available": host.available,
        "swap_used": psutil.swap_memory().used,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=3600)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 7200:
        raise ValueError("Sampling duration must be in 1..7200 seconds")
    end = time.monotonic() + args.seconds
    with args.output.open("x", encoding="utf-8") as stream:
        try:
            while time.monotonic() < end:
                stream.write(json.dumps(sample()) + "\n")
                stream.flush()
                time.sleep(2)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
