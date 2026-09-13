"""Bounded subprocess lifecycle and telemetry for calibration candidates."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

from .resources import host_memory


class ProbeComplete(Exception):
    def __init__(self, result):
        super().__init__(result["status"])
        self.result = result


def write_result(path: Path, result: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def stop_process(process) -> None:
    if process.poll() is not None:
        return
    # Only our own new session, never a training service or another user job.
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=5)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        while True:
            try:
                process.wait()
                break
            except KeyboardInterrupt:
                continue


def run_process(args, directory: Path, *, timeout: int) -> dict:
    directory.mkdir(parents=True, exist_ok=False)
    request = directory / "request.json"
    request.write_text(json.dumps(vars(args)), encoding="utf-8")
    request.chmod(0o600)
    before = host_memory()
    minimum = before.available
    rss_peak = 0
    reason = None
    process = None
    started = time.monotonic()
    try:
        if minimum <= before.reserve:
            return {"status": "host_limit", "host_before": before.to_dict()}
        environment = os.environ.copy()
        environment.update(
            {
                "WANDB_MODE": "disabled",
                "HF_HUB_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            }
        )
        with (directory / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "library.training.auto_block_swap.worker",
                    str(request),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=(os.name == "posix"),
            )
            child = psutil.Process(process.pid)
            while process.poll() is None:
                current = host_memory()
                minimum = min(minimum, current.available)
                try:
                    family = [child, *child.children(recursive=True)]
                    rss_peak = max(rss_peak, sum(p.memory_info().rss for p in family))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                if current.available <= current.reserve:
                    reason = "host_limit"
                elif time.monotonic() - started >= timeout:
                    reason = "timeout"
                if reason:
                    stop_process(process)
                    break
                time.sleep(0.05)
        result_path = directory / "result.json"
        result = (
            json.loads(result_path.read_text())
            if result_path.exists()
            else {"status": "error"}
        )
        if reason:
            result = {"status": reason}
        elif process.returncode != 0:
            result = {"status": "error", "returncode": process.returncode}
        after = host_memory()
        result.update(
            {
                "host_before": before.to_dict(),
                "host_after": after.to_dict(),
                "host_min_available": minimum,
                "host_rss_peak": rss_peak,
                "host_reserve": max(before.reserve, after.reserve),
                "swap_io_bytes": max(0, after.swap_in - before.swap_in)
                + max(0, after.swap_out - before.swap_out),
                "elapsed_seconds": time.monotonic() - started,
                "swap_io_limit_bytes": int(
                    float(getattr(args, "auto_block_swap_swap_io_limit_mb", 1024.0))
                    * 1024**2
                ),
            }
        )
        write_result(directory / "result.json", result)
        return result
    finally:
        if process is not None:
            stop_process(process)
        request.unlink(missing_ok=True)
