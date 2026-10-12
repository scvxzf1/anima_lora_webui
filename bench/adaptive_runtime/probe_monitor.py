"""Resource-monitored, disposable worker process; no training-process discovery."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

from library.training.adaptive_runtime.process import _cancellation_signals
from library.training.auto_block_swap.process import stop_process, write_result
from library.training.auto_block_swap.resources import host_memory


def _smi(*args):
    run = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True,
                         timeout=5, check=True)
    return run.stdout.strip()


def gpu_snapshot(uuid):
    fields = ("uuid", "name", "compute_cap", "memory.used", "memory.free", "memory.total",
              "utilization.gpu", "temperature.gpu", "power.draw", "clocks.sm")
    rows = _smi(f"--id={uuid}", f"--query-gpu={','.join(fields)}",
                "--format=csv,noheader,nounits").splitlines()
    matching = [row for row in rows if row.split(",", 1)[0].strip() == uuid]
    if len(matching) != 1:
        raise ValueError("Selected GPU UUID missing or ambiguous")
    values = [value.strip() for value in matching[0].split(",")]
    if len(values) != len(fields):
        raise ValueError("Invalid nvidia-smi GPU telemetry")
    result = dict(zip(fields, values, strict=True))
    for key in fields[3:]:
        result[key] = float(result[key])
    return result


def check_gpu_idle(uuid, *, gpu=gpu_snapshot, compute_apps=None):
    snapshot = gpu(uuid)
    apps = (_smi("--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits")
            if compute_apps is None else compute_apps())
    if any(len(parts := [part.strip() for part in row.split(",")]) >= 2 and parts[1] == uuid
           for row in apps.splitlines() if row.strip()):
        raise RuntimeError("Selected GPU has an existing compute process")
    return snapshot


def _progress(path):
    try:
        report = json.loads(path.read_text())
        updates = report.get("updates", [])
        completed = updates.get("effective", 0) if isinstance(updates, dict) else len(updates)
        return {"stage": report.get("stage"), "step": report.get("step"),
                "completed_steps": completed}
    except (OSError, ValueError, TypeError):
        return {"stage": None, "step": None, "completed_steps": 0}


def _rss_tree(pid):
    try:
        parent = psutil.Process(pid)
        return sum(item.memory_info().rss for item in [parent, *parent.children(recursive=True)])
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def run_monitored(argv, directory, gpu_uuid, timeout, *, resources=host_memory,
                  gpu=gpu_snapshot, env=None, interval=1.0):
    """Require pre-created attempt directory; never touch unrelated processes."""
    directory = Path(directory)
    if not directory.is_dir() or (directory / "worker.log").exists():
        raise FileExistsError(directory / "worker.log")
    if timeout <= 0 or interval <= 0 or not isinstance(argv, list):
        raise ValueError("Invalid monitored worker request")
    result_path = directory / "result.json"
    started = time.monotonic()
    peak_rss = peak_gpu = 0
    min_host_available = None
    reason = None
    process = None
    samples = 0
    telemetry_failures = consecutive_failures = 0
    last_valid_gpu = started
    environment = {**os.environ, **(env or {}), "CUDA_VISIBLE_DEVICES": gpu_uuid}
    with _cancellation_signals() as cancelled:
        try:
            with (directory / "worker.log").open("xb") as log, (directory / "telemetry.jsonl").open("x") as stream:
                before = resources()
                if before.available <= before.reserve:
                    reason = "host_limit"
                else:
                    try:
                        gpu(gpu_uuid)
                        last_valid_gpu = time.monotonic()
                    except Exception as exc:
                        reason = "telemetry_error"
                        log.write(f"GPU telemetry failed before launch: {exc}\n".encode())
                    if reason is None:
                        process = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                                   env=environment, start_new_session=os.name == "posix")
                while process is not None:
                    host = resources()
                    min_host_available = min(host.available, min_host_available or host.available)
                    telemetry_error = None
                    try:
                        gpu_data = gpu(gpu_uuid)
                        last_valid_gpu = time.monotonic()
                        consecutive_failures = 0
                    except Exception as exc:
                        gpu_data = None
                        telemetry_error = f"{type(exc).__name__}: {exc}"
                        telemetry_failures += 1
                        consecutive_failures += 1
                        log.write(f"GPU telemetry failed: {exc}\n".encode())
                        log.flush()
                    rss = _rss_tree(process.pid)
                    peak_rss = max(peak_rss, rss)
                    if gpu_data is not None:
                        peak_gpu = max(peak_gpu, gpu_data["memory.used"])
                    age = time.monotonic() - last_valid_gpu
                    sample = {"time": time.time(), "pid": process.pid, "rss_tree_bytes": rss,
                              "host": {**asdict(host), "reserve": host.reserve}, "gpu": gpu_data,
                              "gpu_telemetry_error": telemetry_error,
                              "last_valid_gpu_age_seconds": age, **_progress(result_path)}
                    stream.write(json.dumps(sample, allow_nan=False) + "\n")
                    stream.flush()
                    samples += 1
                    if cancelled:
                        reason = "cancelled"
                    elif host.available <= host.reserve:
                        reason = "host_limit"
                    elif time.monotonic() - started >= timeout:
                        reason = "timeout"
                    elif consecutive_failures >= 3 or age >= 10:
                        reason = "telemetry_error"
                    if reason:
                        stop_process(process)
                        break
                    if process.poll() is not None:
                        break
                    time.sleep(interval)
        finally:
            if process is not None:
                stop_process(process)
    try:
        result = json.loads(result_path.read_text())
        if not isinstance(result, dict):
            raise ValueError("Worker result is not an object")
    except (OSError, ValueError) as exc:
        result = {"status": "error", "error": f"Missing/invalid worker result: {exc}"}
    code = process.returncode if process is not None else None
    if reason:
        result.update(status=reason, stop_reason=reason)
    elif result.get("status") == "ok" and code != 0:
        result.update(status="error", stop_reason="nonzero_worker_exit")
    elif result.get("status") == "ok" and not samples:
        result.update(status="error", stop_reason="missing_telemetry")
    elif result.get("status") not in {"ok", "cuda_oom", "error"}:
        result.update(status="error", stop_reason="incomplete_worker_result")
    result.update(returncode=code, monitor={"samples": samples, "peak_rss_tree_bytes": peak_rss,
                  "peak_gpu_used_mib": peak_gpu, "min_host_available_bytes": min_host_available,
                  "telemetry_failures": telemetry_failures,
                  "elapsed_seconds": time.monotonic() - started})
    write_result(directory / "supervisor.json", result)
    return result
