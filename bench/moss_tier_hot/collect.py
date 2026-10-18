"""Telemetry collection helpers for the MOSS tier hot test."""
from __future__ import annotations

import csv
import json
import subprocess
import threading
import time
from pathlib import Path

NVIDIA_QUERY = ("index,name,uuid,memory.used,memory.total,utilization.gpu,"
                "temperature.gpu,power.draw,clocks.sm")


def start_gpu_poller(csv_path: Path, uuid: str | None, interval_s: float = 1.0):
    """Poll the whole card once per second until stop() is called."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def loop():
        query = f"nvidia-smi --query-gpu={NVIDIA_QUERY} --format=csv,noheader,nounits"
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t", "index", "name", "uuid", "mem_used_mib", "mem_total_mib",
                             "util_pct", "temp_c", "power_w", "sm_mhz"])
            while not stop.is_set():
                try:
                    out = subprocess.run(query, shell=True, capture_output=True,
                                         text=True, timeout=10).stdout.strip()
                    for line in out.splitlines():
                        parts = [item.strip() for item in line.split(",")]
                        if len(parts) == len(NVIDIA_QUERY.split(",")) and (
                                uuid is None or parts[2] == uuid):
                            writer.writerow([round(time.time(), 3), *parts])
                except Exception:  # pragma: no cover - telemetry must never kill a run
                    pass
                handle.flush()
                stop.wait(interval_s)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    return stop, thread


def _numbers(row: dict) -> dict:
    out = {}
    for key, value in row.items():
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def summarize_gpu_csv(csv_path: Path) -> dict:
    if not csv_path.is_file():
        return {"samples": 0}
    rows = []
    with csv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(_numbers(row))
    if not rows:
        return {"samples": 0}
    return {
        "samples": len(rows),
        "mem_used_mib_peak": max(row.get("mem_used_mib", 0.0) for row in rows),
        "util_pct_peak": max(row.get("util_pct", 0.0) for row in rows),
        "util_pct_mean": sum(row.get("util_pct", 0.0) for row in rows) / len(rows),
        "temp_c_min": min(row.get("temp_c", 0.0) for row in rows),
        "temp_c_max": max(row.get("temp_c", 0.0) for row in rows),
        "power_w_max": max(row.get("power_w", 0.0) for row in rows),
        "sm_mhz_min": min(row.get("sm_mhz", 0.0) for row in rows),
        "sm_mhz_max": max(row.get("sm_mhz", 0.0) for row in rows),
    }


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def step_events(progress_rows: list[dict]) -> dict:
    """Per-step timestamps and allocator peaks from the trainer progress stream.

    The trainer emits ev=step rows with ts (seconds since run start) and the
    cuda/* memory counters, but does not write per-step seconds to JSONL (that
    number only exists in the tqdm line), so step duration is the ts delta.
    """
    start = None
    events = []
    for row in progress_rows:
        if row.get("ev") == "run_start" and start is None:
            try:
                start = float(row.get("ts"))
            except (TypeError, ValueError):
                start = None
        if row.get("ev") != "step":
            continue
        try:
            step = int(row.get("global_step"))
            ts = float(row.get("ts"))
        except (TypeError, ValueError):
            continue
        events.append({
            "step": step,
            "ts": ts,
            "peak_allocated_gb": row.get("cuda/max_memory_allocated_gb"),
            "peak_reserved_gb": row.get("cuda/max_memory_reserved_gb"),
        })
    events.sort(key=lambda item: item["step"])
    durations = []
    previous = start
    for event in events:
        if previous is not None and event["ts"] > previous:
            durations.append((event["step"], round(event["ts"] - previous, 6)))
        previous = event["ts"]
    return {"events": events, "durations": durations}


def peak_memory(memory_rows: list[dict]) -> dict:
    """Best-effort peak extraction from the trainer memory probe stream."""
    peaks: dict[str, float] = {}
    for row in memory_rows:
        for key, value in row.items():
            if not isinstance(value, (int, float)):
                continue
            low = key.lower()
            if "peak" not in low and "reserved" not in low:
                continue
            peaks[key] = max(peaks.get(key, 0.0), float(value))
    return peaks
