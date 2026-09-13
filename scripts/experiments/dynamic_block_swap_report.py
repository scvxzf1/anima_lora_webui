"""Summarize recorded live AUTO runs without starting training or loading weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from library.training.auto_block_swap.process import write_result


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def acceptance(request, completion, updates, events, runtime_count, pressure):
    request = request or {}
    dynamic = request.get("mode", "dynamic") == "dynamic"
    end = next((row for row in reversed(events) if row["event"] == "end"), None)
    expected_steps = request.get("steps", (completion or {}).get("steps"))
    failures = []
    if not completion or completion.get("status") != "completed":
        failures.append("checkpoint_completion_missing_or_failed")
    elif completion.get("finite") is not True:
        failures.append("checkpoint_not_verified_finite")
    if expected_steps is None or [row["step"] for row in updates] != list(
        range(1, expected_steps + 1)
    ):
        failures.append("incomplete_update_sequence")
    if completion and completion.get("steps") != expected_steps:
        failures.append("checkpoint_step_mismatch")
    if dynamic:
        if runtime_count != 1:
            failures.append("runtime_report_missing_or_ambiguous")
        if not end or end.get("completed") is not True:
            failures.append("runtime_not_completed")
        elif end.get("step") != expected_steps:
            failures.append("runtime_end_step_mismatch")
        runtime_steps = [row["step"] for row in events if row["event"] == "update"]
        if runtime_steps != [row["step"] for row in updates]:
            failures.append("runtime_update_sequence_mismatch")
    switches = [row for row in events if row["event"] == "switch"]
    held = next((row for row in pressure if row.get("held_bytes", 0) > 0), None)
    release = next(
        (row for row in pressure if held and row["step"] > held["step"]
         and row.get("held_bytes") == 0), None
    )
    pressure_adjusted = bool(held and release) and any(
        held["step"] <= row["step"] < release["step"]
        and row["reason"] == "gpu_pressure" for row in switches
    )
    recovery_explored = bool(release) and any(
        row["step"] > release["step"] and row["reason"] == "explore"
        for row in switches
    )
    pressure_cycle = pressure_adjusted and recovery_explored
    if request.get("pressure") and not pressure_cycle:
        failures.append("pressure_cycle_incomplete")
    return {
        "passed": not failures,
        "failures": failures,
        "pressure_adjusted": pressure_adjusted,
        "pressure_recovery_explored": recovery_explored,
        "pressure_cycle_completed": pressure_cycle,
        "causal_speedup_proven": False,
    }


def timing_segments(updates):
    segments = []
    current = None
    for row in updates:
        if current is None or row["blocks"] != current["blocks"]:
            current = {"blocks": row["blocks"], "start": row["step"], "samples": {}}
            segments.append(current)
        current["end"] = row["step"]
        key = str(row.get("shapes", "unrecorded"))
        current["samples"].setdefault(key, []).append(row["seconds"])
    for segment in segments:
        samples = segment.pop("samples")
        segment["sample_counts_by_shape"] = {key: len(v) for key, v in samples.items()}
        segment["hot_seconds_by_shape"] = {
            key: statistics.median(values[2:])
            for key, values in samples.items() if len(values) >= 4
        }
    return segments


def summarize(directory):
    updates = rows(directory / "updates.jsonl")
    runtime_paths = list(directory.glob("auto-block-swap/calibration-*/runtime.jsonl"))
    events = rows(runtime_paths[0]) if len(runtime_paths) == 1 else []
    runtime_updates = [row for row in events if row["event"] == "update"]
    switches = [row for row in events if row["event"] == "switch"]
    completion = load_json(directory / "completion.json")
    request = load_json(directory / "request.json")
    pressure = rows(directory / "pressure.jsonl")
    validation = acceptance(
        request, completion, updates, events, len(runtime_paths), pressure
    )
    return {
        "name": directory.name,
        "completion": completion,
        "acceptance": validation,
        "updates": len(updates),
        "observed_blocks": sorted({row["blocks"] for row in updates}),
        "peak_gib": max((row["peak_allocated"] for row in updates), default=0)
        / 1024**3,
        "switches": switches,
        "pressure": pressure,
        "pressure_recovery_explored": validation["pressure_recovery_explored"],
        "runtime_end": next(
            (row for row in reversed(events) if row["event"] == "end"), None
        ),
        "decisions": [
            row
            for index, row in enumerate(runtime_updates)
            if row.get("verdict")
            and (
                index == 0
                or row["verdict"] != runtime_updates[index - 1].get("verdict")
            )
        ],
        "compile_graph_counts": sorted(
            {row["dynamo_unique_graphs"] for row in updates}
        ),
        "segments": timing_segments(runtime_updates or updates),
        "mean_recorded_update_seconds": statistics.mean(
            row["seconds"] for row in updates
        )
        if updates
        else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    results = [
        summarize(directory)
        for directory in sorted(args.root.iterdir())
        if (directory / "request.json").exists()
    ]
    write_result(args.output, {"runs": results})
    for row in results:
        print(
            row["name"],
            row["updates"],
            row["observed_blocks"],
            "passed" if row["acceptance"]["passed"] else "not_accepted",
        )


if __name__ == "__main__":
    main()
