"""Audit recorded startup AUTO runs without treating rejected probes as passes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from library.training.auto_block_swap.coordinator import measurement
from library.training.auto_block_swap.process import write_result


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def summarize(root):
    calibration = read_json(root / "calibration/summary.json")
    acceptance = read_json(root / "acceptance.json")
    completion = read_json(root / "formal/completion.json")
    updates = read_json(root / "hot-updates.json").get("updates", [])
    errors = []
    selected = calibration.get("selected_blocks")
    if calibration.get("status") != "selected":
        errors.append("AUTO selection incomplete")
    else:
        try:
            if not measurement(calibration.get("confirmation", {}), selected).safe:
                errors.append("AUTO confirmation unsafe")
        except (RuntimeError, KeyError) as exc:
            errors.append(str(exc))
    if completion.get("status") != "completed" or acceptance.get("status") != "passed":
        errors.append("Formal acceptance incomplete or failed")
    steps = completion.get("steps", 0)
    if not updates or [r["step"] for r in updates] != list(range(1, steps + 1)):
        errors.append("Formal update sequence incomplete")
    if any(not math.isfinite(r["loss"]) or not math.isfinite(r["seconds"]) or r["seconds"] <= 0 for r in updates):
        errors.append("Nonfinite loss or invalid update timing")
    if any(r["blocks"] != selected for r in updates):
        errors.append("Formal swap count differs from selected count")
    host_reserve = (calibration.get("confirmation") or {}).get("host_reserve", 0)
    if any(r["host_available"] <= r.get("host_reserve", host_reserve) for r in updates):
        errors.append("Formal host reserve violated")
    swap_limit = acceptance.get("swap_io_limit_bytes", 64 * 1024**2)
    if swap_limit and any(r["swap_io"] > swap_limit for r in updates):
        errors.append("Formal paging budget violated")
    if any(r["headroom"] < acceptance.get("gpu_reserve", 0) for r in updates):
        errors.append("Formal GPU reserve violated")
    trials = []
    for trial in calibration.get("trials", []):
        row = {key: trial.get(key) for key in (
            "blocks", "status", "safe", "seconds", "headroom", "gpu_reserve",
            "cpu_master_bytes", "host_rss_peak", "host_min_available", "swap_io_bytes",
        )}
        try:
            row["accepted_resource_probe"] = measurement(trial, trial["blocks"]).safe
        except (RuntimeError, KeyError) as exc:
            row.update(accepted_resource_probe=False, rejection=str(exc))
        trials.append(row)
    return {
        "source": str(root), "valid": not errors, "errors": errors,
        "acceptance": acceptance, "completion": completion,
        "environment": read_json(root / "environment.json"),
        "fixture": read_json(root / "fixture.json"), "trials": trials,
        "confirmation": calibration.get("confirmation"),
        "reused_candidates_from": calibration.get("reused_candidates_from"),
        "failure": read_json(root / "failure.json"),
        "updates": updates,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    write_result(options.output, {"runs": [summarize(root) for root in options.runs]})


if __name__ == "__main__":
    main()
