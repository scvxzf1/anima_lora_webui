"""Summarize saved real-GPU probes without loading models or starting training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from library.training.auto_block_swap.process import write_result

GIB = 1024**3


def summarize_updates(updates: list[dict]) -> dict:
    if not updates:
        return {"completed_updates": 0}
    cases = []
    for case in sorted({row["case"] for row in updates}):
        rows = [row for row in updates if row["case"] == case]
        warm = [row["seconds"] for row in rows if row["update"] > 0]
        hot = [row["seconds"] for row in rows if row["update"] >= 2]
        cases.append(
            {
                "case": case,
                "updates": len(rows),
                "first_seconds": rows[0]["seconds"],
                "warm_median_seconds": statistics.median(warm) if warm else None,
                "hot_median_seconds": statistics.median(hot) if hot else None,
                "early_hot_seconds": statistics.mean(hot[:3]) if hot else None,
                "last_hot_seconds": statistics.mean(hot[-3:]) if hot else None,
                "losses": [row["loss"] for row in rows],
            }
        )
    warm = [case["warm_median_seconds"] for case in cases]
    hot = [case["hot_median_seconds"] for case in cases]
    return {
        "completed_updates": len(updates),
        "warm_seconds": statistics.mean(warm)
        if all(v is not None for v in warm)
        else None,
        "hot_seconds": statistics.mean(hot)
        if all(v is not None for v in hot)
        else None,
        "peak_allocated_gib": max(row["peak_allocated"] for row in updates) / GIB,
        "peak_reserved_gib": max(row["peak_reserved"] for row in updates) / GIB,
        "cases": cases,
    }


def summarize_hardware(rows: list[dict], start: float, end: float) -> dict:
    # Active samples exclude model-loading idle gaps, not other GPU processes.
    active = [
        row["devices"][0]
        for row in rows
        if start <= row["time"] <= end
        and row["devices"]
        and float(row["devices"][0]["utilization.gpu"]) >= 90
    ]
    result = {"active_samples": len(active)}
    for field in ("temperature.gpu", "clocks.sm", "power.draw", "memory.used"):
        values = []
        for row in active:
            try:
                values.append(float(row[field]))
            except (KeyError, ValueError):
                continue
        if values:
            result[field] = {
                "min": min(values),
                "median": statistics.median(values),
                "max": max(values),
            }
    return result


def summarize_probe(path: Path, root: Path, hardware: list[dict]) -> dict | None:
    result = json.loads(path.read_text())
    # The worker writes first; the coordinator adds host telemetry after exit.
    if "elapsed_seconds" not in result:
        return None
    updates = result.get("updates", [])
    live = path.with_name("updates.json")
    if not updates and live.exists():
        updates = json.loads(live.read_text())["updates"]
    end = path.stat().st_mtime
    elapsed = result["elapsed_seconds"]
    record = {
        "source": str(path.relative_to(root)),
        "status": result["status"],
        "safe": result.get("safe"),
        "blocks": result.get("blocks"),
        "elapsed_seconds": elapsed,
        "headroom_gib": result.get("headroom", 0) / GIB,
        "host_min_available_gib": result["host_min_available"] / GIB,
        "process_tree_rss_upper_bound_gib": result["host_rss_peak"] / GIB,
        "swap_io_bytes": result["swap_io_bytes"],
        "cpu_master_gib": result.get("cpu_master_bytes", 0) / GIB,
        "shapes": result.get("cases", []),
        "hardware": summarize_hardware(hardware, end - elapsed, end),
        **summarize_updates(updates),
    }
    if result.get("error"):
        record["error"] = result["error"]
    return record


def summarize_search(path: Path, root: Path) -> dict:
    result = json.loads(path.read_text())
    trials = result["trials"]
    confirmation = result.get("confirmation", {})
    selected = result.get("selected_blocks", result.get("selected"))
    selected_trial = next((row for row in trials if row["blocks"] == selected), {})
    measured_seconds = selected_trial.get("seconds")
    confirmed_seconds = confirmation.get("seconds")
    return {
        "source": str(path.relative_to(root)),
        "selected": selected,
        "confirmation_timing_change_percent": (
            100 * (confirmed_seconds / measured_seconds - 1)
            if measured_seconds and confirmed_seconds
            else None
        ),
        "trials": [
            {
                "blocks": row["blocks"],
                "status": row["status"],
                "safe": row.get("safe"),
                "seconds": row.get("seconds"),
                "headroom_gib": row.get("headroom", 0) / GIB,
                "reused": "reused_from" in row,
            }
            for row in trials
        ],
        "candidate_count": len(trials),
        "fresh_candidates": sum("reused_from" not in row for row in trials),
        "logical_candidate_seconds": sum(row["elapsed_seconds"] for row in trials),
        "fresh_candidate_seconds": sum(
            row["elapsed_seconds"] for row in trials if "reused_from" not in row
        ),
        "inventory_seconds": result.get("inventory", {}).get("elapsed_seconds", 0),
        "confirmation_seconds": confirmation.get("elapsed_seconds"),
        "confirmation_status": confirmation.get("status"),
        "confirmation_safe": confirmation.get("safe"),
        "confirmation_passed_by_variant": result.get("confirmation_passed"),
    }


def compare_losses(first: dict, second: dict) -> dict:
    def shapes(record):
        return [
            (case["width"], case["height"], case["batch_size"])
            for case in record["shapes"]
        ]

    if shapes(first) != shapes(second):
        raise ValueError("Loss comparison requires identical ordered shapes")
    if [c["updates"] for c in first["cases"]] != [
        c["updates"] for c in second["cases"]
    ]:
        raise ValueError("Loss comparison requires identical update counts")
    a = [loss for case in first["cases"] for loss in case["losses"]]
    b = [loss for case in second["cases"] for loss in case["losses"]]
    differences = [abs(x - y) for x, y in zip(a, b, strict=True)]
    return {
        "first": first["source"],
        "second": second["source"],
        "updates": len(differences),
        "max_absolute_loss_difference": max(differences),
        "mean_absolute_loss_difference": statistics.mean(differences),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    root = options.root
    hardware = [
        json.loads(line) for line in (root / "hardware.jsonl").read_text().splitlines()
    ]
    probes = [
        record
        for path in sorted(root.rglob("result.json"))
        if (record := summarize_probe(path, root, hardware)) is not None
    ]
    searches = [
        summarize_search(path, root) for path in sorted(root.glob("*/summary.json"))
    ]
    by_source = {probe["source"]: probe for probe in probes}
    comparisons = []
    baseline = by_source.get("hot-swap-26/result.json")
    if baseline:
        for source, candidate in by_source.items():
            if source.startswith("hot-swap-") and candidate is not baseline:
                comparisons.append(compare_losses(baseline, candidate))
    formal = []
    for path in sorted(root.glob("*/completion.json")):
        result = json.loads(path.read_text())
        checks = path.with_name("checkpoint_checks.json")
        formal.append(
            {
                "source": str(path.relative_to(root)),
                "checkpoint_checks": json.loads(checks.read_text())
                if checks.exists()
                else None,
                **{
                    key: result[key]
                    for key in (
                        "status",
                        "steps",
                        "blocks",
                        "elapsed_seconds",
                        "checkpoints",
                    )
                },
            }
        )
    report = {
        "environment": json.loads((root / "environment.json").read_text()),
        "config_sha256": hashlib.sha256(
            (root / "config.toml").read_bytes()
        ).hexdigest(),
        "probes": probes,
        "searches": searches,
        "loss_comparisons": comparisons,
        "formal": formal,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    write_result(options.output, report)
    print(
        json.dumps(
            {
                "probes": len(probes),
                "searches": len(searches),
                "output": str(options.output),
            }
        )
    )


if __name__ == "__main__":
    main()
