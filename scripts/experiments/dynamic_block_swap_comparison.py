"""Descriptive fixed/live comparison; sequential runs cannot prove causality."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import statistics

from library.training.auto_block_swap.process import write_result
from scripts.experiments.dynamic_block_swap_report import rows, summarize


def measured_updates(directory):
    updates = rows(directory / "updates.jsonl")
    paths = list(directory.glob("auto-block-swap/calibration-*/runtime.jsonl"))
    runtime = rows(paths[0]) if len(paths) == 1 else []
    shapes = {row["step"]: row["shapes"] for row in runtime if row["event"] == "update"}
    for row in updates:
        row.setdefault("shapes", shapes.get(row["step"]))
        if row["shapes"] is None:
            raise ValueError("Comparison requires measured shape metadata")
    return updates


def shape_summary(updates):
    samples = defaultdict(list)
    for row in updates:
        samples[str(row["shapes"])].append(row["seconds"])
    return {
        key: {"n": len(values), "median_seconds": statistics.median(values)}
        for key, values in samples.items()
    }


def compare(dynamic, fixed):
    if not all(summarize(path)["acceptance"]["passed"] for path in (dynamic, fixed)):
        raise ValueError("Comparison requires two accepted complete runs")
    dynamic_rows, fixed_rows = measured_updates(dynamic), measured_updates(fixed)
    paired = list(zip(dynamic_rows, fixed_rows))
    if any(a["step"] != b["step"] or a["shapes"] != b["shapes"] for a, b in paired):
        raise ValueError("Step/shape mismatch invalidates the loss comparison")
    differences = [abs(a["loss"] - b["loss"]) for a, b in paired]
    return {
        "evidence_kind": "sequential_descriptive_comparison_not_causal_speedup",
        "dynamic": dynamic.name,
        "fixed": fixed.name,
        "same_step_loss": {
            "updates": len(paired),
            "max_abs": max(differences),
            "mean_abs": statistics.mean(differences),
            "max_relative": max(
                delta / max(abs(b["loss"]), 1e-12)
                for delta, (_, b) in zip(differences, paired)
            ),
        },
        "fixed_after_update_8": shape_summary([r for r in fixed_rows if r["step"] > 8]),
        "dynamic_after_paired_verdict_66": shape_summary([
            r for r in dynamic_rows if r["step"] > 66
        ]),
        "causal_speedup_proven": False,
        "limits": [
            "Sequential runs are subject to thermal and temporal drift.",
            "Loss agreement is not a convergence or image-quality evaluation.",
            "Timings are synchronized whole-update wall times, not runtime CUDA-event times.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.dynamic, args.fixed)
    write_result(args.output, result)
    print(result["same_step_loss"])


if __name__ == "__main__":
    main()
