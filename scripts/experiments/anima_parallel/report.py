"""Summarize completed PP experiments and compare exported adapter tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch
from safetensors.torch import load_file

from .common import write_json


def compare_adapters(left: Path, right: Path):
    first, second = load_file(left), load_file(right)
    if set(first) != set(second):
        raise ValueError("adapter key sets differ")
    square_error = square_reference = 0.0
    max_abs = 0.0
    changed = count = 0
    for key, value in first.items():
        other = second[key]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError(f"adapter tensor contract differs: {key}")
        difference = value.double() - other.double()
        max_abs = max(max_abs, float(difference.abs().max()))
        square_error += float(difference.square().sum())
        square_reference += float(value.double().square().sum())
        changed += int(torch.count_nonzero(difference))
        count += value.numel()
    return {"left": str(left), "right": str(right), "keys": len(first),
            "max_abs": max_abs, "rel_l2": (square_error / max(square_reference, 1e-60)) ** 0.5,
            "changed_elements": changed, "total_elements": count,
            "exact": changed == 0}


def summarize(path):
    data = json.loads(path.read_text())
    ranks = data["ranks"]
    records = ranks[0]["steps"]
    config = data["config"]
    return {
        "name": path.parent.name, "mode": data["mode"],
        "runtime_version": data.get("runtime_version", 1),
        "microbatches": data["effective_batch"], "schedule": data["schedule"],
        "split": config["split"] if data["mode"] == "pp" else None,
        "checkpoint": config["checkpoint"], "steps": len(records),
        "seconds": data["seconds_per_optimizer_step"],
        "median_seconds": statistics.median(record["seconds"] for record in records),
        "min_seconds": min(record["seconds"] for record in records),
        "max_seconds": max(record["seconds"] for record in records),
        "samples_per_second": data["samples_per_second"],
        "peak_gib": [rank["peak_allocated_bytes"] / (1 << 30) for rank in ranks],
        "loss_first": records[0]["loss"], "loss_last": records[-1]["loss"],
        "communication_per_step": {
            key: value / len(records)
            for key, value in ranks[0]["communication_measured_steps"].items()
        },
        "source": str(path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--compare", type=Path, nargs=2)
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows = [summarize(path) for path in sorted(args.root.glob("*/result.json"))]
    comparison = compare_adapters(*args.compare) if args.compare else None
    write_json(args.root / "summary.json", {"cases": rows, "adapter_comparison": comparison})
    lines = ["# Anima PP Experiment Summary", "",
             "Only completed results are included. Hardware and runtime versions must be matched before claiming speedup.", "",
             "| Case | Runtime | M | Split | Checkpoint | s/update | samples/s | Peak GiB/rank |",
             "|---|---:|---:|---|---|---:|---:|---|"]
    for row in rows:
        peak = "/".join(f"{value:.2f}" for value in row["peak_gib"])
        lines.append(f"| {row['name']} | {row['runtime_version']} | {row['microbatches']} | {row['split']} | {row['checkpoint']} | {row['seconds']:.3f} | {row['samples_per_second']:.3f} | {peak} |")
    if comparison:
        lines.extend(["", "## Adapter Comparison", "", "```json", json.dumps(comparison, indent=2), "```"])
    (args.root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
