"""Audit a successful plain-AdamW training smoke, not resume or quality safety."""

import argparse
from collections import Counter
import json
from pathlib import Path

from safetensors.torch import load_file
import torch

from library.training.auto_block_swap.process import write_result


def audit_training(directory, *, output_name="adaptive-smoke"):
    directory = Path(directory).resolve()
    summary = json.loads((directory / ".adaptive-recovery/summary.json").read_text())
    if summary["status"] != "ok":
        raise ValueError("Expected successful actual training")
    result = summary["attempts"][-1]["result"]
    steps = result["completed_steps"]
    if type(steps) is not int or steps <= 0 or result["status"] != "ok":
        raise ValueError("Missing successful optimizer progress")
    state = Path(result["saved_state"]).resolve()
    if not state.is_relative_to(directory / ".adaptive-recovery"):
        raise ValueError("Snapshot must belong to this run")
    manifest = json.loads((state / "snapshot.json").read_text())
    if manifest["global_step"] != steps or manifest["data_cursor_resume_supported"]:
        raise ValueError("Unexpected snapshot contract")
    for name, size in manifest["files"].items():
        if Path(name).name != name or (state / name).stat().st_size != size:
            raise ValueError("Snapshot inventory mismatch")
    snapshot = load_file(str(state / "model.safetensors"))
    output = load_file(str(directory / f"{output_name}.safetensors"))
    if not snapshot or snapshot.keys() != output.keys():
        raise ValueError("Final adapter keys differ from snapshot")
    for name, value in snapshot.items():
        if not torch.isfinite(value).all() or not torch.equal(value, output[name]):
            raise ValueError(f"Final adapter differs or is nonfinite: {name}")
    optimizer = torch.load(state / "optimizer.bin", map_location="cpu", weights_only=True)
    optimizer_steps = []
    for value in optimizer["state"].values():
        if "exp_avg" not in value or "exp_avg_sq" not in value:
            raise ValueError("Audit supports plain AdamW state only")
        if any(not torch.isfinite(t).all() for t in value.values() if isinstance(t, torch.Tensor)):
            raise ValueError("Nonfinite optimizer state")
        optimizer_steps.append(float(value["step"]))
    if not optimizer_steps or set(optimizer_steps) != {float(steps)}:
        raise ValueError("AdamW progress mismatch")
    scaler = torch.load(state / "scaler.pt", weights_only=True)
    scheduler = torch.load(state / "scheduler.bin", weights_only=True)
    if scaler["_growth_tracker"] != steps % scaler["growth_interval"] or scheduler["last_epoch"] != steps:
        raise ValueError("Scaler/scheduler progress mismatch")
    precision = json.loads((state / "adaptive_precision.json").read_text())
    memory = [json.loads(line) for line in (directory / "memory.jsonl").read_text().splitlines()]
    peak = max(row.get("cuda_max_allocated_gb", 0) for row in memory)
    return {"status": "ok", "scope": "actual_training_smoke_state_audit_not_quality_or_resume",
            "completed_steps": steps, "adapter_tensors": len(snapshot),
            "optimizer_states": len(optimizer_steps), "adapter_snapshot_exact": True,
            "loss_scale": scaler["scale"], "scaler_growth_tracker": scaler["_growth_tracker"],
            "scheduler_last_epoch": scheduler["last_epoch"],
            "precision_counts": dict(Counter(precision["assignments"].values())),
            "peak_allocated_gib": peak, "worker_seconds": result["elapsed_seconds"],
            "data_cursor_resume_supported": False, "precision_calibrated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = audit_training(args.directory)
    write_result(args.directory / "training-audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
