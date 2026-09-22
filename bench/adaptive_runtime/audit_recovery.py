"""Audit completed fixed-input probe recovery artifacts, not production state."""

import argparse
import json
from pathlib import Path

from safetensors.torch import load_file
import torch

from bench.adaptive_runtime.checkpoint import read_checkpoint
from library.training.auto_block_swap.process import write_result


def audit(directory):
    root = Path(directory).resolve()
    summary = json.loads((root / "summary.json").read_text())
    attempts = summary.get("attempts")
    if summary.get("status") != "ok" or not isinstance(attempts, list) or not attempts:
        raise ValueError("Successful completed recovery required")
    if [a.get("attempt") for a in attempts] != list(range(len(attempts))):
        raise ValueError("Noncontiguous recovery attempts")
    previous_checkpoint, resumed = None, False
    resumed_signatures = []
    rows = []
    for index, attempt in enumerate(attempts):
        folder = root / f"attempt-{index:03d}"
        result = json.loads((folder / "result.json").read_text())
        supervisor = json.loads((folder / "supervisor.json").read_text())
        if attempt.get("result") != supervisor or any(supervisor.get(k) != v for k, v in result.items()):
            raise ValueError("Worker/supervisor/history mismatch")
        if (result.get("precision") != summary.get("precision")
                or result.get("loss_scale", 1.0) != summary.get("loss_scale", 1.0)
                or result.get("swap") != attempt["plan"]["blocks_to_swap"]):
            raise ValueError("Recovery precision/scale/memory plan changed")
        if index and result.get("fp32_modules") != rows[0]["fp32_modules"]:
            raise ValueError("Recovery resolved precision profile changed")
        if previous_checkpoint:
            if Path(result.get("resumed_from", "")).resolve() != previous_checkpoint:
                raise ValueError("Committed checkpoint not used for resume")
            source = torch.load(previous_checkpoint, map_location="cpu", weights_only=True)
            source = read_checkpoint(previous_checkpoint, signature=source["signature"])
            resumed_signatures.append(source["signature"])
            progress = result.get("updates", [])
            if progress[:source["step"]] != source["updates"]:
                raise ValueError("Resumed worker did not preserve committed update history")
            if result.get("status") == "ok" and len(progress) <= source["step"]:
                raise ValueError("Resumed worker made no new committed progress")
            resumed = True
        expected = "ok" if index == len(attempts) - 1 else "cuda_oom"
        if result.get("status") != expected:
            raise ValueError("Unexpected recovery worker status")
        if expected == "cuda_oom" and result.get("optimizer_started"):
            path = result.get("committed_checkpoint")
            if not path or not Path(path).resolve().is_relative_to(folder):
                raise ValueError("Missing owned committed checkpoint")
            previous_checkpoint = Path(path).resolve()
        rows.append({"attempt": index, "swap": result["swap"], "status": expected,
                     "optimizer_started": result.get("optimizer_started"),
                     "fp32_modules": result.get("fp32_modules"),
                     "elapsed_seconds": supervisor["elapsed_seconds"]})
    checkpoint = Path(result["committed_checkpoint"]).resolve()
    if not checkpoint.is_relative_to(folder):
        raise ValueError("Final checkpoint is outside final worker directory")
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload = read_checkpoint(checkpoint, signature=raw["signature"])
    if any(signature != payload["signature"] for signature in resumed_signatures):
        raise ValueError("Resumed checkpoint training signature changed")
    if (payload["updates"] != result["updates"]
            or payload["signature"].get("precision") != result["precision"]
            or payload["signature"].get("fp32_modules") != result["fp32_modules"]
            or payload["signature"].get("inputs_sha256") != result["inputs_sha256"]
            or payload["signature"].get("loss_scale", 1.0) != result.get("loss_scale", 1.0)):
        raise ValueError("Final checkpoint/worker contract mismatch")
    output = load_file(str(folder / "result.safetensors"))
    if set(output) != set(payload["network"]) or not all(
            torch.equal(value, payload["network"][name]) for name, value in output.items()):
        raise ValueError("Final checkpoint and adapter output differ")
    states = payload["optimizer"]["state"]
    if not states or any(float(s.get("step", -1)) != payload["step"] for s in states.values()):
        raise ValueError("Optimizer progress differs from committed step")
    return {"status": "audited", "production_ready": False, "attempts": rows,
            "scope": "fixed_input_probe_artifact_consistency_not_trajectory_equivalence",
            "schema": payload["schema"], "step": payload["step"], "scaler": payload.get("scaler"),
            "optimizer_states": len(states), "checkpoint_output_exact": True,
            "checkpoint_resume_observed": resumed,
            "peak_allocated_final_worker": max(u["peak_allocated"] for u in result["updates"]
                                               if u["swap"] == result["swap"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    report = audit(args.directory)
    write_result(args.output, report)
    print(json.dumps({k: report[k] for k in ("status", "step", "checkpoint_resume_observed")}, indent=2))


if __name__ == "__main__":
    main()
