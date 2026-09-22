"""Checkpoint identity for the fixed-input Z-Image probe, not production state."""

from pathlib import Path

import torch

from bench.adaptive_runtime.checkpoint import read_checkpoint, restore_checkpoint
from library.models.z_image.weights import CONFIG_ROOT
from library.training.auto_block_swap.process import write_result


def weight_inventory(path):
    root = Path(path).expanduser().resolve()
    if root.is_file():
        files = [root, CONFIG_ROOT / "transformer" / "config.json"]
    else:
        component = root / "transformer" if (root / "transformer" / "config.json").is_file() else root
        files = sorted(p for p in component.iterdir() if p.is_file()
                       and p.suffix in {".json", ".safetensors", ".bin"})
        if not any(p.suffix in {".safetensors", ".bin"} for p in files):
            raise ValueError("Checkpoint identity requires local transformer weights")
    return [{"path": str(p.resolve()), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in files]


def signature_for(args, report):
    if not args.inputs:
        raise ValueError("Checkpoint recovery requires immutable real cached inputs")
    return {"model_family": "z_image", "inputs_sha256": report["inputs_sha256"],
            "weights": weight_inventory(args.weights), "precision": args.precision,
            "fp32_modules": report["fp32_modules"], "steps": args.steps,
            "resolution": args.resolution, "rank": 4, "alpha": 4, "learning_rate": 1e-4,
            "scheduler": None, "torch": str(torch.__version__)}


def resume_if_requested(args, report, signature, network, optimizer, params, initial, *, scaler=None):
    if not args.resume:
        return initial
    payload = read_checkpoint(args.resume, signature=signature)
    if payload["step"] > args.steps:
        raise ValueError("Checkpoint is beyond the requested training length")
    report.update(stage="resume", optimizer_started=True,
                  committed_checkpoint=str(args.resume.resolve()), resumed_from=str(args.resume))
    write_result(args.output, report)
    initial, report["updates"] = restore_checkpoint(
        payload, network=network, optimizer=optimizer, params=params, device="cuda", scaler=scaler,
    )
    return initial
