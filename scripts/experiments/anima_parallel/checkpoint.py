"""Stage-local experiment checkpoints; never accept a mismatched topology."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def save_stage_checkpoint(
    path: Path, module, optimizer, *, step: int, contract: dict
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "topology": module.topology(),
        "contract": contract,
        "step": step,
        "adapters": module.adapter_state(),
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "cuda_rng": torch.cuda.get_rng_state(next(module.parameters()).device)
        if next(module.parameters()).is_cuda
        else None,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_stage_checkpoint(path: Path, module, optimizer, *, contract: dict) -> int:
    # These are local optimizer/RNG files produced by this experiment, not downloaded weights.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["topology"] != module.topology() or state["contract"] != contract:
        raise ValueError("PP checkpoint topology or experiment contract mismatch")
    module.adapters.load_state_dict(state["adapters"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state(state["cuda_rng"], next(module.parameters()).device)
    return int(state["step"])


def load_distributed_checkpoint(directory, module, optimizer, *, contract, group):
    """All ranks agree on validity before any rank enters pipeline execution."""
    outcome = {"step": None, "error": None}
    try:
        if not (directory / "result.json").is_file():
            raise ValueError("resume directory has no completed experiment result")
        outcome["step"] = load_stage_checkpoint(
            directory / f"stage-rank{module.rank}.pt",
            module,
            optimizer,
            contract=contract,
        )
    except Exception as error:
        outcome["error"] = f"rank {module.rank}: {type(error).__name__}: {error}"
    outcomes = [None] * dist.get_world_size(group)
    dist.all_gather_object(outcomes, outcome, group=group)
    errors = [item["error"] for item in outcomes if item["error"]]
    if errors:
        raise ValueError("PP resume rejected: " + "; ".join(errors))
    steps = {item["step"] for item in outcomes}
    if len(steps) != 1:
        raise ValueError(f"PP checkpoint step mismatch: {outcomes}")
    return int(outcome["step"])


def experiment_contract(args):
    def fingerprint(path):
        stat = path.stat()
        return {
            "path": str(path.resolve()),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }

    return {
        "schema": 1,
        "runtime_version": args.runtime_version,
        "base": fingerprint(args.dit_path),
        "latent": fingerprint(args.latent_path),
        "text": fingerprint(args.text_path),
        "microbatches": args.microbatches,
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "rank_dim": args.rank_dim,
        "alpha": args.alpha,
        "lr": args.lr,
        "attn_mode": args.attn_mode,
        "deterministic_attention": args.deterministic_attention,
        "warmup_steps": args.warmup_steps,
    }
