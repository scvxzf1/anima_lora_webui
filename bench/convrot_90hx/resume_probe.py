"""Deterministic replay resume contract; does not claim dataloader cursor support."""

import argparse
from pathlib import Path
import torch
from .model import build_model, drain
from .inputs import load_input
from .runner import DEFAULT_MODEL, step, save_named
from .manifest import ROOT, GPU_UUID, validate_device, write_json, software_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["bf16", "w8a16", "w8a8"], required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    device = validate_device(GPU_UUID)
    model, network, initial, metadata = build_model(
        Path(DEFAULT_MODEL), args.mode, 12, 0, flash_deterministic=True
    )
    params = [q for q in network.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-5, weight_decay=0.01)
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state["mode"] != args.mode:
            raise ValueError("resume precision mismatch")
        network.load_state_dict(state["network"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng_cpu"])
        torch.cuda.set_rng_state_all(state["rng_cuda"])
        start = state["step"]
    if not start < args.steps:
        raise ValueError("resume target must exceed saved step")
    history = []
    for index in range(start, args.steps):
        # Sample order and noise stream are an explicit function of step.
        batch, input_meta = load_input(
            ROOT / "output/ab-anima-20261002/cache",
            ROOT / "output/ab-anima-20261002/images-1024",
            index % 2,
            17 + index,
            [0.1, 0.5, 0.9][index % 3],
        )
        out, loss, finite = step(
            model, network, optimizer, {k: v.to("cuda") for k, v in batch.items()}, 12
        )
        drain(model)
        record = {
            "step": index + 1,
            "loss": float(loss),
            "finite": finite.tolist(),
            "input_hash": input_meta["input_hash"],
        }
        history.append(record)
        if not all(record["finite"]):
            raise RuntimeError("nonfinite replay")
    drain(model)
    torch.save(
        {
            "mode": args.mode,
            "step": args.steps,
            "network": network.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all(),
        },
        args.out_dir / "state.pt",
    )
    save_named(
        args.out_dir / "parameters.safetensors",
        {n: q for n, q in network.named_parameters() if q.requires_grad},
    )
    write_json(
        args.out_dir / "result.json",
        {
            "status": "PASS",
            "device": device,
            "manifest": software_manifest(),
            "model": metadata,
            "start_step": start,
            "end_step": args.steps,
            "effective_updates": args.steps - start,
            "steps": history,
            "scope": "exact replay cursor and deterministic Flash backward controlled here; production dataloader cursor not restored and default Flash remains unchanged",
        },
    )


if __name__ == "__main__":
    main()
