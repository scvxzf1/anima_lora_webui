"""Continuous optimizer trajectories; complements independent one-update grid."""

import argparse
import os
from pathlib import Path
import torch
from safetensors.torch import load_file
from .inputs import load_input
from .model import build_model, drain
from .runner import DEFAULT_MODEL, step, save_named
from .manifest import (
    ROOT,
    GPU_UUID,
    validate_device,
    software_manifest,
    write_json,
    object_hash,
)


def save_or_link(path, tensors, reference):
    if reference is not None and reference.is_file():
        expected = load_file(str(reference))
        if set(expected) == set(tensors) and all(
            expected[n].dtype == value.dtype
            and expected[n].shape == value.shape
            and torch.equal(expected[n], value.detach().cpu())
            for n, value in tensors.items()
        ):
            os.link(reference, path)
            return "exact_reference_hardlink"
    save_named(path, tensors)
    return "independent_file"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--forward-policy", choices=["torch", "triton_both"], required=True
    )
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    device = validate_device(GPU_UUID)
    model, network, initial, metadata = build_model(
        Path(DEFAULT_MODEL), "w8a8", 12, 0, args.forward_policy, "fp32"
    )
    manifest = software_manifest()
    manifest.update(
        device=device,
        model=metadata,
        initial_hash=object_hash(metadata["initial_parameters"]),
        forward_policy=args.forward_policy,
        seeds=[0, 1, 2],
        steps_per_seed=20,
        timing_valid_for_ranking=False,
    )
    write_json(args.out_dir / "manifest.json", manifest)
    params = {n: p for n, p in network.named_parameters() if p.requires_grad}
    history = []
    for seed in [0, 1, 2]:
        drain(model)
        with torch.no_grad():
            for n, p in params.items():
                p.copy_(initial[n].to(p.device))
        optimizer = torch.optim.AdamW(list(params.values()), lr=2e-5, weight_decay=0.01)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        for index in range(20):
            folder = args.out_dir / f"seed{seed}-step{index + 1:02d}"
            folder.mkdir()
            batch, inputs = load_input(
                ROOT / "output/ab-anima-20261002/cache",
                ROOT / "output/ab-anima-20261002/images-1024",
                index % 2,
                seed * 100 + index,
                [0.1, 0.5, 0.9][index % 3],
            )
            before = {n: p.detach().cpu().clone() for n, p in params.items()}
            out, loss, finite = step(
                model,
                network,
                optimizer,
                {k: v.to("cuda") for k, v in batch.items()},
                12,
            )
            drain(model)
            grads = {n: p.grad for n, p in params.items() if p.grad is not None}
            summary = {
                "name": folder.name,
                "inputs": inputs,
                "loss": float(loss),
                "finite": finite.tolist(),
                "missing_gradients": sorted(set(params) - set(grads)),
                "optimizer_state_reset": index == 0,
                "adapter_reset": index == 0,
            }
            artifacts = {
                "grads": grads,
                "updates": {n: p.detach().cpu() - before[n] for n, p in params.items()},
                "output": {"output": out},
            }
            summary["storage"] = {
                kind: save_or_link(
                    folder / f"{kind}.safetensors",
                    values,
                    args.reference / folder.name / f"{kind}.safetensors"
                    if args.reference
                    else None,
                )
                for kind, values in artifacts.items()
            }
            write_json(folder / "sample.json", summary)
            if summary["missing_gradients"] or not all(summary["finite"]):
                raise RuntimeError(f"failed trajectory {folder.name}")
            history.append(summary)
            print(folder.name, flush=True)
        save_named(args.out_dir / f"seed{seed}-final.safetensors", params)
    write_json(
        args.out_dir / "result.json",
        {
            "status": "PASS",
            "count": len(history),
            "samples": history,
            "scope": "3 seeds x continuous20updates; fixed replay stream; parent B2 gradient quality vs B0 is separately gated",
        },
    )


if __name__ == "__main__":
    main()
