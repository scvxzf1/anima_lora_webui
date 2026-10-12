"""Independent paired full-model gradient/update samples, never throughput data."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import torch

from .assessment import EXPECTED_SAMPLES
from .inputs import load_input
from .manifest import (
    GPU_UUID,
    ROOT,
    file_hash,
    object_hash,
    software_manifest,
    validate_device,
    write_json,
)
from .model import build_model, drain, placement
from .runner import DEFAULT_MODEL, save_named, step
from .telemetry import Telemetry


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["bf16", "w8a16", "w8a8"], required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL))
    parser.add_argument(
        "--cache-dir", type=Path, default=ROOT / "output/ab-anima-20261002/cache"
    )
    parser.add_argument(
        "--image-dir", type=Path, default=ROOT / "output/ab-anima-20261002/images-1024"
    )
    parser.add_argument("--swap", type=int, default=12)
    parser.add_argument("--init-seed", type=int, default=0)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--cache-indices", default="0,1")
    parser.add_argument("--sigmas", default="0.1,0.5,0.9")
    parser.add_argument("--flash-deterministic", action="store_true")
    parser.add_argument(
        "--forward-policy",
        choices=["torch", "triton_quant", "triton_gemm", "triton_both"],
        default="torch",
    )
    parser.add_argument(
        "--backward-policy", choices=["fp32", "bf16", "int8"], default="fp32"
    )
    return parser.parse_args()


def run_sample(args, model, network, initial, cache_index, seed, sigma):
    name = f"cache{cache_index}-seed{seed}-sigma{sigma:g}"
    folder = args.out_dir / name
    folder.mkdir()
    inputs, metadata = load_input(
        args.cache_dir, args.image_dir, cache_index, seed, sigma
    )
    params = {n: p for n, p in network.named_parameters() if p.requires_grad}
    drain(model)
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(initial[n].to(p.device))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    optimizer = torch.optim.AdamW(list(params.values()), lr=2e-5, weight_decay=0.01)
    batch = {k: v.to("cuda:0") for k, v in inputs.items()}
    output, loss, finite = step(model, network, optimizer, batch, args.swap)
    drain(model)
    gradients = {n: p.grad for n, p in params.items() if p.grad is not None}
    save_named(folder / "grads.safetensors", gradients)
    save_named(
        folder / "updates.safetensors",
        {n: p.detach().cpu() - initial[n] for n, p in params.items()},
    )
    save_named(folder / "output.safetensors", {"output": output})
    save_named(folder / "input.safetensors", inputs)
    checks = finite.cpu().tolist()
    summary = {
        "name": name,
        "loss": float(loss),
        "finite": checks,
        "inputs": metadata,
        "missing_gradients": sorted(set(params) - set(gradients)),
        "optimizer_state_reset": True,
        "adapter_reset": True,
        "rng_reset": True,
        "scheduler": "constant/no scheduler state",
    }
    write_json(folder / "sample.json", summary)
    if not all(checks) or not math.isfinite(summary["loss"]):
        raise RuntimeError(f"nonfinite sample {name}")
    if summary["missing_gradients"]:
        raise RuntimeError(
            f"missing gradients in sample {name}: {summary['missing_gradients']}"
        )
    return summary


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    telemetry = None
    started = time.time()
    try:
        device = validate_device(GPU_UUID)
        manifest = software_manifest()
        manifest.update(
            device=device,
            model_path=str(args.model_path),
            model_sha256=file_hash(args.model_path),
            initial_seed=args.init_seed,
            mode=args.mode,
            forward_policy=args.forward_policy,
            backward_policy=args.backward_policy,
            swap=args.swap,
            timing_valid_for_ranking=False,
            expected_samples=[
                f"cache{cache}-seed{seed}-sigma{sigma:g}"
                for cache in map(int, args.cache_indices.split(","))
                for seed in map(int, args.seeds.split(","))
                for sigma in map(float, args.sigmas.split(","))
            ],
        )
        telemetry = Telemetry(args.out_dir / "telemetry.jsonl", GPU_UUID)
        telemetry.start()
        model, network, initial, model_metadata = build_model(
            args.model_path,
            args.mode,
            args.swap,
            args.init_seed,
            args.forward_policy,
            args.backward_policy,
            args.flash_deterministic,
        )
        manifest["model"] = model_metadata
        manifest["initial_hash"] = object_hash(model_metadata["initial_parameters"])
        write_json(args.out_dir / "manifest.json", manifest)
        save_named(args.out_dir / "initial.safetensors", initial)
        samples = []
        for cache_index in map(int, args.cache_indices.split(",")):
            for seed in map(int, args.seeds.split(",")):
                for sigma in map(float, args.sigmas.split(",")):
                    samples.append(
                        run_sample(
                            args, model, network, initial, cache_index, seed, sigma
                        )
                    )
                    write_json(
                        args.out_dir / "progress.json",
                        {
                            "status": "RUNNING",
                            "completed": len(samples),
                            "samples": samples,
                        },
                    )
                    print(samples[-1]["name"], flush=True)
        result = {
            "status": "PASS"
            if (
                len(samples) == len(EXPECTED_SAMPLES)
                and {sample["name"] for sample in samples} == set(EXPECTED_SAMPLES)
            )
            else "INCOMPLETE",
            "samples": samples,
            "count": len(samples),
            "seconds": time.time() - started,
            "placement_after_samples": placement(model, network),
        }
        write_json(args.out_dir / "result.json", result)
    except Exception as exc:
        import traceback

        write_json(
            args.out_dir / "failure.json",
            {"status": "FAIL", "error": str(exc), "traceback": traceback.format_exc()},
        )
        write_json(args.out_dir / "result.json", {"status": "FAIL", "error": str(exc)})
        raise
    finally:
        if telemetry:
            telemetry.stop()


if __name__ == "__main__":
    main()
