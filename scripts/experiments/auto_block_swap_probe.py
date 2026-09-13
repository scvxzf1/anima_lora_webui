"""Bounded real-model AUTO acceptance; outputs stay in an explicit run directory."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from PIL import Image
import toml

from library.training.auto_block_swap.config import probe_arguments
from library.training.auto_block_swap.coordinator import (
    _validate_launch,
    measurement,
    run_calibration,
)
from library.training.auto_block_swap.policy import SwapSearch
from library.training.auto_block_swap.process import run_process, write_result


def build_fixture(root: Path) -> Path:
    """Copy one real image/cache pair per active token family, never edit sources."""
    images = Path("post_image_dataset/resized")
    cache = Path("post_image_dataset/lora")
    selected = {}
    for image in sorted(images.glob("*.png")):
        with Image.open(image) as content:
            width, height = content.size
        if width * height not in (4032 * 256, 4200 * 256):
            continue
        latent = cache / f"{image.stem}_{width:04d}x{height:04d}_anima.npz"
        text = cache / f"{image.stem}_krea2_te.safetensors"
        if latent.is_file() and text.is_file():
            selected.setdefault(width * height, (image, latent, text, width, height))
    if set(selected) != {4032 * 256, 4200 * 256}:
        raise ValueError("Two real cached token families are required")
    manifest = []
    for area, (image, latent, text, width, height) in sorted(selected.items()):
        for source, directory in (
            (image, "images"),
            (latent, "cache"),
            (text, "cache"),
        ):
            destination = root / directory / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        caption = image.with_suffix(".txt")
        if caption.is_file():
            shutil.copy2(caption, root / "images" / caption.name)
        manifest.append(
            {"source": str(image), "width": width, "height": height, "area": area}
        )
    write_result(root / "fixture.json", {"samples": manifest})
    dataset = {
        "general": {"caption_extension": ".txt", "keep_tokens": 0},
        "datasets": [
            {
                "resolution": 1024,
                "batch_size": 1,
                "enable_bucket": True,
                "min_bucket_reso": 672,
                "max_bucket_reso": 1536,
                "bucket_reso_steps": 64,
                "bucket_no_upscale": True,
                "subsets": [
                    {
                        "image_dir": str(root / "images"),
                        "cache_dir": str(root / "cache"),
                        "num_repeats": 1,
                        "caption_dropout_rate": 0.0,
                    }
                ],
            }
        ],
    }
    path = root / "dataset.toml"
    path.write_text(toml.dumps(dataset), encoding="utf-8")
    return path


def prepare_args(root: Path):
    from train import setup_parser
    from library.config.io import read_config_from_file

    root.mkdir(parents=True, exist_ok=True)
    path = root / "config.toml"
    if not path.exists():
        source = Path("output/runs/krea2-variant-300-20260904/configs/plain.toml")
        config = toml.load(source)
        config.update(
            {
                "dataset_config": str(build_fixture(root)),
                "output_dir": str(root / "formal"),
                "output_name": "auto-swap-acceptance",
                "logging_dir": str(root / "logs"),
                "log_with": "tensorboard",
                "auto_block_swap": True,
                "auto_block_swap_max_trials": 6,
                "auto_block_swap_timeout": 900,
                "base_compute": "nf4",
                "seed": 114,
                "max_train_steps": 128,
                "save_every_n_steps": 0,
                "save_every_n_epochs": 0,
                "checkpointing_epochs": 0,
                "config_snapshot": False,
                "max_data_loader_n_workers": 0,
                "persistent_data_loader_workers": False,
                "sample_prompts": "",
                "gradient_flow_probe_jsonl": "off",
                "memory_probe_jsonl": "off",
                "block_swap_profile_jsonl": "off",
            }
        )
        path.write_text(toml.dumps(config), encoding="utf-8")
    parser = setup_parser()
    argv = ["--config_file", str(path)]
    initial = parser.parse_args(argv)
    initial.config_snapshot = False
    args = read_config_from_file(initial, parser, argv=argv)
    args.config_snapshot = False
    _validate_launch(args)
    return args


def run_candidate(args, root, label, blocks, updates=3, inventory=False):
    destination = root / label
    probe = probe_arguments(args, destination, blocks=blocks, inventory=inventory)
    probe._auto_swap_probe["updates_per_case"] = updates
    result = run_process(probe, destination, timeout=args.auto_block_swap_timeout)
    print(json.dumps({"label": label, "blocks": blocks, **compact(result)}), flush=True)
    return result


def compact(result):
    values = {
        k: result[k]
        for k in (
            "status",
            "seconds",
            "headroom",
            "safe",
            "elapsed_seconds",
            "host_rss_peak",
            "host_min_available",
            "swap_io_bytes",
            "error",
            "error_type",
        )
        if k in result
    }
    updates = result.get("updates", [])
    if updates:
        values["updates"] = len(updates)
        values["peak_allocated"] = max(v["peak_allocated"] for v in updates)
        values["pooled_hot_update_median_seconds"] = statistics.median(
            v["seconds"] for v in updates if v["update"] >= 2
        )
    return values


def ablation_measurement(result, blocks, mode):
    measured = measurement(result, blocks)
    # The reserve is a decision threshold, not an allocation. Only remove
    # that threshold; host guards and real CUDA OOM remain failures.
    if mode == "no-reserve" and result["status"] == "ok":
        return replace(measured, safe=True)
    return measured


def ablation(args, root, mode, maximum, limit, reuse=None):
    search = SwapSearch(maximum, max_trials=limit)
    results = []
    previous = {}
    if reuse:
        source = root / reuse / "summary.json"
        previous = {
            row["blocks"]: row for row in json.loads(source.read_text())["trials"]
        }
    candidate = maximum
    started = time.monotonic()
    while candidate is not None and len(results) < limit:
        if candidate in previous:
            result = {**previous[candidate], "reused_from": str(source)}
            print(json.dumps({"reused_blocks": candidate, "mode": mode}), flush=True)
        else:
            result = run_candidate(
                args,
                root,
                f"{mode}/trial-{len(results):02d}-swap-{candidate}",
                candidate,
            )
        measured = ablation_measurement(result, candidate, mode)
        results.append({"blocks": candidate, **result})
        if mode == "fixed-stride":
            if not measured.safe or candidate == 0:
                break
            candidate -= 1
        else:
            search.observe(measured)
            if mode == "no-bisection" and not measured.safe:
                break
            candidate = search.next_candidate()
    successes = [
        r
        for r in results
        if r["status"] == "ok" and (r["safe"] or mode == "no-reserve")
    ]
    fastest = min(r["seconds"] for r in successes)
    selected = max(r["blocks"] for r in successes if r["seconds"] <= fastest * 1.03)
    confirm = run_candidate(args, root, f"{mode}/confirm-swap-{selected}", selected)
    report = {
        "mode": mode,
        "selected": selected,
        "trials": results,
        "confirmation": confirm,
        "elapsed_seconds": time.monotonic() - started,
        "measured_candidate_seconds": sum(row["elapsed_seconds"] for row in results),
        "reused_trials": sum("reused_from" in row for row in results),
        "confirmation_passed": ablation_measurement(confirm, selected, mode).safe,
    }
    (root / mode).mkdir(parents=True, exist_ok=True)
    write_result(root / mode / "summary.json", report)
    if not report["confirmation_passed"]:
        raise RuntimeError("Ablation confirmation failed")


def run_formal(args, root, label, reuse, steps):
    """Accept a production selection, never an intentionally unsafe ablation."""
    from safetensors import safe_open
    from train import AnimaTrainer

    if not reuse or not 1 <= steps <= 64:
        raise ValueError("Formal acceptance requires a selected report and 1..64 steps")
    source = root / reuse / "summary.json"
    report = json.loads(source.read_text())
    if report.get("status") != "selected" or not report["confirmation"].get("safe"):
        raise ValueError("Formal acceptance requires a confirmed AUTO selection")
    destination = root / label
    destination.mkdir(parents=True, exist_ok=False)
    args.output_dir = str(destination)
    args.logging_dir = str(destination / "logs")
    args.max_train_steps = steps
    args.max_train_epochs = None
    args.save_every_n_steps = None
    args.save_every_n_epochs = None
    args.save_state = False
    args.save_state_on_train_end = False
    args.resume = None
    args.checkpointing_epochs = 0
    args._auto_swap_resolved = True
    args.blocks_to_swap = report["selected_blocks"]
    args._auto_swap_report = str(source)
    started = time.monotonic()
    AnimaTrainer().train(args)
    checkpoints = list(destination.glob("*.safetensors"))
    if len(checkpoints) != 1:
        raise RuntimeError("Formal acceptance requires exactly one final checkpoint")
    with safe_open(checkpoints[0], framework="pt", device="cpu") as checkpoint:
        actual_steps = int((checkpoint.metadata() or {}).get("ss_steps", -1))
    if actual_steps != steps:
        raise RuntimeError(
            f"Formal acceptance completed {actual_steps}, expected {steps} updates"
        )
    write_result(
        destination / "completion.json",
        {
            "status": "completed",
            "steps": actual_steps,
            "blocks": args.blocks_to_swap,
            "selection": str(source),
            "elapsed_seconds": time.monotonic() - started,
            "checkpoints": [str(path.relative_to(destination)) for path in checkpoints],
        },
    )


def main():
    from library.training.train_bootstrap import install_stop_signal_handlers

    install_stop_signal_handlers()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--action",
        choices=["inventory", "probe", "calibrate", "ablation", "formal"],
        required=True,
    )
    parser.add_argument("--label", default="probe")
    parser.add_argument("--blocks", type=int, default=26)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument(
        "--mode", choices=["fixed-stride", "no-bisection", "no-reserve"]
    )
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument(
        "--reuse",
        help="Explicit same-scenario summary label for shared candidate measurements",
    )
    parser.add_argument(
        "--reserve-gib",
        type=float,
        default=0,
        help="Hold real VRAM in the coordinator to test competing allocations",
    )
    options = parser.parse_args()
    root = options.output.resolve()
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(root / "compiler-cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    args = prepare_args(root)
    import torch
    from library.training.auto_block_swap.resources import host_memory

    environment_path = root / "environment.json"
    if not environment_path.exists():
        properties = torch.cuda.get_device_properties(0)
        write_result(
            environment_path,
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": str(properties),
                "host": host_memory().to_dict(),
                "nvidia_smi": subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=name,uuid,memory.total,driver_version,temperature.gpu",
                        "--format=csv,noheader",
                    ],
                    text=True,
                ).strip(),
            },
        )
    pressure = None
    if options.reserve_gib:
        if not 0 < options.reserve_gib <= 10:
            raise ValueError("Coordinator VRAM reservation must be in (0,10] GiB")
        pressure = torch.empty(
            int(options.reserve_gib * 1024**3), dtype=torch.uint8, device="cuda"
        )
        pressure.zero_()
        torch.cuda.synchronize()
        print(f"Holding {options.reserve_gib} GiB of real coordinator VRAM", flush=True)
    if options.action == "calibrate":
        destination = root / options.label
        destination.mkdir(parents=True, exist_ok=False)
        args.auto_block_swap_max_trials = options.limit
        print("selected", run_calibration(args, destination), flush=True)
    elif options.action == "ablation":
        ablation(args, root, options.mode, options.blocks, options.limit, options.reuse)
    elif options.action == "formal":
        run_formal(args, root, options.label, options.reuse, options.updates)
    else:
        run_candidate(
            args,
            root,
            options.label,
            options.blocks,
            options.updates,
            inventory=options.action == "inventory",
        )
    del pressure


if __name__ == "__main__":
    main()
