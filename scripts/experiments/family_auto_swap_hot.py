"""Bounded Anima/Z-Image AUTO hot acceptance with isolated real cache copies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

from PIL import Image
import toml

from library.training.auto_block_swap.process import write_result


def prepare_fixture(options):
    root = options.output
    root.mkdir(parents=True, exist_ok=False)
    images, cache = root / "images", root / "cache"
    images.mkdir()
    cache.mkdir()
    image = next(iter(sorted(options.images.glob("*.png"))))
    with Image.open(image) as content:
        width, height = content.size
    suffix = options.family
    sources = [
        options.cache / f"{image.stem}_{width:04d}x{height:04d}_{suffix}.npz",
        options.cache / f"{image.stem}_{suffix}_te.safetensors",
    ]
    for source in [image, image.with_suffix(".txt"), *sources]:
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, (cache if source in sources else images) / source.name)
    dataset = {
        "general": {"caption_extension": ".txt", "keep_tokens": 0},
        "datasets": [{
            "resolution": 1024, "batch_size": 1, "enable_bucket": True,
            "min_bucket_reso": 672, "max_bucket_reso": 1536,
            "bucket_reso_steps": 64, "bucket_no_upscale": True,
            "subsets": [{"image_dir": str(images), "cache_dir": str(cache),
                         "num_repeats": 1, "caption_dropout_rate": 0.0}],
        }],
    }
    (root / "dataset.toml").write_text(toml.dumps(dataset), encoding="utf-8")
    config = toml.load(options.config)
    config.update({
        "model_family": options.family, "pretrained_model_name_or_path": str(options.dit),
        "qwen3": str(options.qwen3), "vae": str(options.vae),
        "dataset_config": str(root / "dataset.toml"), "seed": 114,
        "output_dir": str(root / "formal"), "output_name": "family-auto-hot",
        "logging_dir": None, "log_with": None,
        "base_compute": "bf16", "mixed_precision": "bf16", "attn_mode": "flash",
        "gradient_checkpointing": True, "torch_compile": False,
        "selective_checkpoint": "off", "auto_block_swap": True,
        "auto_block_swap_mode": "startup", "auto_block_swap_max_trials": options.limit,
        "auto_block_swap_timeout": 300,
        "auto_block_swap_vram_reserve_percent": options.reserve_percent,
        "auto_block_swap_preference": options.preference,
        "max_train_steps": options.steps, "max_train_epochs": None,
        "gradient_accumulation_steps": 1, "max_data_loader_n_workers": 0,
        "persistent_data_loader_workers": False, "save_every_n_steps": 0,
        "save_every_n_epochs": 0, "checkpointing_epochs": 0, "save_state": False,
        "save_state_on_train_end": False, "resume": None, "sample_prompts": "",
        "sample_at_first": False, "sample_every_n_steps": None, "sample_every_n_epochs": None,
        "validate_every_n_steps": None, "validate_every_n_epochs": None,
        "memory_probe_jsonl": "off", "block_swap_profile_jsonl": "off",
        "gradient_flow_probe_jsonl": "off", "progress_jsonl": "off",
        "config_snapshot": False, "use_text_cache": True, "use_vae_cache": True,
    })
    (root / "config.toml").write_text(toml.dumps(config), encoding="utf-8")
    write_result(root / "fixture.json", {
        "family": options.family, "width": width, "height": height,
        "sources": [str(path.resolve()) for path in [image, *sources]],
        "source_config": str(options.config), "real_cached_sample": True,
    })


def formal_hot(args, root, steps):
    import torch
    from safetensors import safe_open
    from library.training import loop
    from library.training.auto_block_swap.preferences import gpu_reserve_bytes
    from library.training.auto_block_swap.resources import host_memory
    from scripts.experiments.auto_block_swap_probe import run_formal

    rows = []
    original = loop._run_step
    before = host_memory()
    total = torch.cuda.get_device_properties(0).total_memory

    def recorded(trainer, state, batch):
        torch.cuda.synchronize()
        started = time.perf_counter()
        loss = original(trainer, state, batch)
        torch.cuda.synchronize()
        if not state.accelerator.sync_gradients or state.accelerator.optimizer_step_was_skipped:
            raise RuntimeError("Expected a complete optimizer update")
        free, _ = torch.cuda.mem_get_info()
        host = host_memory()
        row = {
            "step": state.global_step + 1, "seconds": time.perf_counter() - started,
            "loss": float(loss.detach()), "blocks": args.blocks_to_swap,
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
            "headroom": free + torch.cuda.memory_reserved() - torch.cuda.max_memory_allocated(),
            "host_available": host.available,
            "host_reserve": host.reserve,
            "swap_io": host.swap_in + host.swap_out - before.swap_in - before.swap_out,
        }
        rows.append(row)
        write_result(root / "hot-updates.json", {"updates": rows})
        print(json.dumps({"family": args.model_family, **row}), flush=True)
        return loss

    loop._run_step = recorded
    try:
        run_formal(args, root, "formal", "calibration", steps)
    finally:
        loop._run_step = original
    checkpoint = next((root / "formal").glob("*.safetensors"))
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        finite = all(torch.isfinite(handle.get_tensor(key)).all().item() for key in handle.keys())
        family = (handle.metadata() or {}).get("ss_model_family", "anima")
    safe = min(row["headroom"] for row in rows) >= gpu_reserve_bytes(args, total)
    safe = safe and all(row["host_available"] > row["host_reserve"] for row in rows)
    swap_limit = int(float(getattr(args, "auto_block_swap_swap_io_limit_mb", 1024.0)) * 1024**2)
    report = {
        "status": "passed" if finite and safe and len(rows) == steps else "failed",
        "family": family, "steps": len(rows), "finite_checkpoint": finite,
        "selected_blocks": args.blocks_to_swap, "gpu_reserve": gpu_reserve_bytes(args, total),
        "preference": args.auto_block_swap_preference,
        "hot_median_seconds": statistics.median(row["seconds"] for row in rows[2:]),
        "peak_allocated": max(row["peak_allocated"] for row in rows),
        "min_headroom": min(row["headroom"] for row in rows),
        "max_swap_io": max(row["swap_io"] for row in rows),
        "swap_io_limit_bytes": swap_limit,
        "min_host_available": min(row["host_available"] for row in rows),
        "host_reserve": max(row["host_reserve"] for row in rows),
        "compile": False, "evidence": "single_real_sample_startup_auto_not_dynamic_or_speed_ab",
    }
    if family != args.model_family or report["max_swap_io"] > swap_limit:
        report["status"] = "failed"
    write_result(root / "acceptance.json", report)
    if report["status"] != "passed":
        raise RuntimeError("Hot resource/checkpoint acceptance failed")


def main():
    from library.training.auto_block_swap.coordinator import run_calibration
    from library.training.train_bootstrap import install_stop_signal_handlers
    from scripts.experiments.auto_block_swap_probe import prepare_args

    install_stop_signal_handlers()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["anima", "z_image"], required=True)
    for name in ("output", "config", "images", "cache", "dit", "qwen3", "vae"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--reserve-percent", type=float, default=25.0)
    parser.add_argument("--preference", choices=["balanced", "vram"], default="vram")
    options = parser.parse_args()
    if not 1 <= options.limit <= 6 or not 3 <= options.steps <= 32:
        parser.error("Bounded test requires 1..6 candidates and 3..32 formal steps")
    options.output = options.output.resolve()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    prepare_fixture(options)
    args = prepare_args(options.output)
    write_result(options.output / "environment.json", {
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version", "--format=csv,noheader"], text=True),
        "torch_compile": False,
    })
    directory = options.output / "calibration"
    directory.mkdir()
    try:
        run_calibration(args, directory)
        formal_hot(args, options.output, options.steps)
    except BaseException as exc:
        write_result(options.output / "failure.json", {
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
        })
        raise


if __name__ == "__main__":
    main()
