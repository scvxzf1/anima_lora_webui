"""Cache build + 10-step instrumented training run for one matrix row."""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from pathlib import Path

from bench.moss_tier_hot import collect

DEFAULT_PYTHON = ".venv/bin/python"
CACHE_TIMEOUT_SEC = 3600


def _env(repo_root: Path, uuid: str | None, runtime_config: Path) -> dict:
    env = dict(os.environ)
    env["ANIMA_RUNTIME_CONFIG"] = str(runtime_config)
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    if uuid:
        env["CUDA_VISIBLE_DEVICES"] = uuid
    return env


def _run(cmd: list[str], repo_root: Path, env: dict, log_path: Path,
         timeout_sec: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(cmd, cwd=str(repo_root), env=env, stdout=handle,
                              stderr=subprocess.STDOUT, timeout=timeout_sec)
    return int(proc.returncode)


def _clear_row_outputs(assets: dict) -> None:
    """Fresh resize/cache inputs for a row.

    preprocess-config forwards its trailing flags to every stage, and the
    resize stage rejects --overwrite, so the harness clears these two
    harness-owned directories itself instead of forwarding the flag.
    """
    import shutil

    for key in ("data_dir", "cache_dir"):
        path = Path(assets[key])
        if path.is_dir():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def build_cache(repo_root: Path, assets: dict, uuid: str | None, python: str,
                overwrite: bool = True) -> dict:
    """Resize src -> data and build the MOSS VAE + TE caches for one row."""
    env = _env(repo_root, uuid, assets["runtime_config"])
    row = assets["row"]
    if overwrite:
        _clear_row_outputs(assets)
    # preprocess-config forwards its trailing flags to the resize stage only,
    # and that stage is where the tier ladder has to be pinned: without these
    # every row would be resized with base.toml defaults (1024).
    cmd = [python, "tasks.py", "preprocess-config",
           "--dataset_config", str(assets["dataset_config"]),
           "--src", str(assets["src_dir"]),
           "--dit", f"models/moss_image/MOSS-Image-DiT",
           "--qwen3", f"models/moss_image/MOSS-Image-TextEncoder-BF16",
           "--vae", f"models/moss_image/MOSS-Image-VAE",
           "--resolution", str(row.resolution),
           "--min_bucket_reso", str(row.min_reso),
           "--max_bucket_reso", str(row.max_reso),
           "--constant_token_buckets"]
    started = time.time()
    code = _run(cmd, repo_root, env, assets["root"] / "cache.log", CACHE_TIMEOUT_SEC)
    return {"row": row.name, "cache_exit_code": code,
            "cache_seconds": round(time.time() - started, 3),
            "cache_command": cmd, **_cache_provenance(assets)}


def _cache_provenance(assets: dict) -> dict:
    """What preprocessing actually wrote: resized size and latent keys."""
    import numpy as np

    provenance = {}
    images = sorted(Path(assets["data_dir"]).glob("*.png"))
    if images:
        from PIL import Image

        with Image.open(images[0]) as handle:
            provenance["resized_image"] = f"{handle.width}x{handle.height}"
    latents = sorted(Path(assets["cache_dir"]).glob("*_moss_image.npz"))
    if latents:
        with np.load(latents[0]) as handle:
            provenance["latent_keys"] = sorted(handle.files)
    return provenance


def _max_or_none(values: list) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float))]
    return round(max(numbers), 4) if numbers else None


def _progress_stats(scan: dict) -> dict:
    pairs = scan.get("durations") or []
    events = scan.get("events") or []
    if not pairs:
        return {"steps_recorded": len(events)}
    first = pairs[0][1]
    tail = [value for _, value in pairs[1:]] or [first]
    return {
        "steps_recorded": len(events),
        "first_step_seconds": round(first, 6),
        "tail_mean_seconds": round(statistics.fmean(tail), 6),
        "tail_median_seconds": round(statistics.median(tail), 6),
        "tail_min_seconds": round(min(tail), 6),
        "tail_max_seconds": round(max(tail), 6),
        "per_step_seconds": [round(value, 6) for _, value in pairs],
        "step_peak_allocated_gb": _max_or_none(
            [event.get("peak_allocated_gb") for event in events]),
        "step_peak_reserved_gb": _max_or_none(
            [event.get("peak_reserved_gb") for event in events]),
    }


def run_training(repo_root: Path, assets: dict, uuid: str | None, python: str,
                 timeout_sec: int = 7200, cache_info: dict | None = None) -> dict:
    env = _env(repo_root, uuid, assets["runtime_config"])
    stop, thread = collect.start_gpu_poller(assets["root"] / "gpu.csv", uuid)
    started = time.time()
    try:
        cmd = [python, "train.py", "--config_file", str(assets["runtime_config"])]
        code = _run(cmd, repo_root, env, assets["root"] / "train.log", timeout_sec)
    finally:
        stop.set()
        thread.join(timeout=5)
    wall = round(time.time() - started, 3)
    progress = collect.read_jsonl(assets["root"] / "progress.jsonl")
    memory = collect.read_jsonl(assets["root"] / "memory_probe.jsonl")
    result = {"row": assets["row"].name, "train_exit_code": code, "wall_seconds": wall,
              "command": cmd,
              "cache_seconds": (cache_info or {}).get("cache_seconds"),
              "resized_image": (cache_info or {}).get("resized_image"),
              "latent_keys": (cache_info or {}).get("latent_keys"),
              **_progress_stats(collect.step_events(progress)),
              "gpu": collect.summarize_gpu_csv(assets["root"] / "gpu.csv"),
              "memory_peaks": collect.peak_memory(memory)}
    (assets["root"] / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
