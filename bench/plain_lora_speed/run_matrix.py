#!/usr/bin/env python
"""Run short plain-LoRA speed/VRAM benchmark jobs.

This benchmark intentionally uses the GUI plain LoRA variant
(`configs/gui-methods/lora.toml`) instead of `configs/methods/lora.toml`, whose
current default enables heavier LoRA-family experiments. It measures the boring
but useful acceleration path: frozen-module disk caches + slim adapter path +
compile/attention/VRAM/DataLoader strategy comparisons.

GPU safety: this workstation has physical GPU 0 = GTX 1050 4GB and physical
GPU 1 = RTX 3080 Ti Laptop 16GB. Training defaults to CUDA_VISIBLE_DEVICES=1
and refuses GPU 0 unless explicitly overridden.

Examples:
  python -m bench.plain_lora_speed.run_matrix --dry-run --suite rank
  python -m bench.plain_lora_speed.run_matrix --steps 80 --suite baseline
  python -m bench.plain_lora_speed.run_matrix --steps 80 --suite rank --seeds 42
  python -m bench.plain_lora_speed.run_matrix --steps 80 --suite memory
  python -m bench.plain_lora_speed.run_matrix --steps 80 --suite dataloader
  python -m bench.plain_lora_speed.run_matrix --steps 80 --arms baseline rank16 rank32
  python -m bench.plain_lora_speed.run_matrix --steps 80 --suite baseline --profile-steps 10-60
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import statistics
import subprocess
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from bench._common import REPO_ROOT, write_result

DEFAULT_GPU_INDEX = "1"
DEFAULT_ROOT = "output/bench/plain_lora_speed"
DEFAULT_DATASET_CONFIG = "configs/bench/signal_probe_dataset.toml"
DEFAULT_PROMPTS = "configs/bench/signal_probe_prompts.txt"
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_MIN_VRAM_MB = 12000

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class Arm:
    name: str
    variant: str = "lora"
    preset: str = "default"
    extra: tuple[str, ...] = field(default_factory=tuple)
    group: str = "baseline"
    note: str = ""

    @property
    def network_dim(self) -> int | None:
        if "--network_dim" not in self.extra:
            return None
        idx = self.extra.index("--network_dim")
        try:
            return int(self.extra[idx + 1])
        except (IndexError, ValueError):
            return None

    @property
    def workers(self) -> int | None:
        if "--max_data_loader_n_workers" not in self.extra:
            return None
        idx = self.extra.index("--max_data_loader_n_workers")
        try:
            return int(self.extra[idx + 1])
        except (IndexError, ValueError):
            return None


ARMS: dict[str, Arm] = {
    # Plain baseline: lora-gui + gui-methods/lora.toml + preset default.
    "baseline": Arm(
        "baseline",
        group="baseline",
        note="Plain classic LoRA: gui-methods/lora.toml + preset default.",
    ),
    # Rank sweep: keep alpha=dim for old LoRA scaling behavior.
    "rank16": Arm("rank16", extra=("--network_dim", "16", "--network_alpha", "16"), group="rank"),
    "rank32": Arm("rank32", extra=("--network_dim", "32", "--network_alpha", "32"), group="rank"),
    "rank64": Arm("rank64", extra=("--network_dim", "64", "--network_alpha", "64"), group="rank"),
    # VRAM strategy sweep.
    "mem_default": Arm("mem_default", group="memory", note="preset default: no checkpoint/offload/block swap."),
    "mem_balanced16g": Arm(
        "mem_balanced16g",
        preset="balanced_16g",
        group="memory",
        note="balanced_16g: block swap profile path for 16GB edge cases.",
    ),
    "mem_lora8gb": Arm(
        "mem_lora8gb",
        variant="lora-8gb",
        preset="default",
        group="memory",
        note="GUI low-VRAM plain LoRA variant: grad checkpoint + unsloth offload.",
    ),
    # DataLoader/I/O sweep. Only useful after baseline shows GPU starvation.
    "workers2": Arm("workers2", extra=("--max_data_loader_n_workers", "2"), group="dataloader"),
    "workers4": Arm("workers4", extra=("--max_data_loader_n_workers", "4"), group="dataloader"),
    "workers8": Arm("workers8", extra=("--max_data_loader_n_workers", "8"), group="dataloader"),
}

SUITES: dict[str, tuple[str, ...]] = {
    "baseline": ("baseline",),
    "rank": ("rank16", "rank32", "rank64"),
    "memory": ("mem_default", "mem_balanced16g", "mem_lora8gb"),
    "dataloader": ("workers2", "workers4", "workers8"),
    "short": ("baseline", "rank16", "rank32", "rank64", "mem_balanced16g", "mem_lora8gb"),
    "all": tuple(ARMS),
}


@dataclass(frozen=True)
class CacheSubsetSummary:
    image_dir: str
    cache_dir: str
    image_count: int
    vae_cache_count: int
    text_cache_count: int

    @property
    def ready(self) -> bool:
        return self.image_count > 0 and self.vae_cache_count >= self.image_count and self.text_cache_count >= self.image_count


def _parse_csv_ints(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def _parse_step_window(text: str | None) -> tuple[int, int] | None:
    if not text:
        return None
    raw = text.strip()
    if raw.lower() in {"off", "none", "0", ""}:
        return None
    if "-" in raw:
        start, end = raw.split("-", 1)
        a, b = int(start), int(end)
    else:
        a = int(raw)
        b = a
    if a <= 0 or b <= 0 or b < a:
        raise argparse.ArgumentTypeError("step window must be positive and ordered, e.g. 10-60")
    return a, b


def _parse_arms(values: Iterable[str] | None, suite: str | None) -> list[str]:
    out: list[str] = []
    if values:
        for value in values:
            for part in value.split(","):
                part = part.strip()
                if part:
                    if part not in ARMS:
                        known = ", ".join(sorted(ARMS))
                        raise argparse.ArgumentTypeError(f"unknown arm {part!r}; known: {known}")
                    out.append(part)
    if out:
        return out
    suite = suite or "baseline"
    if suite not in SUITES:
        known = ", ".join(sorted(SUITES))
        raise argparse.ArgumentTypeError(f"unknown suite {suite!r}; known: {known}")
    return list(SUITES[suite])


def _gpu_rows() -> list[dict[str, str]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
    except Exception:
        return []
    rows: list[dict[str, str]] = []
    for line in cp.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5:
            rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_total_mb": parts[2],
                    "memory_used_mb": parts[3],
                    "utilization_gpu_pct": parts[4],
                }
            )
    return rows


def _check_gpu(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = _gpu_rows()
    if args.gpu_index == "0" and not args.allow_gpu0:
        raise SystemExit(
            "refusing to use physical GPU 0 without --allow-gpu0; "
            "this machine's GPU 0 is expected to be the 4GB GTX 1050"
        )
    selected = next((r for r in rows if r.get("index") == args.gpu_index), None)
    if selected:
        name = selected.get("name", "")
        total = int(float(selected.get("memory_total_mb", "0") or 0))
        if "1050" in name and not args.allow_gpu0:
            raise SystemExit(f"refusing to train on GTX1050-like GPU: {selected}")
        if total < args.min_vram_mb and not args.allow_low_vram:
            raise SystemExit(
                f"refusing to train on GPU {args.gpu_index} ({name}) with only {total} MB VRAM; "
                "pass --allow-low-vram to override"
            )
    elif rows:
        raise SystemExit(f"GPU index {args.gpu_index} not found in nvidia-smi rows: {rows}")
    return rows


def _verify_torch_mapping(args: argparse.Namespace) -> dict[str, str] | None:
    """Verify the child process sees exactly the selected physical GPU."""
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    code = (
        "import json, torch; "
        "n=torch.cuda.device_count(); "
        "out={'count': n}; "
        "\nif n:\n"
        " p=torch.cuda.get_device_properties(0); "
        " out.update({'name': torch.cuda.get_device_name(0), 'memory_total_mb': p.total_memory//1024//1024}); "
        "\nprint(json.dumps(out))"
    )
    cp = subprocess.run(
        [str(args.python), "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if cp.returncode != 0:
        raise SystemExit(f"torch GPU mapping check failed:\n{cp.stdout}\n{cp.stderr}")
    lines = [ln for ln in cp.stdout.splitlines() if ln.strip().startswith("{")]
    if not lines:
        raise SystemExit(f"torch GPU mapping check produced no JSON:\n{cp.stdout}\n{cp.stderr}")
    info = json.loads(lines[-1])
    if int(info.get("count", 0)) != 1:
        raise SystemExit(f"expected exactly one visible CUDA device, got: {info}")
    name = str(info.get("name", ""))
    total = int(info.get("memory_total_mb", 0) or 0)
    if "1050" in name and not args.allow_gpu0:
        raise SystemExit(f"refusing to train: child torch maps to GTX1050: {info}")
    if total < args.min_vram_mb and not args.allow_low_vram:
        raise SystemExit(f"refusing to train: child torch visible GPU has low VRAM: {info}")
    return info


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)


def _count_cache(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob(pattern) if p.is_file())


def _dataset_cache_summary(dataset_config: str | Path) -> list[CacheSubsetSummary]:
    path = _repo_path(dataset_config)
    if not path.exists():
        raise SystemExit(f"dataset_config not found: {path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    rows: list[CacheSubsetSummary] = []
    for dataset in data.get("datasets", []) or []:
        for subset in dataset.get("subsets", []) or []:
            image_dir = _repo_path(str(subset.get("image_dir", "")))
            cache_dir = _repo_path(str(subset.get("cache_dir") or subset.get("image_dir", "")))
            rows.append(
                CacheSubsetSummary(
                    image_dir=str(image_dir),
                    cache_dir=str(cache_dir),
                    image_count=_count_images(image_dir),
                    vae_cache_count=_count_cache(cache_dir, "*_anima.npz"),
                    text_cache_count=_count_cache(cache_dir, "*_anima_te.safetensors"),
                )
            )
    return rows


def _check_dataset_cache(args: argparse.Namespace) -> list[dict[str, int | str | bool]]:
    summaries = _dataset_cache_summary(args.dataset_config)
    if not summaries:
        raise SystemExit(f"dataset_config has no subsets: {args.dataset_config}")
    rows: list[dict[str, int | str | bool]] = []
    problems: list[str] = []
    for idx, item in enumerate(summaries):
        rec = {
            "subset": idx,
            "image_dir": item.image_dir,
            "cache_dir": item.cache_dir,
            "image_count": item.image_count,
            "vae_cache_count": item.vae_cache_count,
            "text_cache_count": item.text_cache_count,
            "ready": item.ready,
        }
        rows.append(rec)
        if item.image_count <= 0:
            problems.append(f"subset {idx}: no images in {item.image_dir}")
        elif not item.ready:
            problems.append(
                f"subset {idx}: cache incomplete, images={item.image_count}, "
                f"vae={item.vae_cache_count}, text={item.text_cache_count}, cache_dir={item.cache_dir}"
            )
    if problems and not args.allow_missing_cache:
        raise SystemExit(
            "frozen-module cache preflight failed; run `make preprocess` first or pass "
            "--allow-missing-cache if you intentionally want to measure live/cache-miss overhead:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )
    return rows


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _mean(values: list[float]) -> float | None:
    return float(statistics.fmean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return float(ordered[idx])


def _summarize_progress(progress_path: Path, *, metric_window: tuple[int, int] | None, images_per_step: float) -> dict:
    events = _read_jsonl(progress_path)
    steps = [e for e in events if e.get("ev") == "step" and e.get("global_step") is not None and e.get("ts") is not None]
    steps.sort(key=lambda e: int(e.get("global_step", 0)))
    last_step = steps[-1] if steps else None
    intervals: list[tuple[int, float]] = []
    for prev, cur in zip(steps, steps[1:]):
        try:
            step_no = int(cur["global_step"])
            dt = float(cur["ts"]) - float(prev["ts"])
        except (TypeError, ValueError):
            continue
        if dt > 0:
            intervals.append((step_no, dt))
    if metric_window is not None:
        lo, hi = metric_window
        selected = [dt for step_no, dt in intervals if lo <= step_no <= hi]
    else:
        selected = [dt for _, dt in intervals]
    # If a short run has fewer steps than the configured window, fall back to all
    # observed intervals so smoke runs still produce a useful summary.
    if not selected:
        selected = [dt for _, dt in intervals]
    avg_step = _mean(selected)
    median_step = _median(selected)
    p90_step = _quantile(selected, 0.90)
    images_per_hour = (3600.0 * images_per_step / avg_step) if avg_step and avg_step > 0 else None

    def max_field(key: str) -> float | None:
        vals = []
        for e in steps:
            try:
                val = float(e[key])
            except (KeyError, TypeError, ValueError):
                continue
            vals.append(val)
        return max(vals) if vals else None

    return {
        "step_events": len(steps),
        "steps_completed": None if last_step is None else int(last_step.get("global_step", 0)),
        "last_step": last_step,
        "loss": None if last_step is None else last_step.get("loss"),
        "avr_loss": None if last_step is None else last_step.get("avr_loss"),
        "metric_window": None if metric_window is None else f"{metric_window[0]}-{metric_window[1]}",
        "interval_count": len(selected),
        "avg_step_sec": None if avg_step is None else round(avg_step, 6),
        "median_step_sec": None if median_step is None else round(median_step, 6),
        "p90_step_sec": None if p90_step is None else round(p90_step, 6),
        "images_per_hour": None if images_per_hour is None else round(images_per_hour, 3),
        "peak_allocated_gb": max_field("cuda/max_memory_allocated_gb"),
        "peak_reserved_gb": max_field("cuda/max_memory_reserved_gb"),
    }


def _build_train_cmd(args: argparse.Namespace, arm: Arm, seed: int, run_dir: Path, run_name: str) -> tuple[list[str], dict[str, str], dict[str, str]]:
    ckpt_dir = run_dir / "ckpt"
    log_dir = run_dir / "logs"
    progress_path = log_dir / f"{run_name}.progress.jsonl"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(args.python),
        "tasks.py",
        "lora-gui",
        arm.variant,
        "--max_train_steps",
        str(args.steps),
        "--seed",
        str(seed),
        "--output_dir",
        str(ckpt_dir),
        "--output_name",
        run_name,
        "--progress_jsonl",
        str(progress_path),
        "--sample_prompts",
        args.sample_prompts,
        "--dataset_config",
        args.dataset_config,
        # Keep the measured path slim: no validation/sample/CMMD and no mid-run save.
        "--sample_every_n_epochs",
        "0",
        "--sample_every_n_steps",
        "0",
        "--validation_split_num",
        "0",
        "--no-use_cmmd",
        "--save_every_n_epochs",
        "999999",
        "--checkpointing_epochs",
        "0",
        # Make structured metrics dense enough for step-time estimation.
        "--log_every_n_steps",
        "1",
        # Re-state speed-oriented defaults so config drift is obvious in the command.
        "--torch_compile",
        "--attn_mode",
        "flash",
        "--dataloader_pin_memory",
        "--persistent_data_loader_workers",
        *arm.extra,
        *args.extra,
    ]
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    env["PRESET"] = arm.preset
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    paths = {
        "run_dir": str(run_dir),
        "ckpt_dir": str(ckpt_dir),
        "logs_dir": str(log_dir),
        "progress_jsonl": str(progress_path),
        "stdout": str(log_dir / "train.stdout.log"),
        "block_swap_profile_jsonl": str(log_dir / f"{run_name}.block_swap_profile.jsonl"),
        "nsys_report": str(log_dir / f"{run_name}.nsys-rep"),
    }
    if args.profile_steps:
        env["PROFILE_STEPS"] = args.profile_steps
        env["NSYS_OUT"] = str(log_dir / f"{run_name}.nsys-rep")
    return cmd, env, paths


def _printable_cmd(cmd: list[str], env: dict[str, str]) -> str:
    env_keys = ["CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "PRESET"]
    if "PROFILE_STEPS" in env:
        env_keys.extend(["PROFILE_STEPS", "NSYS_OUT"])
    prefix = " ".join(f"{k}={shlex.quote(str(env[k]))}" for k in env_keys if k in env)
    return prefix + " " + " ".join(shlex.quote(c) for c in cmd)


def _run_one(args: argparse.Namespace, arm: Arm, seed: int, gpu_rows: list[dict[str, str]], cache_rows: list[dict]) -> dict:
    run_name = f"{arm.name}_s{seed}_{args.steps}step"
    root = _repo_path(args.output_root).resolve()
    run_dir = root / run_name
    cmd, env, paths = _build_train_cmd(args, arm, seed, run_dir, run_name)
    stdout_path = Path(paths["stdout"])
    summary_path = run_dir / "summary.json"

    printable = _printable_cmd(cmd, env)
    print(f"\n>>> {run_name}\n{printable}", flush=True)

    started = time.time()
    if args.dry_run:
        rc = 0
    else:
        with stdout_path.open("w", encoding="utf-8") as fh:
            fh.write(f"# {printable}\n")
            fh.flush()
            cp = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
            rc = cp.returncode
    elapsed = time.time() - started

    progress_metrics = _summarize_progress(
        Path(paths["progress_jsonl"]),
        metric_window=args.metric_step_window,
        images_per_step=args.images_per_step,
    )
    metrics = {
        "returncode": rc,
        "elapsed_sec": round(elapsed, 3),
        "steps_requested": args.steps,
        "gpu_index_physical": args.gpu_index,
        **progress_metrics,
    }
    record = {
        "run_name": run_name,
        "arm": arm.name,
        "group": arm.group,
        "variant": arm.variant,
        "preset": arm.preset,
        "network_dim": arm.network_dim,
        "workers": arm.workers,
        "seed": seed,
        "note": arm.note,
        "cmd": cmd,
        "env": {k: env[k] for k in ["CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "PRESET", "PROFILE_STEPS", "NSYS_OUT"] if k in env},
        "gpu_rows_before": gpu_rows,
        "torch_child_visible_gpu": getattr(args, "_torch_child_visible_gpu", None),
        "cache_preflight": cache_rows,
        "metrics": metrics,
        "paths": paths,
    }
    summary_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    status = "OK" if rc == 0 else f"FAIL({rc})"
    print(
        f"<<< {run_name} {status} elapsed={elapsed:.1f}s "
        f"avg_step={metrics.get('avg_step_sec')}s img/h={metrics.get('images_per_hour')} "
        f"peak={metrics.get('peak_reserved_gb')}GB avr_loss={metrics.get('avr_loss')}",
        flush=True,
    )
    if rc != 0 and args.stop_on_fail:
        raise SystemExit(rc)
    return record


def _write_index(output_root: str, records: list[dict], args: argparse.Namespace) -> Path:
    root = _repo_path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "runs.csv"
    fieldnames = [
        "run_name",
        "arm",
        "group",
        "variant",
        "preset",
        "seed",
        "network_dim",
        "workers",
        "returncode",
        "elapsed_sec",
        "steps_completed",
        "avg_step_sec",
        "median_step_sec",
        "p90_step_sec",
        "images_per_hour",
        "peak_allocated_gb",
        "peak_reserved_gb",
        "loss",
        "avr_loss",
        "progress_jsonl",
        "stdout",
        "block_swap_profile_jsonl",
        "nsys_report",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            metrics = rec["metrics"]
            paths = rec["paths"]
            writer.writerow(
                {
                    "run_name": rec["run_name"],
                    "arm": rec["arm"],
                    "group": rec["group"],
                    "variant": rec["variant"],
                    "preset": rec["preset"],
                    "seed": rec["seed"],
                    "network_dim": rec["network_dim"],
                    "workers": rec["workers"],
                    "returncode": metrics["returncode"],
                    "elapsed_sec": metrics["elapsed_sec"],
                    "steps_completed": metrics["steps_completed"],
                    "avg_step_sec": metrics["avg_step_sec"],
                    "median_step_sec": metrics["median_step_sec"],
                    "p90_step_sec": metrics["p90_step_sec"],
                    "images_per_hour": metrics["images_per_hour"],
                    "peak_allocated_gb": metrics["peak_allocated_gb"],
                    "peak_reserved_gb": metrics["peak_reserved_gb"],
                    "loss": metrics["loss"],
                    "avr_loss": metrics["avr_loss"],
                    "progress_jsonl": paths["progress_jsonl"],
                    "stdout": paths["stdout"],
                    "block_swap_profile_jsonl": paths["block_swap_profile_jsonl"],
                    "nsys_report": paths["nsys_report"],
                }
            )
    write_result(
        root,
        script=__file__,
        args=args,
        metrics={"num_runs": len(records), "runs_csv": str(csv_path)},
        artifacts=[csv_path],
        label="plain-lora-speed-matrix",
        device=None,
        extra={"records": records},
    )
    return csv_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", choices=sorted(SUITES), default="baseline", help="Named arm suite to run when --arms is omitted.")
    p.add_argument("--arms", nargs="+", default=None, help=f"Arms to run. Known: {', '.join(sorted(ARMS))}")
    p.add_argument("--seeds", type=_parse_csv_ints, default=[42], help="Comma-separated seeds, e.g. 42,43,44")
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--metric-step-window", type=_parse_step_window, default=(10, 60), help="Step interval window for avg/median step time, e.g. 10-60; off=all.")
    p.add_argument("--images-per-step", type=float, default=1.0, help="For images/hour estimation. Default 1 for current LoRA bench dataset.")
    p.add_argument("--profile-steps", default=None, help="Set PROFILE_STEPS for nsys/NVTX capture, e.g. 10-60. Profiling mode stops when the range ends.")
    p.add_argument("--gpu-index", default=DEFAULT_GPU_INDEX, help="Physical GPU index for CUDA_VISIBLE_DEVICES. Default: 1 (RTX 3080 Ti on this host).")
    p.add_argument("--allow-gpu0", action="store_true", help="Allow physical GPU 0. Normally refused because it is the 4GB GTX 1050 on this host.")
    p.add_argument("--min-vram-mb", type=int, default=DEFAULT_MIN_VRAM_MB)
    p.add_argument("--allow-low-vram", action="store_true")
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    p.add_argument("--output-root", default=DEFAULT_ROOT)
    p.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    p.add_argument("--sample-prompts", default=DEFAULT_PROMPTS)
    p.add_argument("--allow-missing-cache", action="store_true", help="Do not refuse if VAE/TE cache files are missing. Use only when intentionally measuring live/cache-miss overhead.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stop-on-fail", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("extra", nargs=argparse.REMAINDER, help="Extra train.py args after --, forwarded to every run.")
    args = p.parse_args()
    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]
    if not args.python.exists():
        raise SystemExit(f"python executable not found: {args.python}")
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.images_per_step <= 0:
        raise SystemExit("--images-per-step must be positive")
    args.arms = _parse_arms(args.arms, args.suite)

    cache_rows = _check_dataset_cache(args)
    gpu_rows = _check_gpu(args)
    torch_gpu = _verify_torch_mapping(args)
    args._torch_child_visible_gpu = torch_gpu

    print("Cache preflight:", json.dumps(cache_rows, ensure_ascii=False), flush=True)
    print("GPU check:", json.dumps(gpu_rows, ensure_ascii=False), flush=True)
    print("Torch child visible GPU:", json.dumps(torch_gpu, ensure_ascii=False), flush=True)
    print(
        f"Using physical GPU {args.gpu_index} via CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={args.gpu_index}",
        flush=True,
    )
    if args.profile_steps:
        print(
            "Profiling mode enabled: train.py exits after PROFILE_STEPS ends, so use this for timing traces, not final loss.",
            flush=True,
        )

    records: list[dict] = []
    for arm_name in args.arms:
        arm = ARMS[arm_name]
        for seed in args.seeds:
            records.append(_run_one(args, arm, seed, gpu_rows, cache_rows))
    csv_path = _write_index(args.output_root, records, args)
    print(f"\nindex: {csv_path}")


if __name__ == "__main__":
    main()
