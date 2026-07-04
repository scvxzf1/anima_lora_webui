#!/usr/bin/env python
"""Run short Anima training jobs and estimate MFU.

Design goals:
- stay inside the existing `bench/` conventions
- reuse `progress_jsonl` for step timing
- reuse `peak_probe_jsonl` only to recover token shapes
- avoid changes to the main training loop

MFU here means:
    estimated_train_step_flops / (avg_step_sec * peak_hw_flops)

The FLOPs term is an Anima-specific estimate over the dominant DiT path, not a
hardware counter. It is mainly for comparing configs on the same machine.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import toml

from bench._common import REPO_ROOT, write_result
from bench.mfu.flops import (
    AnimaModelSpec,
    StepShape,
    estimate_mfu,
    parse_step_shape_from_peak_probe_event,
    total_forward_flops,
    total_train_step_flops,
)
from library.config.io import load_method_preset

DEFAULT_GPU_INDEX = "1"
DEFAULT_ROOT = "output/bench/mfu"
DEFAULT_DRY_RUN_ROOT = "tmp/bench-dry-runs/mfu"
DEFAULT_VARIANT = "lora_signal_probe"
DEFAULT_DATASET_CONFIG = "configs/bench/signal_probe_dataset.toml"
DEFAULT_SINGLE_DATASET_CONFIG = DEFAULT_DATASET_CONFIG
DEFAULT_PROMPTS = "configs/bench/signal_probe_prompts.txt"
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
DEFAULT_MIN_VRAM_MB = 12000
DEFAULT_PEAK_TFLOPS = 181.0
DEFAULT_PEAK_PROBE_LEVEL = "block"
DEFAULT_TRAIN_TIMEOUT_SEC = 3600
GUI_METHODS_SUBDIR = "gui-methods"
LAUNCH_MODE_TASKS_GUI = "tasks-gui"
LAUNCH_MODE_DIRECT = "direct"


@dataclass(frozen=True)
class Arm:
    name: str
    variant: str = "lora"
    preset: str = "default"
    extra: tuple[str, ...] = field(default_factory=tuple)
    group: str = "baseline"
    note: str = ""


ARMS: dict[str, Arm] = {
    "baseline": Arm(
        "baseline",
        variant=DEFAULT_VARIANT,
        group="baseline",
        note="Tracked LoRA signal-probe smoke baseline for MFU command generation and short runs.",
    ),
    "rank16": Arm(
        "rank16",
        variant=DEFAULT_VARIANT,
        extra=("--network_dim", "16", "--network_alpha", "16"),
        group="rank",
        note="Signal-probe baseline with rank=16 override.",
    ),
    "rank64": Arm(
        "rank64",
        variant=DEFAULT_VARIANT,
        extra=("--network_dim", "64", "--network_alpha", "64"),
        group="rank",
        note="Signal-probe baseline with rank=64 override.",
    ),
    "no_compile": Arm(
        "no_compile",
        variant=DEFAULT_VARIANT,
        extra=("--no-torch_compile",),
        group="compile",
        note="Signal-probe baseline with torch.compile disabled.",
    ),
    "balanced16g": Arm(
        "balanced16g",
        variant=DEFAULT_VARIANT,
        preset="balanced_16g",
        group="memory",
        note="Signal-probe baseline under balanced_16g preset.",
    ),
    "plain_lora_ckpt": Arm(
        "plain_lora_ckpt",
        variant="lora-8gb",
        group="baseline",
        note="Tracked plain LoRA low-VRAM variant with gradient checkpointing enabled.",
    ),
}

SUITES: dict[str, tuple[str, ...]] = {
    "baseline": ("baseline",),
    "plain_lora": ("plain_lora_ckpt",),
    "rank": ("baseline", "rank16", "rank64"),
    "compile": ("baseline", "no_compile"),
    "memory": ("baseline", "balanced16g"),
    "all": tuple(ARMS),
}


def _has_cli_option(argv: list[str], option: str) -> bool:
    prefix = option + "="
    return any(item == option or item.startswith(prefix) for item in argv)


def _resolve_output_root(output_root: str, *, dry_run: bool, output_root_explicit: bool) -> str:
    if dry_run and not output_root_explicit:
        return DEFAULT_DRY_RUN_ROOT
    return output_root


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
        lo, hi = int(start), int(end)
    else:
        lo = int(raw)
        hi = lo
    if lo <= 0 or hi <= 0 or hi < lo:
        raise argparse.ArgumentTypeError("step window must be positive and ordered, e.g. 10-60")
    return lo, hi


def _parse_arms(values: Iterable[str] | None, suite: str | None) -> list[str]:
    out: list[str] = []
    if values:
        for value in values:
            for part in value.split(","):
                part = part.strip()
                if not part:
                    continue
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


def _summarize_progress(progress_path: Path, *, metric_window: tuple[int, int] | None) -> dict:
    events = _read_jsonl(progress_path)
    steps = [e for e in events if e.get("ev") == "step" and e.get("global_step") is not None and e.get("ts") is not None]
    steps.sort(key=lambda e: int(e.get("global_step", 0)))
    last_step = steps[-1] if steps else None
    run_end = next((e for e in reversed(events) if e.get("ev") == "run_end"), None)
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
    if not selected:
        selected = [dt for _, dt in intervals]
    avg_step = _mean(selected)
    median_step = _median(selected)
    p90_step = _quantile(selected, 0.90)

    def max_field(key: str) -> float | None:
        vals = []
        for e in steps:
            try:
                vals.append(float(e[key]))
            except (KeyError, TypeError, ValueError):
                continue
        return max(vals) if vals else None

    return {
        "step_events": len(steps),
        "steps_completed": None if last_step is None else int(last_step.get("global_step", 0)),
        "last_step": last_step,
        "loss": None if last_step is None else last_step.get("loss"),
        "avr_loss": None if last_step is None else last_step.get("avr_loss"),
        "run_end_status": None if run_end is None else run_end.get("status"),
        "run_end_error": None if run_end is None else run_end.get("error"),
        "metric_window": None if metric_window is None else f"{metric_window[0]}-{metric_window[1]}",
        "interval_count": len(selected),
        "avg_step_sec": None if avg_step is None else round(avg_step, 6),
        "median_step_sec": None if median_step is None else round(median_step, 6),
        "p90_step_sec": None if p90_step is None else round(p90_step, 6),
        "peak_allocated_gb": max_field("cuda/max_memory_allocated_gb"),
        "peak_reserved_gb": max_field("cuda/max_memory_reserved_gb"),
    }


def _first_block_probe_shape(peak_probe_path: Path) -> StepShape | None:
    for event in _read_jsonl(peak_probe_path):
        if event.get("ev") != "peak_probe":
            continue
        if event.get("label") not in {"block_before", "block_after"}:
            continue
        block_idx = event.get("block_idx")
        if str(block_idx) not in {"0", "0.0"} and block_idx != 0:
            continue
        return parse_step_shape_from_peak_probe_event(event)
    return None


def _write_flat_toml(path: Path, values: dict[str, object]) -> None:
    lines = [
        "# generated by bench.mfu.run_training",
        "",
    ]
    for key in sorted(values):
        value = values[key]
        if value is None:
            continue
        dumped = toml.dumps({key: value}).strip()
        if dumped:
            lines.append(dumped)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _materialize_arm_config(
    run_dir: Path,
    arm: Arm,
    *,
    torch_compile: bool | None = None,
) -> Path:
    merged = load_method_preset(
        arm.variant,
        arm.preset,
        configs_dir=str(REPO_ROOT / "configs"),
        methods_subdir=GUI_METHODS_SUBDIR,
    )
    if torch_compile is not None:
        merged["torch_compile"] = torch_compile
    config_path = run_dir / f"{arm.name}.generated.toml"
    _write_flat_toml(config_path, merged)
    return config_path


def _build_train_cmd(
    args: argparse.Namespace,
    arm: Arm,
    seed: int,
    run_dir: Path,
    run_name: str,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    ckpt_dir = run_dir / "ckpt"
    log_dir = run_dir / "logs"
    progress_path = log_dir / f"{run_name}.progress.jsonl"
    peak_probe_path = log_dir / f"{run_name}.peak_probe.jsonl"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    arm_extra = list(arm.extra)
    force_no_compile = "--no-torch_compile" in arm_extra
    if force_no_compile:
        arm_extra = [item for item in arm_extra if item != "--no-torch_compile"]
    launch_mode = getattr(args, "launch_mode", LAUNCH_MODE_TASKS_GUI)

    config_path = None
    if force_no_compile or launch_mode == LAUNCH_MODE_DIRECT:
        config_path = _materialize_arm_config(
            run_dir,
            arm,
            torch_compile=False if force_no_compile else True,
        )
    elif launch_mode != LAUNCH_MODE_TASKS_GUI:
        raise ValueError(f"unsupported launch mode: {launch_mode}")

    train_args = [
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
        "--peak_probe_jsonl",
        str(peak_probe_path),
        "--peak_probe_max_steps",
        str(args.peak_probe_max_steps),
        "--peak_probe_level",
        args.peak_probe_level,
        "--sample_prompts",
        args.sample_prompts,
        "--dataset_config",
        args.dataset_config,
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
        "--log_every_n_steps",
        "1",
        "--attn_mode",
        "flash",
        "--dataloader_pin_memory",
        "--persistent_data_loader_workers",
    ]
    if config_path is not None:
        train_args = ["--config_file", str(config_path), *train_args]

    if launch_mode == LAUNCH_MODE_TASKS_GUI:
        cmd = [
            str(args.python),
            "tasks.py",
            "lora-gui",
            arm.variant,
            *([] if force_no_compile else ["--torch_compile"]),
            *train_args,
            *arm_extra,
            *args.extra,
        ]
    else:
        cmd = [
            str(args.python),
            "train.py",
            *([] if (force_no_compile or config_path is not None) else ["--method", arm.variant, "--preset", arm.preset, "--methods_subdir", GUI_METHODS_SUBDIR]),
            *([] if force_no_compile else ["--torch_compile"]),
            *train_args,
            *arm_extra,
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
        "peak_probe_jsonl": str(peak_probe_path),
        "stdout": str(log_dir / "train.stdout.log"),
    }
    if config_path is not None:
        paths["config_file"] = str(config_path)
    return cmd, env, paths


def _printable_cmd(cmd: list[str], env: dict[str, str]) -> str:
    env_keys = ["CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "PRESET"]
    prefix = " ".join(f"{k}={shlex.quote(str(env[k]))}" for k in env_keys if k in env)
    return prefix + " " + " ".join(shlex.quote(c) for c in cmd)


def _estimate_metrics(
    *,
    progress_path: Path,
    peak_probe_path: Path,
    metric_window: tuple[int, int] | None,
    peak_tflops: float,
    spec: AnimaModelSpec,
) -> dict:
    progress_metrics = _summarize_progress(progress_path, metric_window=metric_window)
    shape = _first_block_probe_shape(peak_probe_path)
    if shape is None:
        return {
            **progress_metrics,
            "shape": None,
            "forward_flops": None,
            "train_step_flops": None,
            "achieved_tflops": None,
            "mfu": None,
        }
    avg_step = progress_metrics.get("avg_step_sec")
    train_step_flops = total_train_step_flops(shape, spec)
    achieved_tflops = None
    if avg_step and avg_step > 0:
        achieved_tflops = train_step_flops / float(avg_step) / 1e12
    mfu = estimate_mfu(
        shape=shape,
        avg_step_sec=float(avg_step) if avg_step else -1.0,
        peak_flops=peak_tflops * 1e12,
        spec=spec,
    )
    return {
        **progress_metrics,
        "shape": {
            "batch_size": shape.batch_size,
            "time_patches": shape.time_patches,
            "height_patches": shape.height_patches,
            "width_patches": shape.width_patches,
            "token_count": shape.token_count,
        },
        "forward_flops": int(total_forward_flops(shape, spec)),
        "train_step_flops": int(train_step_flops),
        "achieved_tflops": None if achieved_tflops is None else round(achieved_tflops, 6),
        "mfu": None if mfu is None else round(mfu, 6),
    }


def _run_one(
    args: argparse.Namespace,
    arm: Arm,
    seed: int,
    gpu_rows: list[dict[str, str]],
    spec: AnimaModelSpec,
) -> dict:
    run_name = f"{arm.name}_s{seed}_{args.steps}step"
    root = (REPO_ROOT / args.output_root).resolve()
    run_dir = root / run_name
    cmd, env, paths = _build_train_cmd(args, arm, seed, run_dir, run_name)
    stdout_path = Path(paths["stdout"])
    summary_path = run_dir / "summary.json"

    printable = _printable_cmd(cmd, env)
    print(f"\n>>> {run_name}\n{printable}", flush=True)

    started = time.time()
    timed_out = False
    if args.dry_run:
        rc = 0
    else:
        with stdout_path.open("w", encoding="utf-8") as fh:
            fh.write(f"# {printable}\n")
            fh.flush()
            try:
                cp = subprocess.run(
                    cmd,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    timeout=args.train_timeout_sec,
                )
                rc = cp.returncode
            except subprocess.TimeoutExpired:
                timed_out = True
                rc = 124
                fh.write(f"\n# timeout after {args.train_timeout_sec}s\n")
    elapsed = time.time() - started

    m = _estimate_metrics(
        progress_path=Path(paths["progress_jsonl"]),
        peak_probe_path=Path(paths["peak_probe_jsonl"]),
        metric_window=args.metric_step_window,
        peak_tflops=args.peak_tflops,
        spec=spec,
    )
    if timed_out:
        m = {
            **m,
            "run_end_status": "timeout",
            "run_end_error": f"timeout after {args.train_timeout_sec}s",
        }
    metrics = {
        "returncode": rc,
        "elapsed_sec": round(elapsed, 3),
        "steps_requested": args.steps,
        "gpu_index_physical": args.gpu_index,
        "peak_tflops_reference": args.peak_tflops,
        "timed_out": timed_out,
        "train_timeout_sec": args.train_timeout_sec,
        **m,
    }
    record = {
        "run_name": run_name,
        "arm": arm.name,
        "group": arm.group,
        "variant": arm.variant,
        "preset": arm.preset,
        "seed": seed,
        "note": arm.note,
        "cmd": cmd,
        "env": {k: env[k] for k in ["CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "PRESET"] if k in env},
        "gpu_rows_before": gpu_rows,
        "torch_child_visible_gpu": getattr(args, "_torch_child_visible_gpu", None),
        "model_spec": spec.__dict__,
        "metrics": metrics,
        "paths": paths,
    }
    summary_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    status = "OK" if rc == 0 else f"FAIL({rc})"
    print(
        f"<<< {run_name} {status} elapsed={elapsed:.1f}s "
        f"avg_step={metrics.get('avg_step_sec')}s "
        f"tflops={metrics.get('achieved_tflops')} "
        f"mfu={metrics.get('mfu')} "
        f"peak={metrics.get('peak_reserved_gb')}GB",
        flush=True,
    )
    if rc != 0 and args.stop_on_fail:
        raise SystemExit(rc)
    return record


def _write_index(output_root: str, records: list[dict], args: argparse.Namespace) -> Path:
    root = (REPO_ROOT / output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "runs.csv"
    fieldnames = [
        "run_name",
        "arm",
        "group",
        "variant",
        "preset",
        "seed",
        "returncode",
        "timed_out",
        "train_timeout_sec",
        "run_end_status",
        "elapsed_sec",
        "steps_completed",
        "avg_step_sec",
        "median_step_sec",
        "p90_step_sec",
        "token_count",
        "forward_flops",
        "train_step_flops",
        "achieved_tflops",
        "mfu",
        "peak_allocated_gb",
        "peak_reserved_gb",
        "progress_jsonl",
        "peak_probe_jsonl",
        "stdout",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            m = rec["metrics"]
            p = rec["paths"]
            shape = m.get("shape") or {}
            writer.writerow(
                {
                    "run_name": rec["run_name"],
                    "arm": rec["arm"],
                    "group": rec["group"],
                    "variant": rec["variant"],
                    "preset": rec["preset"],
                    "seed": rec["seed"],
                    "returncode": m["returncode"],
                    "timed_out": m["timed_out"],
                    "train_timeout_sec": m["train_timeout_sec"],
                    "run_end_status": m.get("run_end_status"),
                    "elapsed_sec": m["elapsed_sec"],
                    "steps_completed": m.get("steps_completed"),
                    "avg_step_sec": m.get("avg_step_sec"),
                    "median_step_sec": m.get("median_step_sec"),
                    "p90_step_sec": m.get("p90_step_sec"),
                    "token_count": shape.get("token_count"),
                    "forward_flops": m.get("forward_flops"),
                    "train_step_flops": m.get("train_step_flops"),
                    "achieved_tflops": m.get("achieved_tflops"),
                    "mfu": m.get("mfu"),
                    "peak_allocated_gb": m.get("peak_allocated_gb"),
                    "peak_reserved_gb": m.get("peak_reserved_gb"),
                    "progress_jsonl": p["progress_jsonl"],
                    "peak_probe_jsonl": p["peak_probe_jsonl"],
                    "stdout": p["stdout"],
                }
            )
    write_result(
        root,
        script=__file__,
        args=args,
        metrics={"num_runs": len(records), "runs_csv": str(csv_path)},
        artifacts=[csv_path],
        label="anima-mfu",
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
    p.add_argument("--peak-tflops", type=float, default=DEFAULT_PEAK_TFLOPS, help="Reference hardware peak throughput in TFLOPS for MFU normalization.")
    p.add_argument("--peak-probe-level", default=DEFAULT_PEAK_PROBE_LEVEL, choices=["block", "ops", "lokr", "full"])
    p.add_argument("--peak-probe-max-steps", type=int, default=1, help="How many early steps to record for shape recovery.")
    p.add_argument("--gpu-index", default=DEFAULT_GPU_INDEX, help="Physical GPU index for CUDA_VISIBLE_DEVICES. Default: 1.")
    p.add_argument("--allow-gpu0", action="store_true", help="Allow physical GPU 0.")
    p.add_argument("--min-vram-mb", type=int, default=DEFAULT_MIN_VRAM_MB)
    p.add_argument("--allow-low-vram", action="store_true")
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    p.add_argument("--output-root", default=DEFAULT_ROOT)
    p.add_argument("--train-timeout-sec", type=int, default=DEFAULT_TRAIN_TIMEOUT_SEC)
    p.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    p.add_argument(
        "--launch-mode",
        choices=[LAUNCH_MODE_TASKS_GUI, LAUNCH_MODE_DIRECT],
        default=LAUNCH_MODE_TASKS_GUI,
        help="tasks-gui = current wrapper path; direct = call train.py directly with a materialized config.",
    )
    p.add_argument(
        "--single-sample-smoke",
        action="store_true",
        help="Shortcut to use the cached single-sample smoke dataset config.",
    )
    p.add_argument("--sample-prompts", default=DEFAULT_PROMPTS)
    p.add_argument("--skip-preflight", action="store_true", help="Skip GPU checks. Useful for command generation only.")
    p.add_argument("--preflight-on-dry-run", action="store_true", help="Run GPU checks even with --dry-run.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stop-on-fail", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("extra", nargs=argparse.REMAINDER, help="Extra train.py args after --, forwarded to every run.")
    raw_argv = sys.argv[1:]
    output_root_explicit = _has_cli_option(raw_argv, "--output-root")
    args = p.parse_args()
    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]
    if not args.python.exists():
        raise SystemExit(f"python executable not found: {args.python}")
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.peak_tflops <= 0:
        raise SystemExit("--peak-tflops must be positive")
    if args.train_timeout_sec <= 0:
        raise SystemExit("--train-timeout-sec must be positive")
    args.output_root = _resolve_output_root(
        args.output_root,
        dry_run=args.dry_run,
        output_root_explicit=output_root_explicit,
    )
    if args.single_sample_smoke:
        args.dataset_config = DEFAULT_SINGLE_DATASET_CONFIG
    args.arms = _parse_arms(args.arms, args.suite)

    run_preflight = not args.skip_preflight and (not args.dry_run or args.preflight_on_dry_run)
    if run_preflight:
        gpu_rows = _check_gpu(args)
        torch_gpu = _verify_torch_mapping(args)
        args._torch_child_visible_gpu = torch_gpu
    else:
        gpu_rows = []
        torch_gpu = None
        args._torch_child_visible_gpu = None
    spec = AnimaModelSpec()

    if gpu_rows:
        print("GPU check:", json.dumps(gpu_rows, ensure_ascii=False), flush=True)
        print("Torch child visible GPU:", json.dumps(torch_gpu, ensure_ascii=False), flush=True)
    print("MFU model spec:", json.dumps(spec.__dict__, ensure_ascii=False), flush=True)
    if gpu_rows:
        print(
            f"Using physical GPU {args.gpu_index} via CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={args.gpu_index}",
            flush=True,
        )

    records: list[dict] = []
    for arm_name in args.arms:
        arm = ARMS[arm_name]
        for seed in args.seeds:
            records.append(_run_one(args, arm, seed, gpu_rows, spec))
    csv_path = _write_index(args.output_root, records, args)
    print(f"\nindex: {csv_path}")


if __name__ == "__main__":
    main()
