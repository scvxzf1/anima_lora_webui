#!/usr/bin/env python
"""Run short training hot tests for arbitrary methods or GUI variants.

This is the small, general-purpose harness for "can I launch it and measure a
few steps?" checks. It intentionally defaults to non-LoKr cases so LoKr-specific
optimization work can proceed independently.

Examples:
  python -m bench.training_hot.run_matrix --dry-run
  python -m bench.training_hot.run_matrix --dry-run --suite compat_runtime
  python -m bench.training_hot.run_matrix --dry-run --suite plugins_nonlokr
  python -m bench.training_hot.run_matrix --steps 12 --case gui:loha
  python -m bench.training_hot.run_matrix --steps 12 --case method:lora:balanced_16g
  python -m bench.training_hot.run_matrix --steps 12 --case config:output/runs/x/config.runtime.toml
  python -m bench.training_hot.run_matrix --steps 12 --case gui:lora -- --torch_compile
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from bench._common import REPO_ROOT, write_result
from bench.plain_lora_speed.run_matrix import (
    DEFAULT_DATASET_CONFIG,
    DEFAULT_GPU_INDEX,
    DEFAULT_MIN_VRAM_MB,
    DEFAULT_PROMPTS,
    DEFAULT_PYTHON,
    _check_dataset_cache,
    _check_gpu,
    _parse_csv_ints,
    _parse_step_window,
    _repo_path,
    _summarize_progress,
    _verify_torch_mapping,
)

DEFAULT_ROOT = "output/bench/training_hot"


@dataclass(frozen=True)
class HotCase:
    name: str
    mode: str
    target: str
    preset: str = "default"
    methods_subdir: str = "methods"
    extra: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""


CASES: dict[str, HotCase] = {
    "gui_lora": HotCase(
        "gui_lora",
        "variant",
        "lora",
        note="GUI plain LoRA smoke path.",
    ),
    "method_lora": HotCase(
        "method_lora",
        "method",
        "lora",
        note="configs/methods/lora.toml smoke path.",
    ),
    "gui_loha": HotCase(
        "gui_loha",
        "variant",
        "loha",
        note="LoHa plugin smoke path.",
    ),
    "gui_glora": HotCase(
        "gui_glora",
        "variant",
        "glora",
        note="GLoRA plugin smoke path.",
    ),
    "gui_vera": HotCase(
        "gui_vera",
        "variant",
        "vera",
        note="VeRA plugin smoke path.",
    ),
    "compat_blockswap_grad_ckpt": HotCase(
        "compat_blockswap_grad_ckpt",
        "variant",
        "lora",
        extra=("--blocks_to_swap", "8", "--gradient_checkpointing"),
        note="Compatibility hot path: block swap + standard full gradient checkpointing.",
    ),
    "compat_blockswap_selective_mlp": HotCase(
        "compat_blockswap_selective_mlp",
        "variant",
        "lora",
        extra=("--blocks_to_swap", "8", "--selective_checkpoint", "mlp_only"),
        note="Compatibility hot path: block swap + selective MLP checkpointing.",
    ),
    "compat_blockswap_cudagraphs": HotCase(
        "compat_blockswap_cudagraphs",
        "variant",
        "lora",
        extra=(
            "--blocks_to_swap",
            "8",
            "--torch_compile",
            "--dynamo_backend",
            "cudagraphs",
        ),
        note="Compatibility hot path: block swap disables unsafe CUDAGraph compile backend.",
    ),
    "compat_blockswap_max_autotune": HotCase(
        "compat_blockswap_max_autotune",
        "variant",
        "lora",
        extra=(
            "--blocks_to_swap",
            "8",
            "--torch_compile",
            "--compile_inductor_mode",
            "max-autotune",
        ),
        note="Compatibility hot path: block swap downgrades max-autotune to no-cudagraphs.",
    ),
}

SUITES: dict[str, tuple[str, ...]] = {
    "baseline": ("gui_lora",),
    "lora_paths": ("gui_lora", "method_lora"),
    "plugins_nonlokr": ("gui_loha", "gui_glora", "gui_vera"),
    "compat_runtime": (
        "compat_blockswap_grad_ckpt",
        "compat_blockswap_selective_mlp",
        "compat_blockswap_cudagraphs",
        "compat_blockswap_max_autotune",
    ),
    "all_nonlokr": ("gui_lora", "method_lora", "gui_loha", "gui_glora", "gui_vera"),
}


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return text or "case"


def _case_from_spec(text: str) -> HotCase:
    if text in CASES:
        return CASES[text]

    head, sep, tail = text.partition(":")
    if not sep:
        known = ", ".join(sorted(CASES))
        raise argparse.ArgumentTypeError(
            f"unknown case {text!r}; use a built-in case ({known}) or gui:/method:/config:"
        )

    mode = head.strip().lower()
    parts = tail.split(":")
    if mode in {"gui", "variant"}:
        if not parts[0]:
            raise argparse.ArgumentTypeError("gui case must be gui:<variant>[:preset]")
        variant = parts[0]
        preset = parts[1] if len(parts) > 1 and parts[1] else "default"
        return HotCase(
            name=f"gui_{_safe_name(variant)}_{_safe_name(preset)}",
            mode="variant",
            target=variant,
            preset=preset,
            note="Custom GUI variant case.",
        )
    if mode == "method":
        if not parts[0]:
            raise argparse.ArgumentTypeError("method case must be method:<name>[:preset[:methods_subdir]]")
        method = parts[0]
        preset = parts[1] if len(parts) > 1 and parts[1] else "default"
        methods_subdir = parts[2] if len(parts) > 2 and parts[2] else "methods"
        return HotCase(
            name=f"method_{_safe_name(method)}_{_safe_name(preset)}",
            mode="method",
            target=method,
            preset=preset,
            methods_subdir=methods_subdir,
            note="Custom method case.",
        )
    if mode in {"config", "config-file", "config_file"}:
        if not parts[0]:
            raise argparse.ArgumentTypeError("config case must be config:<path>")
        path = tail
        return HotCase(
            name=f"config_{_safe_name(Path(path).stem)}",
            mode="config",
            target=path,
            note="Direct --config_file case.",
        )
    raise argparse.ArgumentTypeError(f"unknown case mode {head!r}; expected gui, method, or config")


def _parse_cases(values: Iterable[str] | None, suite: str | None) -> list[HotCase]:
    out: list[HotCase] = []
    if values:
        for value in values:
            for part in value.split(","):
                part = part.strip()
                if part:
                    out.append(_case_from_spec(part))
    if out:
        return out

    suite = suite or "baseline"
    if suite not in SUITES:
        known = ", ".join(sorted(SUITES))
        raise argparse.ArgumentTypeError(f"unknown suite {suite!r}; known: {known}")
    return [CASES[name] for name in SUITES[suite]]


def _common_train_args(args: argparse.Namespace, seed: int, paths: dict[str, str]) -> list[str]:
    train_args = [
        "--max_train_steps",
        str(args.steps),
        "--seed",
        str(seed),
        "--output_dir",
        paths["ckpt_dir"],
        "--output_name",
        paths["run_name"],
        "--progress_jsonl",
        paths["progress_jsonl"],
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
    ]
    if args.profile_steps:
        train_args += ["--profile_steps", args.profile_steps]
    if args.memory_probe:
        train_args += [
            "--memory_probe_jsonl",
            paths["memory_probe_jsonl"],
            "--memory_probe_max_steps",
            str(args.memory_probe_max_steps),
        ]
    if args.block_swap_profile:
        train_args += ["--block_swap_profile_jsonl", paths["block_swap_profile_jsonl"]]
    return train_args


def _build_train_cmd(
    args: argparse.Namespace,
    case: HotCase,
    seed: int,
    run_dir: Path,
    run_name: str,
) -> tuple[list[str], dict[str, str], dict[str, str]]:
    ckpt_dir = run_dir / "ckpt"
    log_dir = run_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "ckpt_dir": str(ckpt_dir),
        "logs_dir": str(log_dir),
        "progress_jsonl": str(log_dir / f"{run_name}.progress.jsonl"),
        "memory_probe_jsonl": str(log_dir / f"{run_name}.memory_probe.jsonl"),
        "block_swap_profile_jsonl": str(log_dir / f"{run_name}.block_swap_profile.jsonl"),
        "stdout": str(log_dir / "train.stdout.log"),
        "summary": str(run_dir / "summary.json"),
    }
    common = _common_train_args(args, seed, paths)

    if case.mode == "variant":
        cmd = [str(args.python), "tasks.py", "lora-gui", case.target, *common]
    elif case.mode == "method":
        cmd = [
            str(args.python),
            "train.py",
            "--method",
            case.target,
            "--preset",
            case.preset,
            "--methods_subdir",
            case.methods_subdir,
            *common,
        ]
    elif case.mode == "config":
        cmd = [str(args.python), "train.py", "--config_file", case.target, *common]
    else:  # pragma: no cover - guarded by parser and built-ins
        raise ValueError(f"unsupported hot case mode: {case.mode}")

    cmd += [*case.extra, *args.extra]

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    env["PRESET"] = case.preset
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return cmd, env, paths


def _printable_cmd(cmd: list[str], env: dict[str, str]) -> str:
    env_keys = ["CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "PRESET"]
    prefix = " ".join(f"{key}={shlex.quote(str(env[key]))}" for key in env_keys if key in env)
    return prefix + " " + " ".join(shlex.quote(part) for part in cmd)


def _dir_size(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    total = 0
    count = 0
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        count += 1
        try:
            total += item.stat().st_size
        except OSError:
            continue
    return total, count


def _run_one(
    args: argparse.Namespace,
    case: HotCase,
    seed: int,
    gpu_rows: list[dict[str, str]],
    cache_rows: list[dict],
) -> dict:
    run_name = f"{case.name}_s{seed}_{args.steps}step"
    run_dir = _repo_path(args.output_root).resolve() / run_name
    cmd, env, paths = _build_train_cmd(args, case, seed, run_dir, run_name)
    printable = _printable_cmd(cmd, env)
    stdout_path = Path(paths["stdout"])

    print(f"\n>>> {run_name}\n{printable}", flush=True)
    started = time.time()
    if args.dry_run:
        stdout_path.write_text(f"# dry-run\n# {printable}\n", encoding="utf-8")
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
    checkpoint_bytes, checkpoint_file_count = _dir_size(Path(paths["ckpt_dir"]))
    metrics = {
        "returncode": rc,
        "elapsed_sec": round(elapsed, 3),
        "steps_requested": args.steps,
        "gpu_index_physical": args.gpu_index,
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_file_count": checkpoint_file_count,
        **progress_metrics,
    }
    record = {
        "run_name": run_name,
        "case": {
            "name": case.name,
            "mode": case.mode,
            "target": case.target,
            "preset": case.preset,
            "methods_subdir": case.methods_subdir,
            "note": case.note,
        },
        "seed": seed,
        "cmd": cmd,
        "env": {
            key: env[key]
            for key in ["CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES", "PRESET"]
            if key in env
        },
        "gpu_rows_before": gpu_rows,
        "torch_child_visible_gpu": getattr(args, "_torch_child_visible_gpu", None),
        "cache_preflight": cache_rows,
        "metrics": metrics,
        "paths": paths,
    }
    Path(paths["summary"]).write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
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
        "case_name",
        "mode",
        "target",
        "preset",
        "methods_subdir",
        "seed",
        "returncode",
        "elapsed_sec",
        "steps_completed",
        "avg_step_sec",
        "median_step_sec",
        "p90_step_sec",
        "images_per_hour",
        "peak_allocated_gb",
        "peak_reserved_gb",
        "checkpoint_bytes",
        "checkpoint_file_count",
        "loss",
        "avr_loss",
        "progress_jsonl",
        "stdout",
        "memory_probe_jsonl",
        "block_swap_profile_jsonl",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            metrics = rec["metrics"]
            paths = rec["paths"]
            case = rec["case"]
            writer.writerow(
                {
                    "run_name": rec["run_name"],
                    "case_name": case["name"],
                    "mode": case["mode"],
                    "target": case["target"],
                    "preset": case["preset"],
                    "methods_subdir": case["methods_subdir"],
                    "seed": rec["seed"],
                    "returncode": metrics["returncode"],
                    "elapsed_sec": metrics["elapsed_sec"],
                    "steps_completed": metrics["steps_completed"],
                    "avg_step_sec": metrics["avg_step_sec"],
                    "median_step_sec": metrics["median_step_sec"],
                    "p90_step_sec": metrics["p90_step_sec"],
                    "images_per_hour": metrics["images_per_hour"],
                    "peak_allocated_gb": metrics["peak_allocated_gb"],
                    "peak_reserved_gb": metrics["peak_reserved_gb"],
                    "checkpoint_bytes": metrics["checkpoint_bytes"],
                    "checkpoint_file_count": metrics["checkpoint_file_count"],
                    "loss": metrics["loss"],
                    "avr_loss": metrics["avr_loss"],
                    "progress_jsonl": paths["progress_jsonl"],
                    "stdout": paths["stdout"],
                    "memory_probe_jsonl": paths["memory_probe_jsonl"],
                    "block_swap_profile_jsonl": paths["block_swap_profile_jsonl"],
                }
            )
    write_result(
        root,
        script=__file__,
        args=args,
        metrics={"num_runs": len(records), "runs_csv": str(csv_path)},
        artifacts=[csv_path],
        label="training-hot-matrix",
        device=None,
        extra={"records": records},
    )
    return csv_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", choices=sorted(SUITES), default="baseline")
    p.add_argument("--case", action="append", default=None, help="Case spec or built-in name. Repeatable, comma-separated OK.")
    p.add_argument("--seeds", type=_parse_csv_ints, default=[42], help="Comma-separated seeds, e.g. 42,43.")
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--metric-step-window", type=_parse_step_window, default=None, help="Step interval window, e.g. 3-10; off=all.")
    p.add_argument("--images-per-step", type=float, default=1.0)
    p.add_argument("--profile-steps", default=None, help="Forward --profile_steps START-END to train.py.")
    p.add_argument("--memory-probe", action="store_true", help="Write memory_probe_jsonl for each run.")
    p.add_argument("--memory-probe-max-steps", type=int, default=2)
    p.add_argument("--block-swap-profile", action="store_true", help="Write block_swap_profile_jsonl for each run.")
    p.add_argument("--gpu-index", default=os.environ.get("BENCH_GPU_INDEX", DEFAULT_GPU_INDEX))
    p.add_argument("--allow-gpu0", action="store_true")
    p.add_argument("--min-vram-mb", type=int, default=DEFAULT_MIN_VRAM_MB)
    p.add_argument("--allow-low-vram", action="store_true")
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    p.add_argument("--output-root", default=DEFAULT_ROOT)
    p.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    p.add_argument("--sample-prompts", default=DEFAULT_PROMPTS)
    p.add_argument("--allow-missing-cache", action="store_true")
    p.add_argument("--skip-preflight", action="store_true", help="Skip cache/GPU checks. Useful for command generation only.")
    p.add_argument("--preflight-on-dry-run", action="store_true", help="Run cache/GPU checks even with --dry-run.")
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

    cases = _parse_cases(args.case, args.suite)
    run_preflight = not args.skip_preflight and (not args.dry_run or args.preflight_on_dry_run)
    if run_preflight:
        cache_rows = _check_dataset_cache(args)
        gpu_rows = _check_gpu(args)
        torch_gpu = _verify_torch_mapping(args)
        args._torch_child_visible_gpu = torch_gpu
    else:
        cache_rows = []
        gpu_rows = []
        args._torch_child_visible_gpu = None

    if cache_rows:
        print("Cache preflight:", json.dumps(cache_rows, ensure_ascii=False), flush=True)
    if gpu_rows:
        print("GPU check:", json.dumps(gpu_rows, ensure_ascii=False), flush=True)
        print(
            "Torch child visible GPU:",
            json.dumps(args._torch_child_visible_gpu, ensure_ascii=False),
            flush=True,
        )

    records: list[dict] = []
    for case in cases:
        for seed in args.seeds:
            records.append(_run_one(args, case, seed, gpu_rows, cache_rows))
    csv_path = _write_index(args.output_root, records, args)
    print(f"\nindex: {csv_path}")


if __name__ == "__main__":
    main()
