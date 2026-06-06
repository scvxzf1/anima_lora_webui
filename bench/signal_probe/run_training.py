#!/usr/bin/env python
"""Run short, repeatable Anima LoRA signal-probe training jobs.

The script is intentionally conservative about GPU selection. This workstation
has GPU 0 = GTX 1050 4GB and GPU 1 = RTX 3080 Ti 16GB; real training must use
physical GPU 1 unless explicitly overridden.

Examples:
  python -m bench.signal_probe.run_training --steps 8 --arms baseline --seeds 42
  python -m bench.signal_probe.run_training --steps 80 --arms baseline bias_p05 dir005
  python -m bench.signal_probe.run_training --dry-run --arms baseline bias_p05
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from bench._common import REPO_ROOT, write_result

DEFAULT_GPU_INDEX = "1"
DEFAULT_METHOD = "lora_signal_probe"
DEFAULT_PROMPTS = "configs/bench/signal_probe_prompts.txt"
DEFAULT_DATASET_CONFIG = "configs/bench/signal_probe_dataset.toml"
DEFAULT_ROOT = "output/bench/fasterdit_signal"
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


@dataclass(frozen=True)
class Arm:
    name: str
    extra: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""


ARMS: dict[str, Arm] = {
    "baseline": Arm("baseline", note="Default sigmoid timestep schedule."),
    "bias_m05": Arm("bias_m05", ("--sigmoid_bias", "-0.5")),
    "bias_p05": Arm("bias_p05", ("--sigmoid_bias", "0.5")),
    "bias_p10": Arm("bias_p10", ("--sigmoid_bias", "1.0")),
    "scale_075": Arm("scale_075", ("--sigmoid_scale", "0.75")),
    "scale_150": Arm("scale_150", ("--sigmoid_scale", "1.5")),
    # Future arms become active after the corresponding training flags land.
    "dir001": Arm("dir001", ("--velocity_direction_loss_weight", "0.01")),
    "dir003": Arm("dir003", ("--velocity_direction_loss_weight", "0.03")),
    "dir005": Arm("dir005", ("--velocity_direction_loss_weight", "0.05")),
    "dir010": Arm("dir010", ("--velocity_direction_loss_weight", "0.10")),
    "minsnr3": Arm("minsnr3", ("--weighting_scheme", "min_snr", "--min_snr_gamma", "3")),
    "minsnr5": Arm("minsnr5", ("--weighting_scheme", "min_snr", "--min_snr_gamma", "5")),
    "minsnr7": Arm("minsnr7", ("--weighting_scheme", "min_snr", "--min_snr_gamma", "7")),
    "p2g05": Arm("p2g05", ("--weighting_scheme", "p2", "--p2_gamma", "0.5")),
    "p2g10": Arm("p2g10", ("--weighting_scheme", "p2", "--p2_gamma", "1.0")),
}


def _parse_csv_ints(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return out


def _parse_arms(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                if part not in ARMS:
                    known = ", ".join(sorted(ARMS))
                    raise argparse.ArgumentTypeError(f"unknown arm {part!r}; known: {known}")
                out.append(part)
    return out or ["baseline"]


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
    if args.gpu_index == DEFAULT_GPU_INDEX:
        selected = next((r for r in rows if r.get("index") == args.gpu_index), None)
        if selected and "1050" in selected.get("name", ""):
            raise SystemExit(
                "refusing to train on GPU 1 because it looks like the GTX 1050; "
                "check nvidia-smi ordering before continuing"
            )
    if args.gpu_index == "0" and not args.allow_gpu0:
        raise SystemExit(
            "refusing to use physical GPU 0 without --allow-gpu0; "
            "this machine's GPU 0 is expected to be the 4GB GTX 1050"
        )
    selected = next((r for r in rows if r.get("index") == args.gpu_index), None)
    if selected:
        total = int(float(selected.get("memory_total_mb", "0") or 0))
        if total < args.min_vram_mb and not args.allow_low_vram:
            raise SystemExit(
                f"refusing to train on GPU {args.gpu_index} ({selected.get('name')}) "
                f"with only {total} MB VRAM; pass --allow-low-vram to override"
            )
    elif rows:
        raise SystemExit(f"GPU index {args.gpu_index} not found in nvidia-smi rows: {rows}")
    return rows



def _verify_torch_mapping(args: argparse.Namespace) -> dict[str, str] | None:
    """Verify the child process will see exactly the intended training GPU."""
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

def _latest_step_event(progress_path: Path) -> dict | None:
    if not progress_path.exists():
        return None
    last = None
    for line in progress_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("ev") == "step":
            last = rec
    return last


def _run_one(args: argparse.Namespace, arm: Arm, seed: int, gpu_rows: list[dict[str, str]]) -> dict:
    run_name = f"{arm.name}_s{seed}_{args.steps}step"
    root = (REPO_ROOT / args.output_root).resolve()
    run_dir = root / run_name
    ckpt_dir = run_dir / "ckpt"
    log_dir = run_dir / "logs"
    progress_path = log_dir / f"{run_name}.progress.jsonl"
    stdout_path = log_dir / "train.stdout.log"
    summary_path = run_dir / "summary.json"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(args.python),
        "tasks.py",
        "lora-gui",
        args.method,
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
        "--sample_every_n_steps",
        str(args.sample_every_n_steps),
        "--sample_sampler",
        args.sample_sampler,
        "--log_every_n_steps",
        "1",
        *arm.extra,
        *args.extra,
    ]
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_index
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    printable = " ".join(
        f"{k}={shlex.quote(v)}"
        for k, v in {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": args.gpu_index}.items()
    )
    printable += " " + " ".join(shlex.quote(c) for c in cmd)
    print(f"\n>>> {run_name}\n{printable}", flush=True)

    started = time.time()
    rc = None
    if args.dry_run:
        rc = 0
    else:
        with stdout_path.open("w", encoding="utf-8") as fh:
            fh.write(f"# {printable}\n")
            fh.flush()
            cp = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
            rc = cp.returncode
    elapsed = time.time() - started
    last_step = _latest_step_event(progress_path)
    metrics = {
        "returncode": rc,
        "elapsed_sec": round(elapsed, 3),
        "steps_requested": args.steps,
        "last_step": last_step,
        "loss": None if last_step is None else last_step.get("loss"),
        "avr_loss": None if last_step is None else last_step.get("avr_loss"),
        "gpu_index_physical": args.gpu_index,
    }
    record = {
        "run_name": run_name,
        "arm": arm.name,
        "seed": seed,
        "cmd": cmd,
        "env": {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": args.gpu_index},
        "gpu_rows_before": gpu_rows,
        "torch_child_visible_gpu": getattr(args, "_torch_child_visible_gpu", None),
        "metrics": metrics,
        "paths": {
            "run_dir": str(run_dir),
            "ckpt_dir": str(ckpt_dir),
            "progress_jsonl": str(progress_path),
            "stdout": str(stdout_path),
        },
    }
    summary_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    status = "OK" if rc == 0 else f"FAIL({rc})"
    print(
        f"<<< {run_name} {status} elapsed={elapsed:.1f}s "
        f"loss={metrics['loss']} avr_loss={metrics['avr_loss']}",
        flush=True,
    )
    if rc != 0 and args.stop_on_fail:
        raise SystemExit(rc)
    return record


def _write_index(output_root: str, records: list[dict], args: argparse.Namespace) -> Path:
    root = (REPO_ROOT / output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / "runs.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "run_name",
                "arm",
                "seed",
                "returncode",
                "elapsed_sec",
                "loss",
                "avr_loss",
                "progress_jsonl",
                "stdout",
            ],
        )
        writer.writeheader()
        for rec in records:
            m = rec["metrics"]
            p = rec["paths"]
            writer.writerow(
                {
                    "run_name": rec["run_name"],
                    "arm": rec["arm"],
                    "seed": rec["seed"],
                    "returncode": m["returncode"],
                    "elapsed_sec": m["elapsed_sec"],
                    "loss": m["loss"],
                    "avr_loss": m["avr_loss"],
                    "progress_jsonl": p["progress_jsonl"],
                    "stdout": p["stdout"],
                }
            )
    write_result(
        root,
        script=__file__,
        args=args,
        metrics={"num_runs": len(records), "runs_csv": str(csv_path)},
        artifacts=[csv_path],
        label="signal-probe-training",
        device=None,
        extra={"records": records},
    )
    return csv_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", default=["baseline"], help=f"Arms to run. Known: {', '.join(sorted(ARMS))}")
    p.add_argument("--seeds", type=_parse_csv_ints, default=[42], help="Comma-separated seeds, e.g. 42,43,44")
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--gpu-index", default=DEFAULT_GPU_INDEX, help="Physical GPU index for CUDA_VISIBLE_DEVICES. Default: 1 (RTX 3080 Ti on this host).")
    p.add_argument("--allow-gpu0", action="store_true", help="Allow physical GPU 0. Normally refused because it is the 4GB GTX 1050 on this host.")
    p.add_argument("--min-vram-mb", type=int, default=12000)
    p.add_argument("--allow-low-vram", action="store_true")
    p.add_argument("--method", default=DEFAULT_METHOD)
    p.add_argument("--python", type=Path, default=DEFAULT_PYTHON, help="Python executable for training. Defaults to the project .venv.")
    p.add_argument("--output-root", default=DEFAULT_ROOT)
    p.add_argument("--sample-prompts", default=DEFAULT_PROMPTS)
    p.add_argument("--dataset-config", default=DEFAULT_DATASET_CONFIG)
    p.add_argument("--sample-every-n-steps", type=int, default=40)
    p.add_argument("--sample-sampler", default="euler")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stop-on-fail", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("extra", nargs=argparse.REMAINDER, help="Extra train.py args after --, forwarded to every run.")
    args = p.parse_args()
    if not args.python.exists():
        raise SystemExit(f"python executable not found: {args.python}")
    if args.extra and args.extra[0] == "--":
        args.extra = args.extra[1:]
    args.arms = _parse_arms(args.arms)

    gpu_rows = _check_gpu(args)
    torch_gpu = _verify_torch_mapping(args)
    args._torch_child_visible_gpu = torch_gpu
    print("GPU check:", json.dumps(gpu_rows, ensure_ascii=False), flush=True)
    print("Torch child visible GPU:", json.dumps(torch_gpu, ensure_ascii=False), flush=True)
    print(
        f"Using physical GPU {args.gpu_index} via "
        f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={args.gpu_index}",
        flush=True,
    )

    records: list[dict] = []
    for arm_name in args.arms:
        arm = ARMS[arm_name]
        for seed in args.seeds:
            records.append(_run_one(args, arm, seed, gpu_rows))
    csv_path = _write_index(args.output_root, records, args)
    print(f"\nindex: {csv_path}")


if __name__ == "__main__":
    main()
