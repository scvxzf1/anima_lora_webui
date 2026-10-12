"""Opt-in, fail-closed Qwen Image 2.1 adapter hot acceptance on GPU 1."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import statistics
import subprocess
import sys
import time

import toml
import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "output/runs/qwen21-stage3-acceptance-20260928/queue"
BASE_CONFIG = REFERENCE / "perf_final_r1_off_torch_s24_n20/attempt-001/config.toml"
DATASET = REFERENCE / "dataset.toml"
DEFAULT_OUTPUT = ROOT / "output/runs/qwen21-variant-hot-acceptance"
VARIANTS = ("lora", "lokr", "loha")
SOURCE_FILES = (
    Path(__file__).resolve(),
    ROOT / "train.py",
    ROOT / "library/config/io.py",
    ROOT / "library/config/schema.py",
    ROOT / "library/models/family_registry.py",
    *sorted((ROOT / "library/models/qwen_image_2_1").glob("*.py")),
    ROOT / "library/training/bootstrap.py",
    ROOT / "library/training/train_session.py",
    ROOT / "library/training/loop.py",
    ROOT / "library/training/checkpoints.py",
    ROOT / "library/training/metadata.py",
    ROOT / "library/training/progress.py",
    ROOT / "library/training/cli_args.py",
    ROOT / "library/training/compat_matrix.py",
    ROOT / "library/runtime/harness.py",
    ROOT / "library/runtime/dynamo.py",
    ROOT / "networks/registry.py",
    ROOT / "networks/registry_api.py",
    ROOT / "networks/core_specs.py",
    ROOT / "networks/lora_save.py",
    ROOT / "networks/plugins/__init__.py",
    *sorted((ROOT / "networks/lora_anima").glob("*.py")),
    *sorted((ROOT / "networks/plugins/lokr").glob("*.py")),
    *sorted((ROOT / "networks/plugins/loha").glob("*.py")),
)
MIN_DISK_GIB = 20
MIN_FREE_GPU_MIB = 9000
TIMEOUT_SECONDS = {"smoke": 2400, "hot": 5400}
GPU_FIELDS = ("timestamp", "temperature.gpu", "clocks.current.sm",
              "clocks.current.memory", "memory.used", "power.draw", "utilization.gpu")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest() -> dict[str, str]:
    return {str(path.relative_to(ROOT)): digest(path) for path in SOURCE_FILES}


def file_fingerprint(path: Path) -> dict:
    stat = path.stat()
    with path.open("rb") as stream:
        head = stream.read(1024 * 1024)
        stream.seek(max(0, stat.st_size - 1024 * 1024))
        tail = stream.read(1024 * 1024)
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "head_sha256": hashlib.sha256(head).hexdigest(),
            "tail_sha256": hashlib.sha256(tail).hexdigest()}


def input_manifest(config: dict) -> dict:
    paths = [Path(config[key]) for key in ("pretrained_model_name_or_path", "qwen3", "vae")]
    dataset = toml.load(DATASET)
    for subset in dataset["datasets"][0]["subsets"]:
        image_dir = Path(subset["image_dir"])
        image = image_dir / subset["path_pattern"]
        paths.append(image)
        caption = image.with_suffix(subset.get("caption_extension", ".txt"))
        if caption.is_file():
            paths.append(caption)
        paths.extend(sorted(path for path in Path(subset["cache_dir"]).iterdir()
                            if path.is_file()))
    return {str(path): file_fingerprint(path) for path in paths}


def load_reference() -> dict:
    config = toml.load(BASE_CONFIG)
    dataset = toml.load(DATASET)
    if (config.get("model_family") != "qwen_image_2_1"
            or config.get("network_module") != "networks.lora_anima"
            or config.get("dataset_config") != str(DATASET)
            or len(dataset.get("datasets", [])) != 1
            or config.get("seed") != 1234):
        raise ValueError("Stage-3 reference contract changed")
    for key, expected in {
        "mixed_precision": "bf16", "base_compute": "bf16",
        "gradient_checkpointing": True, "blocks_to_swap": 24,
        "torch_compile": True, "attn_mode": "torch",
        "qwen_fused_projections": "off",
        "selective_checkpoint": "off", "selective_checkpoint_blocks": "",
        "cpu_offload_checkpointing": False,
        "unsloth_offload_checkpointing": False,
    }.items():
        if config.get(key) != expected:
            raise ValueError(f"Reference {key} changed: {config.get(key)!r}")
    for key in ("pretrained_model_name_or_path", "qwen3", "vae"):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(config[key])
    for subset in dataset["datasets"][0]["subsets"]:
        image_dir = Path(subset["image_dir"])
        if not image_dir.is_dir() or not (image_dir / subset["path_pattern"]).is_file():
            raise FileNotFoundError(f"Stage-3 source image missing: {image_dir}")
        cache = Path(subset["cache_dir"])
        if not cache.is_dir() or not any(cache.iterdir()):
            raise FileNotFoundError(f"Stage-3 cache missing: {cache}")
    return config


def make_config(base: dict, variant: str, phase: str, directory: Path) -> dict:
    if variant not in VARIANTS or phase not in ("smoke", "hot"):
        raise ValueError("Unsupported variant or phase")
    config = dict(base)
    config.update(
        dataset_config=str(directory.parent / "dataset.toml"),
        lora_cache_dir=str(directory.parent / "lora_cache"),
        output_dir=str(directory / "checkpoints"),
        output_name=f"qwen21_{variant}_{phase}",
        logging_dir=str(directory / "logs"),
        progress_jsonl=str(directory / "progress.jsonl"),
        block_swap_profile_jsonl=str(directory / "block_swap.jsonl"),
        max_train_steps=2 if phase == "smoke" else 20,
        network_args=[] if variant == "lora" else [f"use_{variant}=true"],
        memory_probe_jsonl="off",
        peak_probe_jsonl="off",
        gradient_flow_probe_jsonl="off",
        save_state=False,
        save_state_on_train_end=False,
        sample_at_first=False,
    )
    config.pop("resume", None)
    config.pop("save_every_n_steps", None)
    return config


def copy_reference_cache(root: Path) -> None:
    dataset = toml.load(DATASET)
    destination = root / "cache"
    destination.mkdir(exist_ok=False)
    for subset in dataset["datasets"][0]["subsets"]:
        original = Path(subset["cache_dir"])
        for item in original.iterdir():
            if not item.is_file():
                raise RuntimeError(f"Unexpected Stage-3 cache entry: {item}")
            if (destination / item.name).exists():
                raise RuntimeError(f"Duplicate Stage-3 cache filename: {item.name}")
            shutil.copy2(item, destination / item.name)
            if digest(item) != digest(destination / item.name):
                raise RuntimeError(f"Stage-3 cache copy differs: {item.name}")
        subset["cache_dir"] = str(destination)
    (root / "dataset.toml").write_text(toml.dumps(dataset))


def command(directory: Path) -> list[str]:
    return [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "train.py"),
            "--config_file", str(directory / "config.toml")]


def gpu_preflight() -> dict:
    gpu = subprocess.run(
        ["nvidia-smi", "-i", "1", "--query-gpu=name,memory.total,memory.free",
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True,
    ).stdout.strip().split(",")
    if len(gpu) != 3 or "CMP 90HX" not in gpu[0]:
        raise RuntimeError(f"GPU 1 is not CMP 90HX: {gpu!r}")
    if int(gpu[2].strip()) < MIN_FREE_GPU_MIB:
        raise RuntimeError(f"GPU 1 free VRAM below {MIN_FREE_GPU_MIB} MiB: {gpu!r}")
    apps = subprocess.run(
        ["nvidia-smi", "-i", "1", "--query-compute-apps=pid",
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True,
    ).stdout.strip()
    if apps and apps != "No running processes found":
        raise RuntimeError(f"GPU 1 already has compute PIDs: {apps}")
    return {"name": gpu[0].strip(), "total_mib": int(gpu[1]),
            "free_mib": int(gpu[2]), "compute_pids": []}


def disk_preflight(output: Path) -> dict:
    anchor = output
    while not anchor.exists():
        anchor = anchor.parent
    free = shutil.disk_usage(anchor).free
    if free < MIN_DISK_GIB * 1024**3:
        raise RuntimeError(f"Free disk below {MIN_DISK_GIB} GiB: {anchor}")
    return {"free_gib": round(free / 1024**3, 2), "mount_path": str(anchor)}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def summarize(progress: Path) -> dict:
    steps = {}
    for line in progress.read_text().splitlines():
        row = json.loads(line)
        if row.get("ev") == "step" and "loss/current" in row:
            steps[int(row["global_step"])] = row
    expected = range(6, 21)
    if not all(step in steps and step - 1 in steps for step in expected):
        raise RuntimeError("Hot run missing progress steps 5..20")
    seconds = {step: steps[step]["ts"] - steps[step - 1]["ts"] for step in expected}
    values = list(seconds.values())
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise RuntimeError("Nonpositive step interval")
    return {"included_steps": list(expected), "step_seconds": seconds,
            "median_s": statistics.median(values), "mean_s": statistics.mean(values),
            "first5_mean_s": statistics.mean(values[:5]),
            "last5_mean_s": statistics.mean(values[-5:]),
            "peak_allocated_gib": max(steps[s].get("cuda/max_memory_allocated_gb", 0)
                                      for s in expected)}


def verify_progress(progress: Path, count: int) -> None:
    rows = [json.loads(line) for line in progress.read_text().splitlines()]
    # The progress sink may emit a final status-only step row after the last loss.
    steps = [row for row in rows if row.get("ev") == "step" and "loss/current" in row]
    numbers = [row.get("global_step") for row in steps]
    if numbers != list(range(1, count + 1)):
        raise RuntimeError(f"Expected exact progress steps 1..{count}: {numbers}")
    status_rows = [row for row in rows if row.get("ev") == "step" and "loss/current" not in row]
    if any(row.get("global_step") != count for row in status_rows) or len(status_rows) > 1:
        raise RuntimeError(f"Unexpected status-only step rows: {status_rows}")
    for row in steps:
        loss = row.get("loss/current")
        if not isinstance(loss, (int, float)) or not math.isfinite(loss):
            raise RuntimeError(f"Nonfinite or missing loss at step {row['global_step']}")
    ends = [row for row in rows if row.get("ev") == "run_end"]
    if len(ends) != 1 or rows[-1] != ends[0] or ends[0].get("status") != "ok" or ends[0].get("final_step") != count:
        raise RuntimeError(f"Missing successful run_end at step {count}: {progress}")


def verify_smoke(progress: Path) -> None:
    verify_progress(progress, 2)


def verify_checkpoint(path: Path, variant: str) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Missing checkpoint: {path}")
    with safe_open(path, framework="pt", device="cpu") as reader:
        metadata = reader.metadata() or {}
        if metadata.get("ss_network_spec") != variant:
            raise RuntimeError(f"Wrong checkpoint network spec: {metadata.get('ss_network_spec')!r}")
        keys = list(reader.keys())
        suffixes = {
            "lora": (".lora_down.weight", ".lora_up.weight"),
            "lokr": (".lokr_w1", ".lokr_w2", ".lokr_w2_a", ".lokr_w2_b"),
            "loha": (".hada_w1_a", ".hada_w1_b", ".hada_w2_a", ".hada_w2_b"),
        }
        if any(key.endswith(other_suffixes) for key in keys
               for name, other_suffixes in suffixes.items() if name != variant):
            raise RuntimeError(f"Checkpoint contains another adapter variant: {path}")
        groups = {}
        for key in keys:
            for suffix in suffixes[variant]:
                if key.endswith(suffix):
                    groups.setdefault(key.removesuffix(suffix), set()).add(suffix)
                    break
        valid = bool(groups)
        for found in groups.values():
            if variant == "lokr":
                valid &= ".lokr_w1" in found and (
                    ".lokr_w2" in found or {".lokr_w2_a", ".lokr_w2_b"} <= found)
            else:
                valid &= set(suffixes[variant]) <= found
        if not valid:
            raise RuntimeError(f"Missing {variant} checkpoint tensors: {path}")
        for key in keys:
            tensor = reader.get_tensor(key)
            if not torch.is_floating_point(tensor) or not torch.isfinite(tensor).all() or not torch.count_nonzero(tensor):
                raise RuntimeError(f"Invalid trainable checkpoint tensor: {key}")
    return {"path": str(path), "size": path.stat().st_size, "tensor_count": len(keys),
            "network_spec": variant}


def verify_telemetry(path: Path, phase: str) -> dict:
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    minimum = 3 if phase == "smoke" else 10
    if len(rows) < minimum:
        raise RuntimeError(f"Insufficient GPU telemetry: {len(rows)} < {minimum}")
    timestamps = []
    used_memory = []
    for row in rows:
        if len(row) != len(GPU_FIELDS):
            raise RuntimeError(f"Malformed GPU telemetry row: {row!r}")
        stamp = row[0].strip()
        try:
            timestamps.append(datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f"))
        except ValueError:
            timestamps.append(datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S"))
        values = [float(value.strip()) for value in row[1:]]
        if (any(not math.isfinite(value) for value in values)
                or any(value <= 0 for value in values[:3]) or values[3] < 0):
            raise RuntimeError(f"Invalid GPU telemetry values: {row!r}")
        used_memory.append(values[3])
    if not any(value > 0 for value in used_memory):
        raise RuntimeError("GPU telemetry never observed allocated memory")
    if any((right - left).total_seconds() <= 0 or (right - left).total_seconds() > 5
           for left, right in zip(timestamps, timestamps[1:])):
        raise RuntimeError("GPU telemetry has missing or non-monotonic samples")
    return {"samples": len(rows), "first": rows[0][0].strip(), "last": rows[-1][0].strip()}


def _stop_group(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            continue
        else:
            if sig == signal.SIGTERM:
                # The group can still contain children after its leader exits.
                continue
            return
    process.wait()


@contextmanager
def _defer_interrupts():
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    pending = []

    def defer(signum, _frame):
        pending.append(signum)

    try:
        for sig in previous:
            signal.signal(sig, defer)
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        if pending:
            raise KeyboardInterrupt(f"Signal {pending[0]} received during cleanup")


@contextmanager
def _interruptible_run():
    previous = signal.getsignal(signal.SIGTERM)

    def interrupt(_signum, _frame):
        raise KeyboardInterrupt("SIGTERM received")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _compute_pids() -> set[int]:
    result = subprocess.run(["nvidia-smi", "-i", "1", "--query-compute-apps=pid",
                             "--format=csv,noheader,nounits"], text=True,
                            capture_output=True, check=True, timeout=10)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if lines == ["No running processes found"]:
        return set()
    if any(not line.isdigit() for line in lines):
        raise RuntimeError(f"Cannot parse GPU 1 compute PIDs: {lines!r}")
    return {int(line) for line in lines}


def _wait_trainer(process: subprocess.Popen, phase: str) -> int:
    deadline = time.monotonic() + TIMEOUT_SECONDS[phase]
    while process.poll() is None:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{phase} trainer exceeded {TIMEOUT_SECONDS[phase]} seconds")
        for pid in _compute_pids():
            try:
                group = os.getpgid(pid)
            except ProcessLookupError:
                continue
            if group != process.pid:
                raise RuntimeError(f"Unexpected GPU 1 compute PID {pid}; stopping run")
        time.sleep(1)
    return process.returncode


def run_one(base: dict, variant: str, phase: str, root: Path, manifest: dict,
            inputs: dict) -> None:
    if source_manifest() != manifest:
        raise RuntimeError("Training source changed after plan; stopping")
    if input_manifest(base) != inputs:
        raise RuntimeError("Model, image or cache fingerprint changed after plan; stopping")
    gpu = gpu_preflight()
    disk = disk_preflight(root)
    directory = root / f"{variant}-{phase}"
    directory.mkdir()
    config = make_config(base, variant, phase, directory)
    (directory / "config.toml").write_text(toml.dumps(config))
    if toml.load(directory / "config.toml") != config:
        raise RuntimeError("Written variant config differs from requested config")
    argv = command(directory)
    env = os.environ.copy()
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES="1",
               PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    write_json(directory / "command.json", {"argv": argv, "cwd": str(ROOT),
               "environment": {key: env[key] for key in
                               ("CUDA_DEVICE_ORDER", "CUDA_VISIBLE_DEVICES")},
               "gpu_preflight": gpu, "disk_preflight": disk,
               "source_manifest_sha256": digest(root / "source_sha256.json")})
    monitor_command = [
        "nvidia-smi", "-i", "1",
        "--query-gpu=timestamp,temperature.gpu,clocks.current.sm,clocks.current.memory,memory.used,power.draw,utilization.gpu",
        "--format=csv,noheader,nounits", "-l", "1",
    ]
    start = time.time()
    monitor = None
    trainer = None
    code = None
    with (directory / "gpu.csv").open("w") as gpu_out, \
            (directory / "gpu.stderr.log").open("w") as gpu_err, \
            (directory / "stdout.log").open("w") as out, \
            (directory / "stderr.log").open("w") as err:
        try:
            monitor = subprocess.Popen(monitor_command, stdout=gpu_out, stderr=gpu_err,
                                       start_new_session=True)
            trainer = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=out, stderr=err,
                                       start_new_session=True)
            code = _wait_trainer(trainer, phase)
        finally:
            with _defer_interrupts():
                try:
                    _stop_group(trainer)
                finally:
                    _stop_group(monitor)
    exit_record = {"code": code, "start_epoch": start,
                   "end_epoch": time.time(), "gpu_monitor_code": monitor.returncode if monitor else None}
    write_json(directory / "exit.json", exit_record)
    if code != 0:
        raise RuntimeError(f"{variant}-{phase} failed: {code}; see {directory}")
    if source_manifest() != manifest or input_manifest(base) != inputs:
        raise RuntimeError("Training source or inputs changed during run")
    verify_progress(directory / "progress.jsonl", config["max_train_steps"])
    checkpoint = verify_checkpoint(
        Path(config["output_dir"]) / f"{config['output_name']}.safetensors", variant)
    telemetry = verify_telemetry(directory / "gpu.csv", phase)
    write_json(directory / "acceptance.json", {"checkpoint": checkpoint, "telemetry": telemetry})
    if phase == "hot":
        write_json(directory / "summary.json", summarize(directory / "progress.jsonl"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="Read-only plan; never starts training")
    mode.add_argument("--run", action="store_true", help="Run selected jobs serially on GPU 1")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="New output directory; existing paths are refused")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS,
                        help="Adapter variants to include, in the supplied order")
    parser.add_argument("--phases", nargs="+", choices=("smoke", "hot"),
                        default=("smoke", "hot"), help="Phases to include, in the supplied order")
    args = parser.parse_args(argv)
    if len(set(args.variants)) != len(args.variants) or len(set(args.phases)) != len(args.phases):
        parser.error("variants and phases must not contain duplicates")
    base = load_reference()
    if not (ROOT / ".venv/bin/python").is_file():
        raise FileNotFoundError(ROOT / ".venv/bin/python")
    output = args.output.resolve()
    if output == REFERENCE or REFERENCE in output.parents:
        raise ValueError(f"Output must not be inside Stage-3 history: {output}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    manifest = source_manifest()
    inputs = input_manifest(base)
    plan = {"source": str(ROOT), "reference_config": str(BASE_CONFIG),
            "reference_config_sha256": digest(BASE_CONFIG), "dataset": str(DATASET),
            "dataset_sha256": digest(DATASET), "output": str(output),
            "gpu_index": 1, "minimum_free_gpu_mib": MIN_FREE_GPU_MIB,
            "minimum_free_disk_gib": MIN_DISK_GIB,
            "jobs": [{"variant": variant, "phase": phase,
                      "command": command(output / f"{variant}-{phase}"),
                      "network_args": make_config(base, variant, phase,
                                                  output / f"{variant}-{phase}")["network_args"],
                      "max_train_steps": 2 if phase == "smoke" else 20}
                     for variant in args.variants for phase in args.phases],
            "source_sha256": manifest,
            "input_fingerprints": inputs,
            "input_fingerprint_limit": "size/mtime and first/last 1 MiB hashes only; not proof of full-file immutability",
            "trainer_timeouts_seconds": TIMEOUT_SECONDS}
    if args.plan:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0
    gpu_preflight()
    disk_preflight(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    write_json(output / "source_sha256.json", manifest)
    write_json(output / "input_fingerprints.json", inputs)
    try:
        copy_reference_cache(output)
        with _interruptible_run():
            for variant in args.variants:
                for phase in args.phases:
                    run_one(base, variant, phase, output, manifest, inputs)
    except BaseException as exc:
        write_json(output / "failure.json", {"error": str(exc), "time_epoch": time.time()})
        raise
    write_json(output / "complete.json", {"time_epoch": time.time(), "status": "ok"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
