"""Bounded ordinary BF16 train.py baseline using an existing three-step fixture."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys

import toml

from library.training.adaptive_runtime.process import IsolatedRunner
from library.training.auto_block_swap.process import write_result
from library.training.auto_block_swap.resources import host_memory


@dataclass(frozen=True)
class Request:
    config: str


def baseline_config(source, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("BF16 smoke requires a fresh output directory")
    config = toml.load(source)
    if config.get("model_family") != "krea2_raw" or config.get("max_train_steps") != 3:
        raise ValueError("Expected the three-step Krea training fixture")
    if config.get("optimizer_type") != "AdamW" or config.get("save_precision") != "float":
        raise ValueError("Audit requires AdamW and FP32 adapter saves")
    if config.get("resume") or config.get("network_weights") or config.get("artist_filter"):
        raise ValueError("Expected a fresh, unfiltered training fixture")
    config.update(
        mixed_precision="bf16", full_bf16=False, full_fp16=False,
        adaptive_precision="off", adaptive_oom_retry=False,
        output_dir=str(output), output_name="bf16-smoke", save_state_on_train_end=True,
        progress_jsonl=str(output / "progress.jsonl"),
        memory_probe_jsonl=str(output / "memory.jsonl"), memory_probe_max_steps=3,
    )
    return config


def audit(directory):
    """Check real optimizer updates and saved tensors; not a resume/quality test."""
    import torch
    from safetensors.torch import load_file

    directory = Path(directory)
    events = [json.loads(line) for line in (directory / "progress.jsonl").read_text().splitlines()]
    steps = [event for event in events if event["ev"] == "step"]
    end = events[-1]
    if (end.get("ev"), end.get("status"), end.get("final_step")) != ("run_end", "ok", 3):
        raise ValueError("Training did not finish three steps")
    if [step["global_step"] for step in steps] != [1, 2, 3]:
        raise ValueError("Incomplete step events")
    loss_metric = "loss/current" if all("loss/current" in step for step in steps) else "avr_loss"
    losses = [float(step[loss_metric]) for step in steps]
    if not all(math.isfinite(loss) for loss in losses):
        raise ValueError("Nonfinite loss")
    state = directory / "bf16-smoke-state"
    if (state / "scaler.pt").exists() or (state / "adaptive_precision.json").exists():
        raise ValueError("Ordinary BF16 must not use adaptive precision or an FP16 scaler")
    snapshot = load_file(str(state / "model.safetensors"))
    output = load_file(str(directory / "bf16-smoke.safetensors"))
    if not snapshot or snapshot.keys() != output.keys():
        raise ValueError("Adapter snapshot keys differ")
    for name, tensor in snapshot.items():
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError(f"Expected finite FP32 adapter: {name}")
        if not torch.equal(tensor, output[name]):
            raise ValueError(f"Adapter snapshot differs: {name}")
    optimizer = torch.load(state / "optimizer.bin", map_location="cpu", weights_only=True)
    if not optimizer["state"]:
        raise ValueError("Missing optimizer updates")
    for values in optimizer["state"].values():
        if not {"step", "exp_avg", "exp_avg_sq"} <= values.keys() or float(values["step"]) != 3:
            raise ValueError("AdamW update count mismatch")
        if any(not torch.isfinite(t).all() for t in values.values() if isinstance(t, torch.Tensor)):
            raise ValueError("Nonfinite optimizer state")
    scheduler = torch.load(state / "scheduler.bin", map_location="cpu", weights_only=True)
    if scheduler["last_epoch"] != 3:
        raise ValueError("Scheduler progress mismatch")
    if not (state / "random_states_0.pkl").is_file():
        raise ValueError("Missing RNG state")
    memory = [json.loads(line) for line in (directory / "memory.jsonl").read_text().splitlines()]
    prepared = next(row for row in memory if row.get("label") == "accelerator_prepared")
    backbone_dtypes = prepared["unet"]["dtypes"]
    adapter_dtypes = prepared["network"]["all_parameters"]["dtypes"]
    if set(backbone_dtypes) != {"bfloat16"} or set(adapter_dtypes) != {"float32"}:
        raise ValueError("Expected BF16 backbone and FP32 trainable adapter")
    return {
        "status": "ok", "completed_steps": 3, "losses": losses, "loss_metric": loss_metric,
        "step_event_intervals_seconds": [b["ts"] - a["ts"] for a, b in zip(steps, steps[1:])],
        "adapter_tensors": len(snapshot), "adapter_snapshot_exact": True,
        "adapter_dtype": "float32", "optimizer_states": len(optimizer["state"]),
        "backbone_parameter_dtypes": backbone_dtypes, "adapter_parameter_dtypes": adapter_dtypes,
        "scheduler_last_epoch": scheduler["last_epoch"], "scaler_expected": False,
        "peak_allocated_gib": max(row.get("cuda_max_allocated_gb", 0) for row in memory),
        "peak_reserved_gib": max(row.get("cuda_max_reserved_gb", 0) for row in memory),
        "scope": "ordinary_bf16_training_smoke_not_quality_or_resume",
        "precision_calibrated": False, "resume_tested": False,
    }


def worker(request):
    import torch

    request = Path(request)
    config_path = json.loads(request.read_text())["plan"]["config"]
    output = Path(toml.load(config_path)["output_dir"])
    try:
        if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported(including_emulation=False):
            raise ValueError("Select exactly one GPU with native BF16 support")
        properties = torch.cuda.get_device_properties(0)
        device = {
            "name": properties.name, "uuid": str(properties.uuid),
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory, "native_bf16": True,
        }
        write_result(output / "device.json", device)
        torch.cuda.reset_peak_memory_stats()
        sys.argv = ["train.py", "--config_file", config_path]
        runpy.run_path(str(Path(__file__).resolve().parents[2] / "train.py"), run_name="__main__")
        report = audit(output)
        report["device"] = device
        report["worker_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        report["worker_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
        write_result(output / "training-audit.json", report)
    except Exception as exc:
        import traceback

        traceback.print_exc()
        report = {"status": "cuda_oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "error",
                  "error": f"{type(exc).__name__}: {exc}"}
    write_result(request.parent / "result.json", report)


def run(source, output, *, gpu, timeout=600):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be positive and finite")
    source, output = Path(source).resolve(), Path(output).resolve()
    config = baseline_config(source, output)
    output.mkdir(parents=True, exist_ok=False)
    config_path = output / "train.toml"
    config_path.write_text(toml.dumps(config), encoding="utf-8")
    before = host_memory()
    minimum = before.available

    def resources():
        nonlocal minimum
        current = host_memory()
        minimum = min(minimum, current.available)
        return current

    environment = {
        "CUDA_VISIBLE_DEVICES": gpu, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
        "WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    }
    runner = IsolatedRunner(
        output / ".benchmark",
        lambda request: [sys.executable, "-m", "bench.adaptive_runtime.bf16_training_smoke",
                         "--worker", str(request)],
        timeout=timeout, env=environment, resources=resources,
    )
    result = runner(Request(str(config_path)), attempt=0, resume=None)
    after = host_memory()
    result.update(
        source_config=str(source), source_config_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        selected_gpu=gpu, host_before=before.to_dict(), host_after=after.to_dict(),
        host_min_available=minimum, host_reserve=before.reserve,
        concurrent_gpu_workload_possible=True,
    )
    write_result(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--audit", type=Path, help="Audit saved outputs without starting training")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return
    if args.audit:
        report = audit(args.audit)
        report["device"] = json.loads((args.audit / "device.json").read_text())
        execution = json.loads((args.audit / "summary.json").read_text())
        report["execution_summary_status"] = execution["status"]
        report["execution_summary_error"] = execution.get("error")
        report["worker_seconds"] = execution["elapsed_seconds"]
        write_result(args.audit / "training-audit.json", report)
        print(json.dumps(report, indent=2))
        return
    if not args.source_config or not args.output or not args.gpu:
        parser.error("--source-config, --output and --gpu are required")
    result = run(args.source_config, args.output, gpu=args.gpu, timeout=args.timeout)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
