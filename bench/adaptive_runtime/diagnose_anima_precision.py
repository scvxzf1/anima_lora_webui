"""Bounded Anima precision root-cause matrix on one idle GPU."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
from types import SimpleNamespace

from bench.adaptive_runtime.anima_gradient_diagnostics import diagnose
from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime.compare_training import compare
from bench.adaptive_runtime.probe_monitor import check_gpu_idle, run_monitored
from bench.adaptive_runtime.search_anima_precision import (
    ATTENTION, CONDITIONING, MLP, _complete, launch_if_idle, worker_command,
)
from library.training.auto_block_swap.process import write_result


CASES = {
    "mlp-only-fp16": ("fp16-only", MLP, 1024),
    "attention-only-fp16": ("fp16-only", ATTENTION, 1024),
    "conditioning-only-fp16": ("fp16-only", CONDITIONING, 1024),
    "conditioning-attention-scale1": ("fp32", (*CONDITIONING, *ATTENTION), 1),
    "conditioning-attention-scale64": ("fp32", (*CONDITIONING, *ATTENTION), 64),
    "conditioning-attention-scale1024": ("fp32", (*CONDITIONING, *ATTENTION), 1024),
}
DEFAULT_CASES = tuple(CASES)
CASES.update({
    "mlp-layer1-only-fp16": ("fp16-only", ("blocks.*.mlp.layer1",), 1024),
    "mlp-layer2-only-fp16": ("fp16-only", ("blocks.*.mlp.layer2",), 1024),
    "fp32-control": ("fp32", ("*",), 1024),
})
HOST_GPU_FAILURES = {"gpu_unavailable", "host_limit", "telemetry_error", "timeout", "cancelled"}


def selected_cases(names, max_cases):
    if type(max_cases) is not int or not 1 <= max_cases <= 8:
        raise ValueError("--max-cases must be 1..8")
    chosen = list(DEFAULT_CASES) if names is None else names
    if not chosen or len(chosen) > max_cases or len(set(chosen)) != len(chosen):
        raise ValueError("Cases must be nonempty, unique, and within --max-cases")
    if any(name not in CASES for name in chosen):
        raise ValueError("Unknown diagnostic case")
    return chosen


def case_command(args, directory, name, replay):
    mode, patterns, scale = CASES[name]
    worker_args = SimpleNamespace(**{**vars(args), "loss_scale": scale})
    return worker_command(worker_args, directory, "fp16-islands", replay=replay,
                          fp16_patterns=patterns if mode == "fp16-only" else (),
                          patterns=patterns if mode == "fp32" else ())


def _entry(name, directory, result):
    assignments = (result.get("precision_manifest") or {}).get("assignments", {})
    return {"case": name, "directory": str(directory), "result": result,
            "fp16_modules": sorted(k for k, v in assignments.items() if v == "fp16"),
            "fp32_modules": sorted(k for k, v in assignments.items() if v == "fp32"),
            "monitor": result.get("monitor")}


def run(args, *, gpu_check=check_gpu_idle, monitor=run_monitored,
        comparator=compare, diagnostic=diagnose):
    names = selected_cases(args.case, args.max_cases)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.weights.is_file() or not args.inputs.is_file():
        raise ValueError("Existing weights and prepared inputs required")
    if not 0 <= args.swap <= 26 or not 1 <= args.steps <= 3:
        raise ValueError("--swap must be 0..26 and --steps must be 1..3")
    if not math.isfinite(args.worker_timeout) or args.worker_timeout <= 0:
        raise ValueError("Positive finite worker timeout required")
    sigma = getattr(args, "single_sigma", None)
    if sigma is not None and (args.steps != 1 or not math.isfinite(sigma) or not 0 < sigma < 1):
        raise ValueError("--single-sigma requires one step and finite sigma in (0,1)")
    gpu = gpu_check(args.gpu_uuid)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary = {"status": "starting", "production_ready": False,
               "calibration_scope": "fixed_pre_step_state_and_inputs_not_trajectory_certificate",
               "gpu_uuid": args.gpu_uuid, "selected_cases": names, "cases": []}
    reference = args.output_dir / "reference"
    write_result(args.output_dir / "summary.json", summary)
    try:
        write_result(args.output_dir / "environment.json", {
            "python": platform.python_version(), "platform": platform.platform(), "gpu": gpu,
            "weights": {"path": str(args.weights.resolve()), "sha256": sha256_file(args.weights)},
            "inputs": {"path": str(args.inputs.resolve()), "sha256": sha256_file(args.inputs)},
            "swap": args.swap, "steps": args.steps, "single_sigma": sigma, "backend": "torch",
            "probe_sha256": sha256_file(Path(__file__).with_name("probe_anima_train.py"))})
        reference.mkdir()
        reference_args = SimpleNamespace(**{**vars(args), "loss_scale": 1})
        result = launch_if_idle(worker_command(reference_args, reference, "fp32-reference"),
                                reference, args, gpu_check, monitor)
        summary["reference"] = {"directory": str(reference), "result": result}
        if not _complete(result, reference, args.steps, args.gpu_uuid, reference=True):
            summary.update(status="reference_failed", stop_reason=result.get("stop_reason", "incomplete_reference"))
            return summary
        for index, name in enumerate(names, 1):
            directory = args.output_dir / f"case-{index:02d}-{name}"
            directory.mkdir()
            result = launch_if_idle(case_command(args, directory, name, reference / "training-replay"),
                                    directory, args, gpu_check, monitor)
            entry = _entry(name, directory, result)
            summary["cases"].append(entry)
            if result.get("status") in HOST_GPU_FAILURES or result.get("stop_reason") in HOST_GPU_FAILURES:
                entry["outcome"] = "infrastructure_stop"
                summary.update(status="stopped", stop_reason=result.get("stop_reason", result["status"]))
                break
            if result.get("status") != "ok":
                if result.get("failure_kind") == "nonfinite":
                    entry["outcome"] = "numerical_rejection"
                    continue
                entry["outcome"] = "worker_failed"
                summary.update(status="stopped", stop_reason="worker_failed")
                break
            if not _complete(result, directory, args.steps, args.gpu_uuid):
                entry["outcome"] = "incomplete_candidate"
                summary.update(status="stopped", stop_reason="incomplete_candidate")
                break
            comparison = comparator(reference / "training-capture", directory / "training-capture")
            write_result(directory / "comparison.json", comparison)
            entry["comparison"] = comparison
            diagnostics = diagnostic(reference / "training-capture", directory / "training-capture", comparison)
            write_result(directory / "gradient-diagnostics.json", diagnostics)
            entry["outcome"] = ("within_experimental_tolerances" if comparison["within_experimental_tolerances"]
                                else "numerical_rejection")
            write_result(args.output_dir / "summary.json", summary)
        else:
            summary["status"] = "completed"
        return summary
    except BaseException as exc:
        summary.update(status="failed", stop_reason="orchestration_error",
                       error=f"{type(exc).__name__}: {exc}")
        return summary
    finally:
        write_result(args.output_dir / "summary.json", summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--swap", type=int, default=20)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--single-sigma", type=float)
    parser.add_argument("--worker-timeout", type=float, default=300)
    parser.add_argument("--case", action="append", choices=tuple(CASES))
    parser.add_argument("--max-cases", type=int, default=8)
    args = parser.parse_args(argv)
    summary = run(args)
    print(json.dumps({"status": summary["status"], "output_dir": str(args.output_dir)}))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
