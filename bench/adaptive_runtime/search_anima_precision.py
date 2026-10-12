"""Bounded, disposable Anima full-DiT precision search on one idle GPU."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import sys

from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime.compare_training import compare, read_capture
from bench.adaptive_runtime.probe_monitor import check_gpu_idle, run_monitored
from library.training.adaptive_runtime.precision import preferred_candidate
from library.training.adaptive_runtime.profile_search import PrecisionProfile, search_profiles
from library.training.adaptive_runtime.retry import MemoryPlan, RetryLimits
from library.training.auto_block_swap.process import write_result


CONDITIONING = ("t_embedder.*", "blocks.*.adaln_*", "final_layer.*", "x_embedder.proj.1")
ATTENTION = ("blocks.*.self_attn.*", "blocks.*.cross_attn.*")
MLP = ("blocks.*.mlp.*",)
PROFILE_NAMES = ("bf16", "seed", "conditioning", "conditioning-attention", "mlp", "fp32-control")


def profiles_for(names, capability, loss_scale):
    candidate = preferred_candidate(capability)
    if names is None:
        names = {"bf16": ["bf16"], "fp16": ["seed", "conditioning", "conditioning-attention"],
                 "fp32": []}[candidate]
    if len(set(names)) != len(names) or any(n not in PROFILE_NAMES for n in names):
        raise ValueError("Unknown or duplicate profile")
    if "bf16" in names and candidate != "bf16":
        raise ValueError("Native BF16 requires Ampere+; choose explicit FP16 profile")
    choices = {
        "bf16": PrecisionProfile("bf16", "bf16"),
        "seed": PrecisionProfile("seed", "fp16-islands", (), loss_scale),
        "conditioning": PrecisionProfile("conditioning", "fp16-islands", CONDITIONING, loss_scale),
        "conditioning-attention": PrecisionProfile("conditioning-attention", "fp16-islands",
                                                 (*CONDITIONING, *ATTENTION), loss_scale),
        "mlp": PrecisionProfile("mlp", "fp16-islands", MLP, loss_scale),
        "fp32-control": PrecisionProfile("fp32-control", "fp16-islands", ("*",), loss_scale),
    }
    return [choices[name] for name in names], candidate


def worker_command(args, directory, precision, *, patterns=(), replay=None, fp16_patterns=(),
                   backward_counterfactual_suite=False, base_backward_mode="native"):
    argv = [sys.executable, "-m", "bench.adaptive_runtime.probe_anima_train",
            "--weights", str(args.weights), "--inputs", str(args.inputs),
            "--output", str(directory / "result.json"), "--precision", precision,
            "--swap", str(args.swap), "--steps", str(args.steps),
            "--loss-scale", str(args.loss_scale if precision == "fp16-islands" else 1)]
    argv.extend(["--replay-reference", str(replay)] if replay else ["--record-replay"])
    for pattern in patterns:
        argv.extend(["--fp32-pattern", pattern])
    for pattern in fp16_patterns:
        argv.extend(["--fp16-only-pattern", pattern])
    if getattr(args, "capture_local_vjp", False):
        argv.append("--capture-local-vjp")
        argv.extend(["--local-vjp-profile", getattr(args, "local_vjp_profile", "linear")])
    if getattr(args, "capture_block_boundaries", False):
        argv.append("--capture-block-boundaries")
    if getattr(args, "capture_block_vjp", False):
        argv.append("--capture-block-vjp")
        if getattr(args, "block_vjp_indices", None) is not None:
            argv.extend(["--block-vjp-indices", args.block_vjp_indices])
    if getattr(args, "single_sigma", None) is not None:
        argv.extend(["--single-sigma", str(args.single_sigma)])
    if backward_counterfactual_suite:
        argv.extend(["--backward-counterfactual-suite", "--base-backward-mode", base_backward_mode])
    return argv


def _capability(gpu):
    try:
        major, minor = (int(value) for value in gpu["compute_cap"].split("."))
        return major, minor
    except (KeyError, ValueError) as exc:
        raise ValueError("GPU compute capability unavailable") from exc


def _complete(result, directory, steps, uuid, *, reference=False):
    if (result.get("status") != "ok" or result.get("returncode") != 0
            or result.get("disposable_probe") is not True
            or result.get("uuid") != uuid or result.get("steps") != steps
            or [u.get("step") for u in result.get("updates", [])] != list(range(1, steps + 1))):
        return False
    try:
        read_capture(directory / "training-capture")
        if not reference:
            return result.get("replay_mode") == "replay"
        manifest = json.loads((directory / "training-replay" / "manifest.json").read_text())
        return (result.get("replay_mode") == "record" and manifest.get("status") == "captured"
                and manifest.get("steps") == steps)
    except (OSError, ValueError, KeyError):
        return False


def launch_if_idle(argv, directory, args, gpu_check, monitor):
    try:
        gpu_check(args.gpu_uuid)
    except Exception as exc:
        result = {"status": "gpu_unavailable", "disposable_probe": True,
                  "stop_reason": "gpu_prelaunch_check_failed", "error": str(exc)}
        (directory / "worker.log").write_text(str(exc) + "\n", encoding="utf-8")
        write_result(directory / "supervisor.json", result)
        return result
    return monitor(argv, directory, args.gpu_uuid, args.worker_timeout)


def run(args, *, gpu_check=check_gpu_idle, monitor=run_monitored, comparator=compare):
    if getattr(args, "block_vjp_indices", None) is not None and not getattr(args, "capture_block_vjp", False):
        raise ValueError("--block-vjp-indices requires --capture-block-vjp")
    if getattr(args, "block_vjp_indices", None) not in (None, "14", "15,17,27"):
        raise ValueError("Block VJP indices must be 14 or 15,17,27")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.weights.is_file() or not args.inputs.is_file():
        raise ValueError("Existing weights and prepared inputs required")
    if not 0 <= args.swap <= 26 or not 1 <= args.steps <= 5 or args.max_attempts < 1:
        raise ValueError("Invalid swap, steps, or attempt bound")
    if args.worker_timeout <= 0 or not math.isfinite(args.loss_scale) or args.loss_scale < 1:
        raise ValueError("Invalid timeout or loss scale")
    gpu = gpu_check(args.gpu_uuid)
    capability = _capability(gpu)
    profiles, preferred = profiles_for(args.profile, capability, args.loss_scale)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "production_ready": False, "full_model_calibrated": False,
              "precision_calibrated": False, "calibration_scope": "fixed_state_only",
              "gpu_uuid": args.gpu_uuid, "preferred_candidate": preferred, "attempts": []}
    write_result(args.output_dir / "summary.json", report)
    reference = args.output_dir / "reference"
    try:
        write_result(args.output_dir / "environment.json", {
            "python": platform.python_version(), "platform": platform.platform(), "gpu": gpu,
            "weights": {"path": str(args.weights.resolve()), "sha256": sha256_file(args.weights)},
            "inputs": {"path": str(args.inputs.resolve()), "sha256": sha256_file(args.inputs)},
            "swap": args.swap, "steps": args.steps, "backend": "torch"})
        reference.mkdir()
        ref_result = launch_if_idle(worker_command(args, reference, "fp32-reference"), reference,
                                    args, gpu_check, monitor)
        report["reference"] = {"directory": str(reference), "result": ref_result}
        if not _complete(ref_result, reference, args.steps, args.gpu_uuid, reference=True):
            report.update(status="reference_failed", stop_reason="incomplete_reference")
            return report
        if not profiles:
            report.update(status="reference_only", stop_reason="hardware_prefers_fp32")
            return report

        def runner(profile, plan, *, attempt):
            directory = args.output_dir / f"attempt-{attempt:03d}"
            directory.mkdir(exist_ok=False)
            result = launch_if_idle(worker_command(args, directory, profile.precision,
                                                   patterns=profile.fp32_patterns,
                                                   replay=reference / "training-replay"),
                                    directory, args, gpu_check, monitor)
            if result.get("status") == "ok" and not _complete(result, directory, args.steps, args.gpu_uuid):
                result.update(status="error", stop_reason="incomplete_candidate")
            return result

        def comparator_for(attempt):
            directory = args.output_dir / f"attempt-{attempt:03d}"
            comparison = comparator(reference / "training-capture", directory / "training-capture")
            write_result(directory / "comparison.json", comparison)
            return comparison

        def record(progress):
            report.update({key: value for key, value in progress.items() if key != "attempts"})
            report["attempts"] = progress["attempts"]
            write_result(args.output_dir / "summary.json", report)

        outcome = search_profiles(MemoryPlan(args.swap, True),
                                  RetryLimits(max_blocks=args.swap, max_attempts=args.max_attempts),
                                  profiles, runner, comparator_for, record=record)
        report.update(outcome)
        with (args.output_dir / "candidates.jsonl").open("x") as stream:
            for item in outcome["attempts"]:
                stream.write(json.dumps(item, allow_nan=False) + "\n")
        if outcome["status"] == "probe_validated":
            selected = outcome["selected_profile"]
            result = outcome["attempts"][-1]["result"]
            assignments = (result.get("precision_manifest") or {}).get("assignments", {})
            fp32 = sorted(n for n, dtype in assignments.items() if dtype == "fp32")
            fp16 = sorted(n for n, dtype in assignments.items() if dtype == "fp16")
            if selected["name"] == "fp32-control":
                report["status"] = "fp32_control_validated"
            plan = {"precision": selected["precision"], "profile": selected["name"],
                    "fp32_modules": fp32, "fp16_modules": fp16,
                    "dtype_assignments_available": bool(assignments),
                    "base_load_dtype": result.get("base_load_dtype"),
                    "compute_dtype": result.get("compute_dtype"),
                    "adapter_master_dtype": result.get("adapter_master_dtype"),
                    "fp16_linear_fraction": len(fp16) / len(assignments) if assignments else None,
                    "fp16_compute_fraction": None,
                    "performance_validated": False,
                    "loss_scale": selected["loss_scale"], "swap": args.swap,
                    "monitor": result.get("monitor"), "calibration_scope": "fixed_state_only",
                    "production_ready": False, "full_model_calibrated": False}
            write_result(args.output_dir / "selected-plan.json", plan)
        return report
    except BaseException as exc:
        report.update(status="failed", stop_reason="orchestration_error",
                      error=f"{type(exc).__name__}: {exc}")
        return report
    finally:
        write_result(args.output_dir / "summary.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--swap", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--worker-timeout", type=float, default=300)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--loss-scale", type=float, default=1024)
    parser.add_argument("--profile", action="append", choices=PROFILE_NAMES)
    parser.add_argument("--capture-block-vjp", action="store_true")
    parser.add_argument("--block-vjp-indices")
    parser.add_argument("--single-sigma", type=float)
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps({"status": report["status"], "output_dir": str(args.output_dir)}))
    return 0 if report["status"] in {"probe_validated", "fp32_control_validated", "reference_only"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
