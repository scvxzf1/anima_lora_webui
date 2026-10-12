"""Three-worker, fixed-state Anima MLP backward counterfactual diagnosis only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

import torch

from bench.adaptive_runtime.anima_backward_counterfactual import SCOPE, target_names
from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime.compare_training import compare, compare_case, load_case, read_capture
from bench.adaptive_runtime.probe_monitor import check_gpu_idle, run_monitored
from bench.adaptive_runtime.search_anima_precision import _complete, launch_if_idle, worker_command
from library.training.auto_block_swap.process import write_result


def _args(args, precision):
    return SimpleNamespace(**{**vars(args), "steps": 1, "single_sigma": 0.5,
                              "loss_scale": 1 if precision == "fp32-reference" else 1024})


def _command(args, directory, precision, reference=None, *, mode="native"):
    return worker_command(_args(args, precision), directory, precision,
                          replay=reference / "training-replay" if reference else None,
                          fp16_patterns=("blocks.*.mlp.*",) if reference else (),
                          backward_counterfactual_suite=True, base_backward_mode=mode)


def _validate_capture(directory, result, *, precision, mode, environment, reference=False):
    if not _complete(result, directory, 1, environment["gpu_uuid"], reference=reference):
        raise ValueError(f"{directory.name}: incomplete worker/capture/replay")
    manifest, worker = read_capture(directory / "training-capture")
    signature = manifest["signature"]
    helper_sha = environment["helper_sha256"]
    if (manifest.get("counterfactual_mode") != mode
            or manifest.get("counterfactual_helper_sha256") != helper_sha
            or worker.get("counterfactual_mode") != mode
            or worker.get("counterfactual_helper_sha256") != helper_sha
            or worker.get("backward_counterfactual_scope") != SCOPE
            or signature.get("backward_counterfactual_scope") != SCOPE
            or signature.get("backward_counterfactual_helper_sha256") != helper_sha
            or signature.get("precision") != precision
            or signature.get("weights_sha256") != environment["weights_sha256"]
            or signature.get("inputs_sha256") != environment["inputs_sha256"]
            or signature.get("probe_sha256") != environment["probe_sha256"]
            or signature.get("steps") != 1 or signature.get("sigmas") != [0.5]
            or signature.get("loss_scale") != (1 if reference else 1024)
            or worker.get("swap") != environment["swap"]
            or manifest.get("initial_adapter_sha256") is None):
        raise ValueError(f"{directory.name}: counterfactual signature/mode/source contract mismatch")
    if reference:
        if worker.get("precision_manifest") is not None or signature.get("fp32_modules") != []:
            raise ValueError("Reference must remain plain FP32")
    else:
        assignments = (worker.get("precision_manifest") or {}).get("assignments")
        if (not isinstance(assignments, dict) or len(assignments) != 377
                or {name for name, dtype in assignments.items() if dtype == "fp16"} != target_names()
                or sum(dtype == "fp32" for dtype in assignments.values()) != 321
                or sorted(name for name, dtype in assignments.items() if dtype == "fp32")
                != sorted(signature.get("fp32_modules", []))):
            raise ValueError(f"{directory.name}: not the exact 56/321 MLP-only assignment")
    return manifest


def _direct_equivalence(baseline_dir, counterfactual_dir):
    a, _ = read_capture(baseline_dir / "training-capture")
    b, _ = read_capture(counterfactual_dir / "training-capture")
    excluded = {"precision", "fp32_modules", "loss_scale"}
    if ({k: v for k, v in a["signature"].items() if k not in excluded}
            != {k: v for k, v in b["signature"].items() if k not in excluded}
            or a["initial_adapter_sha256"] != b["initial_adapter_sha256"]
            or a["expected_gradients"] != b["expected_gradients"]
            or a["cases"][0].get("replay") != b["cases"][0].get("replay")
            or a["cases"][0].get("sigma") != b["cases"][0].get("sigma")):
        raise ValueError("Baseline/counterfactual replay, inventory or initial state mismatch")
    left = load_case(baseline_dir / "training-capture", a["cases"][0], a["expected_gradients"])
    right = load_case(counterfactual_dir / "training-capture", b["cases"][0], b["expected_gradients"])
    if not torch.equal(left["prediction"], right["prediction"]):
        raise ValueError("Baseline/counterfactual forward predictions differ")
    return {"status": "compared", "prediction_bitwise_equal": True,
            "metrics": compare_case(left, right), "production_ready": False}


def run(args, *, gpu_check=check_gpu_idle, monitor=run_monitored, comparator=compare):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.weights.is_file() or not args.inputs.is_file():
        raise ValueError("Existing weights and prepared inputs required")
    if not 0 <= args.swap <= 26 or not math.isfinite(args.worker_timeout) or args.worker_timeout <= 0:
        raise ValueError("Invalid swap or worker timeout")
    gpu = gpu_check(args.gpu_uuid)
    environment = {"gpu_uuid": args.gpu_uuid, "gpu": gpu, "swap": args.swap,
                   "weights_sha256": sha256_file(args.weights), "inputs_sha256": sha256_file(args.inputs),
                   "probe_sha256": sha256_file(Path(__file__).with_name("probe_anima_train.py")),
                   "helper_sha256": sha256_file(Path(__file__).with_name("anima_backward_counterfactual.py")),
                   "python": platform.python_version(), "platform": platform.platform()}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_result(args.output_dir / "environment.json", environment)
    summary = {"status": "starting", "scope": "counterfactual_not_precision_certificate",
               "production_ready": False, "gpu_uuid": args.gpu_uuid, "workers": {}}
    try:
        directories = {name: args.output_dir / name for name in ("reference", "baseline", "counterfactual")}
        complete = {}
        for name, precision, mode in (("reference", "fp32-reference", "native"),
                                      ("baseline", "fp16-islands", "native"),
                                      ("counterfactual", "fp16-islands", "fp32")):
            directory = directories[name]
            directory.mkdir(exist_ok=False)
            result = launch_if_idle(_command(args, directory, precision,
                                              directories["reference"] if name != "reference" else None,
                                              mode=mode), directory, args, gpu_check, monitor)
            summary["workers"][name] = {"result": result, "directory": str(directory)}
            write_result(args.output_dir / "summary.json", summary)
            if name != "reference" and result.get("failure_kind") == "nonfinite":
                summary["workers"][name]["outcome"] = "numerical_rejection"
                complete[name] = False
                continue
            _validate_capture(directory, result, precision=precision, mode=mode,
                              environment=environment, reference=name == "reference")
            complete[name] = True
        reference = directories["reference"] / "training-capture"
        for name in ("baseline", "counterfactual"):
            if not complete[name]:
                continue
            comparison = comparator(reference, directories[name] / "training-capture")
            summary[f"reference_vs_{name}"] = comparison
            write_result(args.output_dir / f"reference-vs-{name}.json", comparison)
        if complete["baseline"] and complete["counterfactual"]:
            direct = _direct_equivalence(directories["baseline"], directories["counterfactual"])
            summary["baseline_vs_counterfactual"] = direct
            write_result(args.output_dir / "baseline-vs-counterfactual.json", direct)
        summary["status"] = "completed" if all(complete.values()) else "numerical_rejection"
    except BaseException as exc:
        summary.update(status="failed", stop_reason=f"{type(exc).__name__}: {exc}")
    finally:
        write_result(args.output_dir / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--swap", type=int, default=20)
    parser.add_argument("--worker-timeout", type=float, default=150)
    args = parser.parse_args(argv)
    summary = run(args)
    print(json.dumps({"status": summary["status"], "output_dir": str(args.output_dir)}))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
