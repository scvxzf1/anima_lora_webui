"""Three-worker fixed-state Anima sampled block-boundary diagnosis only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
from types import SimpleNamespace

from bench.adaptive_runtime import anima_block_boundaries as boundaries
from bench.adaptive_runtime.anima_backward_counterfactual import target_names
from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime.compare_training import compare, read_capture
from bench.adaptive_runtime.probe_monitor import check_gpu_idle, run_monitored
from bench.adaptive_runtime.search_anima_precision import _complete, launch_if_idle, worker_command
from library.training.auto_block_swap.process import write_result


def _worker_args(args, precision):
    return SimpleNamespace(weights=args.weights, inputs=args.inputs, swap=args.swap,
                           steps=1, single_sigma=0.5, capture_block_boundaries=True,
                           loss_scale=1 if precision == "fp32-reference" else 1024)


def _command(args, directory, precision, reference=None, *, control=False):
    return worker_command(_worker_args(args, precision), directory, precision,
                          replay=reference / "training-replay" if reference else None,
                          fp16_patterns=() if not reference or control else ("blocks.*.mlp.*",),
                          patterns=("*",) if control else ())


def _validate(directory, result, environment, *, kind):
    reference = kind == "reference"
    if not _complete(result, directory, 1, environment["gpu_uuid"], reference=reference):
        raise ValueError(f"{kind}: incomplete worker/capture/replay")
    capture, worker = read_capture(directory / "training-capture")
    boundary, _ = boundaries.read_capture(directory / "block-boundaries")
    signature = capture["signature"]
    if (boundary["signature"] != signature or boundary["uuid"] != environment["gpu_uuid"]
            or boundary["block_count"] != 28 or boundary["sample_tokens"] != boundaries.TOKENS
            or boundary["replay"] != capture["cases"][0].get("replay")
            or signature.get("block_boundary_scope") != boundaries.SCOPE
            or signature.get("block_boundary_helper_sha256") != environment["helper_sha256"]
            or signature.get("probe_sha256") != environment["probe_sha256"]
            or signature.get("weights_sha256") != environment["weights_sha256"]
            or signature.get("inputs_sha256") != environment["inputs_sha256"]
            or signature.get("steps") != 1 or signature.get("sigmas") != [0.5]
            or signature.get("backend") != "torch" or signature.get("checkpoint") != "full"
            or signature.get("rank") != 4 or signature.get("alpha") != 4
            or signature.get("loss_scale") != (1 if reference else 1024)
            or worker.get("swap") != environment["swap"]
            or not isinstance(capture.get("initial_adapter_sha256"), str)):
        raise ValueError(f"{kind}: boundary signature/source mismatch")
    assignments = (worker.get("precision_manifest") or {}).get("assignments")
    if reference:
        if worker.get("precision_manifest") is not None or signature.get("precision") != "fp32-reference":
            raise ValueError("Reference must be plain FP32")
    elif (not isinstance(assignments, dict) or len(assignments) != 377
          or sorted(name for name, value in assignments.items() if value == "fp32")
          != sorted(signature.get("fp32_modules", []))):
        raise ValueError(f"{kind}: incomplete precision assignment")
    elif kind == "candidate":
        if ({name for name, value in assignments.items() if value == "fp16"} != target_names()
                or sum(value == "fp32" for value in assignments.values()) != 321):
            raise ValueError("Candidate must be exact 56/321 MLP-only assignment")
    elif set(assignments.values()) != {"fp32"}:
        raise ValueError("Control must have all FP32 base Linear assignments")
    return capture


def _cross_validate(captures):
    reference, candidate, control = [captures[name] for name in ("reference", "candidate", "control")]
    excluded = {"precision", "fp32_modules", "loss_scale"}
    def common(manifest):
        return {k: v for k, v in manifest["signature"].items() if k not in excluded}
    if (common(reference) != common(candidate) or common(reference) != common(control)
            or len({item["initial_adapter_sha256"] for item in captures.values()}) != 1
            or len({tuple(item["expected_gradients"]) for item in captures.values()}) != 1
            or len({json.dumps(item["cases"][0].get("replay"), sort_keys=True)
                    for item in captures.values()}) != 1
            or [item["cases"][0].get("sigma") for item in captures.values()] != [0.5] * 3):
        raise ValueError("Reference/candidate/control initial state or replay mismatch")


def run(args, *, gpu_check=check_gpu_idle, monitor=run_monitored, comparator=compare):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if (not args.weights.is_file() or not args.inputs.is_file() or args.swap != 20
            or not math.isfinite(args.worker_timeout) or args.worker_timeout <= 0
            or args.worker_timeout > 150):
        raise ValueError("Existing inputs, --swap20 and worker timeout <=150 required")
    gpu = gpu_check(args.gpu_uuid)
    environment = {"gpu_uuid": args.gpu_uuid, "gpu": gpu, "swap": 20,
                   "weights_sha256": sha256_file(args.weights), "inputs_sha256": sha256_file(args.inputs),
                   "probe_sha256": sha256_file(Path(__file__).with_name("probe_anima_train.py")),
                   "helper_sha256": sha256_file(Path(boundaries.__file__)),
                   "python": platform.python_version(), "platform": platform.platform(),
                   "sigma": 0.5, "steps": 1, "backend": "torch", "checkpoint": "full"}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_result(args.output_dir / "environment.json", environment)
    summary = {"status": "starting", "scope": boundaries.SCOPE, "production_ready": False,
               "gpu_uuid": args.gpu_uuid, "workers": {}}
    write_result(args.output_dir / "summary.json", summary)
    captures = {}
    try:
        reference = args.output_dir / "reference"
        for kind, precision in (("reference", "fp32-reference"),
                                ("candidate", "fp16-islands"), ("control", "fp16-islands")):
            directory = args.output_dir / kind
            directory.mkdir(exist_ok=False)
            result = launch_if_idle(_command(args, directory, precision,
                                            None if kind == "reference" else reference,
                                            control=kind == "control"),
                                    directory, args, gpu_check, monitor)
            summary["workers"][kind] = {"directory": str(directory), "result": result}
            write_result(args.output_dir / "summary.json", summary)
            captures[kind] = _validate(directory, result, environment, kind=kind)
        _cross_validate(captures)
        for kind in ("candidate", "control"):
            comparison = comparator(reference / "training-capture",
                                    args.output_dir / kind / "training-capture")
            if comparison.get("status") != "compared":
                raise ValueError("Whole-network comparator failed")
            summary[f"reference_vs_{kind}"] = comparison
            write_result(args.output_dir / f"reference-vs-{kind}.json", comparison)
        analysis = boundaries.analyze(*(args.output_dir / kind / "block-boundaries"
                                        for kind in ("reference", "candidate", "control")))
        write_result(args.output_dir / "boundary-analysis.json", analysis)
        summary.update(status="completed", analysis=analysis)
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
    parser.add_argument("--swap20", action="store_true", required=True)
    parser.add_argument("--worker-timeout", type=float, default=150)
    args = parser.parse_args(argv)
    args.swap = 20
    summary = run(args)
    print(json.dumps({"status": summary["status"], "output_dir": str(args.output_dir)}))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
