"""Bounded full-token VJP localization at Anima blocks 15, 17 and 27."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys
import traceback

from bench.adaptive_runtime import anima_block_vjp as vjp
from bench.adaptive_runtime import anima_block_vjp_capture as capture
from bench.adaptive_runtime import diagnose_anima_block_vjp as single
from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime.compare_training import compare
from bench.adaptive_runtime.probe_monitor import check_gpu_idle, run_monitored
from bench.adaptive_runtime.search_anima_precision import _complete, launch_if_idle
from library.training.auto_block_swap.process import write_result


MATRIX_TOTAL_MAX = 1024**3


def capture_command(args, directory, reference=None):
    return [*single._command(args, directory, reference=reference),
            "--block-vjp-indices", "15,17,27"]


def _read_worker(directory, result, environment, *, reference):
    if not _complete(result, directory, 1, environment["gpu_uuid"], reference=reference):
        raise ValueError("Incomplete multi-block training worker")
    manifest, _ = capture.read_multi_capture(directory / "block-vjp")
    expected = {"weights_sha256": environment["weights_sha256"],
                "inputs_sha256": environment["inputs_sha256"],
                "probe_sha256": environment["probe_sha256"],
                "block_vjp_helper_sha256": environment["helper_sha256"]}
    if (manifest["uuid"] != environment["gpu_uuid"]
            or any(manifest["signature"].get(key) != value for key, value in expected.items())):
        raise ValueError("Multi-block capture source or GPU mismatch")
    return manifest


def analysis_command(args, directory, index):
    child = f"block-{index:02d}"
    return [sys.executable, "-m", "bench.adaptive_runtime.diagnose_anima_block_vjp",
            "--analyze", "--block-index", str(index),
            "--reference", str(args.output_dir / "reference/block-vjp" / child),
            "--candidate", str(args.output_dir / "candidate/block-vjp" / child),
            "--comparison", str(args.output_dir / "comparison.json"),
            "--weights", str(args.weights), "--output", str(directory / "result.json")]


def _read_analysis(directory, result, environment, index, capture_root):
    if result.get("status") != "ok" or result.get("returncode") != 0:
        raise ValueError(f"Block {index} analysis worker failed")
    report = json.loads((directory / "result.json").read_text())
    analysis = report.get("analysis", {})
    expected_source = {key: environment[key] for key in
                       ("weights_sha256", "inputs_sha256", "probe_sha256")}
    expected_source["block_vjp_helper_sha256"] = environment["helper_sha256"]
    if (report.get("status") != "ok" or report.get("uuid") != environment["gpu_uuid"]
            or report.get("driver_sha256") != environment["analysis_driver_sha256"]
            or report.get("analysis_sha256") != environment["analysis_sha256"]
            or analysis.get("uuid") != environment["gpu_uuid"]
            or analysis.get("status") != "analyzed" or analysis.get("block_index") != index
            or analysis.get("scope") != vjp.UPSTREAM_SCOPE
            or analysis.get("cell_count") != 8 or analysis.get("tokens") != 4200
            or analysis.get("production_ready") is not False
            or analysis.get("source") != expected_source):
        raise ValueError(f"Block {index} analysis identity/source mismatch")
    single.validate_matrix_evidence(directory / "matrix-evidence", analysis,
                                    capture_root=capture_root)
    return analysis


def run(args, *, gpu_check=check_gpu_idle, monitor=run_monitored, comparator=compare):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if (not args.weights.is_file() or not args.inputs.is_file() or args.swap != 20
            or not math.isfinite(args.worker_timeout) or not 0 < args.worker_timeout <= 150
            or not math.isfinite(args.analysis_timeout) or not 0 < args.analysis_timeout <= 300
            or 2 * args.worker_timeout + 3 * args.analysis_timeout > 1200):
        raise ValueError("Existing inputs, swap20 and bounded 150/300/1200-second timeouts required")
    planned_matrix_bytes = len(capture.MULTI_INDICES) * 8 * math.prod(capture.SHAPE) * 4
    if planned_matrix_bytes > MATRIX_TOTAL_MAX:
        raise ValueError("Planned matrices exceed aggregate 1GiB budget")
    gpu = gpu_check(args.gpu_uuid)
    environment = {"gpu_uuid": args.gpu_uuid, "gpu": gpu, "block_indices": list(capture.MULTI_INDICES),
                   "weights_sha256": sha256_file(args.weights), "inputs_sha256": sha256_file(args.inputs),
                   "probe_sha256": sha256_file(Path(__file__).with_name("probe_anima_train.py")),
                   "helper_sha256": sha256_file(Path(capture.__file__)),
                   "analysis_sha256": sha256_file(Path(vjp.__file__)),
                   "analysis_driver_sha256": sha256_file(Path(single.__file__)),
                   "driver_sha256": sha256_file(Path(__file__)), "steps": 1, "sigma": 0.5,
                   "swap": 20, "capture_budget_per_worker": capture.MAX_BYTES,
                   "matrix_budget_total": MATRIX_TOTAL_MAX}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_result(args.output_dir / "environment.json", environment)
    summary = {"status": "starting", "scope": vjp.UPSTREAM_SCOPE, "production_ready": False,
               "gpu_uuid": args.gpu_uuid, "workers": {}, "analyses": {}, "matrix_disk_bytes": 0}
    write_result(args.output_dir / "summary.json", summary)
    try:
        _run_captures(args, summary, environment, gpu_check, monitor)
        comparison = comparator(args.output_dir / "reference/training-capture",
                                args.output_dir / "candidate/training-capture")
        if comparison.get("status") != "compared":
            raise ValueError("Whole-network comparator failed")
        write_result(args.output_dir / "comparison.json", comparison)
        summary["comparison"] = comparison
        for index in capture.MULTI_INDICES:
            _run_analysis(args, summary, environment, index, comparison, gpu_check, monitor)
            write_result(args.output_dir / "summary.json", summary)
        if set(summary["analyses"]) != {str(index) for index in capture.MULTI_INDICES}:
            raise ValueError("Missing upstream block analysis")
        summary["status"] = "completed"
    except BaseException as exc:
        summary.update(status="failed", stop_reason=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc())
    finally:
        write_result(args.output_dir / "summary.json", summary)
    return summary


def _run_captures(args, summary, environment, gpu_check, monitor):
    for kind in ("reference", "candidate"):
        directory = args.output_dir / kind
        directory.mkdir(exist_ok=False)
        reference = None if kind == "reference" else args.output_dir / "reference"
        result = launch_if_idle(capture_command(args, directory, reference), directory, args,
                                gpu_check, monitor)
        summary["workers"][kind] = {"directory": str(directory), "result": result}
        write_result(args.output_dir / "summary.json", summary)
        _read_worker(directory, result, environment, reference=kind == "reference")


def _run_analysis(args, summary, environment, index, comparison, gpu_check, monitor):
    child = f"block-{index:02d}"
    vjp.validate_pair(args.output_dir / "reference/block-vjp" / child,
                      args.output_dir / "candidate/block-vjp" / child,
                      comparison, block_index=index)
    directory = args.output_dir / f"analysis-block-{index:02d}"
    directory.mkdir(exist_ok=False)
    limits = SimpleNamespace(gpu_uuid=args.gpu_uuid, worker_timeout=args.analysis_timeout)
    result = launch_if_idle(analysis_command(args, directory, index), directory, limits,
                            gpu_check, monitor)
    summary["workers"][f"analysis-{index}"] = {"directory": str(directory), "result": result}
    analysis = _read_analysis(directory, result, environment, index, args.output_dir)
    disk_bytes = (directory / "matrix-evidence/matrix.safetensors").stat().st_size
    if summary["matrix_disk_bytes"] + disk_bytes > MATRIX_TOTAL_MAX:
        raise ValueError("Upstream matrices exceed aggregate 1GiB disk budget")
    summary["matrix_disk_bytes"] += disk_bytes
    summary["analyses"][str(index)] = analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--swap20", action="store_true", required=True)
    parser.add_argument("--worker-timeout", type=float, default=150)
    parser.add_argument("--analysis-timeout", type=float, default=300)
    args = parser.parse_args(argv)
    args.swap = 20
    summary = run(args)
    print(json.dumps({"status": summary["status"], "output_dir": str(args.output_dir)}))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
