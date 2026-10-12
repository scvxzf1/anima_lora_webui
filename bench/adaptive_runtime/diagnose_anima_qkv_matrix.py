"""Monitored block17 actual post-RoPE Q/K/V activation matrix."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import traceback

import torch

from bench.adaptive_runtime import anima_qkv_matrix as matrix
from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime.probe_anima_train import device_uuid
from bench.adaptive_runtime.probe_monitor import check_gpu_idle, run_monitored
from bench.adaptive_runtime.search_anima_precision import launch_if_idle
from library.training.auto_block_swap.process import write_result


def run(args, *, gpu_check=check_gpu_idle, monitor=run_monitored):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if (not all(path.is_dir() for path in (args.qknorm_run, args.core_run, args.source_run, args.stage_run))
            or not args.weights.is_file() or not 0 < args.analysis_timeout <= 300):
        raise ValueError("Existing source evidence/weights and timeout <=300 required")
    gpu = gpu_check(args.gpu_uuid)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    environment = {"gpu_uuid": args.gpu_uuid, "gpu": gpu, "training_steps": 0,
                   "qknorm_run": str(args.qknorm_run), "core_run": str(args.core_run),
                   "source_run": str(args.source_run), "stage_run": str(args.stage_run),
                   "weights_sha256": sha256_file(args.weights), **matrix.code_hashes(),
                   "driver_sha256": sha256_file(Path(__file__))}
    write_result(args.output_dir / "environment.json", environment)
    summary = {"status": "starting", "scope": matrix.SCOPE, "production_ready": False}
    try:
        command = [sys.executable, "-m", "bench.adaptive_runtime.diagnose_anima_qkv_matrix", "--analyze",
                   "--qknorm-run", str(args.qknorm_run), "--core-run", str(args.core_run),
                   "--source-run", str(args.source_run), "--stage-run", str(args.stage_run),
                   "--weights", str(args.weights), "--output", str(args.output_dir / "result.json")]
        limits = SimpleNamespace(gpu_uuid=args.gpu_uuid, worker_timeout=args.analysis_timeout)
        worker = launch_if_idle(command, args.output_dir, limits, gpu_check, monitor)
        summary["worker"] = worker
        if worker.get("status") != "ok" or worker.get("returncode") != 0:
            raise ValueError("QKV matrix worker failed")
        report = json.loads((args.output_dir / "result.json").read_text())
        analysis = report.get("analysis", {})
        if (report.get("status") != "ok" or report.get("uuid") != args.gpu_uuid
                or analysis.get("uuid") != args.gpu_uuid
                or report.get("driver_sha256") != environment["driver_sha256"]
                or any(analysis.get("source", {}).get(key) != value for key, value in matrix.code_hashes().items())
                or analysis.get("source", {}).get("qknorm_source", {}).get("core_source", {})
                   .get("upstream_source", {}).get("signature", {}).get("weights_sha256") != environment["weights_sha256"]):
            raise ValueError("QKV matrix GPU/code/weights binding mismatch")
        source, _, _, old, historical = matrix.source_evidence(
            args.qknorm_run, args.core_run, args.source_run, args.stage_run, args.weights)
        if source != analysis.get("source"):
            raise ValueError("QKV matrix source changed during worker")
        verified = matrix.validate_files(args.output_dir, analysis)
        matrix.validate_statistics(analysis, verified, old, historical)
        summary.update(status="completed", analysis=analysis)
    except BaseException as exc:
        summary.update(status="failed", stop_reason=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc())
    finally:
        write_result(args.output_dir / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qknorm-run", type=Path, required=True)
    parser.add_argument("--core-run", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--stage-run", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--analysis-timeout", type=float, default=300)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    if args.analyze:
        if args.output is None:
            parser.error("Analysis requires output")
        try:
            if args.output.exists() or not torch.cuda.is_available():
                raise ValueError("Fresh output and monitored CUDA required")
            uuid = device_uuid(torch.cuda.get_device_properties(0).uuid)
            analysis = matrix.analyze(args.qknorm_run, args.core_run, args.source_run,
                                      args.stage_run, args.weights, args.output.parent, torch.device("cuda:0"))
            if analysis["uuid"] != uuid:
                raise ValueError("QKV matrix source and CUDA GPU mismatch")
            report = {"status": "ok", "uuid": uuid, "analysis": analysis,
                      "driver_sha256": sha256_file(Path(__file__)), "production_ready": False}
        except BaseException as exc:
            report = {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}
        write_result(args.output, report)
        return 0 if report["status"] == "ok" else 1
    if args.output_dir is None or not args.gpu_uuid:
        parser.error("Monitored run requires output-dir and GPU UUID")
    summary = run(args)
    print(json.dumps({"status": summary["status"], "output_dir": str(args.output_dir)}))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
