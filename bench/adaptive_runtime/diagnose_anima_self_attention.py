"""Monitored analysis-only block17 normalization/self-attention split."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import traceback

import torch

from bench.adaptive_runtime import anima_self_attention_paths as paths
from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime.probe_anima_train import device_uuid
from bench.adaptive_runtime.probe_monitor import check_gpu_idle, run_monitored
from bench.adaptive_runtime.search_anima_precision import launch_if_idle
from library.training.auto_block_swap.process import write_result


def run(args, *, paths=paths, gpu_check=check_gpu_idle, monitor=run_monitored):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if (not args.source_run.is_dir() or not args.stage_run.is_dir() or not args.weights.is_file()
            or not 0 < args.analysis_timeout <= 300):
        raise ValueError("Existing upstream/stage evidence and weights, timeout <=300 required")
    gpu = gpu_check(args.gpu_uuid)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    environment = {"gpu_uuid": args.gpu_uuid, "gpu": gpu, "training_steps": 0,
                   "source_run": str(args.source_run), "stage_run": str(args.stage_run),
                   "weights_sha256": sha256_file(args.weights), **paths.code_hashes(),
                   "driver_sha256": sha256_file(Path(__file__))}
    write_result(args.output_dir / "environment.json", environment)
    summary = {"status": "starting", "scope": paths.SCOPE, "production_ready": False}
    try:
        command = [sys.executable, "-m", "bench.adaptive_runtime.diagnose_anima_self_attention",
                   "--component", args.component,
                   "--analyze", "--source-run", str(args.source_run), "--stage-run", str(args.stage_run),
                   "--weights", str(args.weights), "--output", str(args.output_dir / "result.json")]
        limits = SimpleNamespace(gpu_uuid=args.gpu_uuid, worker_timeout=args.analysis_timeout)
        worker = launch_if_idle(command, args.output_dir, limits, gpu_check, monitor)
        summary["worker"] = worker
        if worker.get("status") != "ok" or worker.get("returncode") != 0:
            raise ValueError("Self-attention worker failed")
        report = json.loads((args.output_dir / "result.json").read_text())
        analysis = report.get("analysis", {})
        if (report.get("status") != "ok" or report.get("uuid") != args.gpu_uuid
                or analysis.get("uuid") != args.gpu_uuid
                or report.get("driver_sha256") != environment["driver_sha256"]
                or any(analysis.get("source", {}).get(key) != environment[key] for key in paths.code_hashes())
                or analysis.get("source", {}).get("upstream_source", {}).get("signature", {}).get("weights_sha256")
                != environment["weights_sha256"]):
            raise ValueError("Self-attention GPU or source code mismatch")
        source, _, captured, prior = paths.source_evidence(args.source_run, args.stage_run, args.weights)
        if source != analysis.get("source"):
            raise ValueError("Self-attention result not bound to requested source/replay")
        verified = paths.validate_files(args.output_dir, analysis)
        paths.validate_statistics(analysis, verified)
        terminal = prior["reference"]["cotangent.1"] * prior["reference"]["gate.self_attn"]
        if (not torch.equal(verified["matrix"]["terminal"], terminal)
                or any(not torch.equal(verified[side]["x"], captured[side]["x"])
                       or not torch.equal(verified[side]["gate"], prior["reference"]["gate.self_attn"])
                       for side in paths.SIDES)):
            raise ValueError("Self-attention input/gate/terminal not bound to source")
        summary.update(status="completed", analysis=analysis)
    except BaseException as exc:
        summary.update(status="failed", stop_reason=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc())
    finally:
        write_result(args.output_dir / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--component", choices=("normalization", "projection-core"), default="normalization")
    parser.add_argument("--stage-run", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--analysis-timeout", type=float, default=300)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    analysis_paths = paths
    if args.component == "projection-core":
        from bench.adaptive_runtime import anima_attention_core_paths
        analysis_paths = anima_attention_core_paths
    # Historical training comparisons use four CPU threads and exact JSON equality.
    torch.set_num_threads(4)
    if args.analyze:
        if args.output is None:
            parser.error("Analysis requires output")
        try:
            if args.output.exists() or not torch.cuda.is_available():
                raise ValueError("Fresh output and monitored CUDA required")
            uuid = device_uuid(torch.cuda.get_device_properties(0).uuid)
            analysis = analysis_paths.analyze(args.source_run, args.stage_run, args.weights,
                                     args.output.parent, torch.device("cuda:0"))
            if analysis["uuid"] != uuid:
                raise ValueError("Self-attention source and CUDA GPU mismatch")
            report = {"status": "ok", "uuid": uuid, "analysis": analysis,
                      "driver_sha256": sha256_file(Path(__file__)), "production_ready": False}
        except BaseException as exc:
            report = {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}
        write_result(args.output, report)
        return 0 if report["status"] == "ok" else 1
    if args.output_dir is None or not args.gpu_uuid:
        parser.error("Monitored run requires output-dir and GPU UUID")
    summary = run(args, paths=analysis_paths)
    print(json.dumps({"status": summary["status"], "output_dir": str(args.output_dir)}))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
