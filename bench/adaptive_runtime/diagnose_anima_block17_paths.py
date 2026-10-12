"""Monitored, analysis-only block17 internal path localization on saved evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import traceback

from safetensors.torch import load_file
import torch

from bench.adaptive_runtime import anima_block17_paths as paths
from bench.adaptive_runtime import anima_block_vjp_capture as capture
from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime.probe_anima_train import device_uuid
from bench.adaptive_runtime.probe_monitor import check_gpu_idle, run_monitored
from bench.adaptive_runtime.search_anima_precision import launch_if_idle
from library.training.auto_block_swap.process import write_result


def validate_files(directory, analysis):
    directory = Path(directory)
    if (analysis.get("status") != "analyzed" or analysis.get("scope") != paths.SCOPE
            or analysis.get("production_ready") is not False
            or analysis.get("shape") != list(capture.SHAPE)
            or set(analysis.get("files", {})) != {"reference", "candidate", "matrix"}
            or set(analysis.get("stages", {})) != set(paths.STAGES)):
        raise ValueError("Incomplete internal stage analysis")
    total = 0
    for kind in ("reference", "candidate", "matrix"):
        folder = "stage-matrix" if kind == "matrix" else f"trace-{kind}"
        manifest = json.loads((directory / folder / "manifest.json").read_text())
        if (manifest != analysis["files"][kind] or manifest.get("schema") != "anima_block17_stages_v1"
                or manifest.get("scope") != paths.SCOPE or manifest.get("name") != folder
                or manifest.get("file") != "evidence.safetensors"
                or manifest.get("production_ready") is not False
                or manifest.get("source") != analysis["source"]):
            raise ValueError("Stage evidence manifest mismatch")
        path = directory / folder / "evidence.safetensors"
        size = path.stat().st_size
        if size > capture.MAX_BYTES or size != manifest.get("disk_bytes") or sha256_file(path) != manifest.get("sha256"):
            raise ValueError("Stage evidence hash or budget mismatch")
        tensors = load_file(str(path), device="cpu")
        expected = ({f"{part}.{stage}.{side}" for part in ("complete", "branch")
                     for stage in paths.STAGES for side in ("reference", "candidate")}
                    if kind == "matrix" else
                    {f"{part}.{i}" for part in ("activation", "cotangent") for i in range(4)}
                    | {f"gate.{stage}" for stage in paths.STAGES})
        if (set(tensors) != expected
                or manifest.get("inventory") != {key: list(value.shape) for key, value in tensors.items()}
                or any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
                       or (not key.startswith("gate.") and tuple(value.shape) != capture.SHAPE)
                       or (key.startswith("gate.") and tuple(value.shape) != (1, 1, 1, 1, capture.SHAPE[-1]))
                       for key, value in tensors.items())
                or sum(value.numel() * value.element_size() for value in tensors.values()) != manifest.get("tensor_bytes")):
            raise ValueError("Stage tensor inventory, dtype or shape mismatch")
        total += size
    if total > paths.TOTAL_MAX_BYTES or total != analysis.get("disk_bytes"):
        raise ValueError("Stage aggregate 1GiB evidence budget mismatch")
    for side in ("reference", "candidate"):
        fidelity = analysis.get("fidelity", {}).get(side, {})
        identities = fidelity.get("residual_identity", {})
        if fidelity.get("forward_bitwise_equal") is not True or set(identities) != set(paths.STAGES):
            raise ValueError("Missing stage fidelity evidence")
        metrics = [fidelity.get("whole_block", {}), fidelity.get("source_matrix", {}), *identities.values()]
        if any(item.get("finite") is not True or item.get("relative_l2") is None
               or not 0 <= item["relative_l2"] <= paths.vjp.GRADIENT_RELATIVE_L2_MAX for item in metrics):
            raise ValueError("Stage reconstruction fidelity failed")


def validate_source_binding(args, source):
    manifests, _, _ = paths.source_evidence(args.source_run, args.weights)
    reference = manifests["reference"]
    expected = {"capture_sha256": {side: item["sha256"] for side, item in manifests.items()},
                "signature": reference["signature"], "replay": reference["replay"],
                "block_index": 17, "policy": "reference", "scale": 1}
    if any(source.get(key) != value for key, value in expected.items()):
        raise ValueError("Stage result not bound to requested capture/replay/policy")
    block_path = Path(paths.__file__).resolve().parents[2] / "library/anima/models.py"
    if source.get("block_source_sha256") != sha256_file(block_path):
        raise ValueError("Stage block source changed during analysis")


def run(args, *, gpu_check=check_gpu_idle, monitor=run_monitored):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if (not args.source_run.is_dir() or not args.weights.is_file()
            or not 0 < args.analysis_timeout <= 300):
        raise ValueError("Existing source evidence/checkpoint and timeout <=300 required")
    gpu = gpu_check(args.gpu_uuid)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    environment = {"gpu_uuid": args.gpu_uuid, "gpu": gpu, "source_run": str(args.source_run),
                   "weights_sha256": sha256_file(args.weights),
                   "analysis_sha256": sha256_file(Path(paths.__file__)),
                   "helper_sha256": sha256_file(Path(paths.__file__).with_name("anima_block_stages.py")),
                   "driver_sha256": sha256_file(Path(__file__)), "training_steps": 0}
    write_result(args.output_dir / "environment.json", environment)
    summary = {"status": "starting", "scope": paths.SCOPE, "production_ready": False}
    try:
        command = [sys.executable, "-m", "bench.adaptive_runtime.diagnose_anima_block17_paths",
                   "--analyze", "--source-run", str(args.source_run), "--weights", str(args.weights),
                   "--output", str(args.output_dir / "result.json")]
        limits = SimpleNamespace(gpu_uuid=args.gpu_uuid, worker_timeout=args.analysis_timeout)
        result = launch_if_idle(command, args.output_dir, limits, gpu_check, monitor)
        summary["worker"] = result
        if result.get("status") != "ok" or result.get("returncode") != 0:
            raise ValueError("Internal stage worker failed")
        report = json.loads((args.output_dir / "result.json").read_text())
        analysis = report.get("analysis", {})
        if (report.get("driver_sha256") != environment["driver_sha256"]
                or report.get("status") != "ok" or report.get("uuid") != args.gpu_uuid
                or analysis.get("uuid") != args.gpu_uuid
                or analysis.get("source", {}).get("analysis_sha256") != environment["analysis_sha256"]
                or analysis["source"].get("helper_sha256") != environment["helper_sha256"]
                or analysis["source"].get("signature", {}).get("weights_sha256") != environment["weights_sha256"]):
            raise ValueError("Internal stage result source or GPU mismatch")
        validate_source_binding(args, analysis["source"])
        validate_files(args.output_dir, analysis)
        summary.update(status="completed", analysis=analysis)
    except BaseException as exc:
        summary.update(status="failed", stop_reason=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc())
    finally:
        write_result(args.output_dir / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--analysis-timeout", type=float, default=300)
    args = parser.parse_args(argv)
    # Match compare_training's recorded FP64 reduction order: source binding
    # requires exact report equality, including the final reduction bits.
    torch.set_num_threads(4)
    if args.analyze:
        if args.output is None:
            parser.error("Analysis requires output")
        try:
            if args.output.exists() or not torch.cuda.is_available():
                raise ValueError("Fresh output and monitored CUDA required")
            uuid = device_uuid(torch.cuda.get_device_properties(0).uuid)
            analysis = paths.analyze(args.source_run, args.weights, args.output.parent, torch.device("cuda:0"))
            if analysis["uuid"] != uuid:
                raise ValueError("Stage source and analysis GPU mismatch")
            result = {"status": "ok", "uuid": uuid, "analysis": analysis,
                      "driver_sha256": sha256_file(Path(__file__)), "production_ready": False}
        except BaseException as exc:
            result = {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}
        write_result(args.output, result)
        return 0 if result["status"] == "ok" else 1
    if args.output_dir is None or not args.gpu_uuid:
        parser.error("Monitored run requires fresh output-dir and GPU UUID")
    summary = run(args)
    print(json.dumps({"status": summary["status"], "output_dir": str(args.output_dir)}))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
