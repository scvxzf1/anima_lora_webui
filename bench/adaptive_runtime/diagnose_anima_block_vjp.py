"""Bounded two-capture plus monitored full-token block-14 VJP diagnosis."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
from types import SimpleNamespace
import sys
import traceback

import torch
from safetensors.torch import load_file

from bench.adaptive_runtime import anima_block_vjp as vjp
from bench.adaptive_runtime import anima_block_vjp_capture as capture
from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime.compare_training import compare
from bench.adaptive_runtime.probe_anima_train import device_uuid
from bench.adaptive_runtime.probe_monitor import check_gpu_idle, run_monitored
from bench.adaptive_runtime.search_anima_precision import MLP, _complete, launch_if_idle, worker_command
from library.training.auto_block_swap.process import write_result


def _command(args, directory, *, reference=None):
    common = SimpleNamespace(weights=args.weights, inputs=args.inputs, swap=20, steps=1,
                             single_sigma=0.5, loss_scale=1 if reference is None else 1024)
    precision = "fp32-reference" if reference is None else "fp16-islands"
    command = worker_command(common, directory, precision,
                             replay=None if reference is None else reference / "training-replay",
                             fp16_patterns=() if reference is None else MLP)
    return [*command, "--capture-block-vjp"]


def validate_matrix_evidence(directory, analysis, *, capture_root=None):
    """Verify the persisted matrix and bind it to this run's captures."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    expected_cells = {"/".join((policy, x_side, g_side))
                      for policy in ("reference", "candidate")
                      for x_side in ("reference", "candidate")
                      for g_side in ("reference", "candidate")}
    shape = list(capture.SHAPE)
    expected_bytes = 8 * math.prod(shape) * 4
    block_index = analysis.get("block_index", 14)
    if (manifest != analysis.get("matrix_evidence")
            or manifest.get("schema") != "anima_block_vjp_matrix_v1"
            or manifest.get("scope") != vjp.scope_for_block(block_index)
            or manifest.get("block_index", 14) != block_index
            or manifest.get("production_ready") is not False
            or manifest.get("axis_order") != ["policy", "activation_source", "cotangent_source"]
            or manifest.get("file") != "matrix.safetensors"
            or manifest.get("shape") != shape
            or set(manifest.get("cells", [])) != expected_cells
            or len(manifest["cells"]) != 8
            or type(manifest.get("tensor_bytes")) is not int
            or manifest["tensor_bytes"] != expected_bytes
            or expected_bytes > capture.MAX_BYTES
            or manifest.get("replay") != analysis.get("replay")):
        raise ValueError("Incomplete block VJP matrix manifest")
    source = manifest.get("source")
    if (not isinstance(source, dict)
            or any(source.get(key) != value for key, value in analysis["source"].items())):
        raise ValueError("Block VJP matrix source mismatch")

    root = directory.parent.parent if directory.parent.name == "analysis" else directory.parent
    if capture_root is not None:
        root = Path(capture_root)
    for side in ("reference", "candidate"):
        capture_dir = root / side / "block-vjp"
        if block_index != 14:
            capture_dir = capture_dir / f"block-{block_index:02d}"
        captured = json.loads((capture_dir / "manifest.json").read_text())
        capture_file = capture_dir / "capture.safetensors"
        state = analysis["state"][side]
        if (captured.get("block_index", 14) != block_index
                or captured.get("file") != capture_file.name
                or capture_file.stat().st_size > capture.MAX_BYTES
                or sha256_file(capture_file) != captured.get("sha256")
                or manifest.get("capture_sha256", {}).get(side) != captured.get("sha256")
                or captured.get("replay") != analysis["replay"]
                or any(captured.get(key) != value for key, value in state.items())
                or any(captured.get("signature", {}).get(key) != value
                       for key, value in analysis["source"].items())
                or (side == "reference" and captured.get("signature") != source)):
            raise ValueError(f"Block VJP matrix {side} capture binding mismatch")

    path = directory / "matrix.safetensors"
    if path.stat().st_size > capture.MAX_BYTES or sha256_file(path) != manifest.get("sha256"):
        raise ValueError("Block VJP matrix hash/budget mismatch")
    tensors = load_file(str(path), device="cpu")
    inventory = {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                       "bytes": value.numel() * value.element_size()}
                 for key, value in tensors.items()}
    if (set(tensors) != expected_cells
            or any(value.dtype != torch.float32 or list(value.shape) != shape
                   or not bool(torch.isfinite(value).all()) for value in tensors.values())
            or sum(item["bytes"] for item in inventory.values()) != expected_bytes):
        raise ValueError("Invalid full-shape VJP matrix tensor inventory")
    return inventory


def run(args, *, gpu_check=check_gpu_idle, monitor=run_monitored, comparator=compare):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if (not args.weights.is_file() or not args.inputs.is_file() or args.swap != 20
            or getattr(args, "block_index", 14) != 14
            or not 0 < args.worker_timeout <= 150 or not math.isfinite(args.worker_timeout)
            or not 0 < args.analysis_timeout <= 300 or not math.isfinite(args.analysis_timeout)
            or 2 * args.worker_timeout + args.analysis_timeout > 720):
        raise ValueError("Existing inputs, swap20 and bounded 150/300/720-second timeouts required")
    gpu = gpu_check(args.gpu_uuid)
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "gpu_uuid": args.gpu_uuid, "gpu": gpu, "swap": 20, "steps": 1,
                   "sigma": 0.5, "backend": "torch", "checkpoint": "full",
                   "weights_sha256": sha256_file(args.weights),
                   "inputs_sha256": sha256_file(args.inputs),
                   "probe_sha256": sha256_file(Path(__file__).with_name("probe_anima_train.py")),
                   "helper_sha256": sha256_file(Path(capture.__file__)),
                   "analysis_sha256": sha256_file(Path(vjp.__file__)),
                   "driver_sha256": sha256_file(Path(__file__))}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_result(args.output_dir / "environment.json", environment)
    summary = {"status": "starting", "scope": vjp.SCOPE, "production_ready": False,
               "gpu_uuid": args.gpu_uuid, "workers": {}}
    write_result(args.output_dir / "summary.json", summary)
    try:
        reference = args.output_dir / "reference"
        for kind in ("reference", "candidate"):
            directory = args.output_dir / kind
            directory.mkdir(exist_ok=False)
            result = launch_if_idle(_command(args, directory, reference=None if kind == "reference" else reference),
                                    directory, args, gpu_check, monitor)
            summary["workers"][kind] = {"directory": str(directory), "result": result}
            write_result(args.output_dir / "summary.json", summary)
            if not _complete(result, directory, 1, args.gpu_uuid, reference=kind == "reference"):
                raise ValueError(f"Incomplete {kind} worker/training capture/replay")
            manifest, _ = capture.read_capture(directory / "block-vjp")
            if (manifest.get("uuid") != args.gpu_uuid
                    or manifest["signature"].get("weights_sha256") != environment["weights_sha256"]
                    or manifest["signature"].get("inputs_sha256") != environment["inputs_sha256"]
                    or manifest["signature"].get("probe_sha256") != environment["probe_sha256"]
                    or manifest["signature"].get("block_vjp_helper_sha256") != environment["helper_sha256"]):
                raise ValueError(f"{kind} capture environment/source mismatch")
        comparison = comparator(reference / "training-capture",
                                args.output_dir / "candidate" / "training-capture")
        if comparison.get("status") != "compared":
            raise ValueError("Whole-network comparator did not complete")
        write_result(args.output_dir / "comparison.json", comparison)
        vjp.validate_pair(reference / "block-vjp", args.output_dir / "candidate" / "block-vjp",
                          comparison)
        analysis_dir = args.output_dir / "analysis"
        analysis_dir.mkdir(exist_ok=False)
        command = [sys.executable, "-m", "bench.adaptive_runtime.diagnose_anima_block_vjp",
                   "--analyze", "--reference", str(reference / "block-vjp"),
                   "--candidate", str(args.output_dir / "candidate" / "block-vjp"),
                   "--comparison", str(args.output_dir / "comparison.json"),
                   "--weights", str(args.weights), "--output", str(analysis_dir / "result.json")]
        limits = SimpleNamespace(gpu_uuid=args.gpu_uuid, worker_timeout=args.analysis_timeout)
        result = launch_if_idle(command, analysis_dir, limits, gpu_check, monitor)
        summary["workers"]["analysis"] = {"directory": str(analysis_dir), "result": result}
        if result.get("status") != "ok" or result.get("returncode") != 0:
            raise ValueError(f"Monitored block analysis failed: {result.get('stop_reason', 'incomplete')}")
        report = json.loads((analysis_dir / "result.json").read_text())
        analysis = report.get("analysis", {})
        if (report.get("status") != "ok" or report.get("uuid") != args.gpu_uuid
                or analysis.get("status") != "analyzed" or analysis.get("uuid") != args.gpu_uuid
                or analysis.get("cell_count") != 8 or analysis.get("scope") != vjp.SCOPE
                or analysis.get("production_ready") is not False
                or report.get("driver_sha256") != environment["driver_sha256"]
                or report.get("analysis_sha256") != environment["analysis_sha256"]
                or analysis.get("source") != {key: environment[key] for key in
                    ("weights_sha256", "inputs_sha256", "probe_sha256")}
                    | {"block_vjp_helper_sha256": environment["helper_sha256"]}):
            raise ValueError("Incomplete block VJP result")
        validate_matrix_evidence(analysis_dir / "matrix-evidence", analysis)
        summary.update(status="completed", comparison=comparison, analysis=analysis)
    except BaseException as exc:
        summary.update(status="failed", stop_reason=f"{type(exc).__name__}: {exc}",
                       traceback=traceback.format_exc())
    finally:
        write_result(args.output_dir / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--block-index", type=int, choices=(14, 15, 17, 27), default=14)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--swap20", action="store_true")
    parser.add_argument("--worker-timeout", type=float, default=150)
    parser.add_argument("--analysis-timeout", type=float, default=300)
    args = parser.parse_args(argv)
    if args.analyze:
        if not all((args.reference, args.candidate, args.comparison, args.output)):
            parser.error("Analysis requires both captures, comparison, output and weights")
        try:
            if args.output.exists() or not torch.cuda.is_available():
                raise ValueError("Analysis output must not exist and monitored CUDA is required")
            uuid = device_uuid(torch.cuda.get_device_properties(0).uuid)
            comparison = json.loads(args.comparison.read_text())
            analysis = vjp.analyze(args.reference, args.candidate, comparison,
                                   weights=args.weights, device=torch.device("cuda:0"),
                                   matrix_directory=args.output.parent / "matrix-evidence",
                                   block_index=args.block_index)
            if analysis["uuid"] != uuid:
                raise ValueError("Analysis GPU UUID differs from capture")
            result = {"status": "ok", "analysis": analysis, "uuid": uuid, "production_ready": False,
                      "driver_sha256": sha256_file(Path(__file__)),
                      "analysis_sha256": sha256_file(Path(vjp.__file__))}
        except BaseException as exc:
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                      "traceback": traceback.format_exc(), "production_ready": False}
        write_result(args.output, result)
        print(json.dumps({"status": result["status"]}))
        return 0 if result["status"] == "ok" else 1
    if not all((args.inputs, args.output_dir, args.gpu_uuid, args.swap20)):
        parser.error("Probe requires inputs, output-dir, gpu-uuid and --swap20")
    args.swap = 20
    summary = run(args)
    print(json.dumps({"status": summary["status"], "output_dir": str(args.output_dir)}))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
