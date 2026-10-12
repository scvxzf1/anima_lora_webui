"""One-reference, one-MLP-candidate, one-analysis monitored Anima local VJP probe."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
from types import SimpleNamespace
import sys

import torch

from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime.anima_local_vjp import analyze, read_manifest
from bench.adaptive_runtime.compare_training import compare
from bench.adaptive_runtime.probe_anima_train import device_uuid
from bench.adaptive_runtime.probe_monitor import check_gpu_idle, run_monitored
from bench.adaptive_runtime.search_anima_precision import MLP, _complete, launch_if_idle, worker_command
from library.training.auto_block_swap.process import write_result


def run(args, *, gpu_check=check_gpu_idle, monitor=run_monitored, comparator=compare):
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if (not args.weights.is_file() or not args.inputs.is_file() or
            not 0 <= args.swap <= 26 or not math.isfinite(args.worker_timeout) or
            args.worker_timeout <= 0):
        raise ValueError("Existing inputs, swap 0..26 and positive timeout required")
    gpu_check(args.gpu_uuid)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    environment = {"python": platform.python_version(), "platform": platform.platform(),
                   "gpu_uuid": args.gpu_uuid, "gpu": gpu_check(args.gpu_uuid),
                   "weights": {"path": str(args.weights.resolve()), "sha256": sha256_file(args.weights)},
                   "inputs": {"path": str(args.inputs.resolve()), "sha256": sha256_file(args.inputs)},
                   "probe_sha256": sha256_file(Path(__file__).with_name("probe_anima_train.py")),
                   "local_vjp_source_sha256": sha256_file(Path(__file__).with_name("anima_local_vjp.py")),
                   "mlp_path_source_sha256": sha256_file(Path(__file__).with_name("anima_mlp_path.py")),
                   "local_vjp_profile": args.local_vjp_profile,
                   "swap": args.swap, "steps": 1, "sigma": 0.5, "backend": "torch"}
    write_result(args.output_dir / "environment.json", environment)
    summary = {"status": "starting", "production_ready": False,
               "scope": "local_fixed_cotangent_not_whole_network_bound_or_quality_certificate",
               "gpu_uuid": args.gpu_uuid, "workers": []}
    write_result(args.output_dir / "summary.json", summary)
    common = SimpleNamespace(weights=args.weights, inputs=args.inputs, swap=args.swap,
                             steps=1, single_sigma=0.5, capture_local_vjp=True,
                             local_vjp_profile=args.local_vjp_profile)
    try:
        reference = args.output_dir / "reference"
        reference.mkdir()
        common.loss_scale = 1
        result = launch_if_idle(worker_command(common, reference, "fp32-reference"),
                                reference, args, gpu_check, monitor)
        summary["workers"].append({"kind": "reference", "result": result})
        if not _complete(result, reference, 1, args.gpu_uuid, reference=True):
            summary.update(status="reference_failed", stop_reason=result.get("stop_reason", "incomplete"))
            return summary
        ref_manifest = read_manifest(reference / "local-vjp")
        if (ref_manifest["uuid"] != args.gpu_uuid
                or ref_manifest["signature"].get("weights_sha256") != environment["weights"]["sha256"]
                or ref_manifest["signature"].get("inputs_sha256") != environment["inputs"]["sha256"]
                or ref_manifest["signature"].get("probe_sha256") != environment["probe_sha256"]
                or ref_manifest["signature"].get("local_vjp_source_sha256")
                    != environment["local_vjp_source_sha256"]
                or ref_manifest["signature"].get("local_vjp_profile") != args.local_vjp_profile
                or (args.local_vjp_profile == "mlp-path" and
                    ref_manifest["signature"].get("mlp_path_source_sha256") !=
                    environment["mlp_path_source_sha256"])):
            raise ValueError("Reference local VJP environment mismatch")
        candidate = args.output_dir / "candidate"
        candidate.mkdir()
        common.loss_scale = 1024
        result = launch_if_idle(worker_command(common, candidate, "fp16-islands",
                                               replay=reference / "training-replay", fp16_patterns=MLP),
                                candidate, args, gpu_check, monitor)
        summary["workers"].append({"kind": "candidate", "result": result})
        if not _complete(result, candidate, 1, args.gpu_uuid):
            summary.update(status="candidate_failed", stop_reason=result.get("stop_reason", "incomplete"))
            return summary
        trial_manifest = read_manifest(candidate / "local-vjp")
        if (trial_manifest["uuid"] != args.gpu_uuid or
                trial_manifest["signature"].get("weights_sha256") != environment["weights"]["sha256"] or
                trial_manifest["signature"].get("inputs_sha256") != environment["inputs"]["sha256"] or
                trial_manifest["signature"].get("probe_sha256") != environment["probe_sha256"] or
                trial_manifest["signature"].get("local_vjp_source_sha256")
                    != environment["local_vjp_source_sha256"] or
                trial_manifest["signature"].get("local_vjp_profile") != args.local_vjp_profile or
                (args.local_vjp_profile == "mlp-path" and
                 trial_manifest["signature"].get("mlp_path_source_sha256") !=
                 environment["mlp_path_source_sha256"])):
            raise ValueError("Candidate local VJP environment mismatch")
        comparison = comparator(reference / "training-capture", candidate / "training-capture")
        if comparison.get("status") != "compared":
            raise ValueError("Whole-network capture comparison failed")
        write_result(args.output_dir / "comparison.json", comparison)
        analysis = args.output_dir / "analysis"
        analysis.mkdir()
        command = [sys.executable, "-m", "bench.adaptive_runtime.diagnose_anima_local_vjp",
                   "--analyze", "--reference", str(reference / "local-vjp"),
                   "--candidate", str(candidate / "local-vjp"),
                   "--comparison", str(args.output_dir / "comparison.json"),
                   "--output", str(analysis / "result.json")]
        result = launch_if_idle(command, analysis, args, gpu_check, monitor)
        summary["workers"].append({"kind": "analysis", "result": result})
        if result.get("status") != "ok" or result.get("returncode") != 0:
            summary.update(status="analysis_failed", stop_reason=result.get("stop_reason", "incomplete"))
            return summary
        report = json.loads((analysis / "result.json").read_text())
        if (report.get("status") != "ok" or report.get("analysis", {}).get("status") != "analyzed"
                or report.get("uuid") != args.gpu_uuid
                or report["analysis"].get("uuid") != args.gpu_uuid):
            raise ValueError("Incomplete local VJP analysis")
        summary.update(status="completed", comparison=comparison, analysis=report["analysis"])
        return summary
    except BaseException as exc:
        summary.update(status="failed", stop_reason="orchestration_error",
                       error=f"{type(exc).__name__}: {exc}")
        return summary
    finally:
        write_result(args.output_dir / "summary.json", summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--swap", type=int, default=20)
    parser.add_argument("--worker-timeout", type=float, default=150)
    parser.add_argument("--local-vjp-profile", choices=("linear", "mlp-path"), default="linear")
    args = parser.parse_args(argv)
    if args.analyze:
        if not all((args.reference, args.candidate, args.comparison, args.output)):
            parser.error("Analysis requires reference, candidate, comparison and output")
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if not torch.cuda.is_available():
                raise ValueError("Analysis requires monitored CUDA worker")
            uuid = device_uuid(torch.cuda.get_device_properties(0).uuid)
            comparison = json.loads(args.comparison.read_text())
            report = analyze(args.reference, args.candidate, comparison, device=torch.device("cuda:0"))
            if report["uuid"] != uuid:
                raise ValueError("Analysis CUDA device UUID mismatch")
            result = {"status": "ok", "analysis": report, "uuid": uuid, "production_ready": False}
        except BaseException as exc:
            result = {"status": "error", "error": f"{type(exc).__name__}: {exc}",
                      "production_ready": False}
        write_result(args.output, result)
        print(json.dumps({"status": result["status"]}))
        return 0 if result["status"] == "ok" else 1
    if not all((args.weights, args.inputs, args.output_dir, args.gpu_uuid)):
        parser.error("Probe requires weights, inputs, output-dir and gpu-uuid")
    summary = run(args)
    print(json.dumps({"status": summary["status"], "output_dir": str(args.output_dir)}))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
