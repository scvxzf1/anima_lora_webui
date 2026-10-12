"""Monitored bench-only local post-RoPE K intervention on fixed block17 inputs."""

import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import traceback

import torch

from bench.adaptive_runtime import anima_k_local as local
from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime.probe_anima_train import device_uuid
from bench.adaptive_runtime.probe_monitor import check_gpu_idle, run_monitored
from bench.adaptive_runtime.search_anima_precision import launch_if_idle
from library.training.auto_block_swap.process import write_result


SOURCE_KEYS = (
    "matrix_run",
    "qknorm_run",
    "core_run",
    "source_run",
    "stage_run",
    "run",
    "distribution_run",
    "activation_run",
)
MAX_OUTPUT_BYTES = 1024**3


def run(args, *, gpu_check=check_gpu_idle, monitor=run_monitored):
    sources = [getattr(args, key).resolve() for key in SOURCE_KEYS]
    output = args.output_dir.resolve()
    if (
        args.output_dir.exists()
        or any(not path.is_dir() for path in sources)
        or not args.weights.is_file()
        or not math.isfinite(args.analysis_timeout)
        or args.analysis_timeout < 420
        or any(
            output == path or output in path.parents or path in output.parents
            for path in (*sources, args.weights.resolve())
        )
    ):
        raise ValueError(
            "Fresh disjoint output, sources/weights, and timeout >=420 required"
        )
    gpu = gpu_check(args.gpu_uuid)
    environment = {
        "gpu_uuid": args.gpu_uuid,
        "gpu": gpu,
        "training_steps": 0,
        **{key: str(getattr(args, key)) for key in SOURCE_KEYS},
        "weights_sha256": sha256_file(args.weights),
        "activation_report_sha256": sha256_file(
            args.activation_run / "activation-distribution.json"
        ),
        "distribution_report_sha256": sha256_file(
            args.distribution_run / "distribution.json"
        ),
        "source_summary_sha256": {
            key: sha256_file(getattr(args, key) / "summary.json")
            for key in SOURCE_KEYS[:6]
        },
        **local.code_hashes(),
        "driver_sha256": sha256_file(Path(__file__)),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_result(args.output_dir / "environment.json", environment)
    summary = {"status": "starting", "scope": local.SCOPE, "production_ready": False}
    try:
        command = [
            sys.executable,
            "-m",
            "bench.adaptive_runtime.diagnose_anima_k_local",
            "--analyze",
        ]
        for key in SOURCE_KEYS:
            command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
        command.extend(
            [
                "--weights",
                str(args.weights),
                "--output",
                str(args.output_dir / "result.json"),
            ]
        )
        limits = SimpleNamespace(
            gpu_uuid=args.gpu_uuid, worker_timeout=args.analysis_timeout
        )
        worker = launch_if_idle(command, args.output_dir, limits, gpu_check, monitor)
        summary["worker"] = worker
        if worker.get("status") != "ok" or worker.get("returncode") != 0:
            raise ValueError("Local K worker failed")
        result = json.loads((args.output_dir / "result.json").read_text())
        analysis = result.get("analysis", {})
        if (
            result.get("status") != "ok"
            or result.get("uuid") != args.gpu_uuid
            or result.get("driver_sha256") != environment["driver_sha256"]
            or analysis.get("uuid") != args.gpu_uuid
            or analysis.get("analysis_code_sha256") != local.code_hashes()
            or analysis.get("source_binding", {}).get("activation_report_sha256")
            != environment["activation_report_sha256"]
            or analysis.get("source_binding", {}).get("distribution_report_sha256")
            != environment["distribution_report_sha256"]
            or analysis.get("forward_bitwise_equal") != {"reference_core": True}
            or set(analysis.get("replay", {}))
            != {f"{name}_v{side}" for name in local.CASES for side in ("R", "C")}
            or set(analysis.get("cases", {})) != {"R", "C"}
            or any(
                set(analysis["cases"][side]) != set(local.CASES) for side in ("R", "C")
            )
            or analysis.get("disk_bytes_c") != 0
        ):
            raise ValueError("Local K result/GPU/source/coverage binding invalid")
        if set(analysis.get("endpoint", {})) != {
            f"{name}_v{side}_{origin}"
            for name in ("baseline", "full")
            for side in ("R", "C")
            for origin in ("matrix", "interpolation")
        } | {"baseline_vR_historical"}:
            raise ValueError("Local K endpoint coverage incomplete")
        if any(
            item.get("finite") is not True
            or item.get("relative_l2") is None
            or not 0 <= item["relative_l2"] <= 1e-4
            for item in (*analysis["replay"].values(), *analysis["endpoint"].values())
        ):
            raise ValueError("Local K replay/endpoint fidelity incomplete")
        for key in SOURCE_KEYS:
            if str(getattr(args, key)) != environment[key]:
                raise ValueError("Source path changed")
        if sha256_file(args.weights) != environment["weights_sha256"]:
            raise ValueError("Weight changed during worker")
        if (
            sha256_file(args.activation_run / "activation-distribution.json")
            != environment["activation_report_sha256"]
        ):
            raise ValueError("Activation report changed during worker")
        if (
            sha256_file(args.distribution_run / "distribution.json")
            != environment["distribution_report_sha256"]
        ):
            raise ValueError("Distribution report changed during worker")
        if any(
            sha256_file(getattr(args, key) / "summary.json") != digest
            for key, digest in environment["source_summary_sha256"].items()
        ):
            raise ValueError("Source summary changed during worker")
        if (
            sum(
                path.stat().st_size
                for path in args.output_dir.rglob("*")
                if path.is_file()
            )
            > MAX_OUTPUT_BYTES
        ):
            raise ValueError("Local K aggregate output exceeds 1GiB")
        summary.update(status="completed", analysis=analysis)
    except BaseException as exc:
        summary.update(
            status="failed",
            stop_reason=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    finally:
        write_result(args.output_dir / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (*SOURCE_KEYS, "weights"):
        parser.add_argument("--" + key.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--gpu-uuid", default="GPU-806debd9-d254-da57-d515-ccae15b4ac6a"
    )
    parser.add_argument("--analysis-timeout", type=float, default=1200)
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
            analysis = local.analyze(
                args.matrix_run,
                args.qknorm_run,
                args.core_run,
                args.source_run,
                args.stage_run,
                args.run,
                args.distribution_run,
                args.activation_run,
                args.weights,
                torch.device("cuda:0"),
            )
            if analysis["uuid"] != uuid:
                raise ValueError("Source and CUDA GPU mismatch")
            report = {
                "status": "ok",
                "uuid": uuid,
                "analysis": analysis,
                "driver_sha256": sha256_file(Path(__file__)),
                "production_ready": False,
            }
        except BaseException as exc:
            report = {
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        write_result(args.output, report)
        return 0 if report["status"] == "ok" else 1
    if args.output_dir is None:
        parser.error("Monitored run requires output-dir")
    summary = run(args)
    print(json.dumps({"status": summary["status"], "output_dir": str(args.output_dir)}))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
