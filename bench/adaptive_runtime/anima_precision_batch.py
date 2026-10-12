"""Bounded Anima MLP2 precision screening, with reusable monitored workers."""

import argparse
import json
import math
from pathlib import Path
import time

import torch

from bench.adaptive_runtime import anima_precision_batch_policy as policy
from bench.adaptive_runtime import anima_update_comparison as updates
from bench.adaptive_runtime import probe_anima_train as probe
from bench.adaptive_runtime import search_anima_precision as search
from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime.compare_training import compare
from library.training.adaptive_runtime.profile_search import comparison_gate
from library.training.auto_block_swap.process import write_result


SCOPE = "anima_mlp2_bounded_initial_state_single_sigma_screening"


def identities(args):
    return {
        "weights_sha256": sha256_file(args.weights),
        "inputs_sha256": sha256_file(args.inputs),
        "probe_sha256": sha256_file(Path(probe.__file__)),
        "update_sha256": sha256_file(Path(updates.__file__)),
        "policy_sha256": sha256_file(Path(policy.__file__)),
        "batch_sha256": sha256_file(Path(__file__)),
        "dependency_sha256": {
            name: sha256_file(Path(__file__).resolve().parents[2] / name)
            for name in (
                "bench/adaptive_runtime/replay.py",
                "bench/adaptive_runtime/training_capture.py",
                "bench/adaptive_runtime/scaling.py",
                "bench/adaptive_runtime/probe_monitor.py",
                "library/training/adaptive_runtime/precision.py",
                "library/training/adaptive_runtime/profile_search.py",
                "library/training/adaptive_runtime/training_precision.py",
                "library/training/adaptive_runtime/islands.py",
                "library/anima/models.py",
                "library/anima/weights.py",
                "networks/attention_dispatch.py",
                "networks/lora_modules/lora.py",
                "networks/lora_anima/network.py",
                "networks/attn_fuse.py",
            )
        },
        "comparison_sha256": sha256_file(
            Path(search.compare.__module__.replace(".", "/")).with_suffix(".py")
        ),
        "gpu_uuid": args.gpu_uuid,
        "swap": args.swap,
        "loss_scale": args.loss_scale,
        "steps": 1,
        "single_sigma": 0.5,
        "backend": "torch",
    }


def command(args, directory, precision, *, reference=None, blocks=None):
    argv = search.worker_command(
        args,
        directory,
        precision,
        patterns=("*",) if precision == "fp16-islands" and blocks is None else (),
        fp16_patterns=tuple(f"blocks.{i}.mlp.layer2" for i in blocks or ()),
        replay=reference,
    )
    return [*argv, "--capture-update"]


def worker(
    args, directory, precision, *, reference=None, blocks=None, gpu_check, monitor
):
    directory.mkdir(exist_ok=False)
    result = search.launch_if_idle(
        command(args, directory, precision, reference=reference, blocks=blocks),
        directory,
        args,
        gpu_check,
        monitor,
    )
    if not search._complete(
        result, directory, 1, args.gpu_uuid, reference=reference is None
    ):
        raise ValueError(
            f"Incomplete {directory.name} worker: {result.get('status')}/{result.get('returncode')}"
        )
    if precision == "fp16-islands":
        assignments = (result.get("precision_manifest") or {}).get("assignments", {})
        actual = {name for name, dtype in assignments.items() if dtype == "fp16"}
        expected = {f"blocks.{i}.mlp.layer2" for i in blocks or ()}
        if not assignments or actual != expected:
            raise ValueError("Realized FP16 assignments differ from candidate")
    return result


def evaluate(reference, directory, comparator, update_comparator):
    comparison = comparator(
        reference / "training-capture", directory / "training-capture"
    )
    if comparison_gate(comparison) is None:
        raise ValueError("Invalid fixed-state gradient comparison")
    update = update_comparator(reference, directory)
    if (
        update.get("status") != "compared"
        or update.get("scope") != "initial_empty_adamw_same_state_single_step"
    ):
        raise ValueError("Invalid same-state update comparison")
    write_result(directory / "comparison.json", comparison)
    write_result(directory / "update-comparison.json", update)
    return comparison, update


def evidence_hashes(directory):
    names = (
        "result.json",
        "supervisor.json",
        "comparison.json",
        "update-comparison.json",
        "training-capture/manifest.json",
        "update-capture/manifest.json",
    )
    return {name: sha256_file(directory / name) for name in names}


def measurements(result):
    step = result["updates"][0]
    return {
        "monitor": result.get("monitor"),
        "captured_step_seconds": step["seconds"],
        "peak_allocated": step["peak_allocated"],
        "peak_reserved": step["peak_reserved"],
        "performance_validated": False,
        "timing_scope": "single_cold_step_includes_capture_and_update_io",
    }


def check_output_budget(root):
    if sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) > 2 * 1024**3:
        raise ValueError("Batch output exceeds checked 2GiB budget")


def preflight(args, gpu_check):
    root = args.output_dir.resolve()
    if (
        not args.weights.is_file()
        or not args.inputs.is_file()
        or args.swap < 0
        or args.swap > 26
        or args.max_new_candidates < 1
        or args.max_new_candidates > 4
        or args.steps != 1
        or args.single_sigma != 0.5
        or not math.isfinite(args.loss_scale)
        or args.loss_scale < 1
        or not math.isfinite(args.worker_timeout)
        or args.worker_timeout <= 0
        or not math.isfinite(args.batch_timeout)
        or args.batch_timeout <= 0
        or any(
            root == p or root in p.parents or p in root.parents
            for p in (args.weights.resolve(), args.inputs.resolve())
        )
    ):
        raise ValueError(
            "Invalid batch inputs, fixed configuration, disjoint output, or budget"
        )
    if root.exists() != args.continue_run:
        raise ValueError(
            "Fresh output required, or explicitly continue an existing batch"
        )
    return (
        gpu_check(args.gpu_uuid),
        identities(args),
        policy.weight_inventory(args.weights),
    )


def load_existing(root, identity, comparator, update_comparator):
    environment = json.loads((root / "environment.json").read_text())
    report = json.loads((root / "summary.json").read_text())
    if (
        environment.get("identity") != identity
        or report.get("scope") != SCOPE
        or report.get("production_ready") is not False
        or report.get("status") != "screening"
        or not report.get("controls_complete")
    ):
        raise ValueError("Only a bound completed screening segment may continue")
    for item in [*report["controls"], *report["candidates"]]:
        directory = root / item["directory"]
        if evidence_hashes(directory) != item["evidence_sha256"]:
            raise ValueError("Existing worker evidence changed")
        comparison, update = evaluate(
            root / "reference", directory, comparator, update_comparator
        )
        if comparison != item["comparison"] or update != item["update_comparison"]:
            raise ValueError("Existing numerical evidence changed")
    return report


def run(
    args,
    *,
    gpu_check=search.check_gpu_idle,
    monitor=search.run_monitored,
    comparator=compare,
    update_comparator=updates.compare_update,
):
    torch.set_num_threads(4)
    started = time.monotonic()
    gpu, identity, inventory = preflight(args, gpu_check)
    root = args.output_dir
    if args.continue_run:
        report = load_existing(root, identity, comparator, update_comparator)
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_result(
            root / "environment.json",
            {
                "identity": identity,
                "gpu": gpu,
                "weights": str(args.weights.resolve()),
                "inputs": str(args.inputs.resolve()),
                "candidate_budget": policy.MAX_CANDIDATES,
            },
        )
        report = {
            "status": "starting",
            "scope": SCOPE,
            "production_ready": False,
            "controls": [],
            "candidates": [],
            "controls_complete": False,
            "pending": [],
            "segments": [],
        }
    original_timeout = args.worker_timeout
    segment = {"new_candidates": 0, "preflight_seconds": time.monotonic() - started}
    report["segments"].append(segment)

    def launch(directory, precision, **kwargs):
        check_output_budget(root)
        remaining = args.batch_timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Segment end-to-end budget exhausted")
        args.worker_timeout = min(original_timeout, remaining)
        result = worker(
            args, directory, precision, gpu_check=gpu_check, monitor=monitor, **kwargs
        )
        check_output_budget(root)
        return result

    try:
        reference = root / "reference"
        if not report["controls_complete"]:
            result = launch(reference, "fp32-reference")
            # Validate the reference update evidence independently before any candidate.
            updates._read(reference)
            report["reference"] = {
                "directory": "reference",
                **measurements(result),
                "result_sha256": sha256_file(reference / "result.json"),
            }
            for name, precision in (
                ("fp32-control", "fp16-islands"),
                ("bf16-control", "bf16"),
            ):
                if precision == "bf16" and search._capability(gpu)[0] < 8:
                    raise ValueError("Batch 1 BF16 diagnostic control requires Ampere+")
                directory = root / name
                result = launch(
                    directory, precision, reference=reference / "training-replay"
                )
                comparison, update = evaluate(
                    reference, directory, comparator, update_comparator
                )
                if (
                    name == "fp32-control"
                    and not comparison["within_experimental_tolerances"]
                ):
                    raise ValueError(
                        "FP32 control did not reproduce numerical reference"
                    )
                report["controls"].append(
                    {
                        "directory": name,
                        "comparison": comparison,
                        "update_comparison": update,
                        **measurements(result),
                        "evidence_sha256": evidence_hashes(directory),
                    }
                )
                write_result(root / "summary.json", report)
            report["controls_complete"] = True
        for _ in range(args.max_new_candidates):
            if not report["pending"]:
                report["pending"] = policy.next_stage(report["candidates"])
            if not report["pending"]:
                break
            choice = report["pending"][0]
            directory = root / f"candidate-{len(report['candidates']) + 1:02d}"
            report.update(status="running", active_candidate=choice)
            write_result(root / "summary.json", report)
            result = launch(
                directory,
                "fp16-islands",
                reference=reference / "training-replay",
                blocks=choice["blocks"],
            )
            comparison, update = evaluate(
                reference, directory, comparator, update_comparator
            )
            report["candidates"].append(
                {
                    "candidate": choice,
                    "directory": directory.name,
                    "comparison": comparison,
                    "update_comparison": update,
                    **measurements(result),
                    "storage_estimate": policy.storage_estimate(
                        choice["blocks"], inventory
                    ),
                    "fp32_modules": result["fp32_modules"],
                    "evidence_sha256": evidence_hashes(directory),
                }
            )
            report["pending"].pop(0)
            segment["new_candidates"] += 1
            check_output_budget(root)
            write_result(root / "summary.json", report)
        if not report["pending"]:
            report["pending"] = policy.next_stage(report["candidates"])
        report["status"] = "screening" if report["pending"] else "completed"
        passed = [
            item
            for item in report["candidates"]
            if item["comparison"]["within_experimental_tolerances"]
        ]
        report["passing_candidates"] = [item["directory"] for item in passed]
        report["decision"] = (
            "continue_within_budget"
            if report["pending"]
            else "shortlist_requires_heldout_and_training_validation"
            if passed
            else "no_candidate_passed_within_linear_search_budget"
        )
        report["shortlist"] = [
            item["directory"]
            for item in sorted(
                passed,
                key=lambda item: (
                    -item["storage_estimate"]["weight_payload_reduction_fraction"],
                    policy.score(item),
                ),
            )[:2]
        ]
        with (root / "candidate-table.json").open("w") as stream:
            json.dump(report["candidates"], stream, indent=2, allow_nan=False)
    except BaseException as exc:
        report.update(status="failed", stop_reason=f"{type(exc).__name__}: {exc}")
    finally:
        segment["elapsed_seconds"] = time.monotonic() - started
        args.worker_timeout = original_timeout
        write_result(root / "summary.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "inputs", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--continue-run", action="store_true")
    parser.add_argument("--swap", type=int, default=20)
    parser.add_argument("--loss-scale", type=float, default=1024)
    parser.add_argument("--worker-timeout", type=float, default=150)
    parser.add_argument("--batch-timeout", type=float, default=480)
    parser.add_argument("--max-new-candidates", type=int, default=4)
    args = parser.parse_args(argv)
    args.steps, args.single_sigma = 1, 0.5
    report = run(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report.get("decision"),
                "candidates": len(report["candidates"]),
                "shortlist": report.get("shortlist"),
            }
        )
    )
    return 0 if report["status"] in {"screening", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
