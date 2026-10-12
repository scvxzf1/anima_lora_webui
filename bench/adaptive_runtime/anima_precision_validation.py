"""Bounded two-bucket, restored-state validation of the Batch 1 Anima shortlist."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from bench.adaptive_runtime import anima_update_comparison as warmup_updates
from bench.adaptive_runtime import anima_validation_probe as validation
from bench.adaptive_runtime import anima_validation_updates as updates
from bench.adaptive_runtime import search_anima_precision as search
from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime.anima_validation_state import StateBundle
from bench.adaptive_runtime.anima_validation_evidence import validate_validation_sources, validate_warmup_bundle
from bench.adaptive_runtime.compare_training import compare
from library.training.auto_block_swap.process import write_result


SCOPE = "anima_batch2_two_bucket_independent_state_screening"
LIMIT = 4 * 1024**3
PATTERNS = {
    "candidate-05": tuple(f"blocks.{i}.mlp.layer2" for i in range(14, 28)),
    "candidate-04": tuple(f"blocks.{i}.mlp.layer2" for i in range(21, 28)),
}
ROLES = ("representative", "heldout")
TASKS = ((None, "warmup", "warmup"),) + tuple(
    (role, name, f"{role}-{name}")
    for role in ROLES
    for name in ("reference", "fp32-control", *PATTERNS)
)
SOURCE_FILES = (*validation.SOURCE_FILES,
                "bench/adaptive_runtime/compare_training.py",
                "bench/adaptive_runtime/anima_precision_validation.py",
                "bench/adaptive_runtime/search_anima_precision.py",
                "bench/adaptive_runtime/probe_monitor.py",
                "library/training/adaptive_runtime/precision.py",
                "library/training/adaptive_runtime/islands.py",
                "networks/attention_dispatch.py")


def _selection(args):
    manifest = json.loads(args.selection.read_text())
    rows = manifest.get("selection", [])
    if (manifest.get("schema") != "adaptive_anima_batch2_selection_v1"
            or manifest.get("status") != "ready_for_probe"
            or manifest.get("model_family") != "anima"
            or len(rows) != 2 or {r.get("role") for r in rows} != set(ROLES)
            or {r.get("anima_token_count") for r in rows} != {4032, 4200}
            or any(r.get("stem") == manifest.get("batch1_excluded_stem") for r in rows)):
        raise ValueError("Selection must preregister two distinct new Anima buckets")
    selected = {}
    for row in rows:
        role = row["role"]
        path = Path(row["inputs_path"]).resolve()
        if (path.parent != args.selection.resolve().parent
                or path.name != f"{role}-{row['anima_token_count']}-inputs.safetensors"
                or sha256_file(path) != row["inputs_sha256"]
                or not path.with_suffix(".json").is_file()):
            raise ValueError(f"Invalid {role} input package or selection digest")
        selected[role] = path
    if selected["representative"] == selected["heldout"]:
        raise ValueError("Validation roles must use different inputs")
    return selected


def identity(args, selected):
    root = Path(__file__).resolve().parents[2]
    files = {name: sha256_file(root / name) for name in SOURCE_FILES}
    return {
        "selection_sha256": sha256_file(args.selection),
        "inputs": {role: {"sha256": sha256_file(path),
                          "sidecar_sha256": sha256_file(path.with_suffix(".json"))}
                   for role, path in selected.items()},
        "warmup_inputs_sha256": sha256_file(args.warmup_inputs),
        "warmup_sidecar_sha256": sha256_file(args.warmup_inputs.with_suffix(".json")),
        "weights_sha256": sha256_file(args.weights), "source_sha256": files,
        "gpu_uuid": args.gpu_uuid, "backend": "torch", "checkpoint": "full",
        "swap": args.swap, "loss_scale": args.loss_scale,
        "patterns": {k: list(v) for k, v in PATTERNS.items()},
        "case_schedule": [{"state_label": label, "sigma": sigma,
                           "case_id": updates.case_id(label, sigma)}
                          for label, sigma in updates.CASES],
    }


def command(args, directory, role, name, selected):
    warmup = role is None
    reference = warmup or name == "reference"
    probe_args = argparse.Namespace(weights=args.weights,
                                    inputs=args.warmup_inputs if warmup else selected[role],
                                    swap=args.swap, steps=1 if warmup else 6,
                                    single_sigma=0.5 if warmup else None,
                                    loss_scale=args.loss_scale)
    replay = None if reference else args.output_dir / f"{role}-reference" / "training-replay"
    argv = search.worker_command(
        probe_args, directory, "fp32-reference" if reference else "fp16-islands",
        patterns=("*",) if name == "fp32-control" else (),
        fp16_patterns=PATTERNS.get(name, ()), replay=replay,
    )
    state_dir = args.output_dir / "validation-states"
    return [*argv, "--capture-update", "--export-validation-states" if warmup
            else "--validation-states", str(state_dir)]


def _output_bytes(root):
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def _budget(root):
    size = _output_bytes(root)
    if size > LIMIT:
        raise ValueError(f"Batch output exceeds checked 4GiB budget ({size} bytes)")
    return size


def _evidence(directory):
    return {str(p.relative_to(directory)): sha256_file(p)
            for p in sorted(directory.rglob("*")) if p.is_file()}


def _bundle(args, identity_data):
    state_identity = {"family": "anima", "weights_sha256": identity_data["weights_sha256"],
                      "source_code_sha256": validation.source_code_sha256(),
                      "backend": "torch", "rank": 4, "alpha": 4,
                      "optimizer": "torch.optim.AdamW", "scheduler": "none"}
    bundle = StateBundle(args.output_dir / "validation-states",
                         identity=state_identity, uuid=args.gpu_uuid)
    if bundle.manifest["provenance"]["warmup_inputs_sha256"] != identity_data["warmup_inputs_sha256"]:
        raise ValueError("Warmup state bundle has wrong source inputs")
    return bundle


def _assignments(result, name):
    if name == "warmup" or name == "reference":
        return
    assignments = (result.get("precision_manifest") or {}).get("assignments", {})
    actual = {module for module, dtype in assignments.items() if dtype == "fp16"}
    expected = set(PATTERNS.get(name, ()))
    if not assignments or actual != expected:
        raise ValueError(f"{name} realized FP16 modules differ: {sorted(actual ^ expected)}")


def _validate_worker(args, directory, role, name, result, identity_data, comparator,
                     update_comparator):
    reference = role is None or name == "reference"
    steps = 1 if role is None else 6
    if not search._complete(result, directory, steps, args.gpu_uuid, reference=reference):
        raise ValueError(f"Incomplete {directory.name} worker: {result.get('status')}/{result.get('returncode')}")
    _assignments(result, name)
    if role is None:
        warmup_updates._read(directory)
        validate_warmup_bundle(directory, _bundle(args, identity_data))
        return None, None
    updates._read(directory)
    validate_validation_sources(directory, args.output_dir / f"{role}-reference",
                                _bundle(args, identity_data))
    if len(result["updates"]) != 6 or any(not step.get("effective_update") for step in result["updates"]):
        raise ValueError("Validation requires six effective updates")
    if reference:
        return None, None
    refdir = args.output_dir / f"{role}-reference"
    comparison = comparator(refdir / "training-capture", directory / "training-capture")
    update = update_comparator(refdir, directory)
    if (comparison.get("status") != "compared" or len(comparison.get("cases", [])) != 6
            or update.get("status") != "compared" or len(update.get("cases", [])) != 6
            or [(x.get("state_label"), x.get("sigma")) for x in update["cases"]]
            != list(updates.CASES)
            ):
        raise ValueError("Incomplete six-case comparison or ineffective update")
    return comparison, update


def _case_rows(role, name, comparison, update, result):
    rows = []
    for (label, sigma), numerical, delta, step in zip(
            updates.CASES, comparison["cases"], update["cases"], result["updates"], strict=True):
        rows.append({"role": role, "candidate": name, "state": label, "sigma": sigma,
                     "strict_pass": numerical["within_experimental_tolerances"],
                     "prediction": numerical["prediction"],
                     "input_gradient": numerical["input_gradient"],
                     "adapter_gradient": numerical["adapter_gradient"],
                     "actual_delta": delta["global"], "delta_groups": delta["groups"],
                     "effective_update": step["effective_update"],
                     "compute_seconds": step["compute_seconds"],
                     "capture_io_seconds": step["capture_io_seconds"],
                     "step_seconds": step["seconds"],
                     "peak_allocated": step["peak_allocated"],
                     "peak_reserved": step["peak_reserved"]})
    return rows


def _decision(report):
    completed = {x["directory"] for x in report["workers"]}
    if len(completed) != len(TASKS):
        return "pending"
    control_ok = all(all(row["strict_pass"] for row in report["validation_table"]
                         if row["role"] == role and row["candidate"] == "fp32-control")
                     for role in ROLES)
    eligible = [name for name in PATTERNS if control_ok and all(
        row["strict_pass"] for row in report["validation_table"] if row["candidate"] == name)
        and sum(row["candidate"] == name for row in report["validation_table"]) == 12]
    report["eligible_candidates"] = eligible
    return "eligible_requires_production_training" if eligible else "no_candidate_passed_heldout_budget"


def _preflight(args, gpu_check):
    if (args.swap != 20 or args.loss_scale != 1024 or args.max_new_workers not in (1, 2)
            or not math.isfinite(args.worker_timeout) or not 0 < args.worker_timeout <= 240
            or not math.isfinite(args.batch_timeout) or not 0 < args.batch_timeout <= 480
            or not args.weights.is_file() or not args.warmup_inputs.is_file()
            or not args.warmup_inputs.with_suffix(".json").is_file()):
        raise ValueError("Invalid fixed Batch 2 configuration or inputs")
    selected = _selection(args)
    root = args.output_dir.resolve()
    sources = [args.weights.resolve(), args.warmup_inputs.resolve(),
               args.selection.resolve(), *selected.values()]
    if any(root == path or root in path.parents or path in root.parents for path in sources):
        raise ValueError("Output and input trees must be disjoint")
    if root.exists() != args.continue_run:
        raise ValueError("Fresh output required or explicit continuation")
    gpu = gpu_check(args.gpu_uuid)
    return selected, gpu, identity(args, selected)


def _reload(args, identity_data, comparator, update_comparator):
    root = args.output_dir
    env = json.loads((root / "environment.json").read_text())
    report = json.loads((root / "summary.json").read_text())
    if (env.get("identity") != identity_data or report.get("scope") != SCOPE
            or report.get("status") != "screening" or report.get("production_ready") is not False):
        raise ValueError("Continuation identity/status mismatch")
    completed = []
    for role, name, directory_name in TASKS:
        if len(completed) == len(report["workers"]):
            break
        item = report["workers"][len(completed)]
        directory = root / directory_name
        if item.get("directory") != directory_name or _evidence(directory) != item.get("evidence_hashes"):
            raise ValueError("Existing worker evidence changed")
        result = json.loads((directory / "supervisor.json").read_text())
        comparison, update = _validate_worker(args, directory, role, name, result,
                                                identity_data, comparator, update_comparator)
        if comparison != item.get("comparison") or update != item.get("update_comparison"):
            raise ValueError("Existing numerical evidence changed")
        completed.append(directory_name)
    if len(completed) != len(report["workers"]):
        raise ValueError("Worker sequence changed")
    rows = [row for item in report["workers"] for row in item.get("cases", [])]
    if rows != report.get("validation_table"):
        raise ValueError("Validation table changed")
    _budget(root)
    return report


def run(args, *, gpu_check=search.check_gpu_idle, monitor=search.run_monitored,
        comparator=compare, update_comparator=updates.compare_validation_updates):
    started = time.monotonic()
    selected, gpu, identity_data = _preflight(args, gpu_check)
    root = args.output_dir
    if args.continue_run:
        report = _reload(args, identity_data, comparator, update_comparator)
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_result(root / "environment.json", {"identity": identity_data, "gpu": gpu,
                     "selection": str(args.selection.resolve()),
                     "weights": str(args.weights.resolve()), "warmup_inputs": str(args.warmup_inputs.resolve()),
                     "inputs": {k: str(v) for k, v in selected.items()}})
        report = {"scope": SCOPE, "status": "screening", "decision": "pending",
                  "production_ready": False, "performance_validated": False,
                  "timing_scope": "cold_workers_compute_excludes_capture_io_not_steady_state",
                  "workers": [], "validation_table": [], "segments": []}
    segment = {"started_after_preflight_seconds": time.monotonic() - started,
               "new_workers": 0}
    report["segments"].append(segment)
    write_result(root / "summary.json", report)
    active_result = None
    try:
        for role, name, directory_name in TASKS[len(report["workers"]):]:
            if segment["new_workers"] >= args.max_new_workers:
                break
            _budget(root)
            remaining = args.batch_timeout - (time.monotonic() - started)
            if remaining <= 0:
                break
            directory = root / directory_name
            directory.mkdir(exist_ok=False)
            argv = command(args, directory, role, name, selected)
            launch_args = argparse.Namespace(gpu_uuid=args.gpu_uuid,
                                             worker_timeout=min(args.worker_timeout, remaining))
            result = search.launch_if_idle(argv, directory, launch_args, gpu_check, monitor)
            active_result = result
            _budget(root)
            comparison, update = _validate_worker(args, directory, role, name, result,
                                                    identity_data, comparator, update_comparator)
            if comparison is not None:
                write_result(directory / "comparison.json", comparison)
                write_result(directory / "update-comparison.json", update)
            cases = [] if comparison is None else _case_rows(role, name, comparison, update, result)
            item = {"directory": directory_name, "role": role, "name": name,
                    "result_status": result["status"], "monitor": result.get("monitor"),
                    "fp32_modules": result.get("fp32_modules"),
                    "fp16_modules": sorted(module for module, dtype in
                                           (result.get("precision_manifest") or {}).get("assignments", {}).items()
                                           if dtype == "fp16"),
                    "comparison": comparison, "update_comparison": update, "cases": cases,
                    "effective_updates": 1 if role is None else sum(bool(x.get("effective_update")) for x in result["updates"]),
                    "compute_seconds": sum(x.get("compute_seconds", 0) for x in result["updates"]),
                    "capture_io_seconds": sum(x.get("capture_io_seconds", 0) for x in result["updates"]),
                    "evidence_hashes": _evidence(directory)}
            report["workers"].append(item)
            report["validation_table"].extend(cases)
            segment["new_workers"] += 1
            active_result = None
            write_result(root / "summary.json", report)
        report["decision"] = _decision(report)
        report["status"] = "completed" if len(report["workers"]) == len(TASKS) else "screening"
    except BaseException as exc:
        report.update(status="failed", decision="pending",
                      stop_reason=f"{type(exc).__name__}: {exc}",
                      partial_result=None if active_result is None else {
                          "status": active_result.get("status"),
                          "returncode": active_result.get("returncode"),
                          "error": active_result.get("error"),
                          "monitor": active_result.get("monitor")},
                      partial_worker=TASKS[len(report["workers"])][2]
                      if len(report["workers"]) < len(TASKS) else None)
    finally:
        segment["elapsed_seconds"] = time.monotonic() - started
        segment["output_bytes"] = _output_bytes(root)
        write_result(root / "summary.json", report)
        write_result(root / "validation-table.json", report["validation_table"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "warmup-inputs", "selection", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--swap", type=int, default=20)
    parser.add_argument("--loss-scale", type=float, default=1024)
    parser.add_argument("--worker-timeout", type=float, default=240)
    parser.add_argument("--batch-timeout", type=float, default=480)
    parser.add_argument("--max-new-workers", type=int, default=2)
    parser.add_argument("--continue-run", action="store_true")
    args = parser.parse_args(argv)
    report = run(args)
    print(json.dumps({"status": report["status"], "decision": report["decision"],
                      "workers": len(report["workers"])}))
    return 0 if report["status"] in {"screening", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
