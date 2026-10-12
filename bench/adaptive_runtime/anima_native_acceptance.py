"""Prepare, run and audit one fixed native FP16 candidate against saved references."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from bench.adaptive_runtime import anima_native_validation as native
from bench.adaptive_runtime import anima_precision_validation as existing
from bench.adaptive_runtime import anima_validation_probe as validation
from bench.adaptive_runtime import anima_validation_updates as updates
from bench.adaptive_runtime import compare_training
from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime.anima_validation_evidence import validate_validation_sources
from bench.adaptive_runtime.anima_validation_state import StateBundle
from bench.adaptive_runtime.probe_monitor import check_gpu_idle, run_monitored
from library.training.auto_block_swap.process import write_result

SCHEMA = "anima_native_fp16_acceptance_plan_v1"
SOURCE_ID = "1f5ea4b1d29e337500a5270075fe9f9fd2d7b9a8a0c1e5d0eab50c51c8caed71"
ROLES = existing.ROLES
ENV_KEY = "ANIMA_NATIVE_MONITORED_PLAN_SHA256"


def read(path):
    return json.loads(Path(path).read_text())


def file_hashes(directory):
    directory = Path(directory)
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlink evidence refused")
        if path.is_file():
            files[str(path.relative_to(directory))] = sha256_file(path)
    return files


def current_sources():
    root = Path(__file__).resolve().parents[2]
    return {name: sha256_file(root / name) for name in (*existing.SOURCE_FILES,
        "bench/adaptive_runtime/anima_native_validation.py",
        "bench/adaptive_runtime/anima_native_acceptance.py")}


def preflight(args, *, validate_tensors=True):
    if args.gpu_uuid != native.GPU_UUID or validation.source_code_sha256() != SOURCE_ID:
        raise ValueError("Wrong GPU UUID or changed canonical source identity")
    quality = read(args.quality_protocol)
    if (quality.get("schema") != "anima_native_fp16_quality_protocol_v1"
            or quality.get("quality_status") != "not_run"
            or quality.get("future_run_authorized") is not False
            or quality.get("scope", {}).get("gpu_uuid") != args.gpu_uuid):
        raise ValueError("Preregistered, unexecuted quality protocol required")
    selected = existing._selection(args)
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    if (source == output or source in output.parents or output in source.parents
            or any(output == p.resolve() or output in p.resolve().parents or p.resolve() in output.parents
                   for p in (args.weights, args.selection, *selected.values()))):
        raise ValueError("Output must be disjoint from immutable source trees")
    if not args.weights.is_file():
        raise ValueError("Existing real source weights required")
    audit = read(source / "final-audit.json")
    if audit.get("passed") is not True or audit.get("cases") != 36:
        raise ValueError("Original Batch 2 final audit must pass")
    digest = sha256_file(args.weights)
    identity = validation.state_identity({"weights_sha256": digest})
    bundle = (StateBundle(source / "validation-states", identity=identity, uuid=args.gpu_uuid)
              if validate_tensors else None)
    evidence = {"validation-states": file_hashes(source / "validation-states"),
                "final-audit.json": sha256_file(source / "final-audit.json")}
    for role in ROLES:
        directory = source / f"{role}-reference"
        manifest, worker = compare_training.read_capture(directory / "training-capture")
        if (worker.get("inputs_sha256") != sha256_file(selected[role])
                or worker.get("weights_sha256") != digest or worker.get("uuid") != args.gpu_uuid
                or worker["signature"].get("validation_source_code_sha256") != SOURCE_ID):
            raise ValueError("Reference/input/source binding mismatch")
        if validate_tensors:
            for case in manifest["cases"]:
                compare_training.load_case(directory / "training-capture", case, manifest["expected_gradients"])
            validate_validation_sources(directory, directory, bundle)
        evidence[f"{role}-reference"] = file_hashes(directory)
    return {"schema": SCHEMA, "gpu_uuid": args.gpu_uuid, "policy": native.POLICY,
            "source_code_sha256": SOURCE_ID, "source_dir": str(source),
            "selection": str(args.selection.resolve()), "selection_sha256": sha256_file(args.selection),
            "quality_protocol": str(args.quality_protocol.resolve()),
            "quality_protocol_sha256": sha256_file(args.quality_protocol),
            "weights": str(args.weights.resolve()), "weights_sha256": digest,
            "output_dir": str(output), "sources": current_sources(), "source_evidence": evidence,
            "inputs": {role: {"path": str(path.resolve()), "sha256": sha256_file(path),
                              "sidecar_sha256": sha256_file(path.with_suffix(".json"))}
                       for role, path in selected.items()},
            "case_schedule": [{"role": role, "state_label": label, "sigma": sigma,
                               "case_id": updates.case_id(label, sigma)}
                              for role in ROLES for label, sigma in updates.CASES],
            "limits": {"worker_seconds": 240, "batch_seconds": 480, "outer_seconds": 540,
                       "output_bytes_checked": existing.LIMIT}, "quality": "not_run"}


def prepare(args):
    if args.output_dir.exists():
        raise FileExistsError("Fresh acceptance output required; continuation/overwriting refused")
    plan = preflight(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_result(args.output_dir / "plan.json", plan)
    return plan


def verify_plan(args):
    path = args.output_dir / "plan.json"
    plan = read(path)
    # prepare validates all tensor/state/replay semantics once. Rehash every frozen
    # artifact here: unchanged bytes preserve that validation without repeated loads.
    if plan != preflight(args, validate_tensors=False):
        raise ValueError("Frozen plan or current source artifacts changed")
    return plan


def execution_for(plan, plan_sha, role):
    return {"role": role, "plan_sha256": plan_sha, "runtime_source_sha256": plan["sources"],
            "canonical_source_code_sha256": SOURCE_ID,
            "selection_sha256": plan["selection_sha256"],
            "state_manifest_sha256": plan["source_evidence"]["validation-states"]["manifest.json"],
            "reference_evidence_sha256": plan["source_evidence"][f"{role}-reference"],
            "input_package_sha256": plan["inputs"][role]["sha256"]}


def worker_args(plan, role):
    source = Path(plan["source_dir"])
    return argparse.Namespace(weights=Path(plan["weights"]), inputs=Path(plan["inputs"][role]["path"]),
        output=Path(plan["output_dir"]) / role / "result.json", precision=native.PRECISION,
        swap=20, steps=6, loss_scale=1024, fp32_pattern=[], fp16_only_pattern=[],
        capture_update=True, validation_states=source / "validation-states",
        export_validation_states=None, record_replay=False,
        replay_reference=source / f"{role}-reference/training-replay", single_sigma=None,
        capture_local_vjp=False, capture_block_boundaries=False, capture_block_vjp=False,
        backward_counterfactual_suite=False, block_vjp_indices=None, local_vjp_profile="linear",
        base_backward_mode="native", resume=False)


def audit_worker(plan, plan_sha, role):
    directory = Path(plan["output_dir"]) / role
    reference = Path(plan["source_dir"]) / f"{role}-reference"
    worker = read(directory / "result.json")
    execution = execution_for(plan, plan_sha, role)
    native.validate_native_result(worker, execution)
    bundle = StateBundle(Path(plan["source_dir"]) / "validation-states",
                         identity=validation.state_identity(worker), uuid=plan["gpu_uuid"])
    chain = validate_validation_sources(directory, reference, bundle)
    numerical = native.compare_native(reference, directory, execution)
    actual = updates.compare_validation_updates(reference, directory)
    rows = [{"role": role, "state_label": label, "sigma": sigma,
             "case_id": updates.case_id(label, sigma),
             "strict_pass": case["within_experimental_tolerances"],
             **{key: case[key] for key in ("prediction", "input_gradient", "adapter_gradient")},
             "actual_update": delta["global"], "actual_update_groups": delta["groups"]}
            for (label, sigma), case, delta in zip(updates.CASES, numerical["cases"], actual["cases"], strict=True)]
    return {"role": role, "chain": chain, "cases": rows,
            "result_sha256": sha256_file(directory / "result.json"), "evidence": file_hashes(directory)}


def validate_rows(rows):
    expected = {(role, label, sigma) for role in ROLES for label, sigma in updates.CASES}
    actual = [(r["role"], r["state_label"], r["sigma"]) for r in rows]
    if len(actual) != 12 or set(actual) != expected:
        raise ValueError("Acceptance requires exactly twelve unique role/state/sigma cases")


def run(args, *, gpu_check=check_gpu_idle, monitor=run_monitored):
    plan = verify_plan(args)
    root = args.output_dir
    if (root / "summary.json").exists() or any((root / role).exists() for role in ROLES):
        raise FileExistsError("Acceptance run cannot overwrite or continue old results")
    plan_sha = sha256_file(root / "plan.json")
    summary = {"status": "running", "quality": "not_run", "production_ready": False,
               "evidence_integrity": "pending", "chain": "pending", "strict_diagnostic": "pending",
               "workers": [], "cases": [], "plan_sha256": plan_sha}
    started = time.monotonic()
    try:
        for role in ROLES:
            existing._budget(root)
            remaining = 480 - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("Frozen batch budget exhausted")
            gpu_check(args.gpu_uuid)
            directory = root / role
            directory.mkdir(exist_ok=False)
            command = [sys.executable, "-m", "bench.adaptive_runtime.anima_native_acceptance",
                       "--worker", role, "--source-dir", str(args.source_dir),
                       "--selection", str(args.selection), "--weights", str(args.weights),
                       "--quality-protocol", str(args.quality_protocol),
                       "--output-dir", str(root), "--gpu-uuid", args.gpu_uuid]
            result = monitor(command, directory, args.gpu_uuid, min(240, remaining),
                             env={ENV_KEY: plan_sha})
            existing._budget(root)
            if result.get("status") != "ok" or result.get("returncode") != 0 or not result.get("monitor", {}).get("samples"):
                raise RuntimeError(f"{role} monitored worker failed: {result.get('status')}")
            item = audit_worker(plan, plan_sha, role)
            summary["workers"].append(item)
            summary["cases"].extend(item["cases"])
            write_result(root / "summary.json", summary)
        validate_rows(summary["cases"])
        summary.update(status="completed", evidence_integrity="pass", chain="pass",
                       strict_diagnostic="pass" if all(r["strict_pass"] for r in summary["cases"]) else "fail")
    except BaseException as exc:
        summary.update(status="failed", stop_reason=f"{type(exc).__name__}: {exc}",
                       evidence_integrity="incomplete", chain="incomplete")
    finally:
        completed = {(r["role"], r["state_label"], r["sigma"]) for r in summary["cases"]}
        summary["unfinished_cases"] = [row for row in plan["case_schedule"]
            if (row["role"], row["state_label"], row["sigma"]) not in completed]
        summary["elapsed_seconds"] = time.monotonic() - started
        summary["output_bytes"] = existing._output_bytes(root)
        write_result(root / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-dir", "selection", "weights", "output-dir", "quality-protocol"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--worker", choices=ROLES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        plan = verify_plan(args)
        digest = sha256_file(args.output_dir / "plan.json")
        if os.environ.get(ENV_KEY) != digest or os.environ.get("CUDA_VISIBLE_DEVICES") != native.GPU_UUID:
            raise ValueError("Private worker requires monitored frozen-plan environment")
        result = native.run_worker(worker_args(plan, args.worker), execution_for(plan, digest, args.worker))
    elif args.run:
        result = run(args)
    else:
        prepare(args)
        result = {"status": "prepared", "quality": "not_run", "gpu_run": False}
    print(json.dumps({k: result[k] for k in ("status", "quality") if k in result}))
    return 0 if result["status"] in {"prepared", "completed", "ok"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
