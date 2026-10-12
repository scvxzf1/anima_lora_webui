"""Six independent, restored-state Anima optimizer update comparisons."""

from __future__ import annotations

import json
import math
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch

from bench.adaptive_runtime.anima_update_comparison import _group, sha256_file
from bench.adaptive_runtime.anima_validation_state import optimizer_snapshot
from bench.adaptive_runtime.replay import tensor_digest
from library.training.adaptive_runtime.precision import vector_metrics
from library.training.auto_block_swap.process import write_result


SCHEMA = "anima_validation_updates_v1"
SIGMAS = (0.2, 0.5, 0.8)
CASES = tuple((label, sigma) for label in ("initial", "nonzero") for sigma in SIGMAS)
FILE_LIMIT = 128 * 1024**2
TOTAL_LIMIT = 256 * 1024**2


def case_id(label, sigma):
    if (label, sigma) not in CASES:
        raise ValueError("Invalid validation case")
    return f"{label}-{sigma:.1f}"


def _parameters(network):
    params = {name: p for name, p in network.named_parameters() if p.requires_grad}
    if not params:
        raise ValueError("No trainable parameters")
    return params


class ValidationUpdateCapture:
    def __init__(self, directory, *, network, signature, uuid, steps=6):
        if steps != 6 or not uuid or signature.get("steps") != 6 or signature.get("sigmas") != list(SIGMAS):
            raise ValueError("Validation updates require six fixed cases")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.names = list(_parameters(network))
        self.pending = None
        self.manifest = {"schema": SCHEMA, "status": "capturing", "signature": signature,
                         "uuid": uuid, "expected_deltas": [f"delta.{n}" for n in self.names],
                         "cases": [], "tensor_bytes": 0}
        self._write()

    def _write(self):
        write_result(self.directory / "manifest.json", self.manifest)

    def before(self, *, network, optimizer, step, case_id, state_label, sigma,
               replay, training_case, state_evidence, clip_max_norm):
        index = len(self.manifest["cases"])
        if (self.pending is not None or step != index + 1 or index >= 6
                or (state_label, sigma) != CASES[index] or case_id != globals()["case_id"](state_label, sigma)
                or clip_max_norm != 1.0 or not isinstance(replay, dict)
                or set(replay) != {"snapshot_sha256", "state_sha256", "inputs_sha256", "rng_sha256"}
                or training_case.get("step") != step or training_case.get("sigma") != sigma
                or training_case.get("replay") != replay or not training_case.get("sha256")):
            raise ValueError("Validation case/training/replay contract mismatch")
        params = _parameters(network)
        if list(params) != self.names:
            raise ValueError("Parameter inventory changed")
        groups, _, optimizer_sha = optimizer_snapshot(network, optimizer, expected_step=0 if state_label == "initial" else 1)
        param_sha = tensor_digest({f"parameter.{n}": p for n, p in params.items()})
        expected_evidence = {"bundle_manifest_sha256", "state_file_sha256", "state_label", "parameter_sha256", "optimizer_sha256"}
        if (not isinstance(state_evidence, dict) or set(state_evidence) != expected_evidence
                or state_evidence["state_label"] != state_label or state_evidence["parameter_sha256"] != param_sha
                or state_evidence["optimizer_sha256"] != optimizer_sha
                or any(not isinstance(state_evidence[k], str) or len(state_evidence[k]) != 64 for k in expected_evidence - {"state_label"})):
            raise ValueError("Restored state evidence mismatch")
        self.pending = {"step": step, "case_id": case_id, "state_label": state_label, "sigma": sigma,
                        "replay": replay, "training_capture_sha256": training_case["sha256"],
                        "state_evidence": state_evidence, "pre_step_parameter_sha256": param_sha,
                        "pre_step_optimizer_sha256": optimizer_sha, "optimizer_options": groups,
                        "clip": {"max_norm": 1.0, "norm_type": 2.0},
                        "originals": {n: p.detach().cpu().clone() for n, p in params.items()}}

    def after(self, *, network, optimizer, gradient_norm):
        if self.pending is None or not math.isfinite(float(gradient_norm)):
            raise ValueError("Missing pre-step state or gradient norm")
        case = self.pending
        params = _parameters(network)
        if list(params) != self.names:
            raise ValueError("Parameter inventory changed")
        optimizer_snapshot(network, optimizer, expected_step=1 if case["state_label"] == "initial" else 2)
        tensors = {f"delta.{name}": (params[name].detach().cpu() - case["originals"][name]).float().contiguous()
                   for name in self.names}
        if any(not bool(torch.isfinite(v).all()) for v in tensors.values()):
            raise ValueError("Nonfinite update delta")
        size = sum(v.numel() * v.element_size() for v in tensors.values())
        if size > FILE_LIMIT or self.manifest["tensor_bytes"] + size > TOTAL_LIMIT:
            raise ValueError("Update byte budget exceeded")
        path = self.directory / f"step-{case['step']:06d}.safetensors"
        save_file(tensors, str(path))
        if path.stat().st_size > FILE_LIMIT:
            raise ValueError("Update file byte budget exceeded")
        record = {k: v for k, v in case.items() if k != "originals"}
        record.update(file=path.name, sha256=sha256_file(path), gradient_norm=float(gradient_norm), tensor_bytes=size)
        self.manifest["cases"].append(record)
        self.manifest["tensor_bytes"] += size
        self.pending = None
        self._write()

    def close(self):
        self.manifest["status"] = "captured" if len(self.manifest["cases"]) == 6 and self.pending is None else "partial"
        self._write()

    def bind_result(self, worker_dir):
        if self.manifest["status"] != "captured":
            return
        worker = Path(worker_dir)
        self.manifest["result_sha256"] = sha256_file(worker / "result.json")
        self.manifest["training_manifest_sha256"] = sha256_file(worker / "training-capture" / "manifest.json")
        self._write()


def _safe_file(directory, name, expected):
    path = directory / name
    if name != expected or directory.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError("Unsafe capture path")
    if path.stat().st_size > FILE_LIMIT:
        raise ValueError("Capture file budget exceeded")
    return path


def _validate_summary(summary, case, tensors):
    expected = {"case_id": case["case_id"], "state_label": case["state_label"],
                "prestate_evidence": case["state_evidence"], "update_sha256": case["sha256"],
                "gradient_norm": case["gradient_norm"]}
    maximum = max(float(value.abs().max()) for value in tensors.values())
    if (any(summary.get(key) != value for key, value in expected.items())
            or summary.get("effective_update") is not True or maximum <= 0
            or summary.get("adapter_max_delta") != maximum
            or any(not isinstance(summary.get(key), (float, int))
                   or not math.isfinite(summary[key]) or summary[key] <= 0
                   for key in ("loss_scale_before", "loss_scale_after"))):
        raise ValueError("Result update summary differs from captured actual update")


def _read(worker_dir):
    worker_dir = Path(worker_dir)
    directory = worker_dir / "update-capture"
    manifest_path = directory / "manifest.json"
    training_path = worker_dir / "training-capture" / "manifest.json"
    result_path = worker_dir / "result.json"
    if any(p.is_symlink() or p.stat().st_size > 1024**2 for p in (manifest_path, training_path, result_path)):
        raise ValueError("Unsafe worker manifest")
    m = json.loads(manifest_path.read_text())
    t = json.loads(training_path.read_text())
    w = json.loads(result_path.read_text())
    if (m.get("schema") != SCHEMA or m.get("status") != "captured" or t.get("status") != "captured"
            or w.get("status") != "ok" or m.get("result_sha256") != sha256_file(result_path)
            or m.get("training_manifest_sha256") != sha256_file(training_path)
            or m.get("signature") != t.get("signature") or m.get("signature") != w.get("signature")
            or m.get("uuid") != w.get("uuid") or len(m.get("cases", [])) != 6
            or len(t.get("cases", [])) != 6 or [x.get("step") for x in w.get("updates", [])] != list(range(1, 7))
            or m["signature"].get("steps") != 6 or m["signature"].get("sigmas") != list(SIGMAS)):
        raise ValueError("Incomplete or unbound validation worker")
    expected = m.get("expected_deltas")
    if (not isinstance(expected, list) or not expected or len(set(expected)) != len(expected)
            or set(expected) != {"delta." + n.removeprefix("gradient.") for n in t.get("expected_gradients", [])}):
        raise ValueError("Update tensor inventory mismatch")
    used, cases = 0, []
    state_by_label = {}
    for i, (case, training) in enumerate(zip(m["cases"], t["cases"], strict=True)):
        label, sigma = CASES[i]
        path = _safe_file(directory, case.get("file"), f"step-{i+1:06d}.safetensors")
        training_file = _safe_file(worker_dir / "training-capture", training.get("file"), f"step-{i+1:06d}.safetensors")
        if (case.get("step") != i + 1 or case.get("case_id") != case_id(label, sigma)
                or case.get("state_label") != label or case.get("sigma") != sigma
                or training.get("step") != i + 1 or training.get("sigma") != sigma
                or case.get("training_capture_sha256") != training.get("sha256")
                or sha256_file(training_file) != training.get("sha256")
                or case.get("replay") != training.get("replay")
                or case.get("clip") != {"max_norm": 1.0, "norm_type": 2.0}
                or case.get("sha256") != sha256_file(path)
                or not math.isfinite(case.get("gradient_norm", float("nan")))):
            raise ValueError("Validation case binding mismatch")
        evidence = case.get("state_evidence")
        if (not isinstance(evidence, dict)
                or set(evidence) != {"bundle_manifest_sha256", "state_file_sha256", "state_label", "parameter_sha256", "optimizer_sha256"}
                or evidence["state_label"] != label
                or any(not isinstance(evidence[k], str) or len(evidence[k]) != 64 for k in evidence if k != "state_label")
                or evidence["parameter_sha256"] != case.get("pre_step_parameter_sha256")
                or evidence["optimizer_sha256"] != case.get("pre_step_optimizer_sha256")
                or (label in state_by_label and state_by_label[label] != evidence)
                or not isinstance(case.get("optimizer_options"), list)):
            raise ValueError("State evidence binding mismatch")
        state_by_label[label] = evidence
        tensors = load_file(str(path))
        size = sum(v.numel() * v.element_size() for v in tensors.values())
        used += size
        if (set(tensors) != set(expected) or size != case.get("tensor_bytes") or used > TOTAL_LIMIT
                or any(v.dtype != torch.float32 or not v.numel() or not bool(torch.isfinite(v).all()) for v in tensors.values())):
            raise ValueError("Invalid update tensors")
        _validate_summary(w["updates"][i], case, tensors)
        cases.append((case, tensors))
    if used != m.get("tensor_bytes"):
        raise ValueError("Update total bytes mismatch")
    return m, w, cases


def compare_validation_updates(reference_worker, candidate_worker):
    ref, ref_worker, a = _read(reference_worker)
    trial, trial_worker, b = _read(candidate_worker)
    excluded = {"precision", "fp32_modules", "loss_scale"}
    if (ref["signature"].get("precision") != "fp32-reference" or ref_worker.get("replay_mode") != "record"
            or trial_worker.get("replay_mode") != "replay" or ref["uuid"] != trial["uuid"]
            or {k: v for k, v in ref["signature"].items() if k not in excluded}
            != {k: v for k, v in trial["signature"].items() if k not in excluded}):
        raise ValueError("Validation worker identity mismatch")
    results = []
    for (left, x), (right, y) in zip(a, b, strict=True):
        evidence = ("case_id", "sigma", "state_label", "replay", "state_evidence", "pre_step_parameter_sha256",
                    "pre_step_optimizer_sha256", "optimizer_options", "clip")
        if (any(left.get(k) != right.get(k) for k in evidence) or set(x) != set(y)
                or any(x[k].shape != y[k].shape for k in x)):
            raise ValueError("Same-state validation update mismatch")
        groups = {name: [k for k in x if _group(k) == name]
                  for name in ("mlp", "self_attention", "cross_attention", "other")}
        def metrics(names):
            return vector_metrics((x[k], y[k]) for k in names)
        results.append({"case_id": left["case_id"], "state_label": left["state_label"], "sigma": left["sigma"],
                        "reference_sha256": left["sha256"], "candidate_sha256": right["sha256"],
                        "global": metrics(x), "groups": {name: metrics(names) for name, names in groups.items()}})
    return {"status": "compared", "scope": "six_independent_restored_adamw_updates", "production_ready": False,
            "cases": results, "parameter_count": len(a[0][1]), "tensor_inventory": sorted(a[0][1])}
