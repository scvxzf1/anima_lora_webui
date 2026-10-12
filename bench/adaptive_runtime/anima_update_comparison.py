"""Bounded, same-state AdamW update evidence for one Anima replay step."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch

from bench.adaptive_runtime.capture import capture_path
from bench.adaptive_runtime.replay import MODE
from bench.adaptive_runtime.training_capture import parameter_digest
from library.training.adaptive_runtime.precision import vector_metrics
from library.training.auto_block_swap.process import write_result


SCHEMA = "anima_initial_adamw_update_v1"
FILE_LIMIT = 512 * 1024**2
TOTAL_LIMIT = 1024**3


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _names(network):
    names = [name for name, param in network.named_parameters() if param.requires_grad]
    if not names or len(names) != len(set(names)):
        raise ValueError("Expected unique trainable adapter parameters")
    return names


def _optimizer_contract(optimizer, network):
    named = {
        id(param): name
        for name, param in network.named_parameters()
        if param.requires_grad
    }
    groups = []
    seen = []
    for group in optimizer.param_groups:
        names = [named.get(id(param)) for param in group["params"]]
        if any(name is None for name in names):
            raise ValueError("Optimizer contains non-adapter parameter")
        seen.extend(names)
        options = {key: value for key, value in group.items() if key != "params"}
        if any(
            not isinstance(value, (bool, int, float, str, type(None), tuple))
            for value in options.values()
        ):
            raise ValueError("Unsupported optimizer group option")
        groups.append({"parameters": names, **options})
    if len(seen) != len(set(seen)) or set(seen) != set(named.values()):
        raise ValueError("Optimizer parameter inventory mismatch")
    if optimizer.__class__ is not torch.optim.AdamW:
        raise ValueError("Initial update protocol requires torch AdamW")
    if optimizer.state and any(bool(state) for state in optimizer.state.values()):
        raise ValueError(
            "Initial update protocol rejects nonempty AdamW moments or steps"
        )
    return {"optimizer": "torch.optim.AdamW", "state": "empty", "param_groups": groups}


class UpdateCapture:
    def __init__(self, directory, *, network, signature, uuid):
        if signature.get("comparison_mode") != MODE or not signature.get(
            "capture_update"
        ):
            raise ValueError("Update capture requires fixed-state replay signature")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.names = _names(network)
        self.manifest = {
            "schema": SCHEMA,
            "status": "capturing",
            "signature": signature,
            "uuid": uuid,
            "expected_deltas": [f"delta.{name}" for name in self.names],
            "cases": [],
        }
        self.pending = None
        self._write()

    def _write(self):
        write_result(self.directory / "manifest.json", self.manifest)

    def before(
        self, *, network, optimizer, step, sigma, replay, training_case, clip_max_norm
    ):
        if self.pending is not None or step != 1 or self.manifest["cases"]:
            raise ValueError("Initial update capture requires exactly one step")
        if not isinstance(replay, dict) or set(replay) != {
            "snapshot_sha256",
            "state_sha256",
            "inputs_sha256",
            "rng_sha256",
        }:
            raise ValueError("Update capture requires replay evidence")
        if training_case.get("step") != step or training_case.get("replay") != replay:
            raise ValueError("Update capture and training gradient case differ")
        if clip_max_norm != 1.0:
            raise ValueError("Unexpected gradient clip policy")
        params = dict(network.named_parameters())
        originals = {name: params[name].detach().cpu().clone() for name in self.names}
        self.pending = {
            "originals": originals,
            "step": step,
            "sigma": sigma,
            "replay": replay,
            "training_capture_sha256": training_case["sha256"],
            "pre_step_parameter_sha256": parameter_digest(network),
            "optimizer_contract": _optimizer_contract(optimizer, network),
            "clip": {"max_norm": clip_max_norm, "norm_type": 2.0},
        }

    def after(self, *, network, optimizer, gradient_norm):
        if self.pending is None or not math.isfinite(float(gradient_norm)):
            raise ValueError("Missing pre-step state or finite gradient norm")
        case = self.pending
        params = dict(network.named_parameters())
        if any(
            float(optimizer.state[params[name]].get("step", 0)) != 1
            for name in self.names
        ):
            raise ValueError("Incomplete initial AdamW update")
        tensors = {
            f"delta.{name}": (params[name].detach().cpu() - case["originals"][name])
            .to(torch.float32)
            .contiguous()
            for name in self.names
        }
        if any(
            value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
            for value in tensors.values()
        ):
            raise ValueError("Update deltas must be finite FP32")
        size = sum(value.numel() * value.element_size() for value in tensors.values())
        if size > FILE_LIMIT or size > TOTAL_LIMIT:
            raise ValueError("Update capture byte budget exceeded")
        path = self.directory / "step-000001.safetensors"
        save_file(tensors, str(path))
        if path.stat().st_size > FILE_LIMIT:
            raise ValueError("Update capture file budget exceeded")
        case = {key: value for key, value in case.items() if key != "originals"}
        case.update(
            file=path.name,
            sha256=sha256_file(path),
            gradient_norm=float(gradient_norm),
            tensor_bytes=size,
        )
        self.manifest["cases"].append(case)
        self.pending = None
        self._write()

    def close(self):
        self.manifest["status"] = (
            "captured" if len(self.manifest["cases"]) == 1 else "partial"
        )
        self._write()

    def bind_result(self, worker_dir):
        if self.manifest["status"] != "captured":
            return
        worker_dir = Path(worker_dir)
        self.manifest["result_sha256"] = sha256_file(worker_dir / "result.json")
        self.manifest["training_manifest_sha256"] = sha256_file(
            worker_dir / "training-capture" / "manifest.json"
        )
        self._write()


def _read(worker_dir):
    worker_dir = Path(worker_dir)
    directory = worker_dir / "update-capture"
    manifest = json.loads((directory / "manifest.json").read_text())
    worker = json.loads((worker_dir / "result.json").read_text())
    training = json.loads(
        (worker_dir / "training-capture" / "manifest.json").read_text()
    )
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("status") != "captured"
        or worker.get("status") != "ok"
        or training.get("status") != "captured"
        or len(manifest.get("cases", [])) != 1
        or len(training.get("cases", [])) != 1
        or [u.get("step") for u in worker.get("updates", [])] != [1]
        or manifest.get("result_sha256") != sha256_file(worker_dir / "result.json")
        or manifest.get("training_manifest_sha256")
        != sha256_file(worker_dir / "training-capture" / "manifest.json")
        or manifest.get("signature") != training.get("signature")
        or worker.get("signature") != manifest.get("signature")
        or manifest.get("uuid") != worker.get("uuid")
    ):
        raise ValueError("Incomplete or unbound update/training/worker evidence")
    signature = manifest["signature"]
    case, training_case = manifest["cases"][0], training["cases"][0]
    training_path = capture_path(worker_dir / "training-capture", training_case["file"])
    if training_path.stat().st_size > FILE_LIMIT or sha256_file(
        training_path
    ) != training_case.get("sha256"):
        raise ValueError("Training capture case digest mismatch")
    if (
        signature.get("comparison_mode") != MODE
        or not signature.get("capture_update")
        or signature.get("steps") != 1
        or signature.get("sigmas") != [0.5]
        or signature.get("update_source_sha256") != sha256_file(__file__)
        or worker.get("weights_sha256") != signature.get("weights_sha256")
        or worker.get("inputs_sha256") != signature.get("inputs_sha256")
        or case.get("step") != 1
        or case.get("sigma") != 0.5
        or case.get("training_capture_sha256") != training_case.get("sha256")
        or case.get("replay") != training_case.get("replay")
        or case.get("clip") != {"max_norm": 1.0, "norm_type": 2.0}
        or case.get("optimizer_contract", {}).get("state") != "empty"
        or case.get("optimizer_contract", {}).get("optimizer") != "torch.optim.AdamW"
        or not isinstance(case.get("pre_step_parameter_sha256"), str)
        or len(case["pre_step_parameter_sha256"]) != 64
        or not math.isfinite(case.get("gradient_norm", float("nan")))
    ):
        raise ValueError("Update capture contract mismatch")
    expected = manifest.get("expected_deltas")
    if (
        not isinstance(expected, list)
        or not expected
        or len(expected) != len(set(expected))
        or any(
            not isinstance(name, str) or not name.startswith("delta.")
            for name in expected
        )
        or set(expected)
        != {
            "delta." + name.removeprefix("gradient.")
            for name in training.get("expected_gradients", [])
        }
    ):
        raise ValueError("Update tensor inventory mismatch")
    path = capture_path(directory, case["file"])
    if path.stat().st_size > FILE_LIMIT or sha256_file(path) != case["sha256"]:
        raise ValueError("Update capture file budget/digest mismatch")
    tensors = load_file(str(path))
    if (
        set(tensors) != set(expected)
        or sum(t.numel() * t.element_size() for t in tensors.values())
        != case["tensor_bytes"]
        or case["tensor_bytes"] > TOTAL_LIMIT
        or any(
            t.dtype != torch.float32
            or not t.numel()
            or not bool(torch.isfinite(t).all())
            for t in tensors.values()
        )
    ):
        raise ValueError("Invalid update delta tensors")
    return manifest, worker, case, tensors


def _group(name):
    if ".mlp." in name or "_mlp_" in name:
        return "mlp"
    if any(part in name for part in (".self_attn.", ".self_attention.", "_self_attn_")):
        return "self_attention"
    if any(
        part in name for part in (".cross_attn.", ".cross_attention.", "_cross_attn_")
    ):
        return "cross_attention"
    return "other"


def compare_update(reference_worker_dir, candidate_worker_dir):
    ref, ref_worker, a, reference = _read(reference_worker_dir)
    trial, trial_worker, b, candidate = _read(candidate_worker_dir)
    if ref["signature"].get("precision") != "fp32-reference":
        raise ValueError("Update reference must be FP32")
    excluded = {"precision", "fp32_modules", "loss_scale"}
    if (
        {k: v for k, v in ref["signature"].items() if k not in excluded}
        != {k: v for k, v in trial["signature"].items() if k not in excluded}
        or ref["uuid"] != trial["uuid"]
        or ref_worker.get("replay_mode") != "record"
        or trial_worker.get("replay_mode") != "replay"
        or a["replay"] != b["replay"]
        or a["pre_step_parameter_sha256"] != b["pre_step_parameter_sha256"]
        or a["optimizer_contract"] != b["optimizer_contract"]
        or a["clip"] != b["clip"]
        or set(reference) != set(candidate)
        or any(reference[name].shape != candidate[name].shape for name in reference)
    ):
        raise ValueError("Same-state optimizer update contract mismatch")
    groups = {
        name: [key for key in reference if _group(key) == name]
        for name in ("mlp", "self_attention", "cross_attention", "other")
    }

    def metrics(names):
        return vector_metrics((reference[name], candidate[name]) for name in names)

    return {
        "status": "compared",
        "scope": "initial_empty_adamw_same_state_single_step",
        "production_ready": False,
        "reference_sha256": a["sha256"],
        "candidate_sha256": b["sha256"],
        "global": metrics(reference),
        "groups": {name: metrics(names) for name, names in groups.items()},
        "parameter_count": len(reference),
        "tensor_inventory": sorted(reference),
    }
