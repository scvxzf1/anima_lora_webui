"""Bounded, named AdamW state snapshots for independent validation cases."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
from safetensors.torch import load_file, save_file
import torch

from bench.adaptive_runtime.anima_update_comparison import sha256_file
from bench.adaptive_runtime.replay import tensor_digest, tuple_tree
from library.training.auto_block_swap.process import write_result


SCHEMA = "anima_validation_state_v1"
FILE_LIMIT = 128 * 1024**2
TOTAL_LIMIT = 256 * 1024**2
LABELS = ("initial", "nonzero")
OPTIONS = {"lr", "betas", "eps", "weight_decay", "amsgrad", "maximize",
           "foreach", "capturable", "differentiable", "fused", "decoupled_weight_decay"}
IDENTITY_KEYS = {"family", "weights_sha256", "source_code_sha256", "backend",
                 "rank", "alpha", "optimizer", "scheduler"}


def _identity(identity):
    if (not isinstance(identity, dict) or set(identity) != IDENTITY_KEYS
            or identity["family"] != "anima" or identity["optimizer"] != "torch.optim.AdamW"
            or identity["scheduler"] != "none"
            or any(not isinstance(identity[k], str) or len(identity[k]) != 64
                   for k in ("weights_sha256", "source_code_sha256"))
            or not isinstance(identity["backend"], str) or not identity["backend"]
            or type(identity["rank"]) is not int or identity["rank"] < 1
            or type(identity["alpha"]) not in (int, float) or not math.isfinite(identity["alpha"])):
        raise ValueError("Invalid state identity contract")


def _safe_path(directory, name):
    if not isinstance(name, str) or name not in {"initial.safetensors", "nonzero.safetensors"}:
        raise ValueError("Invalid state file name")
    path = directory / name
    if directory.is_symlink() or path.is_symlink() or not path.is_file():
        raise ValueError("Unsafe or missing state path")
    return path


def _params(network):
    named = dict(network.named_parameters())
    if not named or len(named) != len(list(network.named_parameters())):
        raise ValueError("Invalid parameter inventory")
    return named


def optimizer_options(network, optimizer):
    if optimizer.__class__ is not torch.optim.AdamW:
        raise ValueError("State bundle requires torch.optim.AdamW")
    by_id = {id(p): name for name, p in _params(network).items() if p.requires_grad}
    groups, seen = [], []
    for group in optimizer.param_groups:
        if set(group) - {"params"} != OPTIONS:
            raise ValueError("Unsupported AdamW options")
        names = [by_id.get(id(p)) for p in group["params"]]
        if any(name is None for name in names):
            raise ValueError("Optimizer contains unknown parameter")
        options = {k: list(v) if isinstance(v, tuple) else v for k, v in group.items() if k != "params"}
        if (any(type(v) not in (bool, int, float, str, list, type(None)) for v in options.values())
                or any(type(x) not in (int, float) for v in options.values() if isinstance(v, list) for x in v)
                or any(not math.isfinite(v) for v in options.values() if type(v) is float)):
            raise ValueError("Invalid AdamW option")
        groups.append({"parameters": names, "options": options})
        seen.extend(names)
    if len(seen) != len(set(seen)) or set(seen) != set(by_id.values()):
        raise ValueError("Optimizer parameter inventory mismatch")
    return groups


def _step_tensor(state, parameter, options):
    step = state.get("step")
    if not isinstance(step, torch.Tensor) or step.numel() != 1 or step.dtype != torch.float32:
        raise ValueError("AdamW step must be FP32 scalar tensor")
    expected = parameter.device if options.get("capturable") or options.get("fused") else torch.device("cpu")
    if step.device != expected:
        raise ValueError("AdamW step device mismatch")
    return step


def optimizer_snapshot(network, optimizer, *, expected_step=None):
    groups = optimizer_options(network, optimizer)
    params = _params(network)
    options = {name: group["options"] for group in groups for name in group["parameters"]}
    tensors = {}
    for name, param in params.items():
        if name not in options:
            if optimizer.state.get(param):
                raise ValueError("Frozen parameter has optimizer state")
            continue
        state = optimizer.state.get(param, {})
        if expected_step == 0:
            if state:
                raise ValueError("Initial AdamW state must be empty")
            continue
        if set(state) != {"step", "exp_avg", "exp_avg_sq"} | ({"max_exp_avg_sq"} if options[name]["amsgrad"] else set()):
            raise ValueError("Incomplete or unknown AdamW state")
        step = _step_tensor(state, param, options[name])
        if expected_step is not None and float(step.item()) != expected_step:
            raise ValueError("AdamW step mismatch")
        for key, value in state.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError("AdamW state must be tensor")
            if key != "step" and (value.shape != param.shape or value.device != param.device or value.dtype != param.dtype):
                raise ValueError("AdamW moment layout mismatch")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Nonfinite AdamW state")
            tensors[f"optimizer.{name}.{key}"] = value.detach().cpu().contiguous().clone()
    if expected_step == 1 and not any(bool(torch.count_nonzero(v)) for k, v in tensors.items() if k.endswith(("exp_avg", "exp_avg_sq"))):
        raise ValueError("Nonzero state requires nonzero moments")
    return groups, tensors, tensor_digest(tensors)


def _rng_snapshot(device):
    tensors = {"rng.cpu": torch.get_rng_state()}
    if device.type == "cuda":
        tensors["rng.cuda"] = torch.cuda.get_rng_state(device)
    np_state = np.random.get_state()
    metadata = {"python_rng": random.getstate(), "numpy_rng": [np_state[0], np_state[1].tolist(), *np_state[2:]]}
    return tensors, metadata


def _snapshot(network, optimizer, device, step):
    groups, opt_tensors, opt_digest = optimizer_snapshot(network, optimizer, expected_step=step)
    tensors = {f"parameter.{k}": v.detach().cpu().contiguous().clone() for k, v in _params(network).items()}
    tensors.update({f"buffer.{k}": v.detach().cpu().contiguous().clone() for k, v in network.named_buffers()})
    rng, metadata = _rng_snapshot(device)
    tensors.update(opt_tensors)
    tensors.update(rng)
    if any(v.numel() == 0 or (v.is_floating_point() and not bool(torch.isfinite(v).all())) for v in tensors.values()):
        raise ValueError("Empty or nonfinite state tensor")
    metadata.update(groups=groups, step=step, parameter_sha256=tensor_digest({k: v for k, v in tensors.items() if k.startswith("parameter.")}), optimizer_sha256=opt_digest,
                    tensors={k: hashlib.sha256(v.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest() for k, v in tensors.items()},
                    inventory={k: [str(v.dtype), list(v.shape)] for k, v in tensors.items()})
    return tensors, metadata


class StateBundleWriter:
    def __init__(self, directory, *, identity, uuid):
        _identity(identity)
        if not uuid:
            raise ValueError("State bundle requires identity and UUID")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.manifest = {"schema": SCHEMA, "status": "capturing", "identity": identity,
                         "uuid": uuid, "states": {}}
        self._write()

    def _write(self):
        write_result(self.directory / "manifest.json", self.manifest)

    def record(self, label, network, optimizer, device):
        if label not in LABELS or label in self.manifest["states"] or (label == "nonzero" and "initial" not in self.manifest["states"]):
            raise ValueError("Expected initial then nonzero state")
        tensors, metadata = _snapshot(network, optimizer, torch.device(device), LABELS.index(label))
        size = sum(v.numel() * v.element_size() for v in tensors.values())
        if size > FILE_LIMIT or sum(s["tensor_bytes"] for s in self.manifest["states"].values()) + size > TOTAL_LIMIT:
            raise ValueError("State bundle byte budget exceeded")
        path = self.directory / f"{label}.safetensors"
        save_file(tensors, str(path))
        if path.stat().st_size > FILE_LIMIT:
            raise ValueError("State file byte budget exceeded")
        self.manifest["states"][label] = {"file": path.name, "sha256": sha256_file(path), "tensor_bytes": size, **metadata}
        self._write()

    def finish(self, provenance):
        states = self.manifest["states"]
        if (set(states) != set(LABELS) or states["initial"]["parameter_sha256"] == states["nonzero"]["parameter_sha256"]
                or not isinstance(provenance, dict) or not provenance.get("warmup_inputs_sha256")):
            raise ValueError("Incomplete or unchanged warmup state")
        self.manifest.update(status="captured", provenance=provenance)
        self._write()


class StateBundle:
    def __init__(self, directory, *, identity, uuid):
        _identity(identity)
        self.directory = Path(directory)
        if self.directory.is_symlink():
            raise ValueError("Symlink state directory")
        path = self.directory / "manifest.json"
        if path.is_symlink() or path.stat().st_size > 1024**2:
            raise ValueError("Unsafe state manifest")
        self.manifest_sha256 = sha256_file(path)
        self.manifest = json.loads(path.read_text())
        m = self.manifest
        if (set(m) != {"schema", "status", "identity", "uuid", "states", "provenance"}
                or m["schema"] != SCHEMA or m["status"] != "captured" or m["identity"] != identity
                or m["uuid"] != uuid or set(m["states"]) != set(LABELS)
                or not isinstance(m["provenance"], dict) or not m["provenance"].get("warmup_inputs_sha256")):
            raise ValueError("State manifest contract mismatch")
        self.tensors = {}
        used = 0
        for label in LABELS:
            case = m["states"][label]
            path = _safe_path(self.directory, case.get("file"))
            size = path.stat().st_size
            used += size
            if case["file"] != f"{label}.safetensors" or size > FILE_LIMIT or used > TOTAL_LIMIT or sha256_file(path) != case.get("sha256"):
                raise ValueError("State file budget/digest mismatch")
            tensors = load_file(str(path))
            self._validate_case(label, case, tensors)
            self.tensors[label] = tensors
        if m["states"]["initial"]["parameter_sha256"] == m["states"]["nonzero"]["parameter_sha256"]:
            raise ValueError("Unchanged state bundle")

    @staticmethod
    def _validate_case(label, case, tensors):
        required = {"file", "sha256", "tensor_bytes", "python_rng", "numpy_rng", "groups", "step",
                    "parameter_sha256", "optimizer_sha256", "tensors", "inventory"}
        if (set(case) != required or case["step"] != LABELS.index(label) or set(tensors) != set(case["tensors"])
                or set(tensors) != set(case["inventory"]) or case["tensor_bytes"] != sum(v.numel() * v.element_size() for v in tensors.values())
                or any([str(v.dtype), list(v.shape)] != case["inventory"][k] or v.numel() == 0
                       or (v.is_floating_point() and not bool(torch.isfinite(v).all()))
                       or hashlib.sha256(v.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest() != case["tensors"][k]
                       for k, v in tensors.items())):
            raise ValueError("State tensor inventory/digest mismatch")
        params = {k: v for k, v in tensors.items() if k.startswith("parameter.")}
        opt = {k: v for k, v in tensors.items() if k.startswith("optimizer.")}
        names = {k.removeprefix("parameter.") for k in params}
        groups = case["groups"]
        if (not names or not isinstance(groups, list) or any(set(g) != {"parameters", "options"} or set(g["options"]) != OPTIONS for g in groups)
                or not {n for g in groups for n in g["parameters"]} <= names
                or len([n for g in groups for n in g["parameters"]]) != len({n for g in groups for n in g["parameters"]})
                or tensor_digest(params) != case["parameter_sha256"] or tensor_digest(opt) != case["optimizer_sha256"]
                or set(tensors) - set(params) - set(opt) - {k for k in tensors if k.startswith("buffer.")} - {"rng.cpu", "rng.cuda"}
                or (label == "initial" and opt) or (label == "nonzero" and not opt)):
            raise ValueError("State parameter/optimizer contract mismatch")
        for name in (n for g in groups for n in g["parameters"]):
            keys = {k.removeprefix(f"optimizer.{name}.") for k in opt if k.startswith(f"optimizer.{name}.")}
            group = next(g for g in groups if name in g["parameters"])
            expected = {"step", "exp_avg", "exp_avg_sq"} | ({"max_exp_avg_sq"} if group["options"]["amsgrad"] else set())
            if label == "nonzero" and (keys != expected or float(opt[f"optimizer.{name}.step"].item()) != 1):
                raise ValueError("Invalid nonzero AdamW step")
        expected_opt = set()
        if label == "nonzero":
            for group in groups:
                keys = {"step", "exp_avg", "exp_avg_sq"} | ({"max_exp_avg_sq"} if group["options"]["amsgrad"] else set())
                expected_opt.update(f"optimizer.{name}.{key}" for name in group["parameters"] for key in keys)
        if set(opt) != expected_opt:
            raise ValueError("Unexpected named optimizer tensor")
        if label == "nonzero" and not any(bool(torch.count_nonzero(v)) for k, v in opt.items() if k.endswith(("exp_avg", "exp_avg_sq"))):
            raise ValueError("Nonzero moments required")

    def restore(self, label, network, optimizer, device):
        if label not in LABELS:
            raise ValueError("Unknown state label")
        case, tensors = self.manifest["states"][label], self.tensors[label]
        params = _params(network)
        buffers = dict(network.named_buffers())
        if (set(params) != {k.removeprefix("parameter.") for k in tensors if k.startswith("parameter.")}
                or set(buffers) != {k.removeprefix("buffer.") for k in tensors if k.startswith("buffer.")}
                or optimizer_options(network, optimizer) != case["groups"]):
            raise ValueError("Live state inventory/options mismatch")
        device = torch.device(device)
        if (device.type == "cuda") != ("rng.cuda" in tensors):
            raise ValueError("RNG device mismatch")
        with torch.no_grad():
            for kind, mapping in (("parameter", params), ("buffer", buffers)):
                for name, dest in mapping.items():
                    src = tensors[f"{kind}.{name}"]
                    if dest.shape != src.shape or dest.dtype != src.dtype:
                        raise ValueError("Live tensor shape/dtype mismatch")
                    dest.copy_(src)
            optimizer.state.clear()
            options = {n: g["options"] for g in case["groups"] for n in g["parameters"]}
            for name, param in params.items():
                param.grad = None
                if label == "initial" or name not in options:
                    continue
                keys = ("step", "exp_avg", "exp_avg_sq") + (("max_exp_avg_sq",) if options[name]["amsgrad"] else ())
                optimizer.state[param] = {key: tensors[f"optimizer.{name}.{key}"].to(
                    device=(param.device if options[name]["capturable"] or options[name]["fused"] else "cpu") if key == "step" else param.device).clone() for key in keys}
        torch.set_rng_state(tensors["rng.cpu"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(tensors["rng.cuda"], device)
        random.setstate(tuple_tree(case["python_rng"]))
        n = case["numpy_rng"]
        np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
        _, _, actual_opt = optimizer_snapshot(network, optimizer, expected_step=case["step"])
        actual_params = tensor_digest({f"parameter.{k}": v for k, v in params.items()})
        if actual_opt != case["optimizer_sha256"] or actual_params != case["parameter_sha256"]:
            raise ValueError("Restored state digest mismatch")
        return {"bundle_manifest_sha256": self.manifest_sha256, "state_file_sha256": case["sha256"],
                "state_label": label, "parameter_sha256": actual_params, "optimizer_sha256": actual_opt}
