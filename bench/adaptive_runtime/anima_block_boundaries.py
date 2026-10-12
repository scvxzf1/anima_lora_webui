"""Disposable sampled Anima block-boundary cotangents; never a precision plan."""

from __future__ import annotations

import json
import math
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch

from bench.adaptive_runtime.anima_inputs import sha256_file
from library.training.adaptive_runtime.precision import vector_metrics
from library.training.auto_block_swap.process import write_result


SCOPE = "sampled_block_boundary_cotangents_not_causal_or_production_certificate"
TOKENS = 512
MAX_BYTES = 512 * 1024**2


def sampled_indices(count, sample_tokens=TOKENS):
    if count < sample_tokens or sample_tokens < 1:
        raise ValueError("Insufficient block tokens")
    return torch.linspace(0, count - 1, sample_tokens, dtype=torch.float64).long()


def _same_boundary_storage(previous, current):
    return (current is previous or
            (current.device == previous.device
             and current.dtype == previous.dtype
             and current.shape == previous.shape
             and current.stride() == previous.stride()
             and current.storage_offset() == previous.storage_offset()
             and current.untyped_storage().data_ptr() == previous.untyped_storage().data_ptr()
             and current._version == previous._version))


class BlockBoundaryCapture:
    def __init__(self, directory, model, *, signature, uuid, scale, replay,
                 block_count=28, sample_tokens=TOKENS, max_bytes=MAX_BYTES):
        if (signature.get("steps") != 1 or signature.get("sigmas") != [0.5]
                or signature.get("block_boundary_scope") != SCOPE
                or not isinstance(scale, (int, float)) or not math.isfinite(scale)
                or scale < 1 or signature.get("loss_scale") != scale
                or not isinstance(sample_tokens, int) or not 1 <= sample_tokens <= TOKENS
                or not isinstance(max_bytes, int) or not 0 < max_bytes <= MAX_BYTES
                or block_count < 1
                or len(model.blocks) != block_count):
            raise ValueError("Invalid fixed Anima boundary capture contract")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.scale, self.block_count = scale, block_count
        self.sample_tokens, self.max_bytes = sample_tokens, max_bytes
        self.samples, self.shapes, self.handles = {}, {}, []
        self.used_bytes = 0
        self.previous = None
        self.manifest = {"schema": "anima_block_boundaries_v1", "status": "capturing",
                         "scope": SCOPE, "signature": signature, "uuid": uuid,
                         "sigma": 0.5, "replay": replay, "block_count": block_count,
                         "sample_tokens": sample_tokens, "budget_bytes": max_bytes,
                         "boundaries": []}
        write_result(self.directory / "manifest.json", self.manifest)
        for index, block in enumerate(model.blocks):
            self.handles.append(block.register_forward_hook(self._hook(index)))

    def _sample(self, tensor, indices):
        return tensor.detach().flatten(1, -2).index_select(1, indices.to(tensor.device)).float().cpu().contiguous()

    def _capture(self, index, tensor):
        if index in self.samples or not isinstance(tensor, torch.Tensor) or tensor.ndim not in (3, 5):
            raise ValueError(f"Duplicate or invalid block boundary {index}")
        shape = list(tensor.shape)
        if shape[0] != 1 or not tensor.requires_grad:
            raise ValueError(f"Unexpected block boundary shape/grad {index}")
        count = math.prod(shape[1:-1])
        indices = sampled_indices(count, self.sample_tokens)
        value = self._sample(tensor, indices)
        size = value.numel() * 8 + indices.numel() * 8
        if self.used_bytes + size > self.max_bytes:
            raise ValueError("Block-boundary 512MiB capture budget exceeded")
        self.used_bytes += size
        self.samples[index] = {"activation": value, "indices": indices, "gradient": None}
        self.shapes[index] = shape

        def capture_gradient(gradient):
            item = self.samples[index]
            if item["gradient"] is not None or list(gradient.shape) != shape:
                raise ValueError(f"Duplicate or changed block cotangent {index}")
            item["gradient"] = self._sample(gradient, indices).div_(self.scale)
            return gradient

        tensor.register_hook(capture_gradient)

    def _hook(self, index):
        def hook(_module, args, output):
            if len(args) < 1 or not isinstance(output, torch.Tensor):
                raise ValueError("Unexpected block forward signature")
            if index == 0:
                self._capture(0, args[0])
            elif (not _same_boundary_storage(self.previous, args[0])
                  or list(args[0].shape) != self.shapes[index]):
                raise ValueError(f"Nonadjacent block boundary {index}")
            if index + 1 in self.samples:
                raise ValueError("Repeated block module forward")
            self._capture(index + 1, output)
            self.previous = output
        return hook

    def restore(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.previous = None

    def close(self):
        self.restore()
        if set(self.samples) != set(range(self.block_count + 1)) or any(
                item["gradient"] is None for item in self.samples.values()):
            raise ValueError("Missing block boundary activation or cotangent")
        tensors = {}
        for index, item in sorted(self.samples.items()):
            if any(not bool(torch.isfinite(value).all()) for value in item.values()):
                raise ValueError("Nonfinite boundary evidence")
            tensors.update({f"boundary_{index:02d}.{key}": value for key, value in item.items()})
            self.manifest["boundaries"].append({"index": index, "shape": self.shapes[index],
                 "indices": item["indices"].tolist(),
                 "token_layout": "row_major_flatten_nonbatch_nonfeature_axes"})
        path = self.directory / "boundaries.safetensors"
        save_file(tensors, str(path))
        if path.stat().st_size > self.max_bytes:
            raise ValueError("Block-boundary disk budget exceeded")
        self.manifest.update(status="captured", file=path.name, sha256=sha256_file(path),
                             tensor_bytes=self.used_bytes)
        write_result(self.directory / "manifest.json", self.manifest)


def read_capture(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    count = manifest.get("block_count")
    samples = manifest.get("sample_tokens")
    if (manifest.get("schema") != "anima_block_boundaries_v1"
            or manifest.get("status") != "captured" or manifest.get("scope") != SCOPE
            or not isinstance(count, int) or count < 1 or not isinstance(samples, int) or samples < 1
            or manifest.get("budget_bytes") != MAX_BYTES
            or not 0 < manifest.get("tensor_bytes", 0) <= MAX_BYTES
            or manifest.get("sigma") != 0.5
            or manifest.get("file") != "boundaries.safetensors"
            or len(manifest.get("boundaries", [])) != count + 1):
        raise ValueError("Incomplete boundary manifest")
    path = directory / "boundaries.safetensors"
    if path.stat().st_size > MAX_BYTES or sha256_file(path) != manifest.get("sha256"):
        raise ValueError("Boundary file hash/budget mismatch")
    tensors = load_file(str(path))
    expected = {f"boundary_{i:02d}.{key}" for i in range(count + 1)
                for key in ("activation", "gradient", "indices")}
    if set(tensors) != expected:
        raise ValueError("Missing or extra boundary tensor")
    for index, entry in enumerate(manifest["boundaries"]):
        shape = entry.get("shape")
        if (entry.get("index") != index or not isinstance(shape, list) or len(shape) not in (3, 5)
                or shape[0] != 1 or entry.get("token_layout") !=
                "row_major_flatten_nonbatch_nonfeature_axes"):
            raise ValueError("Boundary shape/index contract mismatch")
        indices = sampled_indices(math.prod(shape[1:-1]), samples)
        prefix = f"boundary_{index:02d}."
        if (entry.get("indices") != indices.tolist()
                or not torch.equal(tensors[prefix + "indices"], indices)
                or any(tensors[prefix + key].shape != (1, samples, shape[-1])
                       or tensors[prefix + key].dtype != torch.float32
                       or not bool(torch.isfinite(tensors[prefix + key]).all())
                       for key in ("activation", "gradient"))):
            raise ValueError("Boundary samples changed")
    return manifest, tensors


def analyze(reference, candidate, control):
    captures = [read_capture(directory) for directory in (reference, candidate, control)]
    manifests = [item[0] for item in captures]
    a, b, c = manifests
    excluded = {"precision", "fp32_modules", "loss_scale"}
    def contract(manifest):
        return {k: v for k, v in manifest["signature"].items() if k not in excluded}
    if (contract(a) != contract(b) or contract(a) != contract(c)
            or a["uuid"] != b["uuid"] or a["uuid"] != c["uuid"]
            or a["replay"] != b["replay"] or a["replay"] != c["replay"]
            or a["boundaries"] != b["boundaries"] or a["boundaries"] != c["boundaries"]
            or [m["signature"]["loss_scale"] for m in manifests] != [1, 1024, 1024]):
        raise ValueError("Boundary comparison source/replay/layout contract mismatch")
    rows = []
    first = None
    for index in reversed(range(a["block_count"] + 1)):
        prefix = f"boundary_{index:02d}."
        ref, trial, ctrl = [item[1] for item in captures]
        grad = vector_metrics([(ref[prefix + "gradient"], trial[prefix + "gradient"])])
        control_grad = vector_metrics([(ref[prefix + "gradient"], ctrl[prefix + "gradient"])])
        activation = vector_metrics([(ref[prefix + "activation"], trial[prefix + "activation"])])
        control_activation = vector_metrics([(ref[prefix + "activation"], ctrl[prefix + "activation"])])
        crossed = grad["relative_l2"] is None or grad["relative_l2"] > .02 or grad["cosine"] < .999
        if crossed and first is None:
            first = index
        rows.append({"boundary": index, "gradient": grad, "activation_drift": activation,
                     "control_gradient": control_grad, "control_activation_drift": control_activation,
                     "exceeds_sampled_threshold": crossed})
    return {"status": "analyzed", "scope": SCOPE, "production_ready": False,
            "interpretation": "Sampled adjacent error differences are not causal contributions; do not recommend a plan.",
            "first_exceedance_reverse_order": first, "boundaries_reverse": rows}
