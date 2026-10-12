"""Disposable full-shape Anima block VJP evidence; no weights are exported."""

from __future__ import annotations

from dataclasses import fields
import json
import math
from pathlib import Path

from safetensors.torch import load_file, save_file
import torch

from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime.replay import tensor_digest
from library.training.auto_block_swap.process import write_result
from networks.attention_dispatch import AttentionParams


SCOPE = "full_shape_block14_vjp_disposable_not_production_certificate"
MULTI_SCOPE = "full_shape_blocks15_17_27_vjp_disposable_not_production_certificate"
MAX_BYTES = 512 * 1024**2
SHAPE = (1, 1, 75, 56, 2048)
SCHEMA = "anima_block_vjp_v1"
MULTI_SCHEMA = "anima_multi_block_vjp_v1"
MULTI_INDICES = (15, 17, 27)
_ARGS = ("emb_B_T_D", "crossattn_emb", "attn_params", "rope_cos_sin",
         "adaln_lora_B_T_3D", "use_fp32")
_TENSOR_FIELDS = {"attention_mask", "seqlens", "cu_seqlens"}
_OBJECT_FIELDS = {"crossattn_block_mask", "selfattn_block_mask"}


def validate_block_indices(indices):
    if type(indices) not in (tuple, list) or tuple(indices) not in ((14,), MULTI_INDICES):
        raise ValueError("Block VJP indices must be (14,) or (15, 17, 27) in order")
    return tuple(indices)


def scope_for(indices):
    return SCOPE if validate_block_indices(indices) == (14,) else MULTI_SCOPE


def _tensor_bytes(tensors):
    return sum(value.numel() * value.element_size() for value in tensors.values())


class _AggregateBudget:
    def __init__(self):
        self.reserved = {}

    def reserve(self, index, size):
        if index in self.reserved:
            raise ValueError("Duplicate block VJP budget reservation")
        if sum(self.reserved.values()) + size > MAX_BYTES:
            raise ValueError("Multi-block VJP 512MiB aggregate budget exceeded")
        self.reserved[index] = size


def _block_adapters(network, block_index):
    prefix = f"lora_unet_blocks_{block_index}_"
    adapters = [a for a in network.unet_loras if a.lora_name.startswith(prefix)]
    if not adapters:
        raise ValueError("Missing block LoRA adapters")
    return adapters


def state_digests(model, network, block_index):
    """Digest exact named base parameters and this block's LoRA state only."""
    block = model.blocks[block_index]
    base = {f"blocks.{block_index}.{name}": value
            for name, value in block.named_parameters()}
    if not base or any(value.requires_grad for value in base.values()):
        raise ValueError("Block base must be frozen")
    adapters = _block_adapters(network, block_index)
    state = {f"{adapter.lora_name}.{name}": value
             for adapter in adapters for name, value in adapter.state_dict().items()}
    if not state:
        raise ValueError("Missing block LoRA state")
    return tensor_digest(base), tensor_digest(state)


def _cpu_tensor(value):
    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
        raise ValueError("Missing or nonfinite capture tensor")
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()


def _cpu_conditioning(value):
    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
        raise ValueError("Missing or nonfinite conditioning tensor")
    return value.detach().cpu().contiguous()


def _pack_conditioning(args, kwargs):
    if len(args) > 7 or any(key not in _ARGS for key in kwargs):
        raise ValueError("Unknown block conditioning")
    values = dict(zip(_ARGS, args[1:], strict=False))
    if set(values) & set(kwargs):
        raise ValueError("Duplicate block conditioning")
    values.update(kwargs)
    defaults = {"rope_cos_sin": None, "adaln_lora_B_T_3D": None, "use_fp32": False}
    values = {**defaults, **values}
    if set(values) != set(_ARGS) or type(values["use_fp32"]) is not bool:
        raise ValueError("Incomplete block conditioning")
    attn = values["attn_params"]
    if type(attn) is not AttentionParams:
        raise ValueError("Unknown attention parameter object")
    tensors = {}
    meta = {"attn_params": {}, "rope_cos_sin": values["rope_cos_sin"] is not None,
            "adaln_lora_B_T_3D": values["adaln_lora_B_T_3D"] is not None,
            "use_fp32": values["use_fp32"]}
    for key in ("emb_B_T_D", "crossattn_emb", "adaln_lora_B_T_3D"):
        if values[key] is not None:
            tensors[f"conditioning.{key}"] = _cpu_conditioning(values[key])
    rope = values["rope_cos_sin"]
    if rope is not None:
        if not isinstance(rope, (tuple, list)) or len(rope) != 2:
            raise ValueError("Invalid RoPE tuple")
        for index, value in enumerate(rope):
            tensors[f"conditioning.rope_{index}"] = _cpu_conditioning(value)
    for field in fields(AttentionParams):
        name = field.name
        value = getattr(attn, name)
        if name in _OBJECT_FIELDS:
            if value is not None:
                raise ValueError("Unsupported attention block mask")
            meta["attn_params"][name] = None
        elif isinstance(value, torch.Tensor):
            if name not in _TENSOR_FIELDS:
                raise ValueError("Unknown attention tensor field")
            if value.dtype not in (torch.bool, torch.int32, torch.int64, torch.float32):
                raise ValueError("Unsupported attention tensor dtype")
            tensors[f"conditioning.attn_params.{name}"] = value.detach().cpu().contiguous()
            meta["attn_params"][name] = {"tensor": True, "dtype": str(value.dtype)}
        elif value is None or type(value) in (bool, int, float, str):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Nonfinite attention scalar")
            meta["attn_params"][name] = value
        else:
            raise ValueError("Unknown attention field object")
    return tensors, meta


def _validate_tensors(tensors, meta, *, shape):
    expected = {"x", "y", "g", "gx", "conditioning.emb_B_T_D",
                "conditioning.crossattn_emb"}
    if meta["adaln_lora_B_T_3D"]:
        expected.add("conditioning.adaln_lora_B_T_3D")
    if meta["rope_cos_sin"]:
        expected.update({"conditioning.rope_0", "conditioning.rope_1"})
    for key, value in meta["attn_params"].items():
        if isinstance(value, dict) and value == {"tensor": True, "dtype": value.get("dtype")}:
            expected.add(f"conditioning.attn_params.{key}")
    if set(tensors) != expected:
        raise ValueError("Missing or extra block VJP tensors")
    if any(tuple(tensors[key].shape) != tuple(shape) or tensors[key].dtype != torch.float32
           for key in ("x", "y", "g", "gx")):
        raise ValueError("Block VJP shape/dtype mismatch")
    if (tensors["conditioning.emb_B_T_D"].ndim != 3
            or tensors["conditioning.emb_B_T_D"].shape[0] != 1
            or tensors["conditioning.crossattn_emb"].ndim != 3
            or tensors["conditioning.crossattn_emb"].shape[0] != 1):
        raise ValueError("Block conditioning shape mismatch")
    if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise ValueError("Nonfinite block VJP tensor")


def conditioning(manifest, tensors, device):
    meta = manifest["conditioning"]
    params = {}
    for field in fields(AttentionParams):
        name = field.name
        value = meta["attn_params"][name]
        params[name] = (tensors[f"conditioning.attn_params.{name}"].to(device)
                        if isinstance(value, dict) else value)
    return {"emb_B_T_D": tensors["conditioning.emb_B_T_D"].to(device),
            "crossattn_emb": tensors["conditioning.crossattn_emb"].to(device),
            "attn_params": AttentionParams(**params),
            "rope_cos_sin": tuple(tensors[f"conditioning.rope_{i}"].to(device) for i in range(2))
            if meta["rope_cos_sin"] else None,
            "adaln_lora_B_T_3D": tensors["conditioning.adaln_lora_B_T_3D"].to(device)
            if meta["adaln_lora_B_T_3D"] else None,
            "use_fp32": meta["use_fp32"]}


class BlockVJPCapture:
    def __init__(self, directory, model, network, *, signature, uuid, scale, replay,
                 block_index=14, budget=None):
        indices = validate_block_indices(signature.get("block_vjp_indices", [14]))
        if (block_index not in indices or len(model.blocks) <= block_index
                or signature.get("steps") != 1 or signature.get("sigmas") != [0.5]
                or signature.get("block_vjp_scope") != scope_for(indices)
                or not isinstance(signature.get("block_vjp_helper_sha256"), str)
                or len(signature["block_vjp_helper_sha256"]) != 64
                or not isinstance(scale, (int, float)) or not math.isfinite(scale)
                or scale < 1 or signature.get("loss_scale") != scale or not uuid):
            raise ValueError("Invalid fixed Anima block VJP contract")
        adapters = _block_adapters(network, block_index)
        if any(not hasattr(a, "lora_up") or bool(torch.count_nonzero(a.lora_up.weight.detach()))
               for a in adapters):
            raise ValueError("Block LoRA up weights must be zero")
        base_hash, adapter_hash = state_digests(model, network, block_index)
        block = model.blocks[block_index]
        residual = getattr(block, "fp32_residual", None)
        if (type(residual) is not bool
                or residual is not (signature.get("precision") == "fp16-islands")
                or getattr(block, "gradient_checkpointing", None) is not True):
            raise ValueError("Actual block residual/checkpoint contract mismatch")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.scale = scale
        self.budget = budget
        self.block_index = block_index
        self.tensors = {}
        self.handles = []
        self.seen = False
        self.manifest = {"schema": SCHEMA, "status": "capturing", "scope": scope_for(indices),
                         "signature": signature, "uuid": uuid, "replay": replay,
                         "block_index": block_index, "sigma": 0.5, "loss_scale": scale,
                         "fp32_residual": residual,
                         "base_state_sha256": base_hash, "adapter_state_sha256": adapter_hash,
                         "budget_bytes": MAX_BYTES}
        write_result(self.directory / "manifest.json", self.manifest)
        self.handles.append(block.register_full_backward_hook(lambda _m, _gi, _go: None))
        self.handles.append(block.register_forward_hook(self._forward, with_kwargs=True))

    def _gradient(self, name, gradient):
        if name in self.tensors:
            raise ValueError(f"Duplicate block {name} cotangent")
        self.tensors[name] = _cpu_tensor(gradient).div_(self.scale)
        return gradient

    def _forward(self, _module, args, kwargs, output):
        if self.seen or not args or not isinstance(args[0], torch.Tensor):
            raise ValueError("Duplicate or invalid block forward")
        x = args[0]
        if (tuple(x.shape) != SHAPE or not isinstance(output, torch.Tensor)
                or tuple(output.shape) != SHAPE or not x.requires_grad or not output.requires_grad):
            raise ValueError("Unexpected full-shape block boundary")
        values, meta = _pack_conditioning(args, kwargs)
        values.update(x=_cpu_tensor(x), y=_cpu_tensor(output))
        reservation = _tensor_bytes(values) + 2 * x.numel() * 4
        if reservation > MAX_BYTES:
            raise ValueError("Block VJP 512MiB budget exceeded")
        if self.budget is not None:
            self.budget.reserve(self.block_index, reservation)
        self.tensors.update(values)
        self.manifest["conditioning"] = meta
        self.seen = True
        x.register_hook(lambda gradient: self._gradient("gx", gradient))
        output.register_hook(lambda gradient: self._gradient("g", gradient))

    def restore(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def close(self):
        self.restore()
        if not self.seen or "g" not in self.tensors or "gx" not in self.tensors:
            raise ValueError("Incomplete block VJP forward/cotangent capture")
        _validate_tensors(self.tensors, self.manifest["conditioning"], shape=SHAPE)
        total = _tensor_bytes(self.tensors)
        if total > MAX_BYTES:
            raise ValueError("Block VJP tensor budget exceeded")
        if self.budget is not None and total > self.budget.reserved[self.block_index]:
            raise ValueError("Block VJP exceeded reserved aggregate budget")
        path = self.directory / "capture.safetensors"
        save_file(self.tensors, str(path))
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("Block VJP disk budget exceeded")
        self.manifest.update(status="captured", file=path.name, sha256=sha256_file(path),
                             tensor_bytes=total, tensor_shapes={key: list(value.shape)
                                                                for key, value in self.tensors.items()})
        write_result(self.directory / "manifest.json", self.manifest)


class MultiBlockVJPCapture:
    def __init__(self, directory, model, network, *, signature, uuid, scale, replay,
                 block_indices=MULTI_INDICES):
        indices = validate_block_indices(block_indices)
        if indices != MULTI_INDICES or signature.get("block_vjp_indices") != list(indices):
            raise ValueError("Invalid multi-block VJP index signature")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.captures = []
        self.budget = _AggregateBudget()
        self.manifest = {"schema": MULTI_SCHEMA, "status": "capturing", "scope": MULTI_SCOPE,
                         "signature": signature, "uuid": uuid, "replay": replay,
                         "block_indices": list(indices), "budget_bytes": MAX_BYTES}
        try:
            write_result(self.directory / "manifest.json", self.manifest)
            for index in indices:
                self.captures.append(BlockVJPCapture(
                    self.directory / f"block-{index:02d}", model, network,
                    signature=signature, uuid=uuid, scale=scale, replay=replay,
                    block_index=index, budget=self.budget))
        except BaseException:
            self.restore()
            raise

    def restore(self):
        for capture in self.captures:
            capture.restore()

    def close(self):
        self.restore()
        blocks = {}
        for capture in self.captures:
            capture.close()
            index = capture.block_index
            path = capture.directory / "capture.safetensors"
            blocks[str(index)] = {"directory": capture.directory.name,
                                  "sha256": capture.manifest["sha256"],
                                  "tensor_bytes": capture.manifest["tensor_bytes"],
                                  "disk_bytes": path.stat().st_size}
        tensor_bytes = sum(item["tensor_bytes"] for item in blocks.values())
        disk_bytes = sum(item["disk_bytes"] for item in blocks.values())
        if tensor_bytes > MAX_BYTES or disk_bytes > MAX_BYTES:
            raise ValueError("Multi-block VJP aggregate tensor/disk budget exceeded")
        self.manifest.update(status="captured", blocks=blocks,
                             tensor_bytes=tensor_bytes, disk_bytes=disk_bytes)
        write_result(self.directory / "manifest.json", self.manifest)


def read_capture(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    signature = manifest.get("signature", {})
    indices = validate_block_indices(signature.get("block_vjp_indices", [14]))
    block_index = manifest.get("block_index")
    if (manifest.get("schema") != SCHEMA or manifest.get("status") != "captured"
            or manifest.get("scope") != scope_for(indices) or block_index not in indices
            or manifest.get("sigma") != 0.5 or manifest.get("budget_bytes") != MAX_BYTES
            or signature.get("block_vjp_scope") != scope_for(indices)
            or not isinstance(signature.get("block_vjp_helper_sha256"), str)
            or len(signature["block_vjp_helper_sha256"]) != 64
            or manifest.get("file") != "capture.safetensors"
            or not isinstance(manifest.get("uuid"), str)
            or not isinstance(manifest.get("base_state_sha256"), str)
            or len(manifest["base_state_sha256"]) != 64
            or not isinstance(manifest.get("adapter_state_sha256"), str)
            or len(manifest["adapter_state_sha256"]) != 64
            or type(manifest.get("fp32_residual")) is not bool
            or manifest["fp32_residual"] != (signature.get("precision") == "fp16-islands")
            or not isinstance(manifest.get("conditioning"), dict)
            or type(manifest.get("tensor_bytes")) is not int
            or not 0 < manifest["tensor_bytes"] <= MAX_BYTES):
        raise ValueError("Invalid block VJP manifest")
    meta = manifest["conditioning"]
    if (set(meta) != {"attn_params", "rope_cos_sin", "adaln_lora_B_T_3D", "use_fp32"}
            or type(meta["rope_cos_sin"]) is not bool
            or type(meta["adaln_lora_B_T_3D"]) is not bool
            or type(meta["use_fp32"]) is not bool
            or not isinstance(meta["attn_params"], dict)
            or set(meta["attn_params"]) != {field.name for field in fields(AttentionParams)}
            or any(meta["attn_params"][key] is not None for key in _OBJECT_FIELDS)):
        raise ValueError("Invalid block conditioning metadata")
    path = directory / "capture.safetensors"
    if path.stat().st_size > MAX_BYTES or sha256_file(path) != manifest.get("sha256"):
        raise ValueError("Block VJP hash/budget mismatch")
    tensors = load_file(str(path))
    if manifest.get("tensor_shapes") != {key: list(value.shape) for key, value in tensors.items()}:
        raise ValueError("Block VJP tensor shape metadata mismatch")
    if sum(value.numel() * value.element_size() for value in tensors.values()) != manifest["tensor_bytes"]:
        raise ValueError("Block VJP tensor byte metadata mismatch")
    _validate_tensors(tensors, meta, shape=SHAPE)
    return manifest, tensors


def read_multi_capture(directory):
    directory = Path(directory)
    root = json.loads((directory / "manifest.json").read_text())
    indices = validate_block_indices(root.get("block_indices"))
    if (indices != MULTI_INDICES or root.get("schema") != MULTI_SCHEMA
            or root.get("status") != "captured" or root.get("scope") != MULTI_SCOPE
            or root.get("budget_bytes") != MAX_BYTES
            or root.get("signature", {}).get("block_vjp_indices") != list(indices)
            or root["signature"].get("block_vjp_scope") != MULTI_SCOPE
            or type(root.get("tensor_bytes")) is not int
            or type(root.get("disk_bytes")) is not int
            or not isinstance(root.get("blocks"), dict)
            or set(root["blocks"]) != {str(index) for index in indices}):
        raise ValueError("Invalid multi-block VJP manifest")
    captures = {}
    total_tensors = total_disk = 0
    for index in indices:
        entry = root["blocks"][str(index)]
        name = f"block-{index:02d}"
        if not isinstance(entry, dict) or entry.get("directory") != name:
            raise ValueError("Invalid multi-block VJP child directory")
        child_dir = directory / name
        manifest, tensors = read_capture(child_dir)
        size = (child_dir / "capture.safetensors").stat().st_size
        if (manifest["block_index"] != index or manifest["signature"] != root["signature"]
                or manifest["uuid"] != root.get("uuid")
                or manifest.get("replay") != root.get("replay")
                or entry.get("sha256") != manifest["sha256"]
                or entry.get("tensor_bytes") != manifest["tensor_bytes"]
                or entry.get("disk_bytes") != size):
            raise ValueError("Multi-block VJP child identity mismatch")
        total_tensors += manifest["tensor_bytes"]
        total_disk += size
        captures[index] = manifest, tensors
    if (total_tensors != root["tensor_bytes"] or total_disk != root["disk_bytes"]
            or total_tensors > MAX_BYTES or total_disk > MAX_BYTES):
        raise ValueError("Multi-block VJP aggregate budget mismatch")
    return root, captures
