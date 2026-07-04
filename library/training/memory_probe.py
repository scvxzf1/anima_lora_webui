"""Opt-in JSONL memory probe for tight-VRAM training experiments."""

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_DISABLED_VALUES = {"", "off", "none", "false", "0"}


def _jsonable(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, torch.device):
        return str(value)
    return str(value)


def _counter_to_dict(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items())}


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _device_name(device: torch.device | str | None) -> str:
    try:
        return str(torch.device(device)) if device is not None else "unknown"
    except Exception:
        return str(device)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _summarize_parameters(parameters) -> dict[str, Any]:
    total = 0
    trainable = 0
    total_bytes = 0
    trainable_bytes = 0
    devices: Counter[str] = Counter()
    dtypes: Counter[str] = Counter()
    for param in parameters:
        if param is None:
            continue
        count = int(param.numel())
        bytes_ = _tensor_bytes(param)
        total += count
        total_bytes += bytes_
        if getattr(param, "requires_grad", False):
            trainable += count
            trainable_bytes += bytes_
        devices[_device_name(param.device)] += count
        dtypes[_dtype_name(param.dtype)] += count
    return {
        "param_count": total,
        "trainable_param_count": trainable,
        "param_bytes": total_bytes,
        "trainable_param_bytes": trainable_bytes,
        "devices": _counter_to_dict(devices),
        "dtypes": _counter_to_dict(dtypes),
    }


def _summarize_module_list(modules: list[Any]) -> dict[str, Any]:
    type_counts: Counter[str] = Counter(type(module).__name__ for module in modules)
    params = []
    lokr_factors: Counter[str] = Counter()
    lokr_shapes: Counter[str] = Counter()
    for module in modules:
        params.extend(list(module.parameters(recurse=True)))
        if hasattr(module, "factor"):
            lokr_factors[str(getattr(module, "factor"))] += 1
        if hasattr(module, "lokr_w1") and (
            hasattr(module, "lokr_w2")
            or (hasattr(module, "lokr_w2_a") and hasattr(module, "lokr_w2_b"))
        ):
            w1 = getattr(module, "lokr_w1")
            if hasattr(module, "lokr_w2"):
                w2_shape = tuple(getattr(module, "lokr_w2").shape)
            else:
                w2a = getattr(module, "lokr_w2_a")
                w2b = getattr(module, "lokr_w2_b")
                w2_shape = (tuple(w2a.shape), tuple(w2b.shape))
            lokr_shapes[f"{tuple(w1.shape)}x{w2_shape}"] += 1
    out = {
        "module_count": len(modules),
        "module_types": _counter_to_dict(type_counts),
        **_summarize_parameters(params),
    }
    if lokr_factors:
        out["lokr_factors"] = _counter_to_dict(lokr_factors)
    if lokr_shapes:
        out["lokr_shapes"] = _counter_to_dict(lokr_shapes)
    return out


def _optimizer_state_summary(optimizer) -> dict[str, Any]:
    if optimizer is None:
        return {}
    group_summaries: list[dict[str, Any]] = []
    for idx, group in enumerate(getattr(optimizer, "param_groups", []) or []):
        params = list(group.get("params", []) or [])
        summary = _summarize_parameters(params)
        summary["group_index"] = idx
        summary["lr"] = group.get("lr")
        group_summaries.append(summary)

    state_param_count = 0
    state_tensor_count = 0
    state_bytes = 0
    state_devices: Counter[str] = Counter()
    state_dtypes: Counter[str] = Counter()
    for state in getattr(optimizer, "state", {}).values():
        state_param_count += 1
        if not isinstance(state, dict):
            continue
        for value in state.values():
            if not isinstance(value, torch.Tensor):
                continue
            state_tensor_count += 1
            state_bytes += _tensor_bytes(value)
            state_devices[_device_name(value.device)] += int(value.numel())
            state_dtypes[_dtype_name(value.dtype)] += int(value.numel())

    return {
        "optimizer_type": type(optimizer).__name__,
        "param_group_count": len(group_summaries),
        "param_groups": group_summaries,
        "state_param_count": state_param_count,
        "state_tensor_count": state_tensor_count,
        "state_tensor_bytes": state_bytes,
        "state_devices": _counter_to_dict(state_devices),
        "state_dtypes": _counter_to_dict(state_dtypes),
    }


class MemoryProbe:
    """Append-only memory and adapter summary stream.

    The probe is deliberately best-effort: failed writes or CUDA queries must
    never change training behavior.
    """

    def __init__(self, path: str, *, max_steps: int = 2, t0: Optional[float] = None):
        self.path = str(path)
        self.max_steps = max(0, int(max_steps or 0))
        self._t0 = t0 if t0 is not None else time.time()
        self._seq = 0
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)

    @staticmethod
    def resolve_path(args) -> Optional[str]:
        explicit = getattr(args, "memory_probe_jsonl", None)
        if explicit is None:
            return None
        value = str(explicit).strip()
        if value.lower() in _DISABLED_VALUES:
            return None
        if value.lower() != "auto":
            return value
        output_dir = getattr(args, "output_dir", None)
        if not output_dir:
            return None
        output_name = getattr(args, "output_name", None) or "run"
        parent = os.path.dirname(os.path.normpath(output_dir))
        logs_dir = os.path.join(parent or output_dir, "logs")
        return os.path.join(logs_dir, f"{output_name}.memory_probe.jsonl")

    @classmethod
    def from_args(
        cls,
        args,
        *,
        is_main_process: bool,
        t0: Optional[float] = None,
    ) -> Optional["MemoryProbe"]:
        if not is_main_process:
            return None
        path = cls.resolve_path(args)
        if path is None:
            return None
        return cls(
            path,
            max_steps=int(getattr(args, "memory_probe_max_steps", 2) or 0),
            t0=t0,
        )

    def should_record_step(self, step: int) -> bool:
        return self.max_steps <= 0 or int(step) < self.max_steps

    def write(self, event: dict[str, Any]) -> None:
        try:
            self._seq += 1
            payload = {
                "seq": self._seq,
                "ts": round(time.time() - self._t0, 3),
                **event,
            }
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=_jsonable, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.debug("memory probe write failed: %s", exc)

    def snapshot(
        self,
        label: str,
        *,
        device=None,
        step: Optional[int] = None,
        phase: Optional[str] = None,
        **fields: Any,
    ) -> None:
        payload: dict[str, Any] = {
            "ev": "memory",
            "label": label,
        }
        if step is not None:
            payload["step"] = int(step)
        if phase:
            payload["phase"] = phase
        payload.update(fields)
        payload.update(_cuda_memory_snapshot(device))
        self.write(payload)

    def component_summary(
        self,
        label: str,
        *,
        network=None,
        unet=None,
        optimizer=None,
        device=None,
        **fields: Any,
    ) -> None:
        payload: dict[str, Any] = {
            "ev": "component_summary",
            "label": label,
        }
        payload.update(fields)
        if network is not None:
            payload["network"] = _network_summary(network)
        if unet is not None:
            payload["unet"] = _module_summary(unet)
        if optimizer is not None:
            payload["optimizer"] = _optimizer_state_summary(optimizer)
        payload.update(_cuda_memory_snapshot(device))
        self.write(payload)


def _cuda_memory_snapshot(device=None) -> dict[str, Any]:
    try:
        torch_device = torch.device(device) if device is not None else torch.device("cuda")
    except Exception:
        return {}
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        return {}
    try:
        idx = torch_device.index
        if idx is None:
            idx = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
        allocated = torch.cuda.memory_allocated(idx)
        reserved = torch.cuda.memory_reserved(idx)
        max_allocated = torch.cuda.max_memory_allocated(idx)
        max_reserved = torch.cuda.max_memory_reserved(idx)
        device_name = torch.cuda.get_device_name(idx)
    except Exception:
        return {}
    gib = float(1024**3)
    return {
        "cuda_device": int(idx),
        "cuda_name": device_name,
        "cuda_total_gb": total_bytes / gib,
        "cuda_free_gb": free_bytes / gib,
        "cuda_allocated_gb": allocated / gib,
        "cuda_reserved_gb": reserved / gib,
        "cuda_max_allocated_gb": max_allocated / gib,
        "cuda_max_reserved_gb": max_reserved / gib,
    }


def _module_summary(module) -> dict[str, Any]:
    try:
        params = list(module.parameters(recurse=True))
    except Exception:
        params = []
    return {
        "type": type(module).__name__,
        **_summarize_parameters(params),
    }


def _network_summary(network) -> dict[str, Any]:
    text_loras = list(getattr(network, "text_encoder_loras", []) or [])
    unet_loras = list(getattr(network, "unet_loras", []) or [])
    text_refts = list(getattr(network, "text_encoder_refts", []) or [])
    unet_refts = list(getattr(network, "unet_refts", []) or [])
    return {
        "type": type(network).__name__,
        "all_parameters": _module_summary(network),
        "text_encoder_loras": _summarize_module_list(text_loras),
        "unet_loras": _summarize_module_list(unet_loras),
        "text_encoder_refts": _summarize_module_list(text_refts),
        "unet_refts": _summarize_module_list(unet_refts),
    }
