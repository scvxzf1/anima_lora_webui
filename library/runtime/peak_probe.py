"""Fine-grained CUDA peak probe for tight-VRAM DiT / LoKr experiments."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_DISABLED_VALUES = {"", "off", "none", "false", "0"}
_BLOCK_RE = re.compile(r"(?:^|[._])blocks[._](\d+)(?:[._]|$)")


def _jsonable(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    if isinstance(value, torch.Size):
        return list(value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, torch.device):
        return str(value)
    return str(value)


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


def _infer_location(fields: dict[str, Any]) -> None:
    name = str(fields.get("original_name") or fields.get("lora_name") or "")
    if "block_idx" not in fields:
        match = _BLOCK_RE.search(name)
        if match:
            fields["block_idx"] = int(match.group(1))
    if "op_name" in fields and "block_phase" in fields:
        return

    normalized = name.replace("_", ".")
    if ".self.attn." in normalized or ".self_attn." in normalized:
        fields.setdefault("block_phase", "self_attn")
    elif ".cross.attn." in normalized or ".cross_attn." in normalized:
        fields.setdefault("block_phase", "cross_attn")
    elif ".mlp." in normalized:
        fields.setdefault("block_phase", "mlp")
    elif ".final.layer." in normalized or ".final_layer." in normalized:
        fields.setdefault("block_phase", "final_projection")
    elif ".x.embedder." in normalized or ".x_embedder." in normalized:
        fields.setdefault("block_phase", "patch_embed")

    if "op_name" not in fields:
        for marker in (
            "mlp.layer1",
            "mlp.layer2",
            "self.attn.qkv.proj",
            "self_attn.qkv_proj",
            "self.attn.output.proj",
            "self_attn.output_proj",
            "cross.attn.q.proj",
            "cross_attn.q_proj",
            "cross.attn.kv.proj",
            "cross_attn.kv_proj",
            "cross.attn.output.proj",
            "cross_attn.output_proj",
            "final.layer.linear",
            "final_layer.linear",
            "x.embedder.proj.1",
            "x_embedder.proj.1",
        ):
            if marker in normalized:
                fields["op_name"] = marker.replace(".", "_")
                break


class PeakProbe:
    """Append-only fine-grained peak memory stream.

    The probe is intentionally best-effort and opt-in. It is meant for short
    diagnostic runs where the extra Python graph-break points are acceptable.
    """

    def __init__(
        self,
        path: str,
        *,
        max_steps: int = 2,
        level: str = "block",
        t0: Optional[float] = None,
    ):
        self.path = str(path)
        self.max_steps = max(0, int(max_steps or 0))
        self.level = str(level or "block").strip().lower()
        if self.level not in {"block", "ops", "lokr", "full"}:
            self.level = "block"
        self.record_block_boundaries = self.level in {"block", "ops", "lokr", "full"}
        self.record_block_ops = self.level in {"ops", "full"}
        self.record_lokr = self.level in {"lokr", "full"}
        self._t0 = t0 if t0 is not None else time.time()
        self._seq = 0
        self._active_step: Optional[int] = None
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)

    @staticmethod
    def resolve_path(args) -> Optional[str]:
        explicit = getattr(args, "peak_probe_jsonl", None)
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
        return os.path.join(logs_dir, f"{output_name}.peak_probe.jsonl")

    @classmethod
    def from_args(
        cls,
        args,
        *,
        is_main_process: bool,
        t0: Optional[float] = None,
    ) -> Optional["PeakProbe"]:
        if not is_main_process:
            return None
        path = cls.resolve_path(args)
        if path is None:
            return None
        return cls(
            path,
            max_steps=int(getattr(args, "peak_probe_max_steps", 2) or 0),
            level=str(getattr(args, "peak_probe_level", "block") or "block"),
            t0=t0,
        )

    def should_record_step(self, step: int) -> bool:
        return self.max_steps <= 0 or int(step) < self.max_steps

    @property
    def active(self) -> bool:
        return self._active_step is not None

    def begin_step(self, step: int, *, device=None) -> None:
        if not self.should_record_step(step):
            self._active_step = None
            return
        self._active_step = int(step)
        self.record("step_begin", device=device, phase="train")

    def end_step(self, *, device=None) -> None:
        if not self.active:
            return
        self.record("step_end", device=device, phase="train")
        self._active_step = None

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
            logger.debug("peak probe write failed: %s", exc)

    def record(
        self,
        label: str,
        *,
        device=None,
        tensor: Optional[torch.Tensor] = None,
        **fields: Any,
    ) -> None:
        if not self.active:
            return
        if tensor is not None:
            fields.setdefault("tensor_shape", tuple(tensor.shape))
            fields.setdefault("tensor_dtype", tensor.dtype)
            fields.setdefault("tensor_device", tensor.device)
            device = tensor.device
        _infer_location(fields)
        payload: dict[str, Any] = {
            "ev": "peak_probe",
            "label": label,
            "step": self._active_step,
            **fields,
        }
        payload.update(_cuda_memory_snapshot(device))
        self.write(payload)


@torch.compiler.disable(recursive=True)
def record_peak_probe_event(
    probe: Optional[PeakProbe],
    label: str,
    *,
    tensor: Optional[torch.Tensor] = None,
    **fields: Any,
) -> None:
    if probe is None:
        return
    probe.record(label, tensor=tensor, **fields)
