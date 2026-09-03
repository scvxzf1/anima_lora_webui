"""Autograd-aware tensor-parallel collectives with optional INT8 transport."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from time import perf_counter

import torch
import torch.distributed as dist


@dataclass
class CommunicationStats:
    payload_bytes: int = 0
    wire_bytes: int = 0
    collective_calls: int = 0
    quantize_seconds: float = 0.0
    collective_seconds: float = 0.0
    dequantize_seconds: float = 0.0
    _events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = field(
        default_factory=list, repr=False
    )

    def record_cuda(self, category: str, start: torch.cuda.Event, end: torch.cuda.Event) -> None:
        self._events.append((category, start, end))

    def finalize(self, device: torch.device) -> None:
        if device.type != "cuda" or not self._events:
            return
        torch.cuda.synchronize(device)
        for category, start, end in self._events:
            seconds = start.elapsed_time(end) / 1000.0
            if category == "quantize":
                self.quantize_seconds += seconds
            elif category == "collective":
                self.collective_seconds += seconds
            elif category == "dequantize":
                self.dequantize_seconds += seconds
        self._events.clear()

    def as_dict(self) -> dict[str, float | int]:
        return {
            "payload_bytes_per_rank": self.payload_bytes,
            "wire_bytes_per_rank": self.wire_bytes,
            "collective_calls": self.collective_calls,
            "quantize_seconds": self.quantize_seconds,
            "collective_seconds": self.collective_seconds,
            "dequantize_seconds": self.dequantize_seconds,
            "communication_seconds": (
                self.quantize_seconds
                + self.collective_seconds
                + self.dequantize_seconds
            ),
        }


_MODE = "bf16"
_STATS: CommunicationStats | None = None


def configure_collectives(mode: str, stats: CommunicationStats | None) -> None:
    global _MODE, _STATS
    if mode not in {"bf16", "int8"}:
        raise ValueError(f"unsupported collective mode: {mode}")
    _MODE = mode
    _STATS = stats


def _event_pair(device: torch.device) -> tuple[torch.cuda.Event, torch.cuda.Event] | None:
    if device.type != "cuda":
        return None
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _recorded(category: str, device: torch.device, fn):
    events = _event_pair(device)
    if events is None:
        start = perf_counter()
        result = fn()
        elapsed = perf_counter() - start
        if _STATS is not None:
            attr = f"{category}_seconds"
            setattr(_STATS, attr, getattr(_STATS, attr) + elapsed)
        return result
    start, end = events
    start.record()
    result = fn()
    end.record()
    if _STATS is not None:
        _STATS.record_cuda(category, start, end)
    return result


def _bf16_sum(value: torch.Tensor) -> torch.Tensor:
    result = value.contiguous().clone()
    if _STATS is not None:
        size = result.numel() * result.element_size()
        _STATS.payload_bytes += size
        _STATS.wire_bytes += size * (dist.get_world_size() - 1)
        _STATS.collective_calls += 1
    return _recorded("collective", result.device, lambda: _all_reduce(result))


def _all_reduce(value: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def _int8_sum(value: torch.Tensor) -> torch.Tensor:
    world = dist.get_world_size()
    source_dtype = value.dtype
    group_size = math.gcd(128, value.shape[-1])

    def quantize():
        grouped = value.detach().float().reshape(*value.shape[:-1], -1, group_size)
        scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min_(1e-12) / 127.0
        quantized = torch.round(grouped / scale).clamp_(-127, 127).to(torch.int8)
        return quantized.contiguous(), scale.to(torch.bfloat16).contiguous()

    quantized, scale = _recorded("quantize", value.device, quantize)
    q_bytes = quantized.view(torch.uint8).flatten()
    scale_bytes = scale.view(torch.uint8).flatten()
    packed = torch.cat((q_bytes, scale_bytes))
    gathered = [torch.empty_like(packed) for _ in range(world)]

    def exchange():
        dist.all_gather(gathered, packed)

    _recorded("collective", value.device, exchange)

    def dequantize():
        result = torch.zeros_like(value, dtype=torch.float32)
        q_numel = quantized.numel()
        for item in gathered:
            q_value = item[:q_numel].view(torch.int8).reshape(quantized.shape)
            q_scale = item[q_numel:].view(torch.bfloat16).reshape(scale.shape)
            result.add_((q_value.float() * q_scale.float()).reshape(value.shape))
        return result.to(dtype=source_dtype)

    result = _recorded("dequantize", value.device, dequantize)
    if _STATS is not None:
        payload = value.numel() * value.element_size()
        wire = packed.numel() * (world - 1)
        _STATS.payload_bytes += payload
        _STATS.wire_bytes += wire
        _STATS.collective_calls += 1
    return result


def reduce_from_tensor_parallel(value: torch.Tensor) -> torch.Tensor:
    return _int8_sum(value) if _MODE == "int8" else _bf16_sum(value)


def reduce_exact(value: torch.Tensor) -> torch.Tensor:
    """Sum a value in its native dtype, independent of transport mode."""
    return _bf16_sum(value)


class _ReduceFromTensorParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        return reduce_from_tensor_parallel(value)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


class _CopyToTensorParallel(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor) -> torch.Tensor:
        return value

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return reduce_from_tensor_parallel(grad_output)


def reduce_forward(value: torch.Tensor) -> torch.Tensor:
    return _ReduceFromTensorParallel.apply(value)


def copy_forward_reduce_backward(value: torch.Tensor) -> torch.Tensor:
    return _CopyToTensorParallel.apply(value)
