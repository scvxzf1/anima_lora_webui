"""Manual distributed synchronization for monkey-patched adapter networks.

Adapter modules are owned by a network container, but their forwards run from
the frozen DiT/text-encoder modules they patch. Wrapping that container in DDP
does not give DDP a matching root forward, so multi-process training keeps the
container unwrapped and synchronizes its optimizer gradients explicitly.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
from accelerate.utils import broadcast


@dataclass(frozen=True)
class GradientSyncResult:
    parameter_count: int = 0
    reduced_parameter_count: int = 0
    materialized_gradient_count: int = 0
    bucket_count: int = 0


@dataclass(frozen=True)
class StateSyncResult:
    parameter_count: int = 0
    buffer_count: int = 0
    bucket_count: int = 0


def optimizer_parameters(optimizer: Any) -> list[torch.nn.Parameter]:
    """Return unique parameters in deterministic optimizer-group order."""
    out: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for param in group.get("params", ()):
            if not isinstance(param, torch.nn.Parameter):
                raise TypeError(
                    "optimizer param groups must contain torch.nn.Parameter objects"
                )
            if id(param) in seen:
                continue
            seen.add(id(param))
            out.append(param)
    return out


def _tensor_buckets(
    tensors: Iterable[torch.Tensor],
) -> list[list[torch.Tensor]]:
    buckets: OrderedDict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = (
        OrderedDict()
    )
    for tensor in tensors:
        if tensor.layout != torch.strided:
            raise RuntimeError(
                "manual adapter synchronization only supports dense strided tensors"
            )
        if tensor.device.type == "meta":
            raise RuntimeError("cannot synchronize adapter tensors on the meta device")
        buckets.setdefault((tensor.device, tensor.dtype), []).append(tensor)
    return list(buckets.values())


@torch.no_grad()
def synchronize_optimizer_state(
    accelerator: Any,
    network: torch.nn.Module,
    optimizer: Any,
    *,
    broadcast_fn: Callable[..., torch.Tensor] = broadcast,
) -> StateSyncResult:
    """Broadcast optimizer parameters and network buffers from rank zero once."""
    if int(getattr(accelerator, "num_processes", 1)) <= 1:
        return StateSyncResult()

    parameters = optimizer_parameters(optimizer)
    seen = {id(param) for param in parameters}
    buffers = [buffer for buffer in network.buffers() if id(buffer) not in seen]
    tensors: list[torch.Tensor] = [*parameters, *buffers]
    buckets = _tensor_buckets(tensors)

    for bucket in buckets:
        nonempty = [tensor for tensor in bucket if tensor.numel()]
        if not nonempty:
            continue
        flat = torch.cat([tensor.detach().reshape(-1) for tensor in nonempty])
        synced = broadcast_fn(flat, from_process=0)
        offset = 0
        for tensor in nonempty:
            end = offset + tensor.numel()
            tensor.copy_(synced[offset:end].view_as(tensor))
            offset = end

    return StateSyncResult(
        parameter_count=len(parameters),
        buffer_count=len(buffers),
        bucket_count=len(buckets),
    )


def prepare_network_for_manual_gradient_sync(
    accelerator: Any,
    network: torch.nn.Module,
    optimizer: Any,
) -> torch.nn.Module:
    """Move/register an adapter network without applying a DDP wrapper."""
    network = accelerator.prepare_model(network, evaluation_mode=True)
    state = synchronize_optimizer_state(accelerator, network, optimizer)
    accelerator.print(
        "manual adapter gradient sync enabled: "
        f"world_size={accelerator.num_processes}, "
        f"parameters={state.parameter_count}, buffers={state.buffer_count}, "
        f"state_buckets={state.bucket_count}"
    )
    return network


@torch.no_grad()
def synchronize_optimizer_gradients(
    accelerator: Any,
    optimizer: Any,
) -> GradientSyncResult:
    """Mean-reduce optimizer gradients while preserving rank-local ``None``.

    Every rank contributes an identically-shaped payload. A missing local
    gradient contributes zeros; if every rank is missing a gradient it remains
    ``None`` so weight decay and optimizer state do not advance for an unused
    parameter. Gradients intentionally remain loss-scaled here: a non-finite
    value then reaches every rank before GradScaler unscales/checks it.
    """
    if int(getattr(accelerator, "num_processes", 1)) <= 1:
        return GradientSyncResult()

    parameters = optimizer_parameters(optimizer)
    buckets = _tensor_buckets(parameters)
    reduced_parameter_count = 0
    materialized_gradient_count = 0

    for bucket in buckets:
        grad_parts: list[torch.Tensor] = []
        presence: list[float] = []
        total_numel = 0
        for param in bucket:
            grad = param.grad
            total_numel += param.numel()
            if grad is None:
                grad_parts.append(param.detach().new_zeros(param.numel()))
                presence.append(0.0)
                continue
            if grad.layout != torch.strided:
                raise RuntimeError(
                    "manual adapter gradient sync does not support sparse gradients"
                )
            if grad.device != param.device or grad.dtype != param.dtype:
                raise RuntimeError(
                    "adapter gradient device/dtype must match its optimizer parameter"
                )
            grad_parts.append(grad.detach().reshape(-1))
            presence.append(1.0)

        if not bucket:
            continue
        presence_tensor = bucket[0].new_tensor(presence)
        payload = torch.cat([*grad_parts, presence_tensor])
        reduced = accelerator.reduce(payload, reduction="mean")
        global_presence = reduced[total_numel:]
        present_on_any_rank = global_presence.ne(0).to(device="cpu").tolist()

        offset = 0
        for index, param in enumerate(bucket):
            end = offset + param.numel()
            if present_on_any_rank[index]:
                synced_grad = reduced[offset:end].view_as(param)
                if param.grad is None:
                    param.grad = synced_grad.clone()
                    materialized_gradient_count += 1
                else:
                    param.grad.copy_(synced_grad)
                reduced_parameter_count += 1
            offset = end

    return GradientSyncResult(
        parameter_count=len(parameters),
        reduced_parameter_count=reduced_parameter_count,
        materialized_gradient_count=materialized_gradient_count,
        bucket_count=len(buckets),
    )
