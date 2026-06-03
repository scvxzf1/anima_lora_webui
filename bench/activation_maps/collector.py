"""Lightweight activation probes for Anima LoRA / LoKr analysis.

The collector records scalar summaries instead of full tensors, so it can be
left on during short inference probes without dumping gigabytes of activations.
It is intentionally model-agnostic where possible; Anima-specific naming is
kept at the hook registration boundary.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import torch

from networks.lora_modules.lora import LoRAModule
from networks.plugins.lokr.module import LoKrModule


EPS = 1e-8


@dataclass
class TensorSummary:
    """Compact tensor statistics safe to serialize as JSON."""

    shape: list[int]
    dtype: str
    device: str
    mean_abs: float
    rms: float
    max_abs: float
    token_energy: list[float] | None = None
    channel_energy: list[float] | None = None


@dataclass
class ActivationEvent:
    """One recorded activation summary."""

    kind: str
    name: str
    summary: TensorSummary
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class AdapterDeltaEvent:
    """Adapter-specific event with base and delta norms."""

    adapter_type: str
    name: str
    base_rms: float
    delta_rms: float
    delta_to_base: float
    delta_summary: TensorSummary
    bottleneck_summary: TensorSummary | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _as_float_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float()


def summarize_tensor(
    tensor: torch.Tensor,
    *,
    token_axis: int | None = None,
    channel_axis: int | None = -1,
    topk: int = 8,
) -> TensorSummary:
    x = _as_float_tensor(tensor)
    flat = x.reshape(-1)
    if flat.numel() == 0:
        mean_abs = rms = max_abs = 0.0
    else:
        mean_abs = float(flat.abs().mean().item())
        rms = float(flat.pow(2).mean().sqrt().item())
        max_abs = float(flat.abs().max().item())

    token_energy = _top_axis_energy(x, token_axis, topk) if token_axis is not None else None
    channel_energy = (
        _top_axis_energy(x, channel_axis, topk) if channel_axis is not None else None
    )
    return TensorSummary(
        shape=list(tensor.shape),
        dtype=str(tensor.dtype).replace("torch.", ""),
        device=str(tensor.device),
        mean_abs=mean_abs,
        rms=rms,
        max_abs=max_abs,
        token_energy=token_energy,
        channel_energy=channel_energy,
    )


def _top_axis_energy(tensor: torch.Tensor, axis: int, topk: int) -> list[float]:
    if tensor.ndim == 0:
        return [float(tensor.abs().item())]
    axis = axis if axis >= 0 else tensor.ndim + axis
    if axis < 0 or axis >= tensor.ndim:
        return []
    dims = tuple(i for i in range(tensor.ndim) if i != axis)
    energy = tensor.pow(2).mean(dim=dims).sqrt().flatten()
    if energy.numel() == 0:
        return []
    k = min(int(topk), int(energy.numel()))
    values = torch.topk(energy, k=k).values
    return [float(v.item()) for v in values]


def block_component_name(module_name: str) -> tuple[int | None, str]:
    """Return ``(block_idx, component)`` for common Anima module names."""

    parts = module_name.split(".")
    if len(parts) >= 3 and parts[0] == "blocks":
        try:
            block_idx = int(parts[1])
        except ValueError:
            block_idx = None
        return block_idx, ".".join(parts[2:])
    return None, module_name


class ActivationCollector(AbstractContextManager["ActivationCollector"]):
    """Register and remove Anima activation hooks as a context manager."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        record_blocks: bool = True,
        record_text_adapter: bool = True,
        record_adapter_delta: bool = True,
        topk: int = 8,
        module_filter: Callable[[str], bool] | None = None,
    ) -> None:
        self.model = model
        self.record_blocks = record_blocks
        self.record_text_adapter = record_text_adapter
        self.record_adapter_delta = record_adapter_delta
        self.topk = int(topk)
        self.module_filter = module_filter
        self.events: list[ActivationEvent] = []
        self.adapter_events: list[AdapterDeltaEvent] = []
        self._handles: list[Any] = []

    def __enter__(self) -> "ActivationCollector":
        self.register()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def register(self) -> None:
        for name, module in self.model.named_modules():
            if self.module_filter is not None and not self.module_filter(name):
                continue
            if self.record_blocks and _is_anima_block(name, module):
                self._handles.append(
                    module.register_forward_hook(self._make_activation_hook("block", name))
                )
            elif self.record_blocks and _is_block_component(name):
                self._handles.append(
                    module.register_forward_hook(
                        self._make_activation_hook("block_component", name)
                    )
                )
            elif self.record_text_adapter and _is_llm_adapter_block(name):
                self._handles.append(
                    module.register_forward_hook(
                        self._make_activation_hook("text_adapter", name)
                    )
                )

        if self.record_adapter_delta:
            self._patch_adapter_forwards()

    def close(self) -> None:
        for handle in reversed(self._handles):
            if isinstance(handle, _ForwardRestore):
                handle.restore()
            else:
                handle.remove()
        self._handles.clear()

    def _make_activation_hook(self, kind: str, name: str):
        def hook(_module, _inputs, output):
            tensor = _first_tensor(output)
            if tensor is None:
                return
            block_idx, component = block_component_name(name)
            self.events.append(
                ActivationEvent(
                    kind=kind,
                    name=name,
                    summary=summarize_tensor(
                        tensor,
                        token_axis=_guess_token_axis(tensor),
                        channel_axis=-1,
                        topk=self.topk,
                    ),
                    meta={"block_idx": block_idx, "component": component},
                )
            )

        return hook

    def _patch_adapter_forwards(self) -> None:
        for module in _iter_adapter_modules(self.model):
            if isinstance(module, LoRAModule):
                target = _adapter_forward_target(module)
                self._handles.append(
                    _ForwardRestore.patch(target, module, self._lora_forward)
                )
            elif isinstance(module, LoKrModule):
                target = _adapter_forward_target(module)
                self._handles.append(
                    _ForwardRestore.patch(target, module, self._lokr_forward)
                )

    def _lora_forward(self, module: LoRAModule, original_forward, x):
        if not module.enabled or getattr(module, "_fused", False):
            return original_forward(x)

        org_forwarded = module.org_forward(x)
        if module.training and module._skip_module():
            return org_forwarded

        x_lora = module._rebalance(x)
        if module.training:
            bottleneck = torch.nn.functional.linear(
                x_lora.float(), module.lora_down.weight.float()
            )
            bottleneck = bottleneck * module._timestep_mask
            if module.dropout is not None:
                bottleneck = torch.nn.functional.dropout(bottleneck, p=module.dropout)
            bottleneck, scale = module._apply_rank_dropout(bottleneck)
            delta = torch.nn.functional.linear(bottleneck, module.lora_up.weight.float())
            out = org_forwarded + (delta * module.multiplier * scale).to(
                org_forwarded.dtype
            )
        else:
            bottleneck = module.lora_down(x_lora)
            delta = module.lora_up(bottleneck)
            out = org_forwarded + delta * module.multiplier * module.scale

        effective_delta = out - org_forwarded
        self._record_adapter_delta(
            "lora",
            module.lora_name,
            org_forwarded,
            effective_delta,
            bottleneck=bottleneck,
            extra={"rank": int(module.lora_dim)},
        )
        return out

    def _lokr_forward(self, module: LoKrModule, original_forward, x):
        if not module.enabled or getattr(module, "_fused", False):
            return original_forward(x)

        org_forwarded = module.org_forward(x)
        if module._skip_module():
            return org_forwarded

        x_lokr = (
            torch.nn.functional.dropout(x, p=module.dropout)
            if module.training and module.dropout
            else x
        )
        if module.training and module.use_custom_lokr_autograd:
            from networks.plugins.lokr.autograd import lokr_project

            delta = lokr_project(
                x_lokr,
                module.lokr_w1,
                module.lokr_w2,
                module.factor,
                module.in_dim,
                module.out_dim,
            )
        else:
            weight = module._compute_weight()
            work_x = x_lokr.float() if module.training else x_lokr
            work_w = weight.float() if module.training else weight.to(x_lokr.dtype)
            delta = torch.nn.functional.linear(work_x, work_w)

        if module.training:
            delta = delta * module._timestep_mask[:, :1]
            delta = delta.to(org_forwarded.dtype)

        out = org_forwarded + delta * module.multiplier * module.scale
        effective_delta = out - org_forwarded
        self._record_adapter_delta(
            "lokr",
            module.lora_name,
            org_forwarded,
            effective_delta,
            bottleneck=None,
            extra={
                "factor": int(module.factor),
                "kron_shape": [int(v) for v in module._compute_weight().shape],
            },
        )
        return out

    def _record_adapter_delta(
        self,
        adapter_type: str,
        name: str,
        base: torch.Tensor,
        delta: torch.Tensor,
        *,
        bottleneck: torch.Tensor | None,
        extra: dict[str, Any],
    ) -> None:
        base_rms = float(_as_float_tensor(base).pow(2).mean().sqrt().item())
        delta_rms = float(_as_float_tensor(delta).pow(2).mean().sqrt().item())
        block_idx, component = _adapter_name_to_block_component(name)
        meta = {
            "block_idx": block_idx,
            "component": component,
            **extra,
        }
        self.adapter_events.append(
            AdapterDeltaEvent(
                adapter_type=adapter_type,
                name=name,
                base_rms=base_rms,
                delta_rms=delta_rms,
                delta_to_base=delta_rms / max(base_rms, EPS),
                delta_summary=summarize_tensor(
                    delta,
                    token_axis=_guess_token_axis(delta),
                    channel_axis=-1,
                    topk=self.topk,
                ),
                bottleneck_summary=(
                    summarize_tensor(
                        bottleneck,
                        token_axis=_guess_token_axis(bottleneck),
                        channel_axis=-1,
                        topk=self.topk,
                    )
                    if bottleneck is not None
                    else None
                ),
                meta=meta,
            )
        )

    def layer_heatmap(self) -> list[dict[str, Any]]:
        """Aggregate adapter delta ratios by block and component."""

        buckets: dict[tuple[int | None, str, str], list[float]] = defaultdict(list)
        for event in self.adapter_events:
            key = (
                event.meta.get("block_idx"),
                str(event.meta.get("component")),
                event.adapter_type,
            )
            buckets[key].append(float(event.delta_to_base))

        rows = []
        for (block_idx, component, adapter_type), values in sorted(
            buckets.items(), key=lambda item: (-1 if item[0][0] is None else item[0][0], item[0][1], item[0][2])
        ):
            tensor = torch.tensor(values, dtype=torch.float32)
            rows.append(
                {
                    "block_idx": block_idx,
                    "component": component,
                    "adapter_type": adapter_type,
                    "count": len(values),
                    "delta_to_base_mean": float(tensor.mean().item()),
                    "delta_to_base_max": float(tensor.max().item()),
                }
            )
        return rows


@dataclass
class _ForwardRestore:
    module: torch.nn.Module
    original_forward: Callable[..., Any]

    @classmethod
    def patch(
        cls,
        target: torch.nn.Module,
        adapter: torch.nn.Module,
        replacement: Callable[..., Any],
    ):
        original_forward = target.forward

        def wrapped(*args, **kwargs):
            return replacement(adapter, original_forward, *args, **kwargs)

        target.forward = wrapped
        return cls(module=target, original_forward=original_forward)

    def restore(self) -> None:
        self.module.forward = self.original_forward


def _is_anima_block(name: str, module: torch.nn.Module) -> bool:
    return module.__class__.__name__ == "Block" and name.startswith("blocks.")


def _is_block_component(name: str) -> bool:
    return name.endswith(
        (
            "self_attn",
            "cross_attn",
            "mlp",
            "layer_norm_self_attn",
            "layer_norm_cross_attn",
            "layer_norm_mlp",
        )
    ) and name.startswith("blocks.")


def _is_llm_adapter_block(name: str) -> bool:
    return name.startswith("llm_adapter.blocks.") and name.count(".") == 2


def _guess_token_axis(tensor: torch.Tensor) -> int | None:
    if tensor.ndim >= 3:
        return 1
    return None


def _adapter_name_to_block_component(name: str) -> tuple[int | None, str]:
    prefix = "lora_unet_blocks_"
    if not name.startswith(prefix):
        return None, name
    rest = name[len(prefix) :]
    block_str, _, component = rest.partition("_")
    try:
        block_idx = int(block_str)
    except ValueError:
        block_idx = None
    component = component.replace("_", ".")
    return block_idx, component


def _iter_adapter_modules(model: torch.nn.Module) -> Iterable[torch.nn.Module]:
    seen: set[int] = set()

    def visit(root: Any):
        if root is None:
            return
        if isinstance(root, (list, tuple, set)):
            for item in root:
                yield from visit(item)
            return
        if not isinstance(root, torch.nn.Module):
            return
        for module in root.modules():
            ident = id(module)
            if ident in seen:
                continue
            seen.add(ident)
            if isinstance(module, (LoRAModule, LoKrModule)):
                yield module

    yield from visit(model)
    yield from visit(getattr(model, "_pgraft_network", None))
    yield from visit(getattr(model, "_hydra_network", None))
    yield from visit(getattr(model, "_hydra_networks", None))


def _adapter_forward_target(adapter: torch.nn.Module) -> torch.nn.Module:
    refs = getattr(adapter, "org_module_ref", None)
    if refs:
        return refs[0]
    return adapter


def events_to_jsonable(events: Iterable[ActivationEvent]) -> list[dict[str, Any]]:
    return [
        {
            "kind": event.kind,
            "name": event.name,
            "summary": event.summary.__dict__,
            "meta": event.meta,
        }
        for event in events
    ]


def adapter_events_to_jsonable(
    events: Iterable[AdapterDeltaEvent],
) -> list[dict[str, Any]]:
    return [
        {
            "adapter_type": event.adapter_type,
            "name": event.name,
            "base_rms": event.base_rms,
            "delta_rms": event.delta_rms,
            "delta_to_base": event.delta_to_base,
            "delta_summary": event.delta_summary.__dict__,
            "bottleneck_summary": (
                event.bottleneck_summary.__dict__
                if event.bottleneck_summary is not None
                else None
            ),
            "meta": event.meta,
        }
        for event in events
    ]
