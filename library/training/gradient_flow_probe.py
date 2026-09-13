"""Opt-in block-level gradient and parameter-update telemetry."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)

_DISABLED_VALUES = {"", "none", "off", "false", "0"}
_BLOCK_RE = re.compile(
    r"(?:^|[._])(?P<kind>blocks|layers)[._](?P<index>\d+)(?:[._]|$)"
)
_EPS = 1e-12


def infer_block_group(parameter_name: str) -> tuple[str, Optional[int]]:
    """Map both original and underscore-normalized LoRA names to a block."""

    match = _BLOCK_RE.search(parameter_name)
    if match is None:
        return "other", None
    index = int(match.group("index"))
    return f"{match.group('kind')}.{index}", index


@dataclass
class _ParameterSnapshot:
    name: str
    group: str
    block_index: Optional[int]
    parameter: torch.nn.Parameter
    before: torch.Tensor


def _new_stats() -> dict[str, Any]:
    return {
        "parameter_tensor_count": 0,
        "trainable_parameter_count": 0,
        "no_grad_tensor_count": 0,
        "parameter_sq": 0.0,
        "grad_sq": 0.0,
        "abs_grad_parameter_dot": 0.0,
        "elementwise_abs_grad_parameter_sum": 0.0,
        "update_sq": 0.0,
        "interval_change_sq": None,
    }


def _accumulate_tensor_scalar(stats: dict[str, Any], key: str, value: torch.Tensor) -> None:
    scalar = float(value.detach().item())
    stats[key] = float(stats[key]) + scalar


def _materialize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    parameter_norm = math.sqrt(max(0.0, float(stats["parameter_sq"])))
    grad_norm = math.sqrt(max(0.0, float(stats["grad_sq"])))
    update_norm = math.sqrt(max(0.0, float(stats["update_sq"])))
    interval_sq = stats["interval_change_sq"]
    interval_norm = None if interval_sq is None else math.sqrt(max(0.0, float(interval_sq)))
    return {
        "parameter_tensor_count": int(stats["parameter_tensor_count"]),
        "trainable_parameter_count": int(stats["trainable_parameter_count"]),
        "no_grad_tensor_count": int(stats["no_grad_tensor_count"]),
        "parameter_norm": parameter_norm,
        "grad_norm": grad_norm,
        "relative_grad_norm": grad_norm / max(parameter_norm, _EPS),
        "abs_grad_parameter_dot": float(stats["abs_grad_parameter_dot"]),
        "elementwise_abs_grad_parameter_sum": float(
            stats["elementwise_abs_grad_parameter_sum"]
        ),
        "update_norm": update_norm,
        "relative_update": update_norm / max(parameter_norm, _EPS),
        "interval_change_norm": interval_norm,
        "relative_interval_change": (
            None if interval_norm is None else interval_norm / max(parameter_norm, _EPS)
        ),
    }


class GradientFlowProbe:
    """Append block-level LoRA gradient/update observations to JSONL.

    Sampling keeps one previous sampled parameter snapshot on-device. During a
    sampled optimizer step a second snapshot is briefly live, allowing both the
    exact one-step update and the change since the preceding sample to be
    measured without changing the training graph.
    """

    def __init__(
        self,
        path: str,
        *,
        every_n_steps: int = 10,
        dense_steps: int = 20,
        ema_beta: float = 0.9,
        t0: Optional[float] = None,
    ) -> None:
        self.path = str(path)
        self.every_n_steps = max(1, int(every_n_steps or 1))
        self.dense_steps = max(0, int(dense_steps or 0))
        self.ema_beta = min(1.0, max(0.0, float(ema_beta)))
        self._t0 = t0 if t0 is not None else time.time()
        self._seq = 0
        self._pending_step: Optional[int] = None
        self._pending: list[_ParameterSnapshot] = []
        self._previous: dict[str, torch.Tensor] = {}
        self._stats: dict[str, dict[str, Any]] = {}
        self._ema: dict[str, dict[str, float]] = {}
        self._disabled = False
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)

    @staticmethod
    def resolve_path(args) -> Optional[str]:
        explicit = getattr(args, "gradient_flow_probe_jsonl", None)
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
        return os.path.join(logs_dir, f"{output_name}.gradient_flow.jsonl")

    @classmethod
    def from_args(
        cls,
        args,
        *,
        is_main_process: bool,
        t0: Optional[float] = None,
    ) -> Optional["GradientFlowProbe"]:
        if not is_main_process:
            return None
        path = cls.resolve_path(args)
        if path is None:
            return None
        probe = cls(
            path,
            every_n_steps=getattr(args, "gradient_flow_probe_every_n_steps", 10),
            dense_steps=getattr(args, "gradient_flow_probe_dense_steps", 20),
            t0=t0,
        )
        probe.write(
            {
                "ev": "gradient_flow_probe_config",
                "every_n_steps": probe.every_n_steps,
                "dense_steps": probe.dense_steps,
                "ema_beta": probe.ema_beta,
            }
        )
        return probe

    def should_record_step(self, step: int, *, max_train_steps: Optional[int] = None) -> bool:
        step = int(step)
        return not self._disabled and (
            step <= self.dense_steps
            or step % self.every_n_steps == 0
            or (max_train_steps is not None and step >= int(max_train_steps))
        )

    def write(self, event: dict[str, Any]) -> None:
        try:
            self._seq += 1
            payload = {
                "seq": self._seq,
                "ts": round(time.time() - self._t0, 3),
                **event,
            }
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.warning("gradient-flow probe write failed: %s", exc)

    @torch.no_grad()
    def capture_before_optimizer(
        self,
        step: int,
        network: torch.nn.Module,
        *,
        grad_scale: float = 1.0,
    ) -> None:
        self.discard_pending()
        grouped: dict[str, dict[str, Any]] = {}
        pending: list[_ParameterSnapshot] = []
        grad_scale = float(grad_scale)
        if not math.isfinite(grad_scale) or grad_scale <= 0:
            grad_scale = 1.0

        for name, parameter in network.named_parameters():
            if not parameter.requires_grad:
                continue
            group, block_index = infer_block_group(name)
            stats = grouped.setdefault(group, _new_stats())
            stats["parameter_tensor_count"] += 1
            stats["trainable_parameter_count"] += int(parameter.numel())

            value = parameter.detach()
            _accumulate_tensor_scalar(stats, "parameter_sq", value.float().square().sum())
            grad = parameter.grad
            if grad is None:
                stats["no_grad_tensor_count"] += 1
            else:
                grad_value = grad.detach().float() / grad_scale
                parameter_value = value.float()
                _accumulate_tensor_scalar(stats, "grad_sq", grad_value.square().sum())
                _accumulate_tensor_scalar(
                    stats,
                    "abs_grad_parameter_dot",
                    (grad_value * parameter_value).sum().abs(),
                )
                _accumulate_tensor_scalar(
                    stats,
                    "elementwise_abs_grad_parameter_sum",
                    (grad_value * parameter_value).abs().sum(),
                )

            pending.append(
                _ParameterSnapshot(
                    name=name,
                    group=group,
                    block_index=block_index,
                    parameter=parameter,
                    before=value.clone(),
                )
            )

        self._pending_step = int(step)
        self._pending = pending
        self._stats = grouped

    @torch.no_grad()
    def capture_after_optimizer(self, step: int) -> None:
        if self._pending_step != int(step):
            return

        for snapshot in self._pending:
            stats = self._stats[snapshot.group]
            current = snapshot.parameter.detach().float()
            update = current - snapshot.before.float()
            _accumulate_tensor_scalar(stats, "update_sq", update.square().sum())
            previous = self._previous.get(snapshot.name)
            if previous is not None and previous.shape == current.shape:
                interval = current - previous.float()
                if stats["interval_change_sq"] is None:
                    stats["interval_change_sq"] = 0.0
                _accumulate_tensor_scalar(
                    stats, "interval_change_sq", interval.square().sum()
                )

        blocks = []
        materialized: dict[str, dict[str, Any]] = {}
        for group, stats in self._stats.items():
            row = _materialize_stats(stats)
            materialized[group] = row

        total_grad_sq = sum(row["grad_norm"] ** 2 for row in materialized.values())
        total_update_sq = sum(row["update_norm"] ** 2 for row in materialized.values())
        total_first_order = sum(
            row["elementwise_abs_grad_parameter_sum"]
            for row in materialized.values()
        )
        for group in sorted(materialized, key=self._group_sort_key):
            row = materialized[group]
            row.update(
                {
                    "group": group,
                    "block_index": infer_block_group(group)[1],
                    "grad_energy_share": (
                        row["grad_norm"] ** 2 / total_grad_sq if total_grad_sq > 0 else 0.0
                    ),
                    "update_energy_share": (
                        row["update_norm"] ** 2 / total_update_sq
                        if total_update_sq > 0
                        else 0.0
                    ),
                    "first_order_share": (
                        row["elementwise_abs_grad_parameter_sum"] / total_first_order
                        if total_first_order > 0
                        else 0.0
                    ),
                }
            )
            row.update(self._update_ema(group, row))
            blocks.append(row)

        self.write(
            {
                "ev": "gradient_flow",
                "step": int(step),
                "blocks": blocks,
            }
        )

        self._previous = {}
        for snapshot in self._pending:
            snapshot.before.copy_(snapshot.parameter.detach())
            self._previous[snapshot.name] = snapshot.before
        self.discard_pending(keep_previous=True)

    def discard_pending(self, *, keep_previous: bool = True) -> None:
        self._pending_step = None
        self._pending = []
        self._stats = {}
        if not keep_previous:
            self._previous = {}

    def fail(self, phase: str, step: int, exc: BaseException) -> None:
        """Disable telemetry after a capture failure without failing training."""

        self.discard_pending(keep_previous=False)
        self._disabled = True
        self.write(
            {
                "ev": "gradient_flow_probe_error",
                "phase": str(phase),
                "step": int(step),
                "exception_type": type(exc).__name__,
                "exception_message": str(exc)[:1000],
                "disabled": True,
            }
        )

    def _update_ema(self, group: str, row: dict[str, Any]) -> dict[str, float]:
        current = {
            "relative_grad_norm_ema": float(row["relative_grad_norm"]),
            "relative_update_ema": float(row["relative_update"]),
            "first_order_share_ema": float(row["first_order_share"]),
        }
        previous = self._ema.get(group)
        if previous is not None:
            for key in (
                "relative_grad_norm_ema",
                "relative_update_ema",
                "first_order_share_ema",
            ):
                current[key] = (
                    self.ema_beta * previous[key]
                    + (1.0 - self.ema_beta) * current[key]
                )
        self._ema[group] = current
        return current

    @staticmethod
    def _group_sort_key(group: str) -> tuple[int, int, str]:
        kind, index = infer_block_group(group)
        if index is None:
            return (2, 0, group)
        return (0 if kind.startswith("blocks.") else 1, index, group)
