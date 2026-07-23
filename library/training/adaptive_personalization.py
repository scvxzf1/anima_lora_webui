"""Observation-only controller for APT-style personalization diagnostics."""

from __future__ import annotations

import math
from typing import Any

import torch


def observer_enabled(args: Any) -> bool:
    return bool(getattr(args, "adaptive_personalization_observe", False))


def should_probe(state: dict[str, Any], args: Any) -> bool:
    """Advance the observation counter and decide whether to run a reference."""
    global_step = state.get("global_step")
    if global_step is not None:
        current = int(global_step)
        if state.get("last_seen_global_step") == current:
            return False
        state["last_seen_global_step"] = current
        state["calls"] = int(state.get("calls", 0)) + 1
        every = max(
            1,
            int(getattr(args, "adaptive_personalization_probe_every_n_steps", 4)),
        )
        return current % every == 0
    calls = int(state.get("calls", 0))
    state["calls"] = calls + 1
    every = max(
        1, int(getattr(args, "adaptive_personalization_probe_every_n_steps", 4))
    )
    return calls % every == 0


def _ensure_bins(state: dict[str, Any], args: Any) -> list[dict[str, Any]]:
    count = max(1, int(getattr(args, "adaptive_personalization_timestep_bins", 10)))
    bins = state.get("bins")
    if not isinstance(bins, list) or len(bins) != count:
        bins = [
            {"count": 0, "ema_delta": None, "ema_base": None, "ema_adapter": None}
            for _ in range(count)
        ]
        state["bins"] = bins
    return bins


def update_observation(
    state: dict[str, Any],
    args: Any,
    *,
    timesteps: torch.Tensor,
    adapter_pred: torch.Tensor,
    base_pred: torch.Tensor,
    target: torch.Tensor,
) -> None:
    """Update per-timestep-bin EMA deltas without changing training behavior."""
    bins = _ensure_bins(state, args)
    adapter = adapter_pred.float()
    base = base_pred.float()
    target_f = target.float()
    if adapter.ndim == 5:
        adapter = adapter.squeeze(2)
    if base.ndim == 5:
        base = base.squeeze(2)
    adapter_loss = (adapter - target_f).pow(2).flatten(1).mean(dim=1)
    base_loss = (base - target_f).pow(2).flatten(1).mean(dim=1)
    bin_count = len(bins)
    indices = torch.clamp(
        (timesteps.detach().float() * bin_count).long(), 0, bin_count - 1
    )
    decay = float(getattr(args, "adaptive_personalization_ema_decay", 0.95))
    decay = min(0.9999, max(0.0, decay))
    for index in range(bin_count):
        selected = indices == index
        if not bool(selected.any()):
            continue
        base_value = float(base_loss[selected].mean().item())
        adapter_value = float(adapter_loss[selected].mean().item())
        delta_value = base_value - adapter_value
        entry = bins[index]
        entry["count"] = int(entry.get("count", 0)) + int(selected.sum().item())
        for key, value in (
            ("ema_base", base_value),
            ("ema_adapter", adapter_value),
            ("ema_delta", delta_value),
        ):
            previous = entry.get(key)
            entry[key] = (
                value
                if previous is None
                else decay * float(previous) + (1.0 - decay) * value
            )
        scale = max(
            0.0, float(getattr(args, "adaptive_personalization_gamma_scale", 1.0))
        )
        entry["gamma"] = 1.0 - math.exp(-scale * max(0.0, float(entry["ema_delta"])))
    state["last_error"] = None


def record_error(state: dict[str, Any], error: BaseException) -> None:
    state["last_error"] = f"{type(error).__name__}: {error}"


def metrics(state: dict[str, Any]) -> dict[str, float]:
    """Flatten controller state into JSONL-safe scalar fields."""
    bins = state.get("bins")
    if not isinstance(bins, list) or not bins:
        return {}
    gammas = [float(item.get("gamma", 0.0) or 0.0) for item in bins]
    observed = [item for item in bins if int(item.get("count", 0) or 0) > 0]
    result: dict[str, float] = {
        "personalization/observer_calls": float(state.get("calls", 0)),
        "personalization/observed_bins": float(len(observed)),
        "personalization/gamma_mean": sum(gammas) / len(gammas),
        "personalization/gamma_max": max(gammas),
    }
    if "last_affine_fraction" in state:
        result["personalization/affine_fraction"] = float(state["last_affine_fraction"])
    for index, item in enumerate(bins):
        result[f"personalization/gamma_bin_{index:02d}"] = float(
            item.get("gamma", 0.0) or 0.0
        )
        result[f"personalization/count_bin_{index:02d}"] = float(
            item.get("count", 0) or 0
        )
        if "denoise_weight" in item:
            result[f"personalization/denoise_weight_bin_{index:02d}"] = float(
                item["denoise_weight"]
            )
    return result


def dynamic_denoise_weights(
    state: dict[str, Any], args: Any, timesteps: torch.Tensor
) -> torch.Tensor | None:
    """Return per-sample APT weights, or ``None`` when the policy is disabled."""
    if not bool(getattr(args, "adaptive_personalization_loss_weighting", False)):
        return None
    bins = _ensure_bins(state, args)
    minimum = min(
        1.0,
        max(
            0.0,
            float(getattr(args, "adaptive_personalization_denoise_weight_min", 0.25)),
        ),
    )
    min_samples = max(
        1, int(getattr(args, "adaptive_personalization_min_bin_samples", 16))
    )
    values = []
    for entry in bins:
        gamma = float(entry.get("gamma", 0.0) or 0.0)
        weight = (
            max(minimum, 1.0 - gamma)
            if int(entry.get("count", 0)) >= min_samples
            else 1.0
        )
        entry["denoise_weight"] = weight
        values.append(weight)
    table = timesteps.new_tensor(values, dtype=torch.float32)
    indices = torch.clamp(
        (timesteps.detach().float() * len(bins)).long(), 0, len(bins) - 1
    )
    return table[indices]
