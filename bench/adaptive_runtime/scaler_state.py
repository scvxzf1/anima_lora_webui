"""Validate AMP state at committed fixed-input probe boundaries."""

import math


def validate_scaler_state(state, *, enabled):
    if not isinstance(state, dict):
        raise ValueError("Invalid checkpoint scaler state")
    if not enabled:
        if state:
            raise ValueError("Disabled checkpoint scaler must have empty state")
        return
    if set(state) != {"scale", "growth_factor", "backoff_factor", "growth_interval", "_growth_tracker"}:
        raise ValueError("Incomplete checkpoint scaler state")
    for name in ("scale", "growth_factor", "backoff_factor"):
        value = state[name]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Nonfinite or invalid checkpoint scaler state")
    if (state["scale"] <= 0 or state["growth_factor"] <= 1
            or not 0 < state["backoff_factor"] < 1
            or type(state["growth_interval"]) is not int or state["growth_interval"] < 1
            or type(state["_growth_tracker"]) is not int
            or not 0 <= state["_growth_tracker"] < state["growth_interval"]):
        raise ValueError("Invalid checkpoint scaler policy or progress")


def scaling_enabled(signature):
    scale = signature.get("loss_scale", 1.0)
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale < 1:
        raise ValueError("Invalid checkpoint loss scale contract")
    return scale != 1
