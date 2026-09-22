"""Atomic complete state for the fixed-input, scheduler-free probe only."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import torch
from torch.utils._pytree import tree_flatten, tree_map
from bench.adaptive_runtime.scaler_state import scaling_enabled, validate_scaler_state


def save_checkpoint(path, *, network, optimizer, initial, updates, signature, device, scaler=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "adaptive_fixed_input_probe_v1", "signature": signature,
        "network": network.state_dict(), "optimizer": optimizer.state_dict(),
        "initial": initial, "updates": list(updates), "step": len(updates),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(device) if torch.device(device).type == "cuda" else None,
    }
    enabled = scaling_enabled(signature)
    if scaler is None and enabled:
        raise ValueError("Scaled checkpoint requires scaler state")
    if scaler is not None:
        if scaler.is_enabled() != enabled:
            raise ValueError("Checkpoint scaler does not match loss scale contract")
        state = scaler.state_dict()
        validate_scaler_state(state, enabled=enabled)
        payload.update(schema="adaptive_fixed_input_probe_v2", scaler=state)
    payload = tree_map(lambda t: t.detach().cpu() if isinstance(t, torch.Tensor) else t, payload)
    fd, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return str(path.resolve())


def read_checkpoint(path, *, signature):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"schema", "signature", "network", "optimizer", "initial", "updates", "step",
                "cpu_rng", "cuda_rng"}
    if isinstance(payload, dict) and payload.get("schema") == "adaptive_fixed_input_probe_v2":
        expected.add("scaler")
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("Incomplete probe checkpoint")
    if payload["schema"] not in {"adaptive_fixed_input_probe_v1", "adaptive_fixed_input_probe_v2"} or payload["signature"] != signature:
        raise ValueError("Checkpoint does not match the input/base/precision/training contract")
    enabled = scaling_enabled(signature)
    if payload["schema"] == "adaptive_fixed_input_probe_v1" and enabled:
        raise ValueError("Legacy checkpoint cannot resume scaled training without scaler state")
    if "scaler" in payload:
        validate_scaler_state(payload["scaler"], enabled=enabled)
    step, updates = payload["step"], payload["updates"]
    if (type(step) is not int or step < 1 or not isinstance(updates, list)
            or len(updates) != step or not all(isinstance(u, dict) for u in updates)
            or [u.get("step") for u in updates] != list(range(1, step + 1))):
        raise ValueError("Invalid checkpoint progress")
    if not isinstance(payload["network"], dict) or not payload["network"]:
        raise ValueError("Missing checkpoint network weights")
    if not all(isinstance(t, torch.Tensor) and torch.isfinite(t).all()
               for t in payload["network"].values()):
        raise ValueError("Invalid checkpoint network weights")
    initial = payload["initial"]
    if (not isinstance(initial, list) or not initial
            or not all(isinstance(t, torch.Tensor) and torch.isfinite(t).all() for t in initial)):
        raise ValueError("Invalid checkpoint initial parameters")
    optimizer = payload["optimizer"]
    if (not isinstance(optimizer, dict) or set(optimizer) != {"state", "param_groups"}
            or not isinstance(optimizer["state"], dict)
            or not isinstance(optimizer["param_groups"], list) or not optimizer["param_groups"]):
        raise ValueError("Invalid checkpoint optimizer state")
    leaves, _ = tree_flatten(optimizer)
    if any(isinstance(t, torch.Tensor) and not torch.isfinite(t).all() for t in leaves):
        raise ValueError("Nonfinite checkpoint optimizer state")
    return payload


def restore_checkpoint(payload, *, network, optimizer, params, device, scaler=None):
    enabled = scaling_enabled(payload["signature"])
    if enabled and (scaler is None or "scaler" not in payload):
        raise ValueError("Scaled restore requires checkpoint and destination scaler state")
    if scaler is not None:
        if scaler.is_enabled() != enabled:
            raise ValueError("Restore scaler does not match checkpoint contract")
        validate_scaler_state(payload.get("scaler", {}), enabled=enabled)
    initial = payload["initial"]
    if (not isinstance(initial, list) or len(initial) != len(params)
            or any(not isinstance(t, torch.Tensor) or t.shape != p.shape
                   for t, p in zip(initial, params, strict=True))):
        raise ValueError("Checkpoint initial parameter contract mismatch")
    network.load_state_dict(payload["network"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None:
        scaler.load_state_dict(payload.get("scaler", {}))
    torch.set_rng_state(payload["cpu_rng"])
    if torch.device(device).type == "cuda":
        if payload["cuda_rng"] is None:
            raise ValueError("Missing checkpoint CUDA random state")
        torch.cuda.set_rng_state(payload["cuda_rng"], device)
    return initial, list(payload["updates"])
