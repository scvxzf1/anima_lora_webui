"""Strict RNG sidecar; validation never advances or replaces live RNG streams."""
import random
import numpy as np
import torch


def capture_rng():
    return {"schema": "recovery_rng_v1", "python": random.getstate(),
            "numpy": np.random.get_state(), "cpu": torch.get_rng_state().clone(),
            "cuda": [s.clone() for s in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_initialized() else [],
            "cuda_initialized": torch.cuda.is_initialized()}


def validate_rng(state):
    if not isinstance(state, dict) or state.get("schema") != "recovery_rng_v1":
        raise ValueError("Invalid RNG schema")
    try:
        random.Random().setstate(state["python"])
        np.random.RandomState().set_state(state["numpy"])
        torch.Generator().set_state(state["cpu"])
        initialized = state["cuda_initialized"]
        states = state["cuda"]
        if type(initialized) is not bool or not isinstance(states, list):
            raise ValueError("Invalid CUDA RNG fields")
        if initialized != torch.cuda.is_initialized():
            raise ValueError("CUDA RNG initialization differs")
        expected = torch.cuda.device_count() if initialized else 0
        if len(states) != expected:
            raise ValueError("CUDA RNG device count differs")
        for i, saved in enumerate(states):
            current = torch.cuda.get_rng_state(i)
            if saved.dtype != torch.uint8 or saved.shape != current.shape or saved.device.type != "cpu":
                raise ValueError("Invalid CUDA RNG tensor")
    except (KeyError, TypeError, RuntimeError, AttributeError) as exc:
        raise ValueError("Invalid RNG state") from exc
    return state


def restore_rng(state):
    validate_rng(state)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["cpu"])
    if state["cuda_initialized"]:
        torch.cuda.set_rng_state_all(state["cuda"])
