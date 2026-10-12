"""Metadata-only identity for native AMP; never copy model or optimizer tensors."""

import hashlib
import json

import torch

SCHEMA = "native_training_precision_v1"


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def base_metadata(model, names=None, *, family="anima"):
    from .native_family import fp32_buffers
    allowed_buffers = fp32_buffers(family)
    tensors = {**{f"parameter:{n}": p for n, p in model.named_parameters(remove_duplicate=False)},
               **{f"buffer:{n}": b for n, b in model.named_buffers(remove_duplicate=False)}}
    selected = tensors if names is None else {n: tensors.get(n) for n in names}
    entries = {}
    for name, tensor in selected.items():
        if tensor is None:
            raise ValueError(f"precision_contract_mismatch: native base tensor missing: {name}")
        if name.startswith("parameter:") and tensor.is_floating_point() and tensor.dtype != torch.float16:
            raise ValueError(f"precision_contract_mismatch: native base parameter must be FP16: {name}")
        if tensor.is_floating_point() and tensor.dtype != torch.float16:
            if name.removeprefix("buffer:") not in allowed_buffers or tensor.dtype != torch.float32:
                raise ValueError(f"precision_contract_mismatch: unsupported native base dtype: {name}")
        entries[name] = {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
                         "numel": tensor.numel()}
    return entries


def execution_metadata(model, network, accelerator, *, family="anima"):
    from .native_family import execution_guards
    from .training_precision import validate_plain_training_network

    validate_plain_training_network(network, label="Native FP16")
    guards = execution_guards(model, family)
    if getattr(accelerator, "native_amp", None) is not True:
        raise ValueError("precision_contract_mismatch: native AMP is disabled")
    if getattr(accelerator, "mixed_precision", None) != "fp16":
        raise ValueError("precision_contract_mismatch: native AMP must use FP16")
    autocast = getattr(accelerator, "autocast_handler", None)
    if autocast is not None and getattr(autocast, "enabled", True) is not True:
        raise ValueError("precision_contract_mismatch: native autocast is disabled")
    scaler = getattr(accelerator, "scaler", None)
    if scaler is None or not scaler.is_enabled():
        raise ValueError("precision_contract_mismatch: native GradScaler is disabled")
    # Scale and growth tracker are dynamic training state, not policy identity.
    # These getters read Python configuration, unlike state_dict(), which
    # fetches the dynamic scale/tracker from CUDA with Tensor.item().
    scaler_policy = {"growth_factor": scaler.get_growth_factor(),
                     "backoff_factor": scaler.get_backoff_factor(),
                     "growth_interval": scaler.get_growth_interval()}
    return {**guards, "native_amp": True,
            "scaler_policy": scaler_policy,
            "lora_fp32_compute": [bool(module.fp32_compute) for module in network.unet_loras],
            "trainables": {n: {"dtype": str(p.dtype), "shape": list(p.shape), "numel": p.numel()}
                           for n, p in network.named_parameters() if p.requires_grad}}
