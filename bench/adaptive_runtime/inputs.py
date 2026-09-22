"""Validated immutable input fixtures shared by precision candidates."""

import hashlib

from safetensors import safe_open
from safetensors.torch import load_file
import torch


def load_z_image_inputs(path, *, resolution, prompt_dim, device, dtype):
    with safe_open(str(path), framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
    if metadata.get("schema") != "adaptive_z_image_inputs_v1":
        raise ValueError("Unknown real-input cache schema")
    cached = load_file(str(path))
    if set(cached) != {"latents", "noise", "prompt"}:
        raise ValueError("Real-input cache requires latents, noise and prompt")
    expected = (1, 16, 1, resolution // 8, resolution // 8)
    if cached["latents"].shape != expected or cached["noise"].shape != expected:
        raise ValueError("Input cache resolution/shape mismatch")
    prompt = cached["prompt"]
    if prompt.ndim != 2 or prompt.shape[1] != prompt_dim or not 1 <= len(prompt) <= 512:
        raise ValueError("Input cache text shape mismatch")
    if not all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in cached.values()):
        raise ValueError("Input cache must contain finite FP32 tensors")
    moved = {key: value.to(device=device, dtype=dtype) for key, value in cached.items()}
    return moved, hashlib.sha256(path.read_bytes()).hexdigest()


def load_krea_inputs(path, *, resolution, device, dtype):
    with safe_open(str(path), framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
    if metadata.get("schema") != "adaptive_krea_inputs_v1":
        raise ValueError("Unknown Krea real-input cache schema")
    cached = load_file(str(path))
    if set(cached) != {"latents", "noise", "hidden", "mask"}:
        raise ValueError("Krea cache requires latents, noise, hidden and mask")
    expected = (1, 16, 1, resolution // 8, resolution // 8)
    if cached["latents"].shape != expected or cached["noise"].shape != expected:
        raise ValueError("Krea input cache resolution/shape mismatch")
    hidden, mask = cached["hidden"], cached["mask"]
    if hidden.shape != (1, 512, 12, 2560) or mask.shape != (1, 512):
        raise ValueError("Krea input cache text shape mismatch")
    if mask.dtype != torch.bool or not mask.any():
        raise ValueError("Krea text mask must be boolean with valid tokens")
    floats = (cached[k] for k in ("latents", "noise", "hidden"))
    if not all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in floats):
        raise ValueError("Krea input cache must contain finite FP32 tensors")
    moved = {key: value.to(device=device, dtype=torch.bool if key == "mask" else dtype)
             for key, value in cached.items()}
    return moved, hashlib.sha256(path.read_bytes()).hexdigest()
