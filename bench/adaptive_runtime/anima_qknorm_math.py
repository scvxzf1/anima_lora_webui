"""Independent FP64 analytic RMSNorm VJP checks for saved Q/K/V cells."""

import numpy as np
from safetensors import safe_open
import torch

from bench.adaptive_runtime.anima_inputs import sha256_file
from library.anima.weights import _strip_net_prefix


SIDES = ("reference", "candidate")
NORM_RELATIVE_L2_MAX = 1e-4


def read_norm_weights(weights, expected_sha256, head_dim):
    if sha256_file(weights) != expected_sha256:
        raise ValueError("QK norm weights SHA mismatch")
    names = {"q": "blocks.17.self_attn.q_norm.weight",
             "k": "blocks.17.self_attn.k_norm.weight"}
    found = {}
    with safe_open(str(weights), framework="pt", device="cpu") as source:
        for key in source.keys():
            stripped = _strip_net_prefix(key)
            for part, wanted in names.items():
                if stripped == wanted:
                    if part in found:
                        raise ValueError("Duplicate QK norm weight")
                    found[part] = source.get_tensor(key).to(dtype=torch.bfloat16).float().numpy()
    if set(found) != set(names) or any(value.shape != (head_dim,) or not np.isfinite(value).all()
                                           for value in found.values()):
        raise ValueError("Missing or malformed block17 QK norm weights")
    return found


def rmsnorm_vjp(x, a, weight, eps):
    x = np.asarray(x, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if x.shape != a.shape or x.shape[-1] != weight.size or weight.ndim != 1 or eps != 1e-6:
        raise ValueError("Invalid RMSNorm analytic VJP contract")
    r = np.reciprocal(np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps))
    g = a * weight
    return r * g - x * (r ** 3) * np.mean(g * x, axis=-1, keepdims=True)


def check_norm_matrix(old, traces, matrix, weights, heads, head_dim, eps=1e-6):
    """Recompute every saved Q/K cell from independent source activations."""
    width = heads * head_dim
    if width <= 0 or any(np.asarray(weights[key]).shape != (head_dim,) for key in ("q", "k")):
        raise ValueError("Invalid QK norm analytic layout")
    checks = {}
    for i in SIDES:
        p = np.asarray(old[i]["p"]).reshape(-1, 3, heads, head_dim)
        for j in SIDES:
            a = np.asarray(traces[j]["a"]).reshape(-1, 3, heads, head_dim)
            saved = np.asarray(matrix[f"norm.{i}.{j}"])
            if saved.shape != np.asarray(old[i]["p"]).shape or p.shape != a.shape:
                raise ValueError("QK norm analytic tensor shape mismatch")
            saved_parts = saved.reshape(-1, 3, heads, head_dim)
            checks[f"norm.{i}.{j}"] = {}
            for index, name in enumerate(("q", "k")):
                predicted = rmsnorm_vjp(p[:, index], a[:, index], weights[name], eps)
                actual = saved_parts[:, index].astype(np.float64)
                delta = float(np.linalg.norm((actual - predicted).ravel()))
                baseline = float(np.linalg.norm(predicted.ravel()))
                relative = delta / baseline if baseline else (0.0 if delta == 0 else None)
                if relative is None or not np.isfinite(relative) or relative > NORM_RELATIVE_L2_MAX:
                    raise ValueError(f"QK norm analytic VJP mismatch: {i}/{j}/{name}: {relative}")
                checks[f"norm.{i}.{j}"][name] = relative
            if not np.array_equal(saved_parts[:, 2], a[:, 2]):
                raise ValueError(f"QK norm Identity V mismatch: {i}/{j}")
            checks[f"norm.{i}.{j}"]["v_bitwise_equal"] = True
    for j in SIDES:
        reference_v = np.asarray(matrix[f"norm.reference.{j}"]).reshape(-1, 3, heads, head_dim)[:, 2]
        candidate_v = np.asarray(matrix[f"norm.candidate.{j}"]).reshape(-1, 3, heads, head_dim)[:, 2]
        if not np.array_equal(reference_v, candidate_v):
            raise ValueError(f"QK norm fixed-cotangent V mismatch: {j}")
    return checks
