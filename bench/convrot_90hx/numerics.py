"""Complete named-vector comparisons; undefined zero metrics stay null."""

from __future__ import annotations
import torch


def vector_metrics(
    reference: torch.Tensor, candidate: torch.Tensor, zero_atol: float = 0.0
) -> dict:
    if reference.shape != candidate.shape:
        raise ValueError("vector shape mismatch")
    a, b = (
        reference.detach().cpu().double().reshape(-1),
        candidate.detach().cpu().double().reshape(-1),
    )
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return {
            "finite": False,
            "relative_l2": None,
            "cosine": None,
            "max_abs": None,
            "reference_norm": None,
            "candidate_norm": None,
            "reference_zero": None,
            "candidate_zero": None,
        }
    an, bn = float(a.norm()), float(b.norm())
    az, bz = an <= zero_atol, bn <= zero_atol
    delta = float((a - b).norm())
    return {
        "finite": True,
        "relative_l2": None if az else delta / an,
        "cosine": None if az or bz else float(torch.dot(a, b) / (an * bn)),
        "max_abs": float((a - b).abs().max()) if a.numel() else 0.0,
        "absolute_l2": delta,
        "reference_norm": an,
        "candidate_norm": bn,
        "reference_zero": az,
        "candidate_zero": bz,
    }


def compare_named(reference: dict, candidate: dict, zero_atol: float = 0.0) -> dict:
    names = sorted(set(reference) | set(candidate))
    missing_reference, missing_candidate, layers = [], [], {}
    complete_a, complete_b = [], []
    for name in names:
        a, b = reference.get(name), candidate.get(name)
        if a is None:
            missing_reference.append(name)
        if b is None:
            missing_candidate.append(name)
        if a is None and b is None:
            continue
        if a is None:
            a = torch.zeros_like(b)
        if b is None:
            b = torch.zeros_like(a)
        layers[name] = vector_metrics(a, b, zero_atol)
        complete_a.append(a.detach().cpu().reshape(-1))
        complete_b.append(b.detach().cpu().reshape(-1))
    aggregate = vector_metrics(
        torch.cat(complete_a).double() if complete_a else torch.empty(0),
        torch.cat(complete_b).double() if complete_b else torch.empty(0),
        zero_atol,
    )
    return {
        "overall": aggregate,
        "layers": layers,
        "missing_reference": missing_reference,
        "missing_candidate": missing_candidate,
        "zero_atol": zero_atol,
        "acceptance": "metrics only; no automatic quality gate",
    }


def tensor_hash(tensor: torch.Tensor) -> str:
    import hashlib

    work = tensor.detach().cpu().contiguous()
    return hashlib.sha256(
        str(work.dtype).encode()
        + str(tuple(work.shape)).encode()
        + work.view(torch.uint8).numpy().tobytes()
    ).hexdigest()
