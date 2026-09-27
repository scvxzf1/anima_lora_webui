"""Calibrate independent floating-point units against an FP32 reference.

This is a local numerical gate, not a claim about end-to-end training quality.
Use representative real activations and validate the assembled model separately.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math

import torch
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten


@dataclass(frozen=True)
class Tolerances:
    output_relative_l2: float = 0.01
    gradient_relative_l2: float = 0.02
    gradient_cosine: float = 0.999

    def __post_init__(self):
        values = asdict(self)
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("Numerical tolerances must be finite")
        if min(self.output_relative_l2, self.gradient_relative_l2) < 0:
            raise ValueError("Relative error tolerances cannot be negative")
        if not -1 <= self.gradient_cosine <= 1:
            raise ValueError("Cosine threshold must be in [-1, 1]")


def floating_tree(value, *, device, dtype):
    def convert(item):
        if not isinstance(item, torch.Tensor):
            return item
        return item.to(device=device, dtype=dtype if item.is_floating_point() else item.dtype)

    return tree_map(convert, value)


def _measure(reference, candidate):
    if len(reference) != len(candidate):
        raise ValueError("Precision candidate changed the output/gradient structure")
    errors, cosines = [], []
    for ref, actual in zip(reference, candidate, strict=True):
        if ref.shape != actual.shape:
            raise ValueError("Precision candidate changed tensor shapes")
        ref, actual = ref.double().flatten(), actual.double().flatten()
        if not torch.isfinite(actual).all():
            return {"finite": False, "relative_l2": None, "cosine": None}
        norm = torch.linalg.vector_norm(ref)
        other = torch.linalg.vector_norm(actual)
        delta = torch.linalg.vector_norm(actual - ref)
        errors.append(float(delta / norm.clamp_min(1e-30)))
        cosine = (ref @ actual) / (norm * other).clamp_min(1e-30)
        cosines.append(1.0 if norm == 0 and other == 0 else float(cosine))
    return {"finite": True, "relative_l2": max(errors, default=0.0),
            "cosine": min(cosines, default=1.0)}


def _prepare_inputs(args, kwargs, *, device, dtype, preserved):
    leaves, spec = tree_flatten((args, kwargs))
    if any(i < 0 or i >= len(leaves) for i in preserved):
        raise ValueError("Preserved input index outside flattened input tree")
    converted, targets, aliases, policies = [], [], {}, {}
    for index, value in enumerate(leaves):
        if not isinstance(value, torch.Tensor):
            converted.append(value)
            continue
        key = id(value)
        policy = index in preserved
        if key in policies and policies[key] != policy:
            raise ValueError("Aliased input has conflicting preserved precision policies")
        policies[key] = policy
        if key not in aliases:
            differentiable = value.is_floating_point() and index not in preserved
            target_dtype = dtype if differentiable else value.dtype
            tensor = value.detach().to(device=device, dtype=target_dtype).clone()
            if differentiable:
                tensor.requires_grad_(True)
                targets.append(tensor)
            aliases[key] = tensor
        converted.append(aliases[key])
    return tree_unflatten(converted, spec), targets


def _evaluate(module, args, kwargs, *, device, dtype, seed, preserved):
    cuda = torch.device(device).type == "cuda"
    index = torch.device(device).index
    devices = [torch.cuda.current_device() if index is None else index] if cuda else []
    with torch.random.fork_rng(devices=devices), torch.enable_grad():
        torch.random.default_generator.manual_seed(seed)
        if cuda:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(seed)
        unit = copy.deepcopy(module).to(device=device, dtype=dtype)
        (args, kwargs), targets = _prepare_inputs(
            args, kwargs, device=device, dtype=dtype, preserved=preserved,
        )
        targets += [p for p in unit.parameters() if p.requires_grad]
        cotangent_rng = torch.Generator(device="cpu").manual_seed(seed)
        with torch.autocast(torch.device(device).type, enabled=False):
            output = unit(*args, **kwargs)
        output_leaves, _ = tree_flatten(output)
        outputs = [x for x in output_leaves if isinstance(x, torch.Tensor) and x.is_floating_point()]
        if not outputs or not targets:
            raise ValueError("Calibration requires floating outputs and differentiable inputs/parameters")
        terms = []
        for value in outputs:
            if value.requires_grad:
                cotangent = torch.randn(value.shape, generator=cotangent_rng).to(device)
                terms.append((value.float() * cotangent).sum() / math.sqrt(value.numel()))
        if not terms:
            raise ValueError("Calibration outputs are disconnected from autograd")
        gradients = torch.autograd.grad(sum(terms), targets, allow_unused=True)
    return ([x.detach().float().cpu() for x in outputs],
            [torch.zeros_like(t, device="cpu", dtype=torch.float32) if g is None
             else g.detach().float().cpu() for t, g in zip(targets, gradients, strict=True)])


def calibrate_unit(module, cases, *, device, candidate="fp16", tolerances=None,
                   preserve_input_indices=()):
    """Cases are (positional_args, keyword_args); the caller's module is untouched.

    Preserved flattened input indices (e.g. RoPE frequencies) retain their original
    dtype and are excluded from input-gradient scoring. OOM is propagated to the
    memory planner, never misclassified as
    numerical sensitivity. Reference errors also fail closed. An FP32 candidate
    only checks reference finiteness; it does not test a low-precision alternative.
    """
    limits = tolerances or Tolerances()
    if candidate not in ("fp16", "bf16", "fp32"):
        raise ValueError("Candidate must be fp16, bf16 or fp32")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[candidate]
    evidence = []
    for index, (args, kwargs) in enumerate(cases):
        reference = _evaluate(module, args, kwargs, device=device, dtype=torch.float32,
                              seed=index, preserved=preserve_input_indices)
        if not all(torch.isfinite(t).all() for group in reference for t in group):
            raise ValueError("Nonfinite FP32 reference; calibration cannot establish safety")
        trial = reference if candidate == "fp32" else _evaluate(
            module, args, kwargs, device=device, dtype=dtype,
            seed=index, preserved=preserve_input_indices)
        output, gradient = [_measure(a, b) for a, b in zip(reference, trial, strict=True)]
        safe = (output["finite"] and gradient["finite"]
                and output["relative_l2"] <= limits.output_relative_l2
                and gradient["relative_l2"] <= limits.gradient_relative_l2
                and gradient["cosine"] >= limits.gradient_cosine)
        evidence.append({"case": index, "safe": safe, "output": output, "gradient": gradient})
    if not evidence:
        raise ValueError("At least one calibration case is required")
    return {"selected": candidate if all(e["safe"] for e in evidence) else "fp32",
            "candidate": candidate, "tolerances": asdict(limits), "cases": evidence,
            "low_precision_tested": candidate != "fp32",
            "validation_scope": ("fp32_reference_finiteness_only" if candidate == "fp32"
                                 else "local_precision_comparison"),
            "preserve_input_indices": list(preserve_input_indices)}


def preferred_candidate(capability: tuple[int, int]) -> str:
    """Default policy, not kernel compatibility or numerical certification."""
    if (not isinstance(capability, (tuple, list)) or len(capability) != 2
            or any(type(value) is not int for value in capability)
            or capability[0] < 1 or not 0 <= capability[1] <= 9):
        raise ValueError("Expected a valid CUDA compute capability")
    major, minor = capability
    if major >= 8:
        return "bf16"
    if major == 7 or (major, minor) == (6, 0):
        return "fp16"
    return "fp32"


def preferred_probe_precision(capability: tuple[int, int]) -> str:
    """Translate the shared hardware policy to existing DiT worker modes."""
    return {"bf16": "bf16", "fp16": "fp16-islands", "fp32": "fp32-reference"}[
        preferred_candidate(capability)]


def resolve_adaptive_precision(args, *, get_capability=None) -> str:
    """Resolve ``adaptive_precision=auto`` into a concrete execution mode.

    The requested value remains ``auto`` for metadata/UI purposes; the
    concrete mode is stored separately so a frozen OOM worker reproduces the
    exact parent decision.  Explicit modes are left untouched.
    """
    requested = str(getattr(args, "adaptive_precision", "off") or "off").strip().lower()
    if requested != "auto":
        return requested
    cached_mode = getattr(args, "adaptive_resolved_mode", None)
    cached_candidate = getattr(args, "adaptive_candidate", None)
    if cached_mode in {"bf16", "fp16_fp32", "fp32"} and cached_candidate in {
        "bf16", "fp16", "fp32"
    }:
        # The parent may call this helper again while entering the supervisor.
        # A resolved request is now a frozen decision; do not probe hardware a
        # second time and risk a different device/backend identity.
        mode, candidate = cached_mode, cached_candidate
    else:
        capability = get_capability
        if capability is None:
            if not torch.cuda.is_available():
                capability = None
            else:
                capability = torch.cuda.get_device_capability()
        elif callable(capability):
            try:
                capability = capability()
            except Exception:
                capability = None
        if capability is None:
            mixed = str(getattr(args, "mixed_precision", "bf16") or "bf16").strip().lower()
            mode = {"bf16": "bf16", "fp16": "fp16_fp32", "no": "fp32"}.get(mixed, "fp32")
            candidate = {"bf16": "bf16", "fp16_fp32": "fp16", "fp32": "fp32"}[mode]
        else:
            candidate = preferred_candidate(tuple(capability))
            mode = {"bf16": "bf16", "fp16": "fp16_fp32", "fp32": "fp32"}[candidate]
    args.adaptive_resolved_mode = mode
    args.adaptive_candidate = candidate
    args.mixed_precision = {"bf16": "bf16", "fp16_fp32": "fp16", "fp32": "no"}[mode]
    family = str(getattr(args, "model_family", "anima") or "anima").strip().lower()
    try:
        from library.models.family_registry import normalize_registered_family

        family = normalize_registered_family(
            family, source="adaptive precision model_family", allow_aliases=True
        )
    except (ImportError, ValueError):
        pass
    if mode == "fp16_fp32" and getattr(args, "block_swap_restore_mode", None) in {None, "slab"}:
        args.block_swap_restore_mode = "foreach"
    if mode in {"fp16_fp32", "fp32"} and family in {"krea2_raw", "z_image"}:
        # Their validated pre-Ampere paths use native SDPA; Flash is BF16-only
        # for Z-Image and the Krea mixed contract intentionally avoids it.
        args.attn_mode = "torch"
    elif mode == "fp32":
        # Native torch SDPA is the portable reference path on pre-SM70 GPUs.
        args.attn_mode = "torch"
    return mode
