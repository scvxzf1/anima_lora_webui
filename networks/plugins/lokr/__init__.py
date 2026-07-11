"""Bundled LoKR plugin registration."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from .autograd import (
    DEFAULT_LOKR_GROUPED_DELTA_BACKEND,
    DEFAULT_LOKR_GROUPED_DELTA_BACKWARD_BACKEND,
    DEFAULT_LOKR_PROJECT_CHUNK_BYTES,
    normalize_lokr_grouped_delta_backward_backend,
    normalize_lokr_grouped_delta_backend,
)
from .module import LoKrModule
from .save import save_lokr_weights
from ...registry_api import (
    ContinueWeightDetectionContext,
    ModuleCreationContext,
    NetworkSpec,
    WeightDetectionContext,
    register_network_spec,
    register_save_handler,
)

DEFAULT_LOKR_RUNTIME_BACKEND = DEFAULT_LOKR_GROUPED_DELTA_BACKEND


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _bool_arg(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    return _truthy(value)


def _selector(kwargs: Mapping[str, Any]) -> bool:
    return _truthy(kwargs.get("use_lokr"))


def _validate(kwargs: Mapping[str, Any]) -> None:
    if not _selector(kwargs):
        return
    if (
        _truthy(kwargs.get("use_ortho"))
        or _truthy(kwargs.get("use_chimera_hydra"))
        or _truthy(kwargs.get("use_vera"))
        or _truthy(kwargs.get("dora_wd"))
        or kwargs.get("use_moe_style") not in (None, False, "", "false", "False")
    ):
        raise ValueError(
            "use_lokr is mutually exclusive with dora_wd, use_ortho, use_vera, "
            "use_moe_style, and use_chimera_hydra."
        )


def _module_kwargs(ctx: ModuleCreationContext) -> dict[str, Any]:
    return {
        "factor": int(ctx.cfg.plugin_args.get("lokr_factor", 8)),
        "lokr_factor_group_size": int(
            ctx.cfg.plugin_args.get("lokr_factor_group_size", 8)
        ),
        "lokr_project_chunk_bytes": int(
            ctx.cfg.plugin_args.get(
                "lokr_project_chunk_bytes", DEFAULT_LOKR_PROJECT_CHUNK_BYTES
            )
        ),
        "lokr_grouped_delta_backend": normalize_lokr_grouped_delta_backend(
            ctx.cfg.plugin_args.get(
                "lokr_grouped_delta_backend", DEFAULT_LOKR_RUNTIME_BACKEND
            )
        ),
        "lokr_grouped_delta_backward_backend": normalize_lokr_grouped_delta_backward_backend(
            ctx.cfg.plugin_args.get(
                "lokr_grouped_delta_backward_backend",
                DEFAULT_LOKR_GROUPED_DELTA_BACKWARD_BACKEND,
            )
        ),
        "lokr_use_einsum": _bool_arg(
            ctx.cfg.plugin_args.get("lokr_use_einsum"), default=True
        ),
        "lokr_decompose_w2": (
            False
            if ctx.cfg.plugin_args.get("lokr_decompose_w2") is None
            else _truthy(ctx.cfg.plugin_args.get("lokr_decompose_w2"))
        ),
    }


def _detect_from_weights(ctx: WeightDetectionContext) -> bool:
    key = ctx.key
    value = ctx.value
    try:
        network_dim = int(float(ctx.metadata.get("ss_network_dim", "")))
    except (TypeError, ValueError):
        network_dim = None
    ctx.state["lokr_network_dim_meta"] = network_dim
    if key.endswith(".lokr_w1"):
        ctx.state["has_lokr"] = True
        ctx.state.setdefault("lokr_module_names", set()).add(ctx.lora_name)
        if value.dim() == 2:
            ctx.state.setdefault("lokr_factors", set()).add(int(value.size(0)))
        ctx.modules_dim[ctx.lora_name] = network_dim or (
            value.size(0) if value.dim() >= 1 else 1
        )
        ctx.modules_alpha.setdefault(
            ctx.lora_name, torch.tensor(float(ctx.modules_dim[ctx.lora_name]))
        )
        ctx.plain_module_names.add(ctx.lora_name)
        return True
    if key.endswith(".lokr_w2"):
        ctx.state["has_lokr"] = True
        ctx.state["lokr_has_full_w2"] = True
        ctx.state.setdefault("lokr_module_names", set()).add(ctx.lora_name)
        ctx.modules_dim.setdefault(
            ctx.lora_name, ctx.state.get("lokr_network_dim_meta") or 1
        )
        ctx.modules_alpha.setdefault(
            ctx.lora_name, torch.tensor(float(ctx.modules_dim[ctx.lora_name]))
        )
        ctx.plain_module_names.add(ctx.lora_name)
        return True
    if key.endswith(".lokr_w2_a") or key.endswith(".lokr_w2_b"):
        ctx.state["has_lokr"] = True
        ctx.state["lokr_has_decomposed_w2"] = True
        ctx.state.setdefault("lokr_module_names", set()).add(ctx.lora_name)
        ctx.modules_dim.setdefault(
            ctx.lora_name, ctx.state.get("lokr_network_dim_meta") or 1
        )
        ctx.modules_alpha.setdefault(
            ctx.lora_name, torch.tensor(float(ctx.modules_dim[ctx.lora_name]))
        )
        ctx.plain_module_names.add(ctx.lora_name)
        return True
    return False


def _finish_weight_detection(
    state: dict[str, Any],
    modules_dim: dict[str, int],
    modules_alpha: dict[str, Any],
) -> dict[str, Any]:
    if not state.get("has_lokr"):
        return {}
    if state.get("lokr_network_dim_meta") is None:
        for lora_name in state.get("lokr_module_names", set()):
            alpha_value = modules_alpha.get(lora_name)
            if isinstance(alpha_value, torch.Tensor) and alpha_value.numel() == 1:
                modules_dim[lora_name] = max(
                    1,
                    int(float(alpha_value.detach().float().cpu().item())),
                )
    factor = next(iter(sorted(state.get("lokr_factors", set()))), 8)
    use_einsum = bool(state.get("lokr_has_decomposed_w2")) or not bool(
        state.get("lokr_has_full_w2")
    )
    plugin_args = {"lokr_factor": factor, "lokr_use_einsum": use_einsum}
    if state.get("lokr_has_decomposed_w2"):
        plugin_args["lokr_decompose_w2"] = True
    return {
        "detected_spec": "lokr",
        "plugin_args": plugin_args,
    }


def _continue_weight_kind(ctx: ContinueWeightDetectionContext) -> str | None:
    if any("lokr_w1" in key or "lokr_w2" in key for key in ctx.lowered_keys):
        return "LoKr"
    if str(ctx.metadata.get("ss_network_spec") or "").strip().lower() == "lokr":
        return "LoKr"
    return None


register_save_handler("lokr", save_lokr_weights)
register_network_spec(
    NetworkSpec(
        name="lokr",
        module_class=LoKrModule,
        save_variant="lokr",
        kwarg_flags=(
            "use_lokr",
            "lokr_factor",
            "lokr_factor_group_size",
            "lokr_project_chunk_bytes",
            "lokr_grouped_delta_backend",
            "lokr_grouped_delta_backward_backend",
            "lokr_use_einsum",
            "lokr_decompose_w2",
        ),
        selector=_selector,
        validate=_validate,
        module_kwargs=_module_kwargs,
        detect_from_weights=_detect_from_weights,
        finish_weight_detection=_finish_weight_detection,
        continue_weight_kind=_continue_weight_kind,
    )
)

__all__ = ["LoKrModule"]
