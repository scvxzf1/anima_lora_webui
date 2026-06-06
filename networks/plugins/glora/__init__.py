"""Bundled GLoRA plugin registration."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from networks.attn_fuse import iter_split_groups
from networks.plugins.glora.module import GLoRAModule
from networks.plugins.glora.save import save_glora_weights
from networks.registry import (
    ContinueWeightDetectionContext,
    NetworkSpec,
    WeightDetectionContext,
    register_network_spec,
    register_save_handler,
)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _selector(kwargs: Mapping[str, Any]) -> bool:
    return _truthy(kwargs.get("use_glora"))


def _validate(kwargs: Mapping[str, Any]) -> None:
    if not _selector(kwargs):
        return
    if _truthy(kwargs.get("use_loha")):
        raise ValueError("use_glora is mutually exclusive with use_loha.")
    if _truthy(kwargs.get("use_lokr")):
        raise ValueError("use_glora is mutually exclusive with use_lokr.")
    if _truthy(kwargs.get("use_vera")):
        raise ValueError("use_glora is mutually exclusive with use_vera.")
    if _truthy(kwargs.get("dora_wd")):
        raise ValueError("use_glora is mutually exclusive with dora_wd.")
    if (
        _truthy(kwargs.get("use_ortho"))
        or _truthy(kwargs.get("use_chimera_hydra"))
        or kwargs.get("use_moe_style") not in (None, False, "", "false", "False")
    ):
        raise ValueError(
            "use_glora is mutually exclusive with use_ortho, "
            "use_moe_style, and use_chimera_hydra."
        )


def _same_tensor(values: list[torch.Tensor]) -> bool:
    return all(torch.equal(values[0], value) for value in values[1:])


def _same_alpha(values: list[Any | None]) -> bool:
    if all(value is None for value in values):
        return True
    if any(value is None for value in values):
        return False

    def _as_float(value: Any) -> float:
        if torch.is_tensor(value):
            return float(value.detach().float().cpu().reshape(-1)[0].item())
        return float(value)

    first = _as_float(values[0])
    return all(_as_float(value) == first for value in values[1:])


def _preprocess_weights(weights_sd: dict[str, Any]) -> dict[str, Any]:
    """Fuse split q/k/v GLoRA keys back to Anima's runtime qkv/kv modules.

    A split GLoRA can be represented as one fused GLoRA only when the input-side
    A path and B down projection are identical across q/k/v. This is true for
    checkpoints exported by this plugin. Arbitrary external split GLoRA files
    may not satisfy it, so reject those explicitly instead of silently loading
    wrong weights.
    """

    for shared_prefix, spec in iter_split_groups(weights_sd, ".a1.weight"):
        suffixes = spec.component_letters
        complete = True
        a1s: list[torch.Tensor] = []
        a2s: list[torch.Tensor] = []
        b1s: list[torch.Tensor] = []
        b2s: list[torch.Tensor] = []
        alphas: list[Any | None] = []
        for suffix in suffixes:
            prefix = f"{shared_prefix}{suffix}_proj"
            keys = (
                f"{prefix}.a1.weight",
                f"{prefix}.a2.weight",
                f"{prefix}.b1.weight",
                f"{prefix}.b2.weight",
            )
            if any(key not in weights_sd for key in keys):
                complete = False
                break
            a1s.append(weights_sd[keys[0]])
            a2s.append(weights_sd[keys[1]])
            b1s.append(weights_sd[keys[2]])
            b2s.append(weights_sd[keys[3]])
            alphas.append(weights_sd.get(f"{prefix}.alpha"))
        if not complete:
            continue

        if not (_same_tensor(a1s) and _same_tensor(a2s) and _same_tensor(b2s)):
            raise RuntimeError(
                "Split GLoRA checkpoint has q/k/v-specific A-path or b2 weights at "
                f"{shared_prefix}*. Anima's training runtime uses a fused "
                f"{spec.fused_letters}_proj module and can only continue-training "
                "GLoRA files whose input-side GLoRA factors are shared across the "
                "split projections."
            )
        if not _same_alpha(alphas):
            raise RuntimeError(
                f"Split GLoRA checkpoint has inconsistent alpha values at {shared_prefix}*."
            )

        fused_prefix = f"{shared_prefix}{spec.fused_letters}_proj"
        weights_sd[f"{fused_prefix}.a1.weight"] = a1s[0].contiguous()
        weights_sd[f"{fused_prefix}.a2.weight"] = a2s[0].contiguous()
        weights_sd[f"{fused_prefix}.b1.weight"] = torch.cat(b1s, dim=0).contiguous()
        weights_sd[f"{fused_prefix}.b2.weight"] = b2s[0].contiguous()
        if alphas[0] is not None:
            weights_sd[f"{fused_prefix}.alpha"] = alphas[0]

        for suffix in suffixes:
            prefix = f"{shared_prefix}{suffix}_proj"
            for leaf in ("a1.weight", "a2.weight", "b1.weight", "b2.weight", "alpha"):
                weights_sd.pop(f"{prefix}.{leaf}", None)

    return weights_sd


def _detect_from_weights(ctx: WeightDetectionContext) -> bool:
    key = ctx.key
    value = ctx.value
    if key.endswith(".a1.weight"):
        ctx.state["has_glora"] = True
        ctx.state.setdefault("glora_module_names", set()).add(ctx.lora_name)
        ctx.modules_dim[ctx.lora_name] = int(value.size(1))
        ctx.modules_alpha.setdefault(
            ctx.lora_name, torch.tensor(float(ctx.modules_dim[ctx.lora_name]))
        )
        ctx.plain_module_names.add(ctx.lora_name)
        return True
    if key.endswith(".a2.weight"):
        ctx.state["has_glora"] = True
        ctx.state.setdefault("glora_module_names", set()).add(ctx.lora_name)
        ctx.modules_dim[ctx.lora_name] = int(value.size(0))
        ctx.modules_alpha.setdefault(
            ctx.lora_name, torch.tensor(float(ctx.modules_dim[ctx.lora_name]))
        )
        ctx.plain_module_names.add(ctx.lora_name)
        return True
    if key.endswith(".b1.weight"):
        ctx.state["has_glora"] = True
        ctx.state.setdefault("glora_module_names", set()).add(ctx.lora_name)
        ctx.modules_dim[ctx.lora_name] = int(value.size(1))
        ctx.modules_alpha.setdefault(
            ctx.lora_name, torch.tensor(float(ctx.modules_dim[ctx.lora_name]))
        )
        ctx.plain_module_names.add(ctx.lora_name)
        return True
    if key.endswith(".b2.weight"):
        ctx.state["has_glora"] = True
        ctx.state.setdefault("glora_module_names", set()).add(ctx.lora_name)
        ctx.modules_dim[ctx.lora_name] = int(value.size(0))
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
    del modules_dim, modules_alpha
    if not state.get("has_glora"):
        return {}
    return {"detected_spec": "glora"}


def _continue_weight_kind(ctx: ContinueWeightDetectionContext) -> str | None:
    if str(ctx.metadata.get("ss_network_spec") or "").strip().lower() == "glora":
        return "GLoRA"
    glora_suffixes = (
        ".a1.weight",
        ".a2.weight",
        ".b1.weight",
        ".b2.weight",
    )
    if any(key.endswith(glora_suffixes) for key in ctx.lowered_keys):
        return "GLoRA"
    return None


register_save_handler("glora", save_glora_weights)
register_network_spec(
    NetworkSpec(
        name="glora",
        module_class=GLoRAModule,
        save_variant="glora",
        kwarg_flags=("use_glora",),
        selector=_selector,
        validate=_validate,
        preprocess_weights=_preprocess_weights,
        detect_from_weights=_detect_from_weights,
        finish_weight_detection=_finish_weight_detection,
        continue_weight_kind=_continue_weight_kind,
    )
)

__all__ = ["GLoRAModule"]
