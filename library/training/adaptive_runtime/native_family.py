"""Family-specific native AMP guards, separate from shared scaler identity."""
from library.models.family_registry import normalize_registered_family

ANIMA_FP32_BUFFERS = frozenset({"pos_embedder.seq", "pos_embedder.dim_spatial_range",
                              "pos_embedder.dim_temporal_range", "_mod_guidance_delta",
                              "_mod_guidance_schedule", "_mod_guidance_final_w"})


def native_family(family="anima"):
    family = normalize_registered_family(family or "anima", source="native FP16 model_family",
                                         allow_aliases=True)
    if family not in {"anima", "krea2_raw"}:
        raise ValueError("native FP16 training supports only Anima and explicit Krea-2")
    return family


def fp32_buffers(family):
    return ANIMA_FP32_BUFFERS if native_family(family) == "anima" else frozenset()


def install_guards(model, family):
    if native_family(family) == "anima":
        model.enable_fp32_residual()
    else:
        from library.models.krea2_raw.dit import SingleMMDiTConfig
        from library.models.krea2_raw.native_precision import enable_native_guards
        if getattr(model, "config", None) != SingleMMDiTConfig.krea2_raw():
            raise ValueError("Native Krea-2 requires the official checkpoint structure")
        enable_native_guards(model)


def execution_guards(model, family):
    family = native_family(family)
    blocks = getattr(model, "blocks", ())
    if family == "anima":
        if len(blocks) not in {28, 40}:
            raise ValueError("precision_contract_mismatch: native Anima requires 28 or 40 blocks")
        guards = {"blocks": len(blocks), "fp32_residual":
                  all(getattr(b, "fp32_residual", None) is True for b in blocks)
                  and getattr(getattr(model, "final_layer", None), "fp32_residual", None) is True}
    else:
        from library.models.krea2_raw.dit import SingleMMDiTConfig
        from library.models.krea2_raw.native_precision import observe_guards
        if getattr(model, "config", None) != SingleMMDiTConfig.krea2_raw():
            raise ValueError("precision_contract_mismatch: native Krea-2 official structure required")
        if len(blocks) != model.config.layers:
            raise ValueError("precision_contract_mismatch: native Krea-2 block count drifted")
        guards = observe_guards(model)
        if any(getattr(m, "attn_mode", "torch") != "torch" for m in model.modules()):
            raise ValueError("precision_contract_mismatch: native Krea-2 attention drifted")
    if any(value is not True for key, value in guards.items() if key not in {"family", "blocks"}):
        raise ValueError("precision_contract_mismatch: native FP32 residual guards missing")
    if any(getattr(b, "gradient_checkpointing", None) is not True
           or getattr(b, "cpu_offload_checkpointing", False)
           or getattr(b, "unsloth_offload_checkpointing", False) for b in blocks):
        raise ValueError("precision_contract_mismatch: native full checkpoint policy drifted")
    return guards


def report_guards(model, family):
    family = normalize_registered_family(family or "anima", source="precision report", allow_aliases=True)
    if family == "krea2_raw":
        from library.models.krea2_raw.native_precision import observe_guards
        return {"evidence": "observed Krea-2 runtime guard attributes", **observe_guards(model)}
    if family != "anima":
        return {"family": family, "evidence": "no native guard policy registered"}
    blocks = list(getattr(model, "blocks", ()))
    return {"evidence": "observed Anima fp32_residual runtime attributes, not inferred from requested mode",
            "blocks_observed": len(blocks),
            "blocks_enabled": sum(getattr(b, "fp32_residual", False) is True for b in blocks),
            "final_layer_enabled": getattr(getattr(model, "final_layer", None), "fp32_residual", None)}
