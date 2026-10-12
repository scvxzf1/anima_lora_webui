"""Fail closed for the initial multi-family mixed NF4 training contract."""
from collections.abc import Mapping


COMPILE_RISK_FAMILIES = {"krea2_raw", "qwen_image_2_1"}
COMPILE_RISK_MESSAGE = (
    "默认 Mixed NF4 + torch.compile 路径未通过完整权重数值验收；Qwen-Image 2.1 默认路径已观察到梯度差异，"
    "受验联合策略需显式启用 qwen_nf4_compile_experiment。"
    "Krea-2 mixed NF4 的编译训练范围尚未验证。确认风险仅允许尝试，不代表数值问题已修复。"
)


def mixed_nf4_training_errors(family, args, *, allow_compile=False):
    """Shared CPU-only rules for preflight and the model runtime guard."""
    values = dict(args) if isinstance(args, Mapping) else vars(args).copy()
    get = values.get
    errors = []
    required = {"base_compute": "bf16", "mixed_precision": "bf16",
                "adaptive_precision": "off", "block_swap_transfer_dtype": "bf16",
                "selective_checkpoint": "off", "network_module": "networks.lora_anima"}
    for key, expected in required.items():
        if str(get(key, expected) or expected).strip().lower() != expected:
            errors.append(f"{key}={expected}")
    if not get("gradient_checkpointing", False):
        errors.append("full gradient_checkpointing")
    for key in ("full_fp16", "adaptive_oom_retry", "adaptive_fp32_modules",
                "nf4_prequantized_path", "cpu_offload_checkpointing", "unsloth_offload_checkpointing"):
        if get(key, False):
            errors.append(f"{key} disabled")
    if get("torch_compile", False) and not (allow_compile and family in COMPILE_RISK_FAMILIES):
        errors.append("torch_compile disabled")
    network_args = get("network_args", None) or []
    if isinstance(network_args, Mapping):
        values.update(network_args)
    else:
        for item in network_args:
            if "=" not in item:
                raise ValueError("Mixed NF4 requires valid plain LoRA network_args")
            key, value = item.split("=", 1)
            values[key] = value
    for key in ("use_ip_adapter", "use_easycontrol", "use_byg", "use_lokr", "use_loha",
                "use_glora", "use_vera", "use_ortho", "use_chimera_hydra", "use_timestep_mask",
                "add_reft", "train_llm_adapter", "dora_wd", "train_adaln"):
        if str(values.get(key, False)).strip().lower() not in {"false", "0", "none", "off", ""}:
            errors.append("plain LoRA")
    if (str(values.get("router_source", "none") or "none").lower() != "none"
            or str(values.get("use_moe_style", False)).lower() not in {"false", "0", "none", "off", ""}
            or int(values.get("step_expert_K", 1) or 1) > 1):
        errors.append("plain LoRA without routing")
    if family == "qwen_image_2_1":
        if str(get("qwen_fused_projections", "off") or "off").lower() != "off":
            errors.append("qwen_fused_projections=off")
        for key in ("qwen_saved_projection_blocks", "qwen_projection_budget_mib", "qwen_segmented_training"):
            if get(key, False):
                errors.append(f"{key} disabled")
    return list(dict.fromkeys(errors))


def validate_mixed_nf4_training(model, args):
    """Validate before installing precision, attention, checkpoint or swap hooks."""
    family = getattr(model, "_mixed_nf4_family", None)
    values = dict(args) if isinstance(args, Mapping) else vars(args)
    experiment = values.get("qwen_nf4_compile_experiment") is True
    if experiment:
        from .qwen_nf4_compile_policy import qwen_nf4_compile_experiment_errors
        if family != "qwen_image_2_1":
            raise ValueError("Qwen NF4 compile experiment requires actual mixed NF4 weights")
        errors = qwen_nf4_compile_experiment_errors(family, args)
        if errors:
            raise ValueError("Qwen NF4 compile experiment requires " + "; ".join(errors))
    if family is None:
        return
    allow = values.get("allow_unverified_nf4_compile") is True
    errors = mixed_nf4_training_errors(family, args, allow_compile=allow or experiment)
    if errors:
        raise ValueError("Mixed NF4 training requires " + "; ".join(errors))
    if allow and values.get("torch_compile") and family in COMPILE_RISK_FAMILIES:
        import logging
        logging.getLogger(__name__).warning(COMPILE_RISK_MESSAGE)
