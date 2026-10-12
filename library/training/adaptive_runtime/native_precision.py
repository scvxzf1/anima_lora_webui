"""Bounded native AMP contract; base loading and FP32 adapters stay unchanged."""

from collections.abc import Mapping
import math

from library.models.family_registry import normalize_registered_family


def native_enabled(config):
    from .training_config import resolved_mode

    return resolved_mode(config) == "native_fp16"


def accelerator_handlers(args):
    from accelerate.utils import GradScalerKwargs

    # Accelerate keeps its normal autocast, clipping and checkpoint ownership.
    # init_scale is construction-only; load_state restores the saved scaler.
    return [GradScalerKwargs(init_scale=getattr(args, "adaptive_loss_scale", 1024.0))]


def configuration_errors(config, *, world_size=1):
    values = dict(config) if isinstance(config, Mapping) else vars(config).copy()
    get = values.get
    errors = []
    try:
        family = normalize_registered_family(
            get("model_family", "anima") or "anima", source="native FP16 model_family",
            allow_aliases=True,
        )
    except ValueError as exc:
        errors.append(str(exc))
    else:
        from .native_family import native_family
        try:
            native_family(family)
        except ValueError as exc:
            errors.append(str(exc))
        if family == "krea2_raw":
            if get("adaptive_precision") == "auto":
                errors.append("Krea-2 native FP16 requires explicit adaptive_precision='native_fp16'")
            if get("adaptive_oom_retry", False):
                errors.append("Krea-2 native FP16 startup OOM retry is not yet supported")
            if get("pipeline_parallel", False):
                errors.append("Krea-2 native FP16 does not support pipeline_parallel")
    required = {"mixed_precision": "fp16", "network_module": "networks.lora_anima",
                "network_train_unet_only": True, "gradient_checkpointing": True,
                "selective_checkpoint": "off", "attn_mode": "torch"}
    for key, expected in required.items():
        if get(key, expected) != expected:
            errors.append(f"native FP16 training requires {key}={expected!r}")
    for key in ("full_fp16", "full_bf16", "torch_compile", "nf4_prequantized_path", "dim_from_weights",
                "cpu_offload_checkpointing", "unsloth_offload_checkpointing"):
        if get(key, False):
            errors.append(f"native FP16 training does not yet support {key}")
    if str(get("base_compute", "bf16") or "bf16").strip().lower() == "nf4":
        errors.append("native FP16 training does not support NF4")
    if get("adaptive_oom_retry", False):
        from .native_retry_config import native_retry_configuration_errors
        errors.extend(native_retry_configuration_errors(config, world_size=world_size))
    if get("adaptive_fp32_modules", []):
        errors.append("native FP16 training does not support adaptive_fp32_modules")
    elif not isinstance(get("adaptive_fp32_modules", []), (list, tuple)):
        errors.append("adaptive_fp32_modules must be an empty list for native FP16")
    if world_size != 1:
        errors.append("native FP16 training currently requires one process")
    scale = get("adaptive_loss_scale", 1024.0)
    if (isinstance(scale, bool) or not isinstance(scale, (int, float))
            or not math.isfinite(scale) or not 1 < scale <= 2**24):
        errors.append("adaptive_loss_scale must be finite and in (1, 2**24]")
    errors.extend(_adapter_errors(values))
    return errors


def _adapter_errors(values):
    # Match the bootstrap precedence for network_args; use the current registry
    # to recognize plugin adapters rather than maintaining another flag list.
    from networks.registry import resolve_network_spec

    kwargs = values.copy()
    for argument in values.get("network_args") or ():
        if not isinstance(argument, str) or "=" not in argument:
            return ["native FP16 training requires valid key=value network_args"]
        key, value = argument.split("=", 1)
        kwargs[key] = value
    try:
        spec = resolve_network_spec(kwargs)
    except ValueError as exc:
        return [f"native FP16 plain LoRA contract: {exc}"]
    errors = []
    if spec.name != "lora":
        errors.append(f"native FP16 training requires plain LoRA, got {spec.name!r}")
    for key in ("add_reft", "use_timestep_mask", "train_llm_adapter",
                "network_train_text_encoder_only"):
        if str(kwargs.get(key, False)).strip().lower() not in {"", "0", "false", "none", "off"}:
            errors.append(f"native FP16 plain LoRA does not yet support {key}")
    if str(kwargs.get("lora_adapter_kind", "lora") or "lora").lower() != "lora":
        errors.append("native FP16 training requires lora_adapter_kind='lora'")
    if str(kwargs.get("router_source", "none") or "none").lower() != "none":
        errors.append("native FP16 plain LoRA requires router_source='none'")
    if str(kwargs.get("num_registers", 0) or 0) != "0":
        errors.append("native FP16 plain LoRA does not yet support num_registers")
    return errors
