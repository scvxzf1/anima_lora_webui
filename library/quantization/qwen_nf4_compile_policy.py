"""Torch-free, fail-closed contract for the opt-in Qwen NF4 experiment."""
from collections.abc import Mapping


def _enabled(value):
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() not in {"false", "0", "none", "off", "", "[]"}


def _nonzero_dropout(value):
    if value is None or str(value).strip().lower() in {"none", ""}:
        return False
    try:
        return float(value) != 0.0
    except (ValueError, TypeError):
        return True


def qwen_nf4_compile_experiment_errors(family, args):
    values = dict(args) if isinstance(args, Mapping) else vars(args).copy()
    if values.get("qwen_nf4_compile_experiment") is not True:
        return []
    from library.quantization.training import mixed_nf4_training_errors

    try:
        errors = mixed_nf4_training_errors(family, args, allow_compile=True)
    except (ValueError, TypeError) as exc:
        return [f"invalid plain LoRA network_args: {exc}"]
    if family != "qwen_image_2_1":
        errors.append("model_family=qwen_image_2_1")
    required = {
        "torch_compile": True, "attn_mode": "torch", "compile_dynamic_seq": False,
        "compile_seq_bands": False, "compile_block_scope": "all",
        "dynamo_backend": "inductor", "blocks_to_swap": 24,
        "block_swap_restore_mode": "foreach", "network_dim": 16,
        "network_alpha": 16, "qwen_image_2_1_task": "edit",
    }
    for key, expected in required.items():
        value = values.get(key)
        if isinstance(expected, bool):
            valid = value is expected
        elif isinstance(expected, int):
            try:
                valid = float(value) == expected
            except (ValueError, TypeError):
                valid = False
        else:
            valid = value == expected
        if not valid:
            errors.append(f"{key}={expected}")
    if values.get("compile_inductor_mode") not in (None, "default"):
        errors.append("compile_inductor_mode=default")
    for key in ("allow_unverified_nf4_compile", "auto_block_swap", "dynamic_block_swap",
                "pipeline_parallel", "network_train_text_encoder_only", "full_bf16",
                "partitioner_recompute_views", "partitioner_aggressive_recomputation",
                "adaptive_oom_retry_resume", "base_weights"):
        if _enabled(values.get(key, False)):
            errors.append(f"{key} disabled")
    network_args = values.get("network_args") or []
    if isinstance(network_args, Mapping):
        values.update(network_args)
    else:
        values.update(item.split("=", 1) for item in network_args)
    for key in ("network_dropout", "dropout", "rank_dropout", "module_dropout"):
        if _nonzero_dropout(values.get(key, 0)):
            errors.append(f"{key} disabled")
    for key in ("lora_fp32_compute", "per_channel_scaling", "use_rslora"):
        if _enabled(values.get(key, False)):
            errors.append(f"{key} disabled")
    if values.get("down_init", "kaiming") != "kaiming":
        errors.append("down_init=kaiming")
    return list(dict.fromkeys(errors))
