"""Restrictions for replaying a native invocation before its first optimizer."""

from collections.abc import Mapping


def native_retry_configuration_errors(config, *, world_size=1):
    from .training_config import retry_bounds_errors

    get = config.get if isinstance(config, Mapping) else lambda k, d=None: getattr(config, k, d)
    errors = retry_bounds_errors(get)
    if world_size != 1:
        errors.append("Native startup OOM retry requires one process")
    if type(get("seed")) is not int:
        errors.append("Native startup OOM retry requires an explicit integer seed")
    for key, expected in (("gradient_accumulation_steps", 1), ("max_data_loader_n_workers", 0),
                          ("base_compute", "bf16"), ("block_swap_transfer_dtype", "bf16"),
                          ("block_swap_restore_mode", "foreach")):
        if get(key, expected) != expected:
            errors.append(f"Native startup OOM retry requires {key}={expected!r}")
    for key in ("use_vae_cache", "use_text_cache", "cache_latents", "cache_text_encoder_outputs",
                "cache_latents_to_disk", "cache_text_encoder_outputs_to_disk"):
        if get(key, True) is not True:
            errors.append(f"Native startup OOM retry requires {key}")
    # checkpointing_epochs also invokes saver.auto_resume before the loop.
    for key in ("resume", "auto_resume", "resume_from_huggingface", "checkpointing_epochs",
                "initial_step", "initial_epoch", "skip_until_initial_step", "sample_at_first",
                "validate_at_first", "validation_at_first", "base_weights", "lora_path",
                "stage_schedule", "stage_schedule_enabled", "pipeline_parallel", "auto_block_swap",
                "convrot_prequant_path", "debug_dataset", "artist_filter", "scale_weight_norms",
                "functional_loss_weight", "vr_loss_weight"):
        if get(key, False):
            errors.append(f"Native startup OOM retry does not support {key}")
    return errors
