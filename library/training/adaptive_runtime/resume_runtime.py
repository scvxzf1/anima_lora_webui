"""Private exact-cursor recovery integration; public resume remains unsupported."""
from pathlib import Path
import json
import random

import torch

from .resume_checkpoint import save_recovery_state, validate_recovery_state, restore_recovery_auxiliary


def enabled(args):
    return bool(getattr(args, "adaptive_oom_retry_resume", False))


def startup_log(args, logger, message):
    """Replayed startup telemetry must not draw from committed training RNG.

    Rich generates random hyperlink IDs while rendering a log record. Only
    isolate these repeated log emissions, never scheduler, data or epoch hooks.
    Initial startup retains its existing RNG behavior.
    """
    if not getattr(args, "_adaptive_training_resume", None):
        logger.info(message, stacklevel=2)
        return
    previous = random.getstate()
    try:
        logger.info(message, stacklevel=2)
    finally:
        random.setstate(previous)


def canonical_device(device):
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu") if device.type == "cpu" else device


def configuration_errors(get, *, world_size=1):
    if not get("adaptive_oom_retry_resume", False):
        return []
    errors = []
    if not (get("adaptive_oom_retry", False) or get("_adaptive_training_worker", None)):
        errors.append("adaptive_oom_retry_resume requires adaptive_oom_retry")
    if world_size != 1 or get("gradient_accumulation_steps", 1) != 1:
        errors.append("Recovery requires one process and accumulation=1")
    if type(get("seed")) is not int or get("max_data_loader_n_workers", 0) != 0:
        errors.append("Recovery requires integer seed and workers=0")
    for key in ("stage_schedule", "stage_schedule_enabled", "dataset_class", "sample_at_first",
                "sample_every_n_steps", "sample_every_n_epochs", "validate_at_first",
                "validate_every_n_steps", "validate_every_n_epochs", "auto_resume", "initial_step",
                "initial_epoch", "skip_until_initial_step", "network_args", "anima_freeze_blocks"):
        if get(key, False):
            errors.append(f"Recovery does not support {key}")
    for key in ("cache_latents_to_disk", "cache_text_encoder_outputs_to_disk"):
        if get(key, True) is not True:
            errors.append(f"Recovery requires {key}")
    return errors


def commit(runtime, state):
    counters = {"global_step": state.global_step, "completed_steps": state.global_step}
    for name in ("attempted_steps", "effective_steps", "skipped_steps"):
        if hasattr(runtime, name):
            counters[name] = getattr(runtime, name)
    precision = {"mode": getattr(runtime.args, "adaptive_resolved_mode", None)
                 or runtime.args.adaptive_precision,
                 "base": getattr(runtime, "_base", None),
                 "execution": getattr(runtime, "_execution", None)}
    saved = save_recovery_state(state, runtime.directory,
                               precision_contract_id=runtime.precision_contract_id,
                               precision_manifest_id=runtime.precision_manifest_id,
                               counters=counters, precision_metadata=precision)
    runtime.saved_state = str(saved)
    runtime.completed_steps = state.global_step
    return saved


def restore(trainer, args, accelerator, loader, model, network, optimizer):
    request = getattr(args, "_adaptive_training_resume", None)
    if request is None:
        return None
    if not enabled(args) or not getattr(args, "_adaptive_training_worker", None):
        raise ValueError("Exact resume requires an isolated recovery worker")
    runtime = trainer._adaptive_training_runtime
    options = dict(owner_root=request["owner_root"],
                   precision_contract_id=runtime.precision_contract_id,
                   precision_manifest_id=runtime.precision_manifest_id,
                   data_identity=loader.dataset.data_identity)
    metadata = validate_recovery_state(request["path"], **options)
    if metadata["scaler_required"] != (accelerator.scaler is not None):
        raise ValueError("Recovery scaler requirement changed")
    cursor = torch.load(Path(request["path"]) / "cursor.pt", weights_only=False, map_location="cpu")
    progress = cursor["loader"]
    data = progress["dataset_state"]
    step = metadata["global_step"]
    if (data["fetched"] != step or data["epoch"] != cursor["iteration"]
            or cursor["iteration"] * len(loader) + progress["_num_yielded"] != step
            or progress["_sampler_iter_yielded"] != progress["_num_yielded"]):
        raise ValueError("Recovery cursor progress differs from committed step")
    counts = metadata["counters"]
    if "attempted_steps" in counts and not (
            counts["attempted_steps"] == counts["effective_steps"] + counts["skipped_steps"] == step):
        raise ValueError("Recovery native counters differ")
    precision = json.loads((Path(request["path"]) / "precision.json").read_text())
    expected = {"mode": getattr(runtime.args, "adaptive_resolved_mode", None) or runtime.args.adaptive_precision,
                "base": getattr(runtime, "_base", None), "execution": getattr(runtime, "_execution", None)}
    from .resume_checkpoint import _precision
    if precision != _precision(expected, runtime.precision_contract_id, runtime.precision_manifest_id, counts):
        raise ValueError("Recovery precision metadata changed")
    accelerator.load_state(request["path"])
    restore_recovery_auxiliary(accelerator, loader, request["path"], **options)
    model = accelerator.unwrap_model(model)
    model.move_to_device_except_swap_blocks(accelerator.device)
    model.prepare_block_swap_before_forward(free_cache=False)
    target_device = canonical_device(accelerator.device)
    for name, parameter in network.named_parameters():
        if parameter.requires_grad and (parameter.dtype != torch.float32
                                       or canonical_device(parameter.device) != target_device):
            raise ValueError(f"Recovery trainable dtype/device differs: {name} "
                             f"{parameter.dtype}/{parameter.device}, expected FP32/{target_device}")
    for parameter, values in optimizer.state.items():
        for name, value in values.items():
            if not isinstance(value, torch.Tensor):
                continue
            if name == "step" and value.numel() == 1:
                continue
            if value.device != parameter.device or (value.is_floating_point() and value.dtype != parameter.dtype):
                raise ValueError("Recovery optimizer dtype/device differs")
    for name, value in metadata["counters"].items():
        if hasattr(runtime, name):
            setattr(runtime, name, value)
    runtime.saved_state = str(Path(request["path"]))
    runtime.optimizer_started = metadata["global_step"] > 0
    runtime.validate_precision(model)
    from library.training.checkpoints import ResumeStartPlan
    return ResumeStartPlan(initial_step=0, epoch_to_start=loader.iteration,
                           global_step=metadata["global_step"])
