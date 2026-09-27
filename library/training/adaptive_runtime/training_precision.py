"""Training-side precision installation and Accelerate checkpoint identity hooks."""

from fnmatch import fnmatchcase
import json
import hashlib
from pathlib import Path

import torch

from .contract import precision_contract_id
from .islands import install_precision_islands


PRECISION_STATE = "adaptive_precision.json"


def _dtype_entries(model, names=None):
    parameters = dict(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers(remove_duplicate=False))
    if names is None:
        names = [*(f"parameter:{name}" for name in parameters),
                 *(f"buffer:{name}" for name in buffers)]
    entries = {}
    for key in names:
        kind, name = key.split(":", 1)
        tensor = parameters.get(name) if kind == "parameter" else buffers.get(name)
        if tensor is None:
            raise ValueError(f"Realized precision manifest is missing {key}")
        entries[key] = str(tensor.dtype)
    return entries


def realized_precision_manifest_id(model, names=None) -> str:
    """Hash the dtypes actually resident in the selected base tensors."""
    payload = json.dumps(_dtype_entries(model, names), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def accelerator_handlers(args):
    from accelerate.utils import AutocastKwargs, GradScalerKwargs

    # Keep native_amp/scaler ownership intact: disabling native_amp would also
    # bypass Accelerate's unscale-before-clipping path.
    return [AutocastKwargs(enabled=False),
            GradScalerKwargs(init_scale=args.adaptive_loss_scale)]


def install_training_precision(model, args, *, model_family=None):
    if any(type(p) is not torch.nn.Parameter or not p.is_floating_point()
           for p in model.parameters()):
        raise ValueError("FP16/FP32 training requires unquantized ordinary base parameters")
    parameters = list(model.named_parameters(remove_duplicate=False))
    if len({id(p) for _, p in parameters}) != len(parameters):
        raise ValueError("Aliased base parameters are not supported in experimental training")
    assignments = {n: "fp16" for n, m in model.named_modules() if type(m) is torch.nn.Linear}
    if not assignments:
        raise ValueError("No plain Linear modules to assign")
    for pattern in getattr(args, "adaptive_fp32_modules", ()):
        matches = [name for name in assignments if fnmatchcase(name, pattern)]
        if not matches:
            raise ValueError(f"Unknown FP32 Linear pattern: {pattern}")
        assignments.update({name: "fp32" for name in matches})
    model.requires_grad_(False)
    install_precision_islands(model, assignments)
    for module in model.modules():
        if type(module) is torch.nn.Linear:
            continue
        for parameter in module.parameters(recurse=False):
            if parameter.is_floating_point():
                parameter.data = parameter.data.float()
        for name, buffer in module.named_buffers(recurse=False):
            if buffer.is_floating_point():
                module._buffers[name] = buffer.float()
    family = model_family or getattr(args, "model_family", "anima")
    manifest = {"schema": "adaptive_training_precision_v1", "model_family": family,
                "assignments": assignments, "residual_dtype": "fp32",
                "initial_loss_scale": args.adaptive_loss_scale}
    model._adaptive_precision_manifest_names = tuple(_dtype_entries(model))
    model._adaptive_precision_manifest_id = realized_precision_manifest_id(
        model, model._adaptive_precision_manifest_names
    )
    model._adaptive_training_precision = manifest
    return manifest


def validate_local_precision_contract(args, expected):
    """Recompute the worker-side request; injected IDs are not trusted."""
    requested = str(getattr(args, "adaptive_precision", "off") or "off").strip().lower()
    if requested == "auto" and getattr(args, "adaptive_resolved_mode", None) not in {
        "bf16", "fp16_fp32", "fp32"
    }:
        raise ValueError("precision_contract_mismatch: worker adaptive mode is unresolved")
    actual = precision_contract_id(args)
    if actual != expected:
        raise ValueError("precision_contract_mismatch: worker precision request drifted")
    return actual


def preserve_precision_cast(model, dtype):
    manifest = getattr(model, "_adaptive_training_precision", None)
    if manifest is None:
        return dtype
    for name, precision in manifest["assignments"].items():
        module = model.get_submodule(name)
        expected = torch.float16 if precision == "fp16" else torch.float32
        if any(p.dtype != expected for p in module.parameters(recurse=False)):
            raise ValueError(f"Training precision drift before accelerator.prepare: {name}")
    return None


def validate_training_network(args, network):
    from .training_config import islands_enabled
    if not islands_enabled(args):
        return
    from networks.lora_modules.lora import LoRAModule

    modules = getattr(network, "unet_loras", ())
    if (not modules or getattr(network, "text_encoder_loras", ())
            or any(type(module) is not LoRAModule for module in modules)):
        raise ValueError("FP16/FP32 training requires plain DiT LoRA modules only")
    allowed = {id(p) for module in modules for p in module.parameters()}
    if any(id(p) not in allowed or p.dtype != torch.float32
           for p in network.parameters() if p.requires_grad):
        raise ValueError("FP16/FP32 training requires only FP32 LoRA trainable parameters")


def register_precision_checkpoint(accelerator, manifest):
    # Canonical independent copy: later memory-plan edits cannot mutate what
    # a checkpoint claims. Swap count is deliberately absent from this contract.
    expected = json.loads(json.dumps(manifest, sort_keys=True))

    def save_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            Path(output_dir, PRECISION_STATE).write_text(json.dumps(expected, sort_keys=True), encoding="utf-8")

    def load_hook(models, input_dir):
        path = Path(input_dir, PRECISION_STATE)
        if not path.is_file() or json.loads(path.read_text(encoding="utf-8")) != expected:
            raise ValueError("Checkpoint FP16/FP32 precision contract is missing or changed")

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)


def reject_unplanned_resume(args):
    from .training_config import islands_enabled

    if (not islands_enabled(args) and Path(args.resume, PRECISION_STATE).is_file()):
        raise ValueError("Checkpoint requires explicit FP16/FP32 training configuration")
