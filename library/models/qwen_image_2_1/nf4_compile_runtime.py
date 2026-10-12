"""Session environment and model checks for the experimental NF4 policy."""
from collections.abc import Mapping
from contextlib import contextmanager
import os

from library.quantization.qwen_nf4_compile_policy import qwen_nf4_compile_experiment_errors


def _values(args):
    return dict(args) if isinstance(args, Mapping) else vars(args)


@contextmanager
def qwen_nf4_compile_session(args):
    """Set deterministic state before CUDA, restoring it on every exit."""
    values = _values(args)
    if values.get("qwen_nf4_compile_experiment") is not True:
        yield
        return
    errors = qwen_nf4_compile_experiment_errors(values.get("model_family"), args)
    if errors:
        raise ValueError("Qwen NF4 compile experiment requires " + "; ".join(errors))
    import torch

    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace is not None and workspace != ":4096:8":
        raise ValueError("Qwen NF4 compile experiment requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    if workspace is None and torch.cuda.is_initialized():
        raise ValueError("Qwen NF4 compile experiment cannot set workspace after CUDA initialization")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Qwen NF4 compile experiment requires single-process training")
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        if workspace is None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=False)
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        if workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace


def _verify_packed_model(model):
    import torch
    from library.quantization.nf4 import BF16AutocastLinear4bit

    packed = [module for module in model.modules() if isinstance(module, BF16AutocastLinear4bit)]
    if len(packed) != 232 or any(
        module.compute_dtype != torch.bfloat16 or module.weight.dtype != torch.uint8
        or getattr(module.weight, "quant_state", None) is None
        for module in packed
    ):
        raise ValueError("Qwen NF4 compile experiment requires actual 232 packed BF16 NF4 modules")


def prepare_qwen_nf4_compile_experiment(model, network, args):
    """Validate the entire model and both patches before mutating forwards."""
    import torch
    from networks.lora_modules.lora import LoRAModule
    from .native_layer_norm import patch_native_layer_norm, validate_native_layer_norm
    from .native_silu import patch_native_silu, validate_native_silu

    errors = qwen_nf4_compile_experiment_errors(_values(args).get("model_family"), args)
    if errors:
        raise ValueError("Qwen NF4 compile experiment requires " + "; ".join(errors))
    if getattr(model, "_mixed_nf4_family", None) != "qwen_image_2_1":
        raise ValueError("Qwen NF4 compile experiment requires actual mixed NF4 weights")
    if (os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
            or not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()):
        raise ValueError("Qwen NF4 compile experiment requires strict deterministic session")
    _verify_packed_model(model)
    blocks = getattr(model, "transformer_blocks", None)
    if not isinstance(blocks, torch.nn.ModuleList) or len(blocks) != 32:
        raise ValueError("Qwen NF4 compile experiment requires 32 transformer blocks")
    if any(getattr(block, "_qwen_segmented_forward", None) is not None for block in blocks):
        raise ValueError("Qwen NF4 compile experiment does not support segmented block forwards")
    if not getattr(model, "gradient_checkpointing", False) or getattr(model, "blocks_to_swap", None) != 24:
        raise ValueError("Qwen NF4 compile experiment requires actual full checkpoint and swap24")
    adapter = getattr(model, "_qwen_image_2_1_block_swap_adapter", None)
    offloader = getattr(adapter, "offloader", None)
    if (offloader is None or offloader.restore_mode != "foreach"
            or offloader.transfer_dtype != "bf16"):
        raise ValueError("Qwen NF4 compile experiment requires actual foreach/BF16 offloader")
    targets = {id(module): f"{index}.{name}" for index, block in enumerate(blocks)
               for name, module in block.named_modules()
               if name in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0")}
    adapters = list(getattr(network, "unet_loras", []))
    if len(targets) != 128 or len(adapters) != 128 or getattr(network, "text_encoder_loras", []):
        raise ValueError("Qwen NF4 compile experiment requires 128 production attention-only adapters")
    seen = set()
    for module in adapters:
        if type(module) is not LoRAModule:
            raise ValueError("Qwen NF4 compile experiment requires plain LoRAModule")
        original = module.org_module_ref[0]
        if id(original) not in targets or id(original) in seen or not hasattr(module, "org_forward"):
            raise ValueError("Qwen NF4 compile experiment requires applied attention-only adapters")
        seen.add(id(original))
        if (module.lora_dim != 16 or module.alpha.item() != 16 or module.multiplier != 1
                or any(getattr(module, key, False) for key in
                       ("dropout", "rank_dropout", "module_dropout", "fp32_compute", "_has_channel_scale", "_fused"))):
            raise ValueError("Qwen NF4 compile experiment requires rank/alpha16 plain adapters without dropout")
    validate_native_layer_norm(model)
    validate_native_silu(model)
    patch_native_layer_norm(model)
    patch_native_silu(model)
