"""Actual BF16 Anima + plain FP32 LoRA, matching fixed validation stack."""

from __future__ import annotations
import os
import re
from dataclasses import asdict
import torch
from .numerics import tensor_hash


def build_model(
    model_path,
    mode: str,
    swap: int,
    seed: int,
    forward_policy="torch",
    backward_policy="fp32",
    flash_deterministic=False,
):
    from library.anima.weights import load_anima_model
    from library.runtime.convrot.apply import apply_convrot_to_lora_network
    from library.runtime.int8_linear import selected_int8_linear_modules
    from networks.lora_anima.factory import create_network

    os.environ.update(
        ANIMA_CONVROT_INT8_GEMM="int_mm",
        ANIMA_CONVROT_STE_TF32="0",
        ANIMA_CONVROT_RHT="dense",
        ANIMA_CONVROT_HADAMARD="sylvester",
    )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if flash_deterministic:
        # Diagnostic process only. Production/default throughput continues
        # using its original Flash algorithm; never mix these timing stacks.
        from functools import partial
        from networks import attention_dispatch

        attention_dispatch.flash_attn_func = partial(
            attention_dispatch.flash_attn_func, deterministic=True
        )
        attention_dispatch.flash_attn_varlen_func = partial(
            attention_dispatch.flash_attn_varlen_func, deterministic=True
        )
    if mode != "w8a8" and (forward_policy != "torch" or backward_policy != "fp32"):
        raise ValueError("experimental policies require w8a8")
    device = torch.device("cuda:0")
    # CPU load avoids a full-base transient peak before swap placement.
    model = load_anima_model(
        device="cpu",
        dit_path=str(model_path),
        attn_mode="flash",
        loading_device="cpu",
        dit_weight_dtype=torch.bfloat16,
    )
    model.requires_grad_(False).train()
    model.reset_mod_guidance()
    model.enable_gradient_checkpointing()
    if not 0 <= swap <= model.num_blocks - 2:
        raise ValueError(f"swap must be in 0..{model.num_blocks - 2}")
    if swap:
        model.enable_block_swap(
            swap, device, transfer_dtype="bf16", restore_mode="foreach"
        )
        model.move_to_device_except_swap_blocks(device)
    else:
        model.to(device=device)
    targets = "|".join(
        re.escape(n) for n in sorted(selected_int8_linear_modules("all"))
    )
    torch.manual_seed(seed)
    network = create_network(
        1.0,
        32,
        32,
        None,
        [],
        model,
        exclude_patterns=[rf"^(?!blocks\.\d+\.({targets})$).*"],
        train_llm_adapter="false",
        lora_fp32_compute="false",
    )
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    network.to(device=device, dtype=torch.float32).train()
    initial = {
        n: p.detach().cpu().clone()
        for n, p in network.named_parameters()
        if p.requires_grad
    }
    patch = None
    if mode != "bf16":
        result = apply_convrot_to_lora_network(
            network,
            mode=mode,
            scope="all",
            group_size=256,
            free_base_weights=True,
            unet=model,
        )
        if not result.patched_count or result.skipped_count:
            raise RuntimeError(
                f"incomplete ConvRot patch: {result.patched_count} patched, {result.skipped_count} skipped"
            )
        patch = asdict(result)
    policy = None
    if forward_policy != "torch" or backward_policy != "fp32":
        from library.runtime.convrot.experimental_policy import (
            apply_experimental_policy,
        )

        policy = apply_experimental_policy(
            network, forward_policy=forward_policy, backward_policy=backward_policy
        )
    if swap:
        model.switch_block_swap_for_training()
    metadata = {
        "initial_parameters": {
            n: {"dtype": str(t.dtype), "shape": list(t.shape), "sha256": tensor_hash(t)}
            for n, t in initial.items()
        },
        "patch": patch,
        "experimental_policy": policy,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "tf32": False,
        "flash_backward_deterministic": flash_deterministic,
        "backend": "int_mm" if mode == "w8a8" else mode,
    }
    return model, network, initial, metadata


def drain(model) -> None:
    offloader = getattr(model, "offloader", None)
    if offloader is not None:
        for index in list(offloader.futures):
            offloader._wait_blocks_move(index, phase="validation_window")
    torch.cuda.synchronize()


def placement(model, network) -> dict:
    from library.runtime.block_swap_payload import block_swap_payload_residency

    payloads = {}
    for name, module in model.named_modules():
        for key, tensor in list(module.named_buffers(recurse=False)):
            if "convrot" in key:
                payloads[f"{name}.{key}"] = {
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "shape": list(tensor.shape),
                }
        if module.__class__.__name__ == "BlockSwapManagedTensor":
            tensor = module.weight
            payloads[name] = {
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "shape": list(tensor.shape),
            }
    return {
        "payload_residency": block_swap_payload_residency(model),
        "payloads": payloads,
        "adapter_dtypes": sorted(
            {str(p.dtype) for p in network.parameters() if p.requires_grad}
        ),
    }
