"""Model loading helpers extracted from AnimaTrainer.

Keeps DiT / TE / VAE load order and freeze-merge behavior identical to train.py.
Family dispatch (stage 6): ``resolve_model_family(args)`` selects Anima
(Qwen3+T5 cross-attn DiT) or Krea-2-Raw (Qwen3-VL single-stream MMDiT). Anima
is the default and unchanged; Krea-2 loads via ``library.models.krea2_raw``.
"""

from __future__ import annotations

from typing import Any

import torch
from accelerate import Accelerator
from torch import nn

from library.anima import weights as anima_utils
from library.env import resolve_model_family
from library.models import qwen_vae as qwen_image_autoencoder_kl
from library.training.probes import maybe_probe_components as _maybe_probe_components
from library.training.train_bootstrap import resolve_block_swap_profile_jsonl
from library.training.v100_flash import (
    flash_attn_v100_doc,
    resolve_debug_finite_checks,
    resolve_v100_flash_stability,
)
import logging

logger = logging.getLogger(__name__)


def load_target_model(trainer, args, weight_dtype, accelerator, load_qwen3=True, load_vae=True):
    family = resolve_model_family(args)
    trainer.is_swapping_blocks = (
        args.blocks_to_swap is not None and args.blocks_to_swap > 0
    )

    # Load text encoder. Anima uses Qwen3 (+T5 adapter downstream); Krea-2 uses
    # Qwen3-VL-4B directly (hidden_dim=2560=DiT txtdim, no adapter). Skipped
    # when every text-encoder output is already cached and no live encoding
    # (sampling / TE training / cache disabled) needs it.
    if load_qwen3:
        if family == "krea2_raw":
            from library.models.krea2_raw.strategy import load_krea2_text_encoder

            logger.info("Loading Krea-2 Qwen3-VL text encoder...")
            qwen3_text_encoder, _ = load_krea2_text_encoder(
                args.qwen3, dtype=weight_dtype, device="cpu"
            )
        else:
            logger.info("Loading Qwen3 text encoder...")
            qwen3_text_encoder, _ = anima_utils.load_qwen3_text_encoder(
                args.qwen3, dtype=weight_dtype, device="cpu"
            )
        qwen3_text_encoder.eval()
    else:
        logger.info(
            "Skipping text encoder load: all text-encoder outputs cached."
        )
        qwen3_text_encoder = None

    # Load VAE. Shared across families (same AutoencoderKLQwenImage).
    if load_vae:
        logger.info(f"Loading {'Krea-2' if family == 'krea2_raw' else 'Anima'} VAE...")
        vae = qwen_image_autoencoder_kl.load_vae(
            args.vae,
            device="cpu",
            disable_mmap=True,
            spatial_chunk_size=args.vae_chunk_size,
            disable_cache=args.vae_disable_cache,
        )
        vae.to(weight_dtype)
        vae.eval()
    else:
        logger.info("Skipping VAE load: all latents cached and no sampling.")
        vae = None

    # Return format: (model_type, text_encoders, vae, unet). model_type is the
    # family name stamped into training metadata (anima | krea2_raw).
    return family, [qwen3_text_encoder], vae, None  # unet loaded lazily


def load_unet_lazily(trainer, args, weight_dtype, accelerator, text_encoders) -> tuple[nn.Module, list[nn.Module]]:
    family = resolve_model_family(args)
    if family == "krea2_raw":
        return _load_krea2_dit(
            trainer, args, weight_dtype, accelerator, text_encoders
        )
    return _load_anima_dit(trainer, args, weight_dtype, accelerator, text_encoders)


def _load_krea2_dit(trainer, args, weight_dtype, accelerator, text_encoders):
    """Krea-2 single-stream MMDiT loader (stage 6).

    Reuses the anima block-swap / probe / selective-checkpoint / VR-loss
    plumbing where the SingleStreamDiT exposes the same hooks as Anima
    (enable_block_swap / enable_selective_checkpointing / enable_peak_probe —
    added in stage 6 block-swap probe). attn_mode / v100 flash guards are
    anima-specific; Krea-2 uses sdpa_kernel(CUDNN_ATTENTION) internally and
    ignores them (logs once).
    """
    from library.models.krea2_raw.weights import load_krea2_dit

    loading_dtype = weight_dtype
    loading_device = "cpu" if trainer.is_swapping_blocks else accelerator.device
    logger.info(
        f"Loading Krea-2 DiT (family=krea2_raw) from "
        f"{args.pretrained_model_name_or_path}..."
    )
    model = load_krea2_dit(
        args.pretrained_model_name_or_path,
        device=loading_device,
        dtype=loading_dtype,
        eval=False,
    )
    _maybe_probe_components(
        trainer,
        "dit_loaded",
        unet=model,
        device=accelerator.device,
        phase="setup",
        loading_device=loading_device,
        loading_dtype=loading_dtype,
    )
    if trainer.peak_probe is not None and hasattr(model, "enable_peak_probe"):
        model.enable_peak_probe(trainer.peak_probe)
        trainer.peak_probe.write(
            {
                "ev": "peak_probe_config",
                "label": "dit_peak_probe_attached",
                "phase": "setup",
                "block_count": len(getattr(model, "blocks", []) or []),
            }
        )

    trainer._use_unsloth_offload_checkpointing = args.unsloth_offload_checkpointing
    selective_checkpoint = getattr(args, "selective_checkpoint", "off") or "off"
    if selective_checkpoint != "off":
        selective_checkpoint_blocks = getattr(
            args, "selective_checkpoint_blocks", ""
        )
        logger.info(
            "enable selective checkpoint: "
            f"{selective_checkpoint}"
            + (
                f" blocks={selective_checkpoint_blocks or 'auto'}"
                if str(selective_checkpoint).startswith("peak_blocks_")
                else ""
            )
        )
        model.enable_selective_checkpointing(
            selective_checkpoint,
            blocks=selective_checkpoint_blocks,
        )

    # Block swap (reuses anima ModelOffloader; SingleStreamBlock forward is
    # transparent to the offloader — stage 6 block-swap probe verified).
    trainer.is_swapping_blocks = (
        args.blocks_to_swap is not None and args.blocks_to_swap > 0
    )
    if trainer.is_swapping_blocks:
        profile_jsonl = resolve_block_swap_profile_jsonl(args)
        logger.info(
            "enable block swap: "
            f"blocks_to_swap={args.blocks_to_swap}, "
            f"transfer_dtype={args.block_swap_transfer_dtype}, "
            f"restore_mode={args.block_swap_restore_mode}, "
            f"profile_jsonl={profile_jsonl or 'off'}"
        )
        model.enable_block_swap(
            args.blocks_to_swap,
            accelerator.device,
            profile_jsonl=profile_jsonl,
            transfer_dtype=args.block_swap_transfer_dtype,
            restore_mode=args.block_swap_restore_mode,
        )
        _maybe_probe_components(
            trainer,
            "block_swap_enabled",
            unet=model,
            device=accelerator.device,
            phase="setup",
            blocks_to_swap=args.blocks_to_swap,
            block_swap_profile_jsonl=profile_jsonl or "off",
            block_swap_transfer_dtype=args.block_swap_transfer_dtype,
            block_swap_restore_mode=args.block_swap_restore_mode,
        )

    if float(getattr(args, "vr_loss_weight", 0.0) or 0.0) > 0.0:
        logger.info(
            f"VR loss enabled (vr_loss_weight={args.vr_loss_weight}); "
            f"using trainable DiT with multiplier=0 as the control variate"
        )

    return model, text_encoders


def _load_anima_dit(trainer, args, weight_dtype, accelerator, text_encoders):
    loading_dtype = weight_dtype
    loading_device = "cpu" if trainer.is_swapping_blocks else accelerator.device

    attn_mode = "torch"
    if args.xformers:
        attn_mode = "xformers"
    if args.attn_mode is not None:
        attn_mode = args.attn_mode

    v100_flash_stability = resolve_v100_flash_stability(args)
    debug_finite_checks = resolve_debug_finite_checks(
        args, v100_flash_stability
    )

    if attn_mode == "flash4":
        # Flash Attention 4 (flash-attention-sm120) is not supported yet.
        raise RuntimeError(
            "attn_mode='flash4' is not supported yet -- the flash-attention-sm120 "
            "kernel is disabled in this build. Use 'flash', 'torch', 'flex', "
            "'sageattn', or 'xformers' instead."
        )
    elif attn_mode == "flash":
        from networks.attention_dispatch import (
            flash_attn,
            flash_attn_func,
            flash_attn_v100_provider,
        )

        if flash_attn_func is None:
            raise RuntimeError(
                "attn_mode='flash' requested but flash_attn is not available."
            )
        flash_doc, doc_reports_v100 = flash_attn_v100_doc(flash_attn)
        is_v100_fork = flash_attn_v100_provider or doc_reports_v100
        try:
            major, minor = torch.cuda.get_device_capability(accelerator.device)
        except Exception:
            major, minor = -1, -1
        logger.info(
            "Using Flash Attention 2 (flash_attn %s), gpu_sm=%s.%s, "
            "v100_fork=%s, v100_flash_stability=%s, debug_finite_checks=%s%s",
            getattr(flash_attn, "__version__", "unknown"),
            major,
            minor,
            is_v100_fork,
            v100_flash_stability,
            debug_finite_checks,
            f", doc={flash_doc}" if flash_doc else "",
        )
        if major == 7 and minor == 0 and is_v100_fork:
            if loading_dtype != torch.float16:
                raise RuntimeError(
                    "flash-attention-v100 only supports FP16 inputs; use "
                    "mixed_precision='fp16' or attn_mode='torch'."
                )
            logger.warning(
                "Detected flash-attention-v100 on Volta/V100. Its strict "
                "production validation is not approved; use attn_mode='torch' "
                "for production and Flash only for diagnostics."
            )
    else:
        logger.info(f"Using attention mode: {attn_mode}")

    # Frozen LoRA: merged into DiT weights at load time (no runtime hooks).
    # Used by postfix runs that train on top of a fixed LoRA.
    lora_weights_list = None
    lora_multipliers = None
    if getattr(args, "lora_path", None):
        from safetensors.torch import load_file

        logger.info(
            f"merging frozen LoRA from {args.lora_path} into DiT weights "
            f"(multiplier={args.lora_multiplier})"
        )
        lora_sd = load_file(args.lora_path)
        lora_sd = {k: v for k, v in lora_sd.items() if k.startswith("lora_unet_")}
        lora_weights_list = [lora_sd]
        lora_multipliers = [args.lora_multiplier]

    # Load DiT
    attn_softmax_scale = getattr(args, "attn_softmax_scale", None)
    logger.info(
        f"Loading Anima DiT model with attn_softmax_scale: {attn_softmax_scale}..."
    )
    model = anima_utils.load_anima_model(
        accelerator.device,
        args.pretrained_model_name_or_path,
        attn_mode,
        loading_device,
        loading_dtype,
        lora_weights_list=lora_weights_list,
        lora_multipliers=lora_multipliers,
        attn_softmax_scale=attn_softmax_scale,
        v100_flash_stability=v100_flash_stability,
        debug_finite_checks=debug_finite_checks,
    )
    _maybe_probe_components(
        trainer,
        "dit_loaded",
        unet=model,
        device=accelerator.device,
        phase="setup",
        attn_mode=attn_mode,
        loading_device=loading_device,
        loading_dtype=loading_dtype,
    )
    if trainer.peak_probe is not None and hasattr(model, "enable_peak_probe"):
        model.enable_peak_probe(trainer.peak_probe)
        trainer.peak_probe.write(
            {
                "ev": "peak_probe_config",
                "label": "dit_peak_probe_attached",
                "phase": "setup",
                "block_count": len(getattr(model, "blocks", []) or []),
            }
        )

    # Store unsloth preference so that when the base trainer calls
    # dit.enable_gradient_checkpointing(cpu_offload=...), we can override to use unsloth.
    trainer._use_unsloth_offload_checkpointing = args.unsloth_offload_checkpointing

    selective_checkpoint = getattr(args, "selective_checkpoint", "off") or "off"
    if selective_checkpoint != "off":
        selective_checkpoint_blocks = getattr(
            args, "selective_checkpoint_blocks", ""
        )
        logger.info(
            "enable selective checkpoint: "
            f"{selective_checkpoint}"
            + (
                f" blocks={selective_checkpoint_blocks or 'auto'}"
                if str(selective_checkpoint).startswith("peak_blocks_")
                else ""
            )
        )
        model.enable_selective_checkpointing(
            selective_checkpoint,
            blocks=selective_checkpoint_blocks,
        )

    # torch.compile is intentionally delayed until after
    # network.apply_to/load_weights and gradient-checkpoint setup. Otherwise
    # Dynamo can trace the bare DiT path instead of the adapter-patched
    # Linear forwards, and checkpoint recompute may not match the original
    # forward graph. See library.runtime.harness.compile_blocks_for_training.

    # Block swap
    trainer.is_swapping_blocks = (
        args.blocks_to_swap is not None and args.blocks_to_swap > 0
    )
    if trainer.is_swapping_blocks:
        profile_jsonl = resolve_block_swap_profile_jsonl(args)
        logger.info(
            "enable block swap: "
            f"blocks_to_swap={args.blocks_to_swap}, "
            f"transfer_dtype={args.block_swap_transfer_dtype}, "
            f"restore_mode={args.block_swap_restore_mode}, "
            f"profile_jsonl={profile_jsonl or 'off'}"
        )
        model.enable_block_swap(
            args.blocks_to_swap,
            accelerator.device,
            profile_jsonl=profile_jsonl,
            transfer_dtype=args.block_swap_transfer_dtype,
            restore_mode=args.block_swap_restore_mode,
        )
        _maybe_probe_components(
            trainer,
            "block_swap_enabled",
            unet=model,
            device=accelerator.device,
            phase="setup",
            blocks_to_swap=args.blocks_to_swap,
            block_swap_profile_jsonl=profile_jsonl or "off",
            block_swap_transfer_dtype=args.block_swap_transfer_dtype,
            block_swap_restore_mode=args.block_swap_restore_mode,
        )

    # Variance-reduced FM loss: the "frozen reference" is the trainable
    # DiT itself with ``network.set_multiplier(0)`` during the no-grad
    # forward — works because base weights are frozen and LoRA-family
    # adapters are additive. See ``get_noise_pred_and_target`` for the
    # bypass. Saves ~5 GB VRAM vs holding a second DiT copy.
    if float(getattr(args, "vr_loss_weight", 0.0) or 0.0) > 0.0:
        logger.info(
            f"VR loss enabled (vr_loss_weight={args.vr_loss_weight}); "
            f"using trainable DiT with multiplier=0 as the control variate"
        )

    return model, text_encoders
