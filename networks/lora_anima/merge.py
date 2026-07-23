"""Merge and baked-weight helpers for the LoRA-family network facade."""

import logging

import torch


logger = logging.getLogger(__name__)


def _all_loras(network):
    return network.text_encoder_loras + network.unet_loras


def fuse_weights(network) -> None:
    """Merge all LoRA deltas into base model weights for zero-overhead inference."""
    for lora in _all_loras(network):
        lora.fuse_weight()


def unfuse_weights(network) -> None:
    """Remove all LoRA deltas from base model weights."""
    for lora in _all_loras(network):
        lora.unfuse_weight()


def is_mergeable(network) -> bool:
    """True only for adapters that collapse into a static Linear delta.

    Register tokens, ReFT hooks, MoE/routing layouts, and Chimera dual-pool
    routers all need live forward hooks and must not claim bakeability.
    """
    cfg = network.cfg
    if int(getattr(cfg, "num_registers", 0) or 0) != 0:
        return False
    if bool(getattr(cfg, "add_reft", False)):
        return False
    if bool(getattr(cfg, "use_chimera_hydra", False)):
        return False
    use_moe_style = getattr(cfg, "use_moe_style", False)
    if use_moe_style not in (False, None, "", "false", "False"):
        return False
    router_source = str(getattr(cfg, "router_source", "none") or "none")
    if router_source not in ("", "none"):
        return False
    if bool(getattr(cfg, "route_per_layer", False)):
        return False
    return True


def merge_lora_weights(network, text_encoders, unet, weights_sd, dtype=None, device=None):
    apply_text_encoder = apply_unet = False
    for key in weights_sd.keys():
        if key.startswith(network.LORA_PREFIX_TEXT_ENCODER):
            apply_text_encoder = True
        elif key.startswith(network.LORA_PREFIX_ANIMA):
            apply_unet = True

    if apply_text_encoder:
        logger.info("enable LoRA for text encoder")
    else:
        network.text_encoder_loras = []

    if apply_unet:
        logger.info("enable LoRA for DiT")
    else:
        network.unet_loras = []

    # Pre-group checkpoint keys by LoRA module prefix (avoid O(modules * keys) scan).
    # Keys are "{module_name}.{param}" where module_name has no dots (dots -> underscores).
    grouped_sd: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in weights_sd.items():
        prefix, dot, suffix = key.partition(".")
        if not dot:
            continue
        if prefix not in grouped_sd:
            grouped_sd[prefix] = {}
        grouped_sd[prefix][suffix] = value

    for lora in _all_loras(network):
        sd_for_lora = grouped_sd.get(lora.lora_name, {})
        if sd_for_lora:
            lora.merge_to(sd_for_lora, dtype, device)

    logger.info("weights are merged")


def backup_weights(network) -> None:
    for lora in _all_loras(network):
        org_module = lora.org_module_ref[0]
        if not hasattr(org_module, "_lora_org_weight"):
            org_module._lora_org_weight = org_module.weight.detach().clone()
            org_module._lora_restored = True


def restore_weights(network) -> None:
    with torch.no_grad():
        for lora in _all_loras(network):
            org_module = lora.org_module_ref[0]
            if not org_module._lora_restored:
                org_module.weight.data.copy_(org_module._lora_org_weight)
                org_module._lora_restored = True


def pre_calculation(network) -> None:
    with torch.no_grad():
        for lora in _all_loras(network):
            org_module = lora.org_module_ref[0]
            lora_weight = lora.get_weight().to(
                org_module.weight.device,
                dtype=org_module.weight.dtype,
            )
            org_module.weight.data.add_(lora_weight)

            org_module._lora_restored = False
            lora.enabled = False
