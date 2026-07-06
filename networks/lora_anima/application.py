"""Apply/lifecycle helpers for the LoRA-family network facade."""


def prepare_network(network, args, *, logger) -> None:
    if getattr(args, "lora_fp32_accumulation", False):
        logger.warning(
            "--lora_fp32_accumulation is deprecated and has no effect; "
            "fp32 accumulation is now unconditional in LoRA/Hydra/ReFT "
            "bottleneck matmuls. Remove the flag from your config."
        )


def set_multiplier(network, multiplier) -> None:
    network.multiplier = multiplier
    for lora in network.text_encoder_loras + network.unet_loras:
        lora.multiplier = network.multiplier
    for reft in network.text_encoder_refts + network.unet_refts:
        reft.multiplier = network.multiplier


def set_enabled(network, is_enabled) -> None:
    for lora in network.text_encoder_loras + network.unet_loras:
        lora.enabled = is_enabled


def set_step_index(network, step_index: int) -> None:
    """Broadcast a hard denoising-step index to step-expert modules."""
    k = int(step_index)
    for lora in network.text_encoder_loras + network.unet_loras:
        set_step = getattr(lora, "set_step", None)
        if set_step is not None:
            set_step(k)


def apply_to(
    network,
    text_encoders,
    unet,
    *,
    apply_text_encoder: bool = True,
    apply_unet: bool = True,
    logger,
) -> None:
    if apply_text_encoder:
        logger.info(
            f"enable LoRA for text encoder: {len(network.text_encoder_loras)} modules"
        )
    else:
        network.text_encoder_loras = []
        network.text_encoder_refts = []

    if apply_unet:
        logger.info(f"enable LoRA for DiT: {len(network.unet_loras)} modules")
    else:
        network.unet_loras = []
        network.unet_refts = []

    for lora in network.text_encoder_loras + network.unet_loras:
        lora.apply_to()
        network.add_module(lora.lora_name, lora)

    # ReFT wraps each selected DiT Block's forward, so the chain is:
    #   Block.__call__ -> ReFT.forward -> original Block.forward
    #   (inside which LoRA-wrapped Linears still fire normally).
    for reft in network.text_encoder_refts + network.unet_refts:
        reft.apply_to()
        network.add_module(reft.lora_name, reft)

    if apply_unet and network.register_injector is not None:
        network.register_injector.apply(unet)
