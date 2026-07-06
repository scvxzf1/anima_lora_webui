"""Optimizer param-group helpers for the LoRA-family network facade."""

import logging
import re


logger = logging.getLogger(__name__)


def set_loraplus_lr_ratio(
    network, loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio
):
    network.loraplus_lr_ratio = loraplus_lr_ratio
    network.loraplus_unet_lr_ratio = loraplus_unet_lr_ratio
    network.loraplus_text_encoder_lr_ratio = loraplus_text_encoder_lr_ratio

    logger.info(
        f"LoRA+ UNet LR Ratio: {network.loraplus_unet_lr_ratio or network.loraplus_lr_ratio}"
    )
    logger.info(
        "LoRA+ Text Encoder LR Ratio: "
        f"{network.loraplus_text_encoder_lr_ratio or network.loraplus_lr_ratio}"
    )


def _assemble_params(network, loras, lr, loraplus_ratio):
    param_groups = {"lora": {}, "plus": {}, "router": {}}
    reg_groups = {}
    reg_lrs_list = (
        list(network.cfg.reg_lrs.items()) if network.cfg.reg_lrs is not None else []
    )
    router_scale = float(network.cfg.router_lr_scale)
    # Chimera content-router multiplier stacks on router_scale. The
    # per-Linear "router.*" group below collects chimera's content router
    # params. Non-chimera runs keep the old multiplier of 1.0.
    content_router_scale = (
        float(network.cfg.content_router_lr_scale)
        if getattr(network.cfg, "use_chimera_hydra", False)
        else 1.0
    )
    router_lr_mult = router_scale * content_router_scale

    def _is_router_param(pname: str) -> bool:
        # named_parameters() yields names like "router.weight" without a
        # leading dot. Sigma features live inside router.weight now, so
        # there is a single path.
        return pname.startswith("router.")

    for lora in loras:
        matched_reg_lr = None
        for i, (regex_str, reg_lr) in enumerate(reg_lrs_list):
            if re.fullmatch(regex_str, lora.original_name):
                matched_reg_lr = (i, reg_lr)
                logger.info(
                    f"Module {lora.original_name} matched regex '{regex_str}' -> LR {reg_lr}"
                )
                break

        for name, param in lora.named_parameters():
            is_router = _is_router_param(name)
            if matched_reg_lr is not None:
                reg_idx, reg_lr = matched_reg_lr
                group_key = f"reg_lr_{reg_idx}"
                if group_key not in reg_groups:
                    reg_groups[group_key] = {
                        "lora": {},
                        "plus": {},
                        "router": {},
                        "lr": reg_lr,
                    }
                if is_router:
                    reg_groups[group_key]["router"][f"{lora.lora_name}.{name}"] = param
                elif loraplus_ratio is not None and (
                    "lora_up" in name
                    or "p_layer" in name
                    or "learned_source" in name
                ):
                    reg_groups[group_key]["plus"][f"{lora.lora_name}.{name}"] = param
                else:
                    reg_groups[group_key]["lora"][f"{lora.lora_name}.{name}"] = param
                continue

            if is_router:
                param_groups["router"][f"{lora.lora_name}.{name}"] = param
            elif loraplus_ratio is not None and (
                "lora_up" in name
                or "p_layer" in name
                or "learned_source" in name
            ):
                param_groups["plus"][f"{lora.lora_name}.{name}"] = param
            else:
                param_groups["lora"][f"{lora.lora_name}.{name}"] = param

    params = []
    descriptions = []
    for group_key, group in reg_groups.items():
        reg_lr = group["lr"]
        for key in ("lora", "plus", "router"):
            param_data = {"params": group[key].values()}
            if len(param_data["params"]) == 0:
                continue
            if key == "plus":
                param_data["lr"] = (
                    reg_lr * loraplus_ratio
                    if loraplus_ratio is not None
                    else reg_lr
                )
            elif key == "router":
                param_data["lr"] = reg_lr * router_lr_mult
            else:
                param_data["lr"] = reg_lr
            if param_data.get("lr", None) == 0 or param_data.get("lr", None) is None:
                logger.info("NO LR skipping!")
                continue
            params.append(param_data)
            desc = f"reg_lr_{group_key.split('_')[-1]}"
            descriptions.append(
                desc
                + (
                    " plus"
                    if key == "plus"
                    else (" router" if key == "router" else "")
                )
            )

    for key in param_groups.keys():
        param_data = {"params": param_groups[key].values()}
        if len(param_data["params"]) == 0:
            continue
        if lr is not None:
            if key == "plus":
                param_data["lr"] = lr * loraplus_ratio
            elif key == "router":
                param_data["lr"] = lr * router_lr_mult
            else:
                param_data["lr"] = lr
        if param_data.get("lr", None) == 0 or param_data.get("lr", None) is None:
            logger.info("NO LR skipping!")
            continue
        params.append(param_data)
        descriptions.append(
            "plus" if key == "plus" else ("router" if key == "router" else "")
        )
    return params, descriptions


def _append_module_groups(network, all_params, lr_descriptions, prefix, loras, lr, ratio):
    params, descriptions = _assemble_params(network, loras, lr, ratio)
    all_params.extend(params)
    lr_descriptions.extend(
        [prefix + (" " + desc if desc else "") for desc in descriptions]
    )


def prepare_lora_optimizer_params(network, text_encoder_lr, unet_lr, default_lr):
    if text_encoder_lr is None or (
        isinstance(text_encoder_lr, list) and len(text_encoder_lr) == 0
    ):
        text_encoder_lr = [default_lr]
    elif isinstance(text_encoder_lr, float) or isinstance(text_encoder_lr, int):
        text_encoder_lr = [float(text_encoder_lr)]
    elif len(text_encoder_lr) == 1:
        pass  # already a list with one element

    network.requires_grad_(True)

    all_params = []
    lr_descriptions = []

    if network.text_encoder_loras:
        loraplus_ratio = (
            network.loraplus_text_encoder_lr_ratio or network.loraplus_lr_ratio
        )
        te1_loras = [
            lora
            for lora in network.text_encoder_loras
            if lora.lora_name.startswith(network.LORA_PREFIX_TEXT_ENCODER)
        ]
        if len(te1_loras) > 0:
            logger.info(
                f"Text Encoder 1 (Qwen3): {len(te1_loras)} modules, LR {text_encoder_lr[0]}"
            )
            _append_module_groups(
                network,
                all_params,
                lr_descriptions,
                "textencoder 1",
                te1_loras,
                text_encoder_lr[0],
                loraplus_ratio,
            )

    if network.unet_loras:
        _append_module_groups(
            network,
            all_params,
            lr_descriptions,
            "unet",
            network.unet_loras,
            unet_lr if unet_lr is not None else default_lr,
            network.loraplus_unet_lr_ratio or network.loraplus_lr_ratio,
        )

    if network.text_encoder_refts:
        _append_module_groups(
            network,
            all_params,
            lr_descriptions,
            "reft textencoder",
            network.text_encoder_refts,
            text_encoder_lr[0],
            network.loraplus_text_encoder_lr_ratio or network.loraplus_lr_ratio,
        )

    if network.unet_refts:
        _append_module_groups(
            network,
            all_params,
            lr_descriptions,
            "reft unet",
            network.unet_refts,
            unet_lr if unet_lr is not None else default_lr,
            network.loraplus_unet_lr_ratio or network.loraplus_lr_ratio,
        )

    # HydraLoRA per-module routers are submodules of HydraLoRAModule instances,
    # so they are already captured by the unet_loras param group above.

    # GlobalRouter lives on the network, so the per-module loop misses it.
    if getattr(network, "global_router", None) is not None:
        gr_params = list(network.global_router.parameters())
        if len(gr_params) > 0:
            router_scale = float(network.cfg.router_lr_scale)
            base_lr = unet_lr if unet_lr is not None else default_lr
            if base_lr is None or base_lr == 0:
                logger.info("GlobalRouter: no base LR, skipping param group")
            else:
                gr_lr = float(base_lr) * router_scale
                all_params.append({"params": gr_params, "lr": gr_lr})
                lr_descriptions.append("global router")
                logger.info(
                    f"GlobalRouter param group: lr={gr_lr:.2e} "
                    f"({router_scale}x of unet_lr={base_lr})"
                )

    if getattr(network, "freq_router", None) is not None:
        fr_params = list(network.freq_router.parameters())
        if len(fr_params) > 0:
            router_scale = float(network.cfg.router_lr_scale)
            freq_scale = float(network.cfg.freq_router_lr_scale)
            base_lr = unet_lr if unet_lr is not None else default_lr
            if base_lr is None or base_lr == 0:
                logger.info("FreqRouter: no base LR, skipping param group")
            else:
                fr_lr = float(base_lr) * router_scale * freq_scale
                all_params.append({"params": fr_params, "lr": fr_lr})
                lr_descriptions.append("chimera freq router")
                logger.info(
                    f"ChimeraHydra FreqRouter param group: lr={fr_lr:.2e} "
                    f"({router_scale}x router_lr_scale x {freq_scale}x "
                    f"freq_router_lr_scale of unet_lr={base_lr})"
                )

    if getattr(network, "content_router", None) is not None:
        cr_params = list(network.content_router.parameters())
        if len(cr_params) > 0:
            router_scale = float(network.cfg.router_lr_scale)
            content_scale = float(network.cfg.content_router_lr_scale)
            base_lr = unet_lr if unet_lr is not None else default_lr
            if base_lr is None or base_lr == 0:
                logger.info("ContentRouter: no base LR, skipping param group")
            else:
                cr_lr = float(base_lr) * router_scale * content_scale
                all_params.append({"params": cr_params, "lr": cr_lr})
                lr_descriptions.append("chimera content router")
                logger.info(
                    f"ChimeraHydra ContentRouter param group: lr={cr_lr:.2e} "
                    f"({router_scale}x router_lr_scale x {content_scale}x "
                    f"content_router_lr_scale of unet_lr={base_lr})"
                )

    if network.register_injector is not None:
        base_lr = unet_lr if unet_lr is not None else default_lr
        if base_lr is None or base_lr == 0:
            logger.info("Register tokens: no base LR, skipping param group")
        else:
            reg_lr = float(base_lr) * float(network.cfg.register_lr_scale)
            all_params.append({"params": [network.register_tokens], "lr": reg_lr})
            lr_descriptions.append("register tokens")
            logger.info(
                f"Register-token param group: lr={reg_lr:.2e} "
                f"({network.cfg.register_lr_scale:g}x of unet_lr={base_lr})"
            )

    return all_params, lr_descriptions
