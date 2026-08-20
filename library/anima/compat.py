"""Compatibility boundaries for expanded Anima checkpoints and adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from library.anima.checkpoint import AnimaCheckpointLayout

ANIMA29_PREVIEW_V1_SHA256 = (
    "0b3020d1b906155f7eb30667622723e87160632c8c7a5f1c93bdce685f2a346d"
)
ADAPTER_ARCH_KEYS = (
    "ss_anima_arch",
    "ss_anima_num_blocks",
    "ss_anima_model_channels",
)


@dataclass(frozen=True)
class AnimaCompatibility:
    supported: bool
    profile: str
    blockers: tuple[str, ...] = ()


def _get(config: Any, key: str, default: Any = None) -> Any:
    return (
        config.get(key, default)
        if isinstance(config, Mapping)
        else getattr(config, key, default)
    )


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _network_options(config: Any) -> dict[str, Any]:
    options = dict(config) if isinstance(config, Mapping) else dict(vars(config))
    for item in _get(config, "network_args", None) or []:
        if isinstance(item, str) and "=" in item:
            key, value = item.split("=", 1)
            options[key] = value
    return options


def classify_anima_training(config: Any) -> AnimaCompatibility:
    options = _network_options(config)
    blockers: list[str] = []
    module = str(_get(config, "network_module", "") or "").strip()
    if module != "networks.lora_anima":
        blockers.append(f"network_module={module or '<unset>'}")

    unsupported_flags = {
        "use_repa": "REPA",
        "use_lokr": "LoKr",
        "use_glokr": "GLoKr",
        "use_loha": "LoHa",
        "use_dylora": "DyLoRA",
        "use_ve": "VeRA",
        "use_chimera_hydra": "ChimeraHydra",
        "use_controlnet": "ControlNet",
        "use_easycontrol": "EasyControl",
        "use_byg": "BYG",
        "pgraft": "P-GRAFT",
        "train_llm_adapter": "train_llm_adapter",
    }
    for key, label in unsupported_flags.items():
        if _bool(options.get(key, _get(config, key, False))):
            blockers.append(label)

    if not _bool(options.get("network_train_unet_only", True)):
        blockers.append("text encoder training")
    if float(_get(config, "vr_loss_weight", 0.0) or 0.0) > 0:
        blockers.append("VR loss")
    if float(options.get("channel_scaling_alpha", 0.0) or 0.0) != 0.0:
        blockers.append("channel scaling")
    base_compute = str(_get(config, "base_compute", "bf16") or "bf16").lower()
    if base_compute not in {"bf16", "fp16", "none", "off", ""}:
        blockers.append(f"ConvRot/base_compute={base_compute}")
    if str(options.get("use_moe_style", False)).lower() not in {
        "",
        "0",
        "false",
        "none",
    }:
        blockers.append("Hydra/MoE routing")
    if str(options.get("router_source", "none") or "none").lower() != "none":
        blockers.append("router_source")

    timestep_mask = _bool(options.get("use_timestep_mask", False))
    use_ortho = _bool(options.get("use_ortho", False))
    down_init = str(options.get("down_init", "kaiming") or "kaiming").lower()
    if not timestep_mask and not use_ortho and down_init == "kaiming":
        profile = "plain_lora"
    elif timestep_mask and use_ortho and down_init == "kaiming":
        profile = "tlora_ortho"
    else:
        profile = "unsupported"
        blockers.append("profile must be Plain LoRA or T-LoRA + OrthoLoRA")
    return AnimaCompatibility(not blockers, profile, tuple(dict.fromkeys(blockers)))


def require_training_compatibility(config: Any, layout: AnimaCheckpointLayout) -> str:
    if layout.num_blocks == 28:
        return "legacy_28"
    result = classify_anima_training(config)
    if not result.supported:
        raise ValueError(
            f"{layout.variant} ({layout.num_blocks} blocks) does not support this "
            f"training config: {', '.join(result.blockers)}"
        )
    return result.profile


def adapter_identity_metadata(
    layout: AnimaCheckpointLayout, base_sha256: str | None = None
) -> dict[str, str]:
    metadata = {
        "ss_anima_arch": layout.arch,
        "ss_anima_num_blocks": str(layout.num_blocks),
        "ss_anima_model_channels": str(layout.model_channels),
    }
    if base_sha256:
        metadata["ss_new_sd_model_hash"] = base_sha256
    return metadata


def validate_adapter_metadata(metadata: Mapping[str, str], unet: Any) -> None:
    layout = getattr(unet, "_anima_checkpoint_layout", None)
    if layout is None:
        return
    present = [key in metadata for key in ADAPTER_ARCH_KEYS]
    if any(present) and not all(present):
        missing = [key for key, exists in zip(ADAPTER_ARCH_KEYS, present) if not exists]
        raise ValueError(
            f"Adapter has incomplete Anima architecture metadata: {missing}"
        )
    if not any(present):
        if layout.num_blocks == 40:
            raise ValueError(
                "40-block Anima requires adapter architecture metadata; "
                "legacy/unstamped adapters are not compatible"
            )
        return
    expected = adapter_identity_metadata(
        layout, getattr(unet, "_anima_base_sha256", None)
    )
    mismatches = [
        f"{key}={metadata.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if key in ADAPTER_ARCH_KEYS and str(metadata.get(key)) != value
    ]
    base_hash = expected.get("ss_new_sd_model_hash")
    stamped_hash = metadata.get("ss_new_sd_model_hash")
    if base_hash and stamped_hash and stamped_hash.lower() != base_hash.lower():
        mismatches.append("ss_new_sd_model_hash does not match the selected base model")
    if mismatches:
        raise ValueError(
            "Adapter is incompatible with the selected Anima model: "
            + "; ".join(mismatches)
        )
