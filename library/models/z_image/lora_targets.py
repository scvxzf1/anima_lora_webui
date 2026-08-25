"""Conservative attention-only LoRA targets for Diffusers Z-Image."""

from __future__ import annotations


def z_image_target_kwargs() -> dict:
    return {
        "unet_target_replace_modules": ["ZImageTransformerBlock"],
        "text_encoder_target_replace_modules": [],
        "include_patterns": None,
        "exclude_patterns": [
            r".*\.feed_forward\..*",
            r".*\.adaLN_modulation\..*",
        ],
        "train_text_encoder": False,
    }


Z_IMAGE_LORA_INJECTION_POINTS = (
    "layers.N.attention.to_q",
    "layers.N.attention.to_k",
    "layers.N.attention.to_v",
    "layers.N.attention.to_out.0",
)
