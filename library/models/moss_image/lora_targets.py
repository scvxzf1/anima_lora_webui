"""Main MOSS attention projections only; include_patterns is an override."""


def moss_image_target_kwargs() -> dict:
    return {
        "unet_target_replace_modules": ["_Block"],
        "text_encoder_target_replace_modules": [],
        "include_patterns": None,
        # Exclude everything except the four exact main-block projections.
        "exclude_patterns": [r"(?!layers\.\d+\.proj\.[qkvo]$).*"],
        "train_text_encoder": False,
    }


MOSS_IMAGE_LORA_INJECTION_POINTS = tuple(
    f"layers.N.proj.{name}" for name in ("q", "k", "v", "o")
)
