"""Attention-only plain LoRA target specification."""


def qwen_image_2_1_target_kwargs() -> dict:
    return {
        "unet_target_replace_modules": ["QwenImage21TransformerBlock"],
        "text_encoder_target_replace_modules": [],
        "include_patterns": None,
        "exclude_patterns": [r".*\.img_mlp\..*", r".*\.img_mod\..*"],
        "train_text_encoder": False,
    }
