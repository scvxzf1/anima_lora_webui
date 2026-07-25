"""Static metadata for WebUI configuration files and GUI helpers."""

from __future__ import annotations

from web.services.config.form_metadata import BASIC_FIELDS, FIELD_HELP, FORM_GROUPS

from library.preprocess.captions import (
    CAPTION_SOURCE_AUTO,
    CAPTION_SOURCE_CAPTIONS_JSON,
    CAPTION_SOURCE_JSON,
    CAPTION_SOURCE_TXT,
)

SUPPORTED_TRAINING_SAMPLE_SAMPLERS = frozenset({"euler", "er_sde", "lcm"})
LEGACY_TRAINING_SAMPLE_SAMPLERS = frozenset({
    "ddim",
    "pndm",
    "lms",
    "euler_a",
    "heun",
    "dpm_2",
    "dpm_2_a",
    "dpmsolver",
    "dpmsolver++",
    "dpmsingle",
    "k_lms",
    "k_euler",
    "k_euler_a",
    "k_dpm_2",
    "k_dpm_2_a",
})

PREPROCESS_ENV_CHECK_KEY = "preprocess_environment"
PREPROCESS_ENV_REQUIRED_FILES = (
    "tasks.py",
    "library/__init__.py",
    "library/preprocess/__init__.py",
    "scripts/__init__.py",
    "scripts/tasks/__init__.py",
    "scripts/tasks/preprocess.py",
    "scripts/preprocess/resize_images.py",
    "scripts/preprocess/cache_latents.py",
    "scripts/preprocess/cache_text_embeddings.py",
)

UI_ONLY_CONFIG_FIELDS = {
    "dataset_config_picker",
    "precision_preference",
}
SPD_NESTED_PATCH_FIELDS = {
    "channel_scaling_alpha": ("network", "channel_scaling_alpha"),
    "weight_decay": ("optim", "weight_decay"),
}
RETIRED_TOP_LEVEL_CONFIG_FIELDS = {
    "per_channel_scaling",
    "repa_layer",
    "repa_lr_scale",
    "repa_weight",
    "trim_crossattn_kv",
    "use_hydra",
    "use_repa",
    "use_sigma_router",
    "use_fei_router",
}

DATASET_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})
DATASET_CAPTION_EXTS = (".txt", ".json", ".caption")
DATASET_PREVIEW_LIMIT = 120
DATASET_CAPTION_MAX_CHARS = 20000
DEFAULT_RESIZED_IMAGE_DIR = "post_image_dataset/resized"
DEFAULT_LORA_CACHE_DIR = "post_image_dataset/lora"
DATASET_SETTING_KEYS = frozenset({
    "resolution",
    "enable_bucket",
    "min_bucket_reso",
    "max_bucket_reso",
    "bucket_reso_steps",
    "bucket_no_upscale",
    "validation_split",
    "validation_split_num",
    "validation_seed",
    "prior_loss_weight",
    "prefer_json_caption",
    "caption_extension",
    "caption_source_mode",
})
PREPROCESS_DATASET_SETTING_ORDER = (
    "resolution",
    "enable_bucket",
    "min_bucket_reso",
    "max_bucket_reso",
    "bucket_reso_steps",
    "bucket_no_upscale",
)
PREPROCESS_DATASET_SETTING_KEYS = frozenset(PREPROCESS_DATASET_SETTING_ORDER)
RUNTIME_PREPROCESS_ATTR_KEY = "preprocess"
NL_TAG_MIX_ATTR_KEY = "nl_tag_mix"
TRIGGER_CLONE_ATTR_KEY = "trigger_clone"
DEFAULT_NL_TAG_MIX_TAG_RATIO = 0.7
NL_TAG_MIX_CLASSIFICATION_METHOD = "caption_text_v1"
CAPTION_SOURCE_MODE_LABELS = {
    CAPTION_SOURCE_AUTO: "自动识别",
    CAPTION_SOURCE_TXT: "sd-scripts .txt",
    CAPTION_SOURCE_JSON: "AnimaLoraToolkit .json",
    CAPTION_SOURCE_CAPTIONS_JSON: "DiffPipeForge captions.json",
}

OUTPUT_RUN_CONFIG_FILES = {
    "original": ("config.original.toml", "原始配置"),
    "runtime": ("config.runtime.toml", "运行时配置"),
    "dataset": ("dataset.runtime.toml", "数据集配置"),
}

SYSTEM_PRESET_FILES = frozenset({
    "configs/base.toml",
    "configs/presets.toml",
})
SYSTEM_DATASET_PRESET_FILES = frozenset({
    "configs/datasets/easycontrol.toml",
    "configs/datasets/ip_adapter.toml",
})
HIDDEN_DATASET_PRESET_FILES = frozenset({
    "configs/datasets/easycontrol.toml",
    "configs/datasets/ip_adapter.toml",
})
HIDDEN_CONFIG_FILES = frozenset({
    "configs/gui-methods/postfix_ortho_cond.toml",
    "configs/methods/postfix.toml",
})
SYSTEM_PRESET_PREFIXES = ("configs/methods/", "configs/gui-methods/")
SYSTEM_MANAGED_FILES = frozenset({
    "configs/web-file-groups.toml",
    "configs/web-user-locks.toml",
})

CONFIG_FILE_LABELS_ZH = {
    "configs/base.toml": "基础公共配置",
    "configs/presets.toml": "训练预设集合",
    "configs/web-file-groups.toml": "Web 配置分组表",
    "configs/datasets/easycontrol.toml": "EasyControl 数据集蓝图",
    "configs/datasets/ip_adapter.toml": "IP-Adapter 数据集蓝图",
    "configs/gui-methods/chimera_hydra.toml": "Chimera Hydra 训练变体",
    "configs/gui-methods/easycontrol.toml": "EasyControl 训练变体",
    "configs/gui-methods/glora.toml": "GLoRA 训练变体",
    "configs/gui-methods/hydralora-8gb.toml": "HydraLoRA 低显存变体",
    "configs/gui-methods/hydralora.toml": "HydraLoRA 训练变体",
    "configs/gui-methods/ip_adapter.toml": "IP-Adapter 训练变体",
    "configs/gui-methods/loha.toml": "LoHa 训练变体",
    "configs/gui-methods/lokr.toml": "LoKr 训练变体",
    "configs/gui-methods/lora-8gb.toml": "LoRA 低显存变体",
    "configs/gui-methods/lora-convrot-vram.toml": "LoRA + ConvRot W8A16 显存档（实验）",
    "configs/gui-methods/lora.toml": "LoRA 标准训练变体",
    "configs/gui-methods/reft.toml": "ReFT 训练变体",
    "configs/gui-methods/soft_tokens.toml": "Soft Tokens 训练变体",
    "configs/gui-methods/tlora-8gb.toml": "T-LoRA 低显存变体",
    "configs/gui-methods/tlora.toml": "T-LoRA 训练变体",
    "configs/gui-methods/tlora_ortho_reft.toml": "T-LoRA + Ortho + ReFT 组合变体",
    "configs/gui-methods/vera.toml": "VeRA 训练变体",
    "configs/methods/chimera.toml": "Chimera 内置方法配置",
    "configs/methods/easycontrol.toml": "EasyControl 内置方法配置",
    "configs/methods/ip_adapter.toml": "IP-Adapter 内置方法配置",
    "configs/methods/lora.toml": "LoRA 内置方法配置",
    "configs/methods/soft_tokens.toml": "Soft Tokens 内置方法配置",
    "configs/methods/spd.toml": "SPD 实验配置",
    "configs/methods/turbo.toml": "Turbo 内置方法配置",
}

SYSTEM_CONFIG_GROUP_IDS = frozenset({
    "web_config",
    "presets",
    "methods",
    "gui_methods",
    "rokkotsu_goddess",
    "imported",
    "datasets",
})
FIXED_SYSTEM_CONFIG_GROUP_IDS = frozenset({
    "web_config",
    "presets",
    "methods",
    "gui_methods",
})
FILE_MOVE_TARGET_GROUPS = frozenset({
    "imported",
    "rokkotsu_goddess",
    "datasets",
})
USER_LOCKABLE_GROUPS = frozenset({
    "imported",
    "rokkotsu_goddess",
    "datasets",
})


def get_field_help() -> dict[str, dict[str, str]]:
    return FIELD_HELP


def get_groups() -> dict[str, list[str]]:
    return {
        "groups": {name: sorted(fields) for name, fields in FORM_GROUPS.items()},
        "basic": sorted(BASIC_FIELDS),
    }


__all__ = [
    "CAPTION_SOURCE_AUTO",
    "CAPTION_SOURCE_CAPTIONS_JSON",
    "CAPTION_SOURCE_JSON",
    "CAPTION_SOURCE_MODE_LABELS",
    "CAPTION_SOURCE_TXT",
    "CONFIG_FILE_LABELS_ZH",
    "DATASET_CAPTION_EXTS",
    "DATASET_CAPTION_MAX_CHARS",
    "DATASET_IMAGE_EXTS",
    "DATASET_PREVIEW_LIMIT",
    "DATASET_SETTING_KEYS",
    "DEFAULT_LORA_CACHE_DIR",
    "DEFAULT_NL_TAG_MIX_TAG_RATIO",
    "DEFAULT_RESIZED_IMAGE_DIR",
    "FILE_MOVE_TARGET_GROUPS",
    "FIXED_SYSTEM_CONFIG_GROUP_IDS",
    "HIDDEN_CONFIG_FILES",
    "HIDDEN_DATASET_PRESET_FILES",
    "LEGACY_TRAINING_SAMPLE_SAMPLERS",
    "NL_TAG_MIX_ATTR_KEY",
    "NL_TAG_MIX_CLASSIFICATION_METHOD",
    "OUTPUT_RUN_CONFIG_FILES",
    "PREPROCESS_DATASET_SETTING_KEYS",
    "PREPROCESS_DATASET_SETTING_ORDER",
    "PREPROCESS_ENV_CHECK_KEY",
    "PREPROCESS_ENV_REQUIRED_FILES",
    "RETIRED_TOP_LEVEL_CONFIG_FIELDS",
    "RUNTIME_PREPROCESS_ATTR_KEY",
    "SPD_NESTED_PATCH_FIELDS",
    "SUPPORTED_TRAINING_SAMPLE_SAMPLERS",
    "SYSTEM_CONFIG_GROUP_IDS",
    "SYSTEM_DATASET_PRESET_FILES",
    "SYSTEM_MANAGED_FILES",
    "SYSTEM_PRESET_FILES",
    "SYSTEM_PRESET_PREFIXES",
    "TRIGGER_CLONE_ATTR_KEY",
    "UI_ONLY_CONFIG_FIELDS",
    "USER_LOCKABLE_GROUPS",
    "get_field_help",
    "get_groups",
]
