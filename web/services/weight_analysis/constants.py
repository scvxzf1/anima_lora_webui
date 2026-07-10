"""Shared constants for static weight analysis."""

from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
WEIGHT_EXTS = {".safetensors"}
MAX_ANALYSIS_WEIGHT_LIMIT = 500
MAX_METADATA_ITEMS = 80
MAX_UPLOAD_WEIGHT_BYTES = 512 * 1024 * 1024
TOP_LAYER_LIMIT = 20

STYLE_PRIORITY = {
    "mlp_layer1": 1.38,
    "cross_attn_k_proj": 1.34,
    "cross_attn_v_proj": 1.34,
    "self_attn_output_proj": 1.24,
    "self_attn_q_proj": 1.12,
    "self_attn_v_proj": 1.12,
    "cross_attn_output_proj": 1.08,
    "mlp_layer2": 0.86,
}
CHARACTER_PRIORITY = {
    "self_attn_q_proj": 1.28,
    "self_attn_v_proj": 1.26,
    "cross_attn_q_proj": 1.20,
    "cross_attn_k_proj": 1.14,
    "cross_attn_v_proj": 1.14,
    "self_attn_k_proj": 1.10,
    "self_attn_output_proj": 1.02,
    "cross_attn_output_proj": 0.98,
    "mlp_layer1": 0.86,
    "mlp_layer2": 0.82,
}
UNSUPPORTED_SPEC_TOKENS = (
    "hydra",
    "chimera",
    "fera",
    "moe",
    "reft",
    "vera",
    "postfix",
    "ip_adapter",
    "easycontrol",
    "soft_tokens",
)
UNSUPPORTED_KEY_FRAGMENTS = (
    ".lora_ups.",
    ".lora_downs.",
    ".lora_up_weight",
    ".lora_down_weight",
    ".router.",
    "freq_router.",
    "content_router.",
    ".s_p",
    ".s_q",
    "vera_lambda_",
)
BLOCK_RE = re.compile(r"(?:^|_)blocks_(?P<block>\d+)_(?P<component>.+)$")
LORA_SUFFIX = ".lora_down.weight"
LOHA_SUFFIX = ".hada_w1_a"
LOKR_SUFFIX = ".lokr_w1"
