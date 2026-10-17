"""MOSS text-cache identity and local component resolution."""

from pathlib import Path

MOSS_TEXT_CACHE_SCHEMA = "moss_image_te_v1"
MOSS_TOKENIZER_VERSION = "qwen3_5_qwen2_v1"
MOSS_TEMPLATE_VERSION = "user_generation_thinking_v1"
MOSS_MAX_LENGTH = 4096
MOSS_HIDDEN_SIZE = 4096


def validate_quantization(quantization: str) -> str:
    if quantization not in {"nf4", "bf16"}:
        raise ValueError("moss_te_quantization must be 'nf4' or 'bf16'")
    return quantization


def text_cache_metadata(quantization: str) -> dict[str, str]:
    return {
        "model_family": "moss_image",
        "cache_schema": MOSS_TEXT_CACHE_SCHEMA,
        "quantization": validate_quantization(quantization),
        "compute_dtype": "bfloat16",
        "tokenizer_version": MOSS_TOKENIZER_VERSION,
        "template_version": MOSS_TEMPLATE_VERSION,
    }


def validate_cache_metadata(metadata, quantization: str, path: str) -> None:
    for key, expected in text_cache_metadata(quantization).items():
        if (metadata or {}).get(key) != expected:
            raise ValueError(f"Incompatible MOSS text cache {path}: {key} must be {expected!r}")


def resolve_text_encoder_path(path: str) -> str:
    root = Path(path).expanduser()
    component = root / "text_encoder" if (root / "text_encoder/config.json").is_file() else root
    if not (component / "config.json").is_file():
        raise FileNotFoundError(f"MOSS requires a local Transformers text_encoder directory: {path}")
    return str(component)


def resolve_tokenizer_path(path: str) -> str:
    root = Path(path).expanduser()
    candidates = (root / "tokenizer", root, root.parent / "tokenizer")
    for candidate in candidates:
        if (candidate / "tokenizer_config.json").is_file():
            return str(candidate)
    raise FileNotFoundError(f"No local MOSS tokenizer at {root} or its adjacent tokenizer directory")
