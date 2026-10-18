"""MOSS text-cache identity and local component resolution."""

from pathlib import Path

# v2 adds the token cap to the cache identity: MOSS stores real-length states,
# so a cache built at one max_token_length must not be read by another. v1
# caches carry no cap and stay readable (legacy tolerance); they are simply not
# proof of the cap they were built with, so rebuild for a v2 stamp.
MOSS_TEXT_CACHE_SCHEMA = "moss_image_te_v2"
MOSS_LEGACY_TEXT_CACHE_SCHEMAS = ("moss_image_te_v1",)
MOSS_TEXT_TOKEN_CAP_KEY = "max_text_tokens"
MOSS_TOKENIZER_VERSION = "qwen3_5_qwen2_v1"
MOSS_TEMPLATE_VERSION = "user_generation_thinking_v1"
MOSS_MAX_LENGTH = 4096
MOSS_HIDDEN_SIZE = 4096


def validate_quantization(quantization: str) -> str:
    if quantization not in {"nf4", "bf16"}:
        raise ValueError("moss_te_quantization must be 'nf4' or 'bf16'")
    return quantization


def resolve_text_token_cap(value) -> int:
    """Token cap that participates in the MOSS text cache identity."""
    if value is None or value == "":
        value = MOSS_MAX_LENGTH
    try:
        cap = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MOSS max_token_length must be an integer, got {value!r}") from exc
    if not 0 < cap <= MOSS_MAX_LENGTH:
        raise ValueError(f"MOSS max_token_length must be between 1 and {MOSS_MAX_LENGTH}")
    return cap


def text_cache_metadata(quantization: str,
                        max_text_tokens=MOSS_MAX_LENGTH) -> dict[str, str]:
    return {
        "model_family": "moss_image",
        "cache_schema": MOSS_TEXT_CACHE_SCHEMA,
        "quantization": validate_quantization(quantization),
        "compute_dtype": "bfloat16",
        "tokenizer_version": MOSS_TOKENIZER_VERSION,
        "template_version": MOSS_TEMPLATE_VERSION,
        MOSS_TEXT_TOKEN_CAP_KEY: str(resolve_text_token_cap(max_text_tokens)),
    }


def _expected_metadata(metadata, quantization: str, max_text_tokens) -> dict[str, str]:
    schema = (metadata or {}).get("cache_schema")
    if schema in MOSS_LEGACY_TEXT_CACHE_SCHEMAS:
        legacy = text_cache_metadata(quantization)
        legacy.pop(MOSS_TEXT_TOKEN_CAP_KEY)
        legacy["cache_schema"] = schema
        return legacy
    return text_cache_metadata(quantization, max_text_tokens)


def validate_cache_metadata(metadata, quantization: str, path: str,
                            max_text_tokens=MOSS_MAX_LENGTH) -> None:
    for key, expected in _expected_metadata(metadata, quantization, max_text_tokens).items():
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
