"""Official Diffusers and ComfyUI single-file loading for Z-Image."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from library.models.latent_space import Z_IMAGE_F8C16_P2


CONFIG_ROOT = Path(__file__).with_name("configs")
_SAFETENSOR_SUFFIX = ".safetensors"
Z_IMAGE_TEXT_WIDTH = 2560


class ZImageCheckpointError(ValueError):
    """Raised when a single-file checkpoint is not a supported Z-Image component."""


def _require_config_value(model, key: str, expected, *, component: str) -> None:
    actual = getattr(getattr(model, "config", None), key, None)
    matches = (
        abs(float(actual) - expected) <= 1e-7
        if isinstance(expected, float) and actual is not None
        else actual == expected
    )
    if not matches:
        raise ZImageCheckpointError(
            f"Z-Image {component} config mismatch: {key}={actual!r}, "
            f"expected {expected!r}"
        )


def validate_z_image_component_config(model, component: str) -> None:
    """Validate only the cross-component geometry required by training."""
    if component == "text_encoder":
        _require_config_value(
            model, "hidden_size", Z_IMAGE_TEXT_WIDTH, component=component
        )
        return
    if component == "vae":
        _require_config_value(
            model,
            "latent_channels",
            Z_IMAGE_F8C16_P2.latent_channels,
            component=component,
        )
        _require_config_value(
            model,
            "shift_factor",
            float(Z_IMAGE_F8C16_P2.shift_factor),
            component=component,
        )
        _require_config_value(
            model,
            "scaling_factor",
            float(Z_IMAGE_F8C16_P2.scaling_factor),
            component=component,
        )
        return
    if component == "transformer":
        _require_config_value(
            model,
            "in_channels",
            Z_IMAGE_F8C16_P2.latent_channels,
            component=component,
        )
        _require_config_value(
            model, "cap_feat_dim", Z_IMAGE_TEXT_WIDTH, component=component
        )
        return
    raise ValueError(f"Unknown Z-Image component: {component}")


def _component_location(path: str, component: str) -> tuple[str, str | None]:
    root = Path(path).expanduser()
    if (root / component / "config.json").is_file():
        return str(root), component
    return str(root), None


def _from_pretrained(model_cls, path: str, component: str, dtype: torch.dtype):
    root, subfolder = _component_location(path, component)
    kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    if subfolder is not None:
        kwargs["subfolder"] = subfolder
    return model_cls.from_pretrained(root, **kwargs)


def _single_file(path: str) -> Path | None:
    candidate = Path(path).expanduser()
    if candidate.is_file() and candidate.suffix.lower() == _SAFETENSOR_SUFFIX:
        return candidate
    return None


def _safetensor_keys(path: Path) -> set[str]:
    from safetensors import safe_open

    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            return set(handle.keys())
    except Exception as exc:
        raise ZImageCheckpointError(
            f"Cannot read Z-Image safetensors header: {path}: {exc}"
        ) from exc


def _count_indexed_blocks(keys: Iterable[str], prefix: str) -> int:
    indices: set[int] = set()
    for key in keys:
        if not key.startswith(prefix):
            continue
        value = key[len(prefix) :].split(".", 1)[0]
        if value.isdigit():
            indices.add(int(value))
    return len(indices)


def validate_z_image_single_file(path: str, component: str) -> None:
    """Validate architecture markers without materializing checkpoint tensors."""
    candidate = _single_file(path)
    if candidate is None:
        raise ZImageCheckpointError(
            f"Z-Image {component} single-file checkpoint must be a .safetensors file"
        )
    keys = _safetensor_keys(candidate)
    if component == "transformer":
        required = {
            "cap_embedder.1.weight",
            "noise_refiner.0.attention.qkv.weight",
            "context_refiner.0.attention.qkv.weight",
            "layers.0.attention.qkv.weight",
        }
        layer_count = _count_indexed_blocks(keys, "layers.")
        has_final_layer = bool(
            {
                "final_layer.linear.weight",
                "all_final_layer.2-1.linear.weight",
            }
            & keys
        )
        if not required <= keys or not has_final_layer or layer_count != 30:
            raise ZImageCheckpointError(
                "Z-Image transformer checkpoint must use the official 30-layer "
                "Diffusers/ComfyUI architecture"
            )
        return
    if component == "text_encoder":
        required = {
            "model.embed_tokens.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.35.self_attn.q_proj.weight",
            "model.norm.weight",
        }
        layer_count = _count_indexed_blocks(keys, "model.layers.")
        if not required <= keys or layer_count != 36:
            raise ZImageCheckpointError(
                "Z-Image text encoder checkpoint must be the official Qwen3-4B model"
            )
        return
    if component == "vae":
        required = {
            "encoder.conv_in.weight",
            "encoder.mid.attn_1.q.weight",
            "decoder.mid.attn_1.q.weight",
            "decoder.conv_out.weight",
        }
        if not required <= keys:
            raise ZImageCheckpointError(
                "Z-Image VAE checkpoint must use the official Flux Autoencoder layout"
            )
        return
    raise ValueError(f"Unknown Z-Image component: {component}")


def _from_single_file(model_cls, path: Path, component: str, dtype: torch.dtype):
    validate_z_image_single_file(str(path), component)
    return model_cls.from_single_file(
        str(path),
        config=str(CONFIG_ROOT),
        subfolder=component,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )


def load_z_image_text_encoder(path: str, *, dtype: torch.dtype, device="cpu"):
    from transformers import Qwen3Config, Qwen3Model

    checkpoint = _single_file(path)
    if checkpoint is None:
        model = _from_pretrained(Qwen3Model, path, "text_encoder", dtype)
    else:
        from safetensors.torch import load_file

        validate_z_image_single_file(str(checkpoint), "text_encoder")
        state_dict = load_file(str(checkpoint), device="cpu")
        config = Qwen3Config.from_pretrained(
            str(CONFIG_ROOT / "text_encoder"), local_files_only=True
        )
        model = Qwen3Model.from_pretrained(
            None,
            config=config,
            state_dict=state_dict,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    validate_z_image_component_config(model, "text_encoder")
    model.config.use_cache = False
    return model.to(device).eval().requires_grad_(False)


def load_z_image_vae(path: str, *, dtype: torch.dtype, device="cpu"):
    from diffusers import AutoencoderKL

    checkpoint = _single_file(path)
    model = (
        _from_single_file(AutoencoderKL, checkpoint, "vae", dtype)
        if checkpoint is not None
        else _from_pretrained(AutoencoderKL, path, "vae", dtype)
    )
    validate_z_image_component_config(model, "vae")
    return model.to(device).eval().requires_grad_(False)


def load_z_image_transformer(path: str, *, dtype: torch.dtype, device="cpu"):
    from diffusers import ZImageTransformer2DModel

    checkpoint = _single_file(path)
    model = (
        _from_single_file(ZImageTransformer2DModel, checkpoint, "transformer", dtype)
        if checkpoint is not None
        else _from_pretrained(ZImageTransformer2DModel, path, "transformer", dtype)
    )
    validate_z_image_component_config(model, "transformer")
    return model.to(device)


def resolve_z_image_tokenizer_path(path: str) -> str:
    """Resolve an official pipeline root, component dir, or sibling tokenizer."""
    candidate = Path(path).expanduser()
    if candidate.is_file():
        candidate = candidate.parent
    if (candidate / "tokenizer" / "tokenizer_config.json").is_file():
        return str(candidate / "tokenizer")
    if (candidate / "tokenizer_config.json").is_file():
        return str(candidate)
    for sibling_name in ("tokenizer", "qwen25_tokenizer"):
        sibling = candidate.parent / sibling_name
        if (sibling / "tokenizer_config.json").is_file():
            return str(sibling)
    for ancestor in (candidate, *candidate.parents):
        if ancestor.name != "models":
            continue
        comfy_tokenizer = (
            ancestor.parent / "comfy" / "text_encoders" / "qwen25_tokenizer"
        )
        if (comfy_tokenizer / "tokenizer_config.json").is_file():
            return str(comfy_tokenizer)
    return str(candidate)
