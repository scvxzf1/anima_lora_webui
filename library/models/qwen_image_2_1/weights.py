"""Strict loading of the three Qwen Image 2.1 ComfyUI components."""

from __future__ import annotations

import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file


class QwenImage21CheckpointError(ValueError):
    pass


def _single_file(path: str, component: str) -> Path:
    file = Path(path).expanduser()
    if not file.is_file() or file.suffix != ".safetensors":
        raise QwenImage21CheckpointError(
            f"Qwen Image 2.1 {component} requires a local .safetensors component file: {path}"
        )
    return file


def _text_config():
    from transformers import Qwen3VLConfig

    return Qwen3VLConfig(
        text_config={
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_hidden_layers": 36,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 151936,
            "max_position_embeddings": 262144,
            "rope_theta": 5000000,
            "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"},
        },
        vision_config={
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "depth": 27,
            "num_heads": 16,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 4096,
            "deepstack_visual_indexes": [8, 16, 24],
        },
        tie_word_embeddings=False,
        image_token_id=151655,
        vision_start_token_id=151652,
        vision_end_token_id=151653,
    )


_RESIDUAL = {"0": "norm1", "2": "conv1", "3": "norm2", "6": "conv2"}


def _vae_key(key: str) -> str:
    head, dot, suffix = key.rpartition(".")
    if not dot:
        return key
    if head in {"conv1", "conv2"}:
        return {"conv1": "quant_conv", "conv2": "post_quant_conv"}[head] + dot + suffix
    for side in ("encoder", "decoder"):
        simple = {"conv1": "conv_in", "head.0": "norm_out", "head.2": "conv_out"}
        for source, target in simple.items():
            if head == f"{side}.{source}":
                return f"{side}.{target}.{suffix}"
        middle = re.fullmatch(rf"{side}\.middle\.(\d+)\.(.+)", head)
        if middle:
            index, tail = middle.groups()
            if index == "1":
                return f"{side}.mid_block.attentions.0.{tail}.{suffix}"
            residual = re.fullmatch(r"residual\.(\d+)", tail)
            if residual and residual.group(1) in _RESIDUAL:
                return f"{side}.mid_block.resnets.{int(index) // 2}.{_RESIDUAL[residual.group(1)]}.{suffix}"
        group = "downsamples" if side == "encoder" else "upsamples"
        block = re.fullmatch(rf"{side}\.{group}\.(\d+)\.{group}\.(\d+)\.(.+)", head)
        if not block:
            continue
        level, item, tail = block.groups()
        if side == "encoder":
            base = f"encoder.down_blocks.{level}"
            if tail.startswith("residual."):
                field = _RESIDUAL[tail.split(".")[1]]
                return f"{base}.resnets.{item}.{field}.{suffix}"
            if tail == "shortcut":
                return f"{base}.resnets.{item}.conv_shortcut.{suffix}"
            return f"{base}.downsampler.{tail}.{suffix}"
        base = f"decoder.up_blocks.{level}"
        if tail.startswith("residual."):
            field = _RESIDUAL[tail.split(".")[1]]
            return f"{base}.resnets.{item}.{field}.{suffix}"
        if tail == "shortcut":
            return f"{base}.resnets.{item}.conv_shortcut.{suffix}"
        return f"{base}.upsampler.{tail}.{suffix}"
    return key


def _text_key(key: str) -> str:
    if key.startswith("model.") and not key.startswith("model.visual."):
        return "model.language_model." + key[len("model.") :]
    return key


def _vae_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    return shape[:2] + shape[3:] if len(shape) == 5 and shape[2] == 1 else shape


def _validate_header(path: Path, model, key_map, *, component: str, shape_map=None) -> None:
    expected = model.state_dict()
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        actual: dict[str, tuple[int, ...]] = {}
        for source in handle.keys():
            target = key_map(source)
            if target in actual:
                raise QwenImage21CheckpointError(f"Duplicate mapped {component} key: {target}")
            shape = tuple(handle.get_slice(source).get_shape())
            actual[target] = shape_map(shape) if shape_map is not None else shape
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))[:5]
        extra = sorted(set(actual) - set(expected))[:5]
        raise QwenImage21CheckpointError(
            f"Qwen Image 2.1 {component} checkpoint keys mismatch: missing={missing}, extra={extra}"
        )
    mismatch = [key for key, shape in actual.items() if shape != tuple(expected[key].shape)]
    if mismatch:
        raise QwenImage21CheckpointError(f"Qwen Image 2.1 {component} shape mismatch: {mismatch[:5]}")


def _materialize_qwen_derived_tensors(model) -> None:
    """Rebuild non-checkpoint frequencies left meta by low-memory construction."""
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21Rope,
        QwenImage21TemporalTimesteps,
    )
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    for module in model.modules():
        inv_freq = getattr(module, "inv_freq", None)
        original_inv_freq = getattr(module, "original_inv_freq", None)
        if isinstance(inv_freq, torch.Tensor) and inv_freq.is_meta:
            compute_axial = getattr(module, "compute_axial_rope_parameters", None)
            if callable(compute_axial):
                inv_freq, attention_scaling = compute_axial(module.config, device="cpu")
            else:
                rope_type = getattr(module, "rope_type", "default")
                if rope_type == "default":
                    initialize_rope = module.compute_default_rope_parameters
                else:
                    initialize_rope = ROPE_INIT_FUNCTIONS[rope_type]
                inv_freq, attention_scaling = initialize_rope(module.config, device="cpu")
            module.inv_freq = inv_freq
            module.attention_scaling = attention_scaling
            original_inv_freq = getattr(module, "original_inv_freq", None)
        if (
            isinstance(original_inv_freq, torch.Tensor)
            and original_inv_freq.is_meta
            and isinstance(inv_freq, torch.Tensor)
            and not inv_freq.is_meta
        ):
            module.original_inv_freq = inv_freq.clone()

        freqs = getattr(module, "freqs", None)
        if isinstance(module, QwenImage21TemporalTimesteps) and freqs.is_meta:
            type(module).__init__(
                module,
                timestep_dim=module.timestep_dim,
                time_factor=module.time_factor,
            )
        elif isinstance(module, QwenImage21Rope) and any(freq.is_meta for freq in freqs):
            type(module).__init__(module, theta=module.theta, axes_dim=module.axes_dim)


def _load_mapped(path: str, model, key_map, *, component: str, device, shape_map=None):
    file = _single_file(path, component)
    _validate_header(file, model, key_map, component=component, shape_map=shape_map)
    state = {
        key_map(key): value.squeeze(2) if shape_map is not None and value.ndim == 5 else value
        for key, value in load_file(str(file), device="cpu").items()
    }
    model.load_state_dict(state, strict=True, assign=True)
    _materialize_qwen_derived_tensors(model)
    return model.to(device)


def load_qwen_image_2_1_text_encoder(path: str, *, dtype: torch.dtype, device="cpu"):
    from transformers import Qwen3VLForConditionalGeneration

    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(_text_config())
    model = _load_mapped(path, model, _text_key, component="text encoder", device=device)
    model.config.use_cache = False
    return model.to(dtype=dtype).eval().requires_grad_(False)


def load_qwen_image_2_1_vae(path: str, *, dtype: torch.dtype, device="cpu"):
    from diffusers import AutoencoderKLQwenImage21

    with torch.device("meta"):
        model = AutoencoderKLQwenImage21()
    model = _load_mapped(path, model, _vae_key, component="VAE", device=device, shape_map=_vae_shape)
    return model.to(dtype=dtype).eval().requires_grad_(False)


def load_qwen_image_2_1_transformer(path: str, *, dtype: torch.dtype, device="cpu"):
    from diffusers import QwenImage21Transformer2DModel
    from diffusers.loaders.single_file_utils import convert_qwen_image21_transformer_checkpoint_to_diffusers

    file = _single_file(path, "transformer")
    with torch.device("meta"):
        model = QwenImage21Transformer2DModel()
    with safe_open(str(file), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        model_state = model.state_dict()
        expected = set(model_state)
        packed = {f"transformer_blocks.{i}.img_mlp.gate_up.weight" for i in range(32)}
        expanded = {
            f"transformer_blocks.{i}.img_mlp.{part}.weight"
            for i in range(32) for part in ("gate_layer", "proj")
        }
        if keys - packed != expected - expanded:
            raise QwenImage21CheckpointError("Qwen Image 2.1 transformer keys do not match the 32-block checkpoint")
        if keys & packed != packed:
            raise QwenImage21CheckpointError("Qwen Image 2.1 transformer lacks packed MLP weights")
        for key in keys - packed:
            if tuple(handle.get_slice(key).get_shape()) != tuple(model_state[key].shape):
                raise QwenImage21CheckpointError(f"Qwen Image 2.1 transformer shape mismatch: {key}")
        for key in packed:
            gate_key = key.replace("gate_up", "gate_layer")
            gate_shape = tuple(model_state[gate_key].shape)
            if tuple(handle.get_slice(key).get_shape()) != (2 * gate_shape[0], *gate_shape[1:]):
                raise QwenImage21CheckpointError(f"Qwen Image 2.1 packed MLP shape mismatch: {key}")
    state = convert_qwen_image21_transformer_checkpoint_to_diffusers(load_file(str(file), device="cpu"))
    if set(state) != expected:
        raise QwenImage21CheckpointError("Converted Qwen Image 2.1 transformer keys do not match Diffusers")
    model.load_state_dict(state, strict=True, assign=True)
    _materialize_qwen_derived_tensors(model)
    return model.to(device=device, dtype=dtype)
