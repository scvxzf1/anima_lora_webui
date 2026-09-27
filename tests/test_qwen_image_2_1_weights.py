"""Low-memory Qwen checkpoint construction keeps derived frequencies usable."""

from __future__ import annotations

import torch

from library.models.qwen_image_2_1.weights import (
    _materialize_qwen_derived_tensors,
    _text_config,
)


def test_qwen_text_encoder_meta_rope_buffers_are_rebuilt():
    from transformers import Qwen3VLForConditionalGeneration

    with torch.device("meta"):
        model = Qwen3VLForConditionalGeneration(_text_config())

    _materialize_qwen_derived_tensors(model)

    ropes = [module for module in model.modules() if hasattr(module, "inv_freq")]
    assert len(ropes) == 2
    assert all(not module.inv_freq.is_meta for module in ropes)
    assert all(not module.original_inv_freq.is_meta for module in ropes)
    assert all(torch.isfinite(module.inv_freq).all() for module in ropes)


def test_qwen_dit_meta_frequency_tensors_are_rebuilt():
    from diffusers import QwenImage21Transformer2DModel

    with torch.device("meta"):
        model = QwenImage21Transformer2DModel()

    _materialize_qwen_derived_tensors(model)

    assert not model.time_text_embed.time_proj.freqs.is_meta
    assert all(not freq.is_meta for freq in model.pos_embed.freqs)
    assert all(torch.isfinite(freq.real).all() for freq in model.pos_embed.freqs)
    assert all(torch.isfinite(freq.imag).all() for freq in model.pos_embed.freqs)
