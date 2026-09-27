from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from library.models.qwen_image_2_1 import attention_backend


def test_flash_routes_causal_text_to_native_and_image_to_varlen(monkeypatch) -> None:
    from diffusers import QwenImage21Transformer2DModel

    real_api = attention_backend._load_diffusers_attention_api()
    routed = []

    def native_only_dispatch(query, key, value, **kwargs):
        requested_backend = kwargs["backend"]
        mask = kwargs.get("attn_mask")
        routed.append((requested_backend.value, mask))
        kwargs["backend"] = real_api.backend_names.NATIVE
        return real_api.dispatch_attention(query, key, value, **kwargs)

    def native_only_flash(query, key, value, *, mask):
        routed.append(("flash_varlen", mask))
        return real_api.dispatch_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=0.0,
            backend=real_api.backend_names.NATIVE,
        )

    test_api = SimpleNamespace(
        backend_names=real_api.backend_names,
        processor_type=real_api.processor_type,
        prepare_qkv=real_api.prepare_qkv,
        dispatch_attention=native_only_dispatch,
    )
    monkeypatch.setattr(attention_backend, "_load_diffusers_attention_api", lambda: test_api)
    monkeypatch.setattr(attention_backend, "_flash_varlen_attention", native_only_flash)
    from networks import attention_dispatch

    monkeypatch.setattr(attention_dispatch, "flash_attn_available_for_dtype", lambda dtype: True)

    model = QwenImage21Transformer2DModel(
        num_layers=1,
        num_attention_heads=1,
        attention_head_dim=128,
        context_in_dim=4096,
        mlp_ratio=1,
    )
    attn = model.transformer_blocks[0].attn
    hidden = torch.randn(1, 12, 128)
    key_valid = torch.tensor([[False, False, False, False, True, True, True, True, True, True, True, True]])
    segments = [(0, 8, True)]

    expected = attn.processor(attn, hidden, segments=segments, key_valid=key_valid)
    assert attention_backend.prepare_qwen_image_2_1_attention(
        model, "flash", dtype=torch.bfloat16
    ) == "flash"
    actual = attn.processor(attn, hidden, segments=segments, key_valid=key_valid)

    assert [backend for backend, _ in routed] == ["native", "flash_varlen"]
    text_mask = routed[0][1][0, 0]
    assert not text_mask.triu(1).any()
    flash_mask = routed[1][1][0]
    assert not flash_mask[:4].any() and flash_mask[4:].all()
    torch.testing.assert_close(actual, expected)


def test_flash_requires_bf16_before_provider_probe(monkeypatch) -> None:
    monkeypatch.setattr(
        attention_backend,
        "_load_diffusers_attention_api",
        lambda: pytest.fail("provider must not be probed for an unsupported dtype"),
    )
    with pytest.raises(RuntimeError, match="only for bf16"):
        attention_backend.prepare_qwen_image_2_1_attention(
            object(), "flash", dtype=torch.float16
        )


@pytest.mark.parametrize(("value", "expected"), [(None, "torch"), ("sdpa", "torch"), ("flash", "flash")])
def test_normalize_qwen_attention_mode(value, expected) -> None:
    assert attention_backend.normalize_qwen_image_2_1_attention_mode(value) == expected
