from __future__ import annotations

from contextlib import contextmanager

import torch

from networks import attention_dispatch
from networks.attention_dispatch import AttentionParams, dispatch_attention
from networks.methods.easycontrol import _extended_target_attention


def test_mem_efficient_mode_forces_efficient_backend(monkeypatch):
    selected = []

    @contextmanager
    def fake_sdpa_kernel(backend):
        selected.append(backend)
        yield

    def fake_sdpa(q, k, v, **kwargs):
        assert q.shape == (1, 2, 4, 8)  # dispatcher converted BLHD -> BHLD
        assert k.shape == q.shape
        assert v.shape == q.shape
        assert kwargs["attn_mask"] is None
        assert kwargs["dropout_p"] == 0.0
        return q

    monkeypatch.setattr(attention_dispatch, "sdpa_kernel", fake_sdpa_kernel)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)

    q = torch.randn(1, 4, 2, 8)
    params = AttentionParams.create_attention_params("mem_efficient")
    assert params.supports_fp32 is False
    out = dispatch_attention([q, q, q], attn_params=params)

    assert selected == [attention_dispatch.SDPBackend.EFFICIENT_ATTENTION]
    assert out.shape == (1, 4, 16)


def test_mem_efficient_mode_normalizes_variable_length_mask(monkeypatch):
    selected = []

    @contextmanager
    def fake_sdpa_kernel(backend):
        selected.append(backend)
        yield

    def fake_sdpa(q, k, v, **kwargs):
        mask = kwargs["attn_mask"]
        assert mask.dtype == torch.bool
        assert mask.shape == (2, 1, 1, 4)
        return q

    monkeypatch.setattr(attention_dispatch, "sdpa_kernel", fake_sdpa_kernel)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)

    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    params = AttentionParams.create_attention_params_from_mask(
        "mem_efficient", img_len=0, attention_mask=mask
    )
    q = torch.randn(2, 4, 2, 8)
    out = dispatch_attention([q, q, q], attn_params=params)

    assert selected == [attention_dispatch.SDPBackend.EFFICIENT_ATTENTION]
    assert out.shape == (2, 4, 16)


def test_easycontrol_mem_efficient_mode_forces_efficient_backend(monkeypatch):
    selected = []

    @contextmanager
    def fake_sdpa_kernel(backend):
        selected.append(backend)
        yield

    def fake_sdpa(q, k, v, **kwargs):
        assert q.shape == (1, 2, 4, 8)
        assert k.shape == (1, 2, 7, 8)
        assert v.shape == k.shape
        assert kwargs["attn_mask"].shape == (1, 1, 1, 7)
        return q

    monkeypatch.setattr(attention_dispatch, "sdpa_kernel", fake_sdpa_kernel)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", fake_sdpa)

    target = torch.randn(1, 4, 2, 8)
    cond = torch.randn(1, 3, 2, 8)
    params = AttentionParams.create_attention_params("mem_efficient")
    out = _extended_target_attention(
        target,
        target,
        target,
        cond,
        cond,
        b_param=torch.tensor(0.0),
        scale=None,
        attn_params=params,
    )

    assert selected == [attention_dispatch.SDPBackend.EFFICIENT_ATTENTION]
    assert out.shape == (1, 4, 16)


def test_sdpa_kernel_context_is_compile_traceable():
    def attention(q, k, v):
        # MATH is available on CPU; tracing behavior is identical to selecting
        # EFFICIENT_ATTENTION on CUDA and keeps this invariant test GPU-free.
        with attention_dispatch.sdpa_kernel(attention_dispatch.SDPBackend.MATH):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v)

    q = torch.randn(1, 2, 4, 8)
    explanation = torch._dynamo.explain(attention)(q, q, q)

    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0


def test_flash_mode_casts_fp32_qkv_to_half(monkeypatch):
    """FlashAttention rejects float32; dispatcher must hard-cast before FA.

    Reproduces the compile + ConvRot scope=all failure mode where inductor
    materializes all of Q/K/V as float32 and the weak Attention.forward guard
    (cast only when q.dtype != v.dtype) does not fire.
    """
    seen = {}

    def fake_flash_attn_func(q, k, v, drop_rate=0.0, softmax_scale=None, **kwargs):
        seen["dtypes"] = (q.dtype, k.dtype, v.dtype)
        seen["shapes"] = (tuple(q.shape), tuple(k.shape), tuple(v.shape))
        # BLHD → return same layout; dispatcher flattens heads after.
        return torch.zeros_like(q, dtype=q.dtype)

    monkeypatch.setattr(attention_dispatch, "flash_attn_func", fake_flash_attn_func)
    # Ensure import-time None does not skip the branch.
    assert attention_dispatch.flash_attn_func is fake_flash_attn_func

    q = torch.randn(1, 4, 2, 8, dtype=torch.float32)
    k = torch.randn(1, 4, 2, 8, dtype=torch.float32)
    v = torch.randn(1, 4, 2, 8, dtype=torch.float32)
    params = AttentionParams.create_attention_params("flash")
    assert params.supports_fp32 is False

    out = dispatch_attention([q, k, v], attn_params=params)

    assert seen["dtypes"] == (torch.bfloat16, torch.bfloat16, torch.bfloat16)
    assert seen["shapes"] == ((1, 4, 2, 8), (1, 4, 2, 8), (1, 4, 2, 8))
    assert out.shape == (1, 4, 16)
    assert out.dtype == torch.bfloat16


def test_flash_mode_preserves_existing_half_dtype(monkeypatch):
    seen = {}

    def fake_flash_attn_func(q, k, v, drop_rate=0.0, softmax_scale=None, **kwargs):
        seen["dtypes"] = (q.dtype, k.dtype, v.dtype)
        return torch.zeros_like(q, dtype=q.dtype)

    monkeypatch.setattr(attention_dispatch, "flash_attn_func", fake_flash_attn_func)

    q = torch.randn(1, 4, 2, 8, dtype=torch.float16)
    params = AttentionParams.create_attention_params("flash")
    out = dispatch_attention([q, q, q], attn_params=params)

    assert seen["dtypes"] == (torch.float16, torch.float16, torch.float16)
    assert out.dtype == torch.float16


def test_attention_with_lse_casts_fp32_qkv(monkeypatch):
    seen = {}

    def fake_flash_attn_func(q, k, v, drop_rate=0.0, softmax_scale=None, **kwargs):
        seen["dtypes"] = (q.dtype, k.dtype, v.dtype)
        out = torch.zeros_like(q, dtype=q.dtype)
        lse = torch.zeros(q.shape[0], q.shape[2], q.shape[1], dtype=torch.float32)
        return out, lse, None

    monkeypatch.setattr(attention_dispatch, "flash_attn_func", fake_flash_attn_func)

    q = torch.randn(1, 4, 2, 8, dtype=torch.float32)
    out, lse = attention_dispatch.attention_with_lse(
        q, q, q, attn_mode="flash", softmax_scale=None
    )

    assert seen["dtypes"] == (torch.bfloat16, torch.bfloat16, torch.bfloat16)
    assert out.dtype == torch.bfloat16
    assert lse.dtype == torch.float32
