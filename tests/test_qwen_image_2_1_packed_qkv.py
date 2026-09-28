"""CPU equivalence and lifecycle contracts for Qwen Image 2.1 packed QKV."""

from __future__ import annotations

import copy
import gc
from types import SimpleNamespace
import weakref

import pytest
import torch
from torch import nn

from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Attention
from library.models.qwen_image_2_1.packed_qkv import (
    FrozenPackedQKV,
    install_qwen_image_2_1_packed_qkv,
    prepare_packed_qkv,
)
from library.models.qwen_image_2_1 import attention_backend
from networks.lora_modules.lora import LoRAModule


def _fixture(dtype=torch.float64, *, blocks=1):
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = QwenImage21Attention(dim=8, heads=2, dim_head=4)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = nn.ModuleList(Block() for _ in range(blocks))

    original = Model().to(dtype=dtype).requires_grad_(False)
    packed = copy.deepcopy(original)
    reference_loras = []
    packed_loras = []
    for original_block, packed_block in zip(original.transformer_blocks, packed.transformer_blocks):
        for name in ("to_q", "to_k", "to_v"):
            ref = LoRAModule(name, getattr(original_block.attn, name), lora_dim=3, alpha=2).to(dtype)
            tgt = LoRAModule(name, getattr(packed_block.attn, name), lora_dim=3, alpha=2).to(dtype)
            with torch.no_grad():
                ref.lora_up.weight.normal_(std=0.1)
            tgt.load_state_dict(ref.state_dict())
            ref.apply_to()
            tgt.apply_to()
            reference_loras.append(ref)
            packed_loras.append(tgt)
    return original, packed, reference_loras, packed_loras


def _run(attn, x):
    key_valid = torch.tensor([[False, False, True, True, True, True, True, True]])
    rotary = torch.polar(torch.ones(8, 2), torch.arange(8).float()[:, None].expand(8, 2))
    return attn(x, segments=[(0, 4, True)], key_valid=key_valid, rotary_emb=rotary)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_native_output_dx_and_each_lora_gradient(dtype):
    original, packed, ref_loras, packed_loras = _fixture(dtype)
    old_keys = [set(lora.state_dict()) for lora in packed_loras]
    ids = [(id(lora.lora_down.weight), id(lora.lora_up.weight)) for lora in packed_loras]
    assert install_qwen_image_2_1_packed_qkv(packed, packed_loras) == 1
    attn_ref = original.transformer_blocks[0].attn
    attn_packed = packed.transformer_blocks[0].attn
    x = torch.randn(1, 8, 8, dtype=dtype)
    x_ref = x.clone().requires_grad_()
    x_packed = x.clone().requires_grad_()
    expected = _run(attn_ref, x_ref)
    actual = _run(attn_packed, x_packed)
    tol = 2e-12 if dtype == torch.float64 else 2e-6
    torch.testing.assert_close(actual, expected, rtol=tol, atol=tol)
    grad = torch.randn_like(expected)
    expected.backward(grad)
    actual.backward(grad)
    torch.testing.assert_close(x_packed.grad, x_ref.grad, rtol=tol, atol=tol)
    for ref, tgt, state_keys, param_ids in zip(ref_loras, packed_loras, old_keys, ids):
        assert set(tgt.state_dict()) == state_keys
        assert (id(tgt.lora_down.weight), id(tgt.lora_up.weight)) == param_ids
        torch.testing.assert_close(tgt.lora_down.weight.grad, ref.lora_down.weight.grad, rtol=tol, atol=tol)
        torch.testing.assert_close(tgt.lora_up.weight.grad, ref.lora_up.weight.grad, rtol=tol, atol=tol)
    assert set(attn_packed.state_dict()).isdisjoint({"to_q.weight", "to_k.weight", "to_v.weight"})
    assert "packed_qkv.weight" in attn_packed.state_dict()
    assert sum(name.endswith("packed_qkv.weight") for name, _ in packed.named_parameters()) == 1


def test_disabled_eval_and_lora_state_roundtrip():
    original, packed, ref_loras, packed_loras = _fixture()
    install_qwen_image_2_1_packed_qkv(packed, packed_loras)
    x = torch.randn(1, 8, 8, dtype=torch.float64)
    for ref, tgt in zip(ref_loras, packed_loras):
        ref.enabled = tgt.enabled = False
    torch.testing.assert_close(_run(packed.transformer_blocks[0].attn, x), _run(original.transformer_blocks[0].attn, x))
    for ref, tgt in zip(ref_loras, packed_loras):
        ref.enabled = tgt.enabled = True
        ref.eval()
        tgt.eval()
    torch.testing.assert_close(_run(packed.transformer_blocks[0].attn, x), _run(original.transformer_blocks[0].attn, x))
    saved = [copy.deepcopy(lora.state_dict()) for lora in packed_loras]
    for lora in packed_loras:
        with torch.no_grad():
            lora.lora_up.weight.zero_()
    for lora, state in zip(packed_loras, saved):
        lora.load_state_dict(state, strict=True)
    torch.testing.assert_close(_run(packed.transformer_blocks[0].attn, x), _run(original.transformer_blocks[0].attn, x))


def test_one_base_gemm_no_per_step_cat_and_rebound_weight(monkeypatch):
    _, packed, _, loras = _fixture()
    old_base = weakref.ref(packed.transformer_blocks[0].attn.to_q)
    install_qwen_image_2_1_packed_qkv(packed, loras)
    gc.collect()
    assert old_base() is None
    projection = packed.transformer_blocks[0].attn.packed_qkv
    assert isinstance(projection, FrozenPackedQKV)
    x = torch.randn(1, 8, 8, dtype=torch.float64)
    import torch.nn.functional as functional

    original_linear = functional.linear
    base_calls = []

    def count_linear(input, weight, bias=None):
        if weight is projection.weight:
            base_calls.append(weight)
        return original_linear(input, weight, bias)

    monkeypatch.setattr(functional, "linear", count_linear)
    monkeypatch.setattr(torch, "cat", lambda *_args, **_kwargs: pytest.fail("forward must not concatenate"))
    first = projection(x)
    assert len(base_calls) == 1
    projection.weight = nn.Parameter(projection.weight.detach().clone() + 0.1, requires_grad=False)
    second = projection(x)
    assert len(base_calls) == 2
    assert any(not torch.equal(a, b) for a, b in zip(first, second))
    assert set(name for name, _ in projection.named_parameters()) == {"weight"}


def test_flash_processor_uses_local_packed_projection(monkeypatch):
    original, packed, _, loras = _fixture()
    api = attention_backend._load_diffusers_attention_api()
    from diffusers.models.attention_dispatch import dispatch_attention_fn

    def native_flash(query, key, value, *, mask):
        return dispatch_attention_fn(query, key, value, attn_mask=mask, dropout_p=0.0, backend=None)

    monkeypatch.setattr(attention_backend, "_flash_varlen_attention", native_flash)
    packed.transformer_blocks[0].attn.set_processor(attention_backend.QwenImage21FlashAttnProcessor(api))
    install_qwen_image_2_1_packed_qkv(packed, loras)
    x = torch.randn(1, 8, 8, dtype=torch.float64)
    torch.testing.assert_close(
        _run(packed.transformer_blocks[0].attn, x),
        _run(original.transformer_blocks[0].attn, x),
    )


def test_cache_semantics_and_atomic_preflight():
    original, packed, _, loras = _fixture(blocks=2)
    second = packed.transformer_blocks[1].attn
    second.set_processor(object())
    with pytest.raises(TypeError, match="native or Qwen Flash"):
        install_qwen_image_2_1_packed_qkv(packed, loras)
    assert all(not hasattr(block.attn, "packed_qkv") for block in packed.transformer_blocks)
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21AttnProcessor

    second.set_processor(QwenImage21AttnProcessor())
    install_qwen_image_2_1_packed_qkv(packed, loras)
    x = torch.randn(1, 8, 8, dtype=torch.float64)

    class Cache:
        def store(self, key, value):
            self.key, self.value = key, value

        def get(self):
            return self.key, self.value

    a = Cache()
    b = Cache()
    args = (x, None, a, "extract", slice(0, 3))
    expected = original.transformer_blocks[0].attn
    from diffusers.models.transformers.transformer_qwenimage21 import _qwenimage21_prepare_qkv

    _qwenimage21_prepare_qkv(expected, *args)
    prepare_packed_qkv(packed.transformer_blocks[0].attn, x, None, b, "extract", slice(0, 3))
    torch.testing.assert_close(a.key, b.key)
    torch.testing.assert_close(a.value, b.value)
    original_cached = _qwenimage21_prepare_qkv(expected, x[:, 3:], None, a, "cached", None)
    packed_cached = prepare_packed_qkv(packed.transformer_blocks[0].attn, x[:, 3:], None, b, "cached", None)
    for left, right in zip(original_cached[:3], packed_cached[:3]):
        torch.testing.assert_close(left, right)


def test_second_block_cat_failure_restores_original_model(monkeypatch):
    original, packed, _, loras = _fixture(blocks=2)
    packed.transformer_blocks[0].attn.to_q.weight.requires_grad_(True)
    before = []
    for block in packed.transformer_blocks:
        attn = block.attn
        before.append((
            tuple(getattr(attn, name) for name in ("to_q", "to_k", "to_v")),
            tuple(getattr(attn, name).weight for name in ("to_q", "to_k", "to_v")),
            tuple(getattr(attn, name).weight.requires_grad for name in ("to_q", "to_k", "to_v")),
            attn.processor,
        ))
    forwards = tuple(lora.org_forward for lora in loras)
    original_cat = torch.cat
    calls = 0

    def fail_second_cat(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected QKV allocation failure")
        return original_cat(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch, "cat", fail_second_cat)
        with pytest.raises(RuntimeError, match="injected QKV allocation failure"):
            install_qwen_image_2_1_packed_qkv(packed, loras, freeze=True)
    assert calls == 2
    for block, (modules, parameters, flags, processor) in zip(packed.transformer_blocks, before):
        attn = block.attn
        assert not hasattr(attn, "packed_qkv")
        assert attn.processor is processor
        for name, module, parameter, flag in zip(("to_q", "to_k", "to_v"), modules, parameters, flags):
            assert getattr(attn, name) is module
            assert module.weight is parameter
            assert parameter.requires_grad is flag
    for lora, forward, parameter in zip(loras, forwards, (p for group in before for p in group[1])):
        assert lora.org_module_ref[0].weight is parameter
        assert lora.org_forward is forward
    x = torch.randn(1, 8, 8, dtype=torch.float64)
    for reference, restored in zip(original.transformer_blocks, packed.transformer_blocks):
        torch.testing.assert_close(_run(restored.attn, x), _run(reference.attn, x))
    assert install_qwen_image_2_1_packed_qkv(packed, loras, freeze=True) == 2


def test_each_committed_block_releases_split_weight_storage(monkeypatch):
    _, packed, _, loras = _fixture(blocks=2)
    old_ptrs = [
        tuple(getattr(block.attn, name).weight.data_ptr() for name in ("to_q", "to_k", "to_v"))
        for block in packed.transformer_blocks
    ]
    original_cat = torch.cat
    calls = 0

    def inspect_next_cat(tensors, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            first = packed.transformer_blocks[0].attn.packed_qkv.weight
            assert all(lora.org_module_ref[0] is None for lora in loras[:3])
            assert all(pointer not in old_ptrs[0] for pointer in (first.data_ptr(),))
            assert all(
                lora_ref.weight.untyped_storage().data_ptr() == first.untyped_storage().data_ptr()
                for lora_ref in first_block_modules
            )
        return original_cat(tensors, *args, **kwargs)

    first_block_modules = tuple(
        getattr(packed.transformer_blocks[0].attn, name) for name in ("to_q", "to_k", "to_v")
    )
    monkeypatch.setattr(torch, "cat", inspect_next_cat)
    assert install_qwen_image_2_1_packed_qkv(packed, loras) == 2
    assert calls == 2


def test_packed_lora_merge_operations_fail_clearly():
    _, packed, _, loras = _fixture()
    install_qwen_image_2_1_packed_qkv(packed, loras)
    network = SimpleNamespace(text_encoder_loras=[], unet_loras=loras, cfg=SimpleNamespace())
    from networks.lora_anima import merge

    assert not merge.is_mergeable(network)
    for operation in (merge.fuse_weights, merge.backup_weights, merge.pre_calculation, merge.restore_weights):
        with pytest.raises(RuntimeError, match="packed QKV"):
            operation(network)
    with pytest.raises(RuntimeError, match="packed QKV"):
        loras[0].fuse_weight()
