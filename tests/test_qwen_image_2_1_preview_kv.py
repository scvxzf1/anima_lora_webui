"""Official tiny-transformer contracts for invocation-local preview KV reuse."""

from types import SimpleNamespace

import pytest
import torch
from diffusers import QwenImage21Transformer2DModel

from library.models.qwen_image_2_1 import sampling
from library.models.qwen_image_2_1.compile import compile_qwen_image_2_1_blocks
from library.models.qwen_image_2_1.family import forward_for_loss
from library.models.qwen_image_2_1.preview_kv import PreviewKVForward, supports_preview_kv


@pytest.fixture(autouse=True)
def single_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny(*, causal=True):
    torch.manual_seed(71)
    return QwenImage21Transformer2DModel(
        num_layers=2, num_attention_heads=1, attention_head_dim=16,
        context_in_dim=4096, axes_dims_rope=(4, 6, 6), mlp_ratio=1,
        causal_condition=causal,
    ).eval()


def conditions(edit=False):
    hidden = torch.randn(1, 6, 4096)
    mask = torch.tensor([[True, True, True, True, True, False]])
    slots = torch.tensor([[False, True, True, False, False, False]])
    refs = (torch.randn(1, 64, 2, 2), torch.randn(1, 64, 2, 2)) if edit else None
    return (hidden, mask, slots), refs


@pytest.mark.parametrize("edit", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_official_extract_cached_target_layout_and_numerics(edit, compiled):
    model = tiny()
    if compiled:
        compile_qwen_image_2_1_blocks(model, backend="eager")
    values, refs = conditions(edit)
    kv = PreviewKVForward(model)
    modes = []
    original = model.forward

    def record(**kwargs):
        modes.append(kwargs.get("kv_cache_mode"))
        return original(**kwargs)

    model.forward = record
    extra = dict(reference_latents=refs, image_slot_mask=values[2] if edit else None)
    with torch.no_grad():
        for sigma in (1.0, 0.6, 0.2):
            latent = torch.randn(1, 64, 2, 4)
            time = torch.tensor([sigma])
            expected = forward_for_loss(model, latent, *values[:2], time, **extra)
            actual = kv(latent, *values[:2], time, **extra)
            assert actual.shape == latent.shape
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    assert modes == [None, "extract", None, "cached", None, "cached"]
    prefix = 6 + (6 if edit else 0)
    assert all(layer.k.shape[1] == prefix for layer in kv.cache.layer_caches)
    layers = list(kv.cache.layer_caches)
    kv.close()
    assert kv.cache is None and kv.inputs is None and kv.reference_tokens is None
    assert all(layer.k is None and layer.v is None for layer in layers)


@pytest.mark.parametrize("edit", [False, True])
def test_sampler_cfg_isolation_lifetime_and_uncached_equivalence(edit):
    model = tiny()
    positive, refs = conditions(edit)
    negative = (positive[0] * -1, positive[1], positive[2])
    seen = []
    original = model.forward

    def record(**kwargs):
        seen.append((kwargs.get("kv_cache"), kwargs.get("kv_cache_mode")))
        return original(**kwargs)

    model.forward = record
    prompt = dict(width=64, height=64, sample_steps=3, guidance_scale=2.5, negative_prompt="")
    kwargs = dict(device=torch.device("cpu"), dtype=torch.float32, seed=14)
    cached = sampling.generate_preview(model, prompt, positive, negative, refs, use_kv_cache=True, **kwargs)
    assert [mode for _, mode in seen] == ["extract", "extract", "cached", "cached", "cached", "cached"]
    positive_cache, negative_cache = seen[0][0], seen[1][0]
    assert positive_cache is not negative_cache
    assert seen[2][0] is positive_cache and seen[3][0] is negative_cache
    assert all(layer.k is None and layer.v is None for cache, _ in seen for layer in cache.layer_caches)
    seen.clear()
    uncached = sampling.generate_preview(model, prompt, positive, negative, refs, use_kv_cache=False, **kwargs)
    assert all(cache is None and mode is None for cache, mode in seen)
    torch.testing.assert_close(cached, uncached, atol=3e-6, rtol=3e-5)
    seen.clear()
    default = sampling.generate_preview(model, prompt, positive, negative, refs, **kwargs)
    assert all(cache is None and mode is None for cache, mode in seen)
    assert torch.equal(default, uncached)
    seen.clear()
    sampling.generate_preview(model, prompt, positive, negative, refs, use_kv_cache=True, **kwargs)
    assert seen[0][0] is not positive_cache and seen[1][0] is not negative_cache
    assert not hasattr(model, "kv_cache")


def test_sampler_exception_disposes_all_branches():
    model = tiny()
    positive, refs = conditions(True)
    seen = []
    original = model.forward

    def fail(**kwargs):
        seen.append(kwargs["kv_cache"])
        result = original(**kwargs)
        if len(seen) == 2:
            raise RuntimeError("negative prefill failed")
        return result

    model.forward = fail
    with pytest.raises(RuntimeError, match="negative prefill"):
        sampling.generate_preview(
            model, dict(width=64, height=64, sample_steps=2, guidance_scale=2, negative_prompt=""),
            positive, positive, refs, device=torch.device("cpu"), dtype=torch.float32, seed=4,
            use_kv_cache=True,
        )
    assert len(seen) == 2 and seen[0] is not seen[1]
    assert all(layer.k is None and layer.v is None for cache in seen for layer in cache.layer_caches)


def test_causal_false_and_injected_forward_fall_back():
    model = tiny(causal=False)
    values, _ = conditions()
    assert not supports_preview_kv(model, forward_for_loss)
    original = model.forward
    seen = []

    def record(**kwargs):
        seen.append(kwargs.get("kv_cache"))
        return original(**kwargs)

    model.forward = record
    kwargs = dict(device=torch.device("cpu"), dtype=torch.float32, seed=1)
    sampling.generate_preview(model, dict(width=64, height=64, sample_steps=2), values, None, None,
                              use_kv_cache=True, **kwargs)
    assert seen == [None, None]
    calls = []

    def injected(dit, latent, hiddens, mask, sigma, **extra):
        assert set(extra) == {"reference_latents", "image_slot_mask"}
        calls.append(dit)
        return torch.zeros_like(latent)

    sampling.generate_preview(None, dict(width=64, height=64, sample_steps=2), values, None, None,
                              forward=injected, use_kv_cache=True, **kwargs)
    assert calls == [None, None]


@pytest.mark.parametrize("variant", ["lora", "loha", "lokr"])
def test_static_adapters_supported_with_real_nonzero_deltas(variant):
    from library.models.qwen_image_2_1.lora_targets import qwen_image_2_1_target_kwargs
    from networks.lora_anima import create_network

    model = tiny()
    network = create_network(
        multiplier=1, network_dim=4, network_alpha=4, vae=None, text_encoders=[], unet=model,
        model_family="qwen_image_2_1", **qwen_image_2_1_target_kwargs(),
        **({f"use_{variant}": "true"} if variant != "lora" else {}),
    )
    network.apply_to([], model, False, True)
    network.eval()
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.normal_(0, 0.1)
    assert supports_preview_kv(model, forward_for_loss)
    values, refs = conditions(True)
    prompt = dict(width=64, height=64, sample_steps=3)
    kwargs = dict(device=torch.device("cpu"), dtype=torch.float32, seed=8)
    cached = sampling.generate_preview(model, prompt, values, None, refs, use_kv_cache=True, **kwargs)
    uncached = sampling.generate_preview(model, prompt, values, None, refs, use_kv_cache=False, **kwargs)
    torch.testing.assert_close(cached, uncached, atol=3e-6, rtol=3e-5)


def test_unknown_dynamic_adapter_and_training_fall_back():
    model = tiny()
    assert supports_preview_kv(model, forward_for_loss)
    model.train()
    assert not supports_preview_kv(model, forward_for_loss)
    model.eval()
    model.transformer_blocks[0].attn.to_k.forward = SimpleNamespace(forward=lambda x: x).forward
    assert not supports_preview_kv(model, forward_for_loss)


def test_training_forward_still_has_gradients_and_no_cache():
    model = tiny().train()
    values, refs = conditions(True)
    latent = torch.randn(1, 64, 2, 4, requires_grad=True)
    output = forward_for_loss(model, latent, *values[:2], torch.tensor([0.4]),
                              reference_latents=refs, image_slot_mask=values[2])
    output.square().mean().backward()
    assert latent.grad is not None and latent.grad.abs().sum() > 0
    assert model.transformer_blocks[0].attn.to_k.weight.grad is not None
    assert not hasattr(model, "kv_cache")


def test_hooks_and_unknown_attention_processors_disable_cache():
    model = tiny()
    handle = model.transformer_blocks[0].attn.to_v.register_forward_hook(lambda module, args, out: out)
    assert not supports_preview_kv(model, forward_for_loss)
    handle.remove()
    assert supports_preview_kv(model, forward_for_loss)
    model.transformer_blocks[0].attn.set_processor(SimpleNamespace())
    assert not supports_preview_kv(model, forward_for_loss)


@pytest.mark.parametrize("change", ["replace", "mutate"])
def test_helper_rejects_changed_conditions(change):
    model = tiny()
    values, refs = conditions(True)
    kv = PreviewKVForward(model)
    latent = torch.randn(1, 64, 2, 4)
    with torch.no_grad():
        kv(latent, *values[:2], torch.tensor([1.0]), reference_latents=refs, image_slot_mask=values[2])
        hidden = values[0].clone() if change == "replace" else values[0].add_(1)
        with pytest.raises(ValueError, match="conditions changed"):
            kv(latent, hidden, values[1], torch.tensor([0.5]), reference_latents=refs, image_slot_mask=values[2])
    kv.close()
