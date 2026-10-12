"""CPU-only official Qwen 2.1 KV numerical oracle; emits JSONL diagnostics.

Run: CUDA_VISIBLE_DEVICES='' .venv/bin/python -m scripts.qwen_image_2_1.kv_numerics
This intentionally uses random small weights, not a production drift tolerance.
"""

import json

import torch
from diffusers import QwenImage21Transformer2DModel
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

from library.models.qwen_image_2_1.preview_kv import PreviewKVForward


def metrics(a, b):
    a, b = a.float(), b.float()
    difference = a - b
    return dict(exact=torch.equal(a, b), max_abs=difference.abs().max().item(),
                rel_l2=(difference.norm() / b.norm().clamp_min(1e-20)).item())


def native_inputs(latent, text, mask, sigma, references, slots):
    """Construct the official API layout independently of training family/helper."""
    batch, _, height, width = latent.shape
    refs = tuple(references or ())
    target = latent.flatten(2).transpose(1, 2)
    packed = torch.cat([*(r.flatten(2).transpose(1, 2) for r in refs), target], dim=1) if refs else target
    shapes = [[(1, r.shape[-2], r.shape[-1]) for r in (*refs, latent)]]
    image_slots = slots if refs else torch.zeros_like(mask)
    img_mask = torch.cat([image_slots, torch.ones(batch, height * width // 4, dtype=torch.bool)], dim=1)
    return dict(hidden_states=packed, encoder_hidden_states=text,
                encoder_hidden_states_mask=mask, timestep=sigma.to(latent.dtype),
                img_shapes=shapes, img_mask=img_mask, return_dict=False)


def native(model, latent, text, mask, sigma, refs, slots, cache=None, mode=None):
    args = native_inputs(latent, text, mask, sigma, refs, slots)
    if cache is not None:
        args.update(kv_cache=cache, kv_cache_mode=mode)
    result = model(**args)[0][:, -latent.shape[-2] * latent.shape[-1]:]
    return result.transpose(1, 2).reshape_as(latent)


def cache_values(cache):
    return torch.cat([value.flatten() for layer in cache.layer_caches for value in (layer.k, layer.v)])


@torch.no_grad()
def case(dtype, layers, text_len, shape, edit, compiled=False):
    torch.manual_seed(71)
    model = QwenImage21Transformer2DModel(
        num_layers=layers, num_attention_heads=2, attention_head_dim=64,
        context_in_dim=4096, axes_dims_rope=(16, 24, 24), mlp_ratio=3,
        causal_condition=True,
    ).to(dtype).eval()
    if compiled:
        from library.models.qwen_image_2_1.compile import compile_qwen_image_2_1_blocks
        compile_qwen_image_2_1_blocks(model, backend="eager")
    z = torch.randn(1, 64, *shape).to(dtype)
    text = torch.randn(1, text_len, 4096).to(dtype)
    mask = torch.ones(1, text_len, dtype=torch.bool)
    mask[:, -1] = False
    slots = torch.zeros_like(mask)
    slots[:, 1:4] = edit
    refs = (torch.randn(1, 64, 2, 2).to(dtype), torch.randn(1, 64, 2, 4).to(dtype)) if edit else None
    branch_texts = (text, -text)
    helpers = [PreviewKVForward(model), PreviewKVForward(model)]
    caches = [QwenImage21KVCache(layers), QwenImage21KVCache(layers)]
    contiguous_caches = [QwenImage21KVCache(layers), QwenImage21KVCache(layers)]
    helper_errors, official_errors, cfg_errors = [], [], []
    input_layout_errors, input_projection_errors = [], []
    prefix_initial = None
    uncached_z = z.clone()
    cached_z = z.clone()
    for step, sigma_value in enumerate((1., .6, .2)):
        sigma = torch.tensor([sigma_value])
        mode = "extract" if step == 0 else "cached"
        branch_c, branch_u = [], []
        for branch, hidden in enumerate(branch_texts):
            direct = native(model, z, hidden, mask, sigma, refs, slots, caches[branch], mode)
            helper = helpers[branch](z, hidden, mask, sigma, reference_latents=refs,
                                     image_slot_mask=slots if edit else None)
            full = native(model, z, hidden, mask, sigma, refs, slots)
            helper_errors.append(metrics(helper, direct))
            official_errors.append(metrics(direct, full))
            assert torch.equal(helper, direct), "Production helper differs from independently built official API"
            layout_args = native_inputs(z, hidden, mask, sigma, refs, slots)
            packed = layout_args["hidden_states"]
            input_projection_errors.append(metrics(model.img_in(packed), model.img_in(packed.contiguous())))
            layout_args.update(hidden_states=packed.contiguous(), kv_cache=contiguous_caches[branch],
                               kv_cache_mode=mode)
            contiguous = model(**layout_args)[0][:, -z.shape[-2] * z.shape[-1]:]
            contiguous = contiguous.transpose(1, 2).reshape_as(z)
            input_layout_errors.append(metrics(contiguous, direct))
            branch_c.append(native(model, cached_z, hidden, mask, sigma, refs, slots, caches[branch], mode))
            branch_u.append(native(model, uncached_z, hidden, mask, sigma, refs, slots))
        cfg_c = branch_c[1] + 3 * (branch_c[0] - branch_c[1])
        cfg_u = branch_u[1] + 3 * (branch_u[0] - branch_u[1])
        cfg_errors.append(metrics(cfg_c, cfg_u))
        cached_z = (cached_z.float() - .4 * cfg_c.float()).to(dtype)
        uncached_z = (uncached_z.float() - .4 * cfg_u.float()).to(dtype)
        if step == 0:
            prefix_initial = cache_values(caches[0]).clone()
        z = torch.randn_like(z)
    repeat_cache = QwenImage21KVCache(layers)
    native(model, z, text, mask, torch.tensor([.37]), refs, slots, repeat_cache, "extract")
    prefix_repeat = metrics(cache_values(repeat_cache), prefix_initial)
    assert prefix_repeat["exact"], "Prefix changed with target values/timestep at identical shape"
    different_z = torch.randn(1, 64, shape[0], shape[1] + 2).to(dtype)
    size_cache = QwenImage21KVCache(layers)
    native(model, different_z, text, mask, torch.tensor([.19]), refs, slots, size_cache, "extract")
    report = dict(dtype=str(dtype), layers=layers, text_len=text_len, target_shape=shape,
                  references=2 if edit else 0, compiled_eager=compiled,
                  helper_vs_official=helper_errors, official_cached_vs_uncached=official_errors,
                  forced_contiguous_vs_native_layout=input_layout_errors,
                  img_in_contiguous_vs_native_layout=input_projection_errors,
                  prefix_same_shape=prefix_repeat,
                  prefix_different_target_size=metrics(cache_values(size_cache), prefix_initial),
                  cfg3_prediction=cfg_errors, toy_euler_final=metrics(cached_z, uncached_z))
    for helper in helpers:
        helper.close()
    print(json.dumps(report), flush=True)


def main():
    torch.set_num_threads(1)
    assert not torch.cuda.is_available(), "Run with CUDA_VISIBLE_DEVICES=''"
    for dtype in (torch.float32, torch.bfloat16):
        for layers in (2, 4):
            for text_len, shape, edit in ((6, (4, 8), False), (17, (8, 8), True), (65, (4, 8), True)):
                case(dtype, layers, text_len, shape, edit)
    case(torch.bfloat16, 2, 17, (4, 8), True, compiled=True)


if __name__ == "__main__":
    main()
