"""Exact isolated Qwen 2.1 attention stages; preparation alone executes projections."""
from __future__ import annotations


def clone_strided(tensor, device, *, leaf=False):
    """Keep prepared BLHD strides fixed even across CPU artifact transfers."""
    import torch
    result = torch.empty_strided(tensor.shape, tensor.stride(), dtype=tensor.dtype, device=device)
    result.copy_(tensor.detach().to(device))
    return result.requires_grad_(leaf)


def segmented_attention(query, key, value, *, segments, key_valid, attention_mask,
                        attention_backend, parallel_config):
    """Diffusers dense processor core, preserving segment and accumulation order.

    Source: transformer_qwenimage21.py:490-548. Returns BLHD before flatten/out.
    """
    import torch
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    seq_len_q = query.shape[1]
    if segments is None:
        hidden_states = dispatch_attention_fn(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0,
            backend=attention_backend, parallel_config=parallel_config,
        )
    else:
        prefix_len = segments[-1][1] if segments else 0
        outputs = []
        for start, end, is_text in segments:
            seg_mask = None
            if is_text:
                seg_len = end - start
                seg_mask = torch.cat(
                    [
                        torch.ones(seg_len, start, dtype=torch.bool, device=query.device),
                        torch.tril(torch.ones(seg_len, seg_len, dtype=torch.bool, device=query.device)),
                    ],
                    dim=1,
                )[None, None]
            if key_valid is not None:
                seg_key_valid = key_valid[:, None, None, :end]
                seg_mask = seg_key_valid if seg_mask is None else (seg_mask & seg_key_valid)
            outputs.append(
                dispatch_attention_fn(
                    query[:, start:end], key[:, :end], value[:, :end],
                    attn_mask=seg_mask, dropout_p=0.0, backend=None,
                    parallel_config=parallel_config,
                )
            )
        outputs.append(
            dispatch_attention_fn(
                query[:, prefix_len:], key, value,
                attn_mask=None if key_valid is None else key_valid[:, None, None, :],
                dropout_p=0.0, backend=None, parallel_config=parallel_config,
            )
        )
        hidden_states = torch.cat(outputs, dim=1)
    return hidden_states[:, :seq_len_q]


def prepare_core(block, captured, phase, *, backend, tree, device="cuda"):
    """Prepare fixed leaves eagerly from a real captured block, without gradients."""
    import torch
    from diffusers.models.attention_dispatch import attention_backend
    from diffusers.models.normalization import RMSNorm
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21AttnProcessor, apply_rotary_emb_qwen,
    )
    if phase not in ("qk-norm", "rope", "sdpa", "qk-rope", "core"):
        raise ValueError(f"Unknown phase: {phase}")
    attn = block.attn
    if type(attn.norm_q) is not RMSNorm or type(attn.norm_k) is not RMSNorm:
        raise TypeError("Require ordinary Diffusers RMSNorm for Q/K")
    if any(norm.weight.dtype != torch.bfloat16 for norm in (attn.norm_q, attn.norm_k)):
        raise TypeError("Require BF16 RMSNorm weights and their real precision-cast path")
    if type(attn.processor) is not QwenImage21AttnProcessor:
        raise TypeError("Require dense segmented QwenImage21AttnProcessor")
    inputs = tree(captured["bound"], lambda tensor: clone_strided(tensor, device))
    if inputs["layer_cache"] is not None or inputs["kv_cache_mode"] is not None:
        raise ValueError("Core probe requires uncached acceptance prefill")
    with attention_backend(backend), torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        mod1 = inputs["modulation"].chunk(2, dim=-1)[0]
        hidden = block._modulate(block.img_norm1(inputs["hidden_states"]),
                                 mod1, inputs["target_token_mask"])[0]
        query = attn.to_q(hidden).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(hidden).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(hidden).unflatten(-1, (attn.heads, -1))
        if any(tensor.dtype != torch.bfloat16 for tensor in (query, key, value)):
            raise TypeError("Expected BF16 Q/K/V projection outputs")
        if phase not in ("qk-norm", "qk-rope", "core"):
            query = attn.norm_q(query).to(value.dtype)
            key = attn.norm_k(key).to(value.dtype)
        rotary = inputs["rotary_emb"]
        if rotary is None:
            raise ValueError("Acceptance core coverage requires complex RoPE")
        if phase == "sdpa":
            query = apply_rotary_emb_qwen(query, rotary, use_real=False)
            key = apply_rotary_emb_qwen(key, rotary, use_real=False)
    leaves = (query, key, value) if phase in ("sdpa", "core") else (query, key)
    if any(tensor.ndim != 4 for tensor in leaves):
        raise ValueError("Require prepared BLHD tensors")
    if phase in ("sdpa", "core"):
        segments = inputs["segments"]
        if (not segments or not any(s[2] for s in segments) or not any(not s[2] for s in segments)
                or segments[-1][1] >= query.shape[1] or inputs["key_valid"] is None):
            raise ValueError("Require text/image prefix segments, target queries and key-valid mask coverage")
    if phase == "qk-norm":
        def forward(query, key):
            return attn.norm_q(query).to(value.dtype), attn.norm_k(key).to(value.dtype)
    elif phase in ("rope", "qk-rope"):
        def forward(query, key):
            if phase == "qk-rope":
                query = attn.norm_q(query).to(value.dtype)
                key = attn.norm_k(key).to(value.dtype)
            return (apply_rotary_emb_qwen(query, rotary, use_real=False),
                    apply_rotary_emb_qwen(key, rotary, use_real=False))
    else:
        def forward(query, key, value):
            if phase == "core":
                query = apply_rotary_emb_qwen(attn.norm_q(query).to(value.dtype), rotary, use_real=False)
                key = apply_rotary_emb_qwen(attn.norm_k(key).to(value.dtype), rotary, use_real=False)
            return segmented_attention(
                query, key, value, segments=inputs["segments"], key_valid=inputs["key_valid"],
                attention_mask=inputs["attention_mask"],
                attention_backend=attn.processor._attention_backend,
                parallel_config=attn.processor._parallel_config,
            )
    prepared = tuple(clone_strided(tensor, "cpu") for tensor in leaves)
    metadata = {"phase": phase, "phase_nf4_modules": 0, "lora_gradient_coverage": False,
                "preparation_compiled": False, "preparation_backend": backend,
                "input_origin": "full-model no_grad capture -> eager norm1/mod1/QKV; earlier isolated cores eager",
                "composite_phase": phase in ("qk-rope", "core"),
                "layout": "BLHD", "norm_class": "RMSNorm", "rope_use_real": False,
                "leaf_shapes": [list(t.shape) for t in leaves],
                "leaf_strides": [list(t.stride()) for t in leaves],
                "norm_weight_dtypes": [str(attn.norm_q.weight.dtype), str(attn.norm_k.weight.dtype)]}
    constants = {key: inputs[key] for key in ("rotary_emb", "segments", "key_valid", "attention_mask")}
    return prepared, forward, metadata, tree(constants, lambda tensor: clone_strided(tensor, "cpu"))
