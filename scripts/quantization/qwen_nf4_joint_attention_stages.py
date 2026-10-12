"""Connected real Qwen attention VJPs, including the shared QKV input join."""
from __future__ import annotations

import time

from scripts.quantization.qwen_nf4_attention_core_stages import clone_strided, segmented_attention
from scripts.quantization.probe_qwen_nf4_projection import projection_target

PHASES = ("qkv", "core", "attention")
QKV_NAMES = tuple(f"projection_{name}" for name in ("query", "key", "value"))


def assert_bytes(actual, expected, *, stride=False):
    """Validate shape, dtype and raw bytes; optionally require original strides."""
    import torch
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError("Source tensor shape/dtype mismatch")
    if stride and actual.stride() != expected.stride():
        raise AssertionError("Source tensor stride mismatch")
    if not torch.equal(actual.detach().cpu().contiguous().view(torch.uint8),
                       expected.detach().cpu().contiguous().view(torch.uint8)):
        raise AssertionError("Source tensor bytes mismatch")


def prepare_joint(block, network, index, captured, loss_record, phase, *, tree,
                  input_layout="shared", input_dtype="original", device="cuda"):
    """Use actual patched projections and uncached captured attention constants."""
    import torch
    from diffusers.models.normalization import RMSNorm
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21AttnProcessor, apply_rotary_emb_qwen,
    )
    if phase not in PHASES or input_layout not in ("shared", "independent"):
        raise ValueError("Unknown joint phase/input layout")
    if input_dtype not in ("original", "bf16"):
        raise ValueError("Unknown input dtype")
    attn = block.attn
    if type(attn.processor) is not QwenImage21AttnProcessor:
        raise TypeError("Require dense QwenImage21AttnProcessor")
    if any(type(norm) is not RMSNorm or norm.weight.dtype != torch.bfloat16
           for norm in (attn.norm_q, attn.norm_k)):
        raise TypeError("Require ordinary BF16-weight Diffusers RMSNorm")
    if not isinstance(attn.to_out[1], torch.nn.Dropout) or attn.to_out[1].p != 0:
        raise ValueError("Require production output dropout with p=0")
    bound = captured["bound"]
    if bound["layer_cache"] is not None or bound["kv_cache_mode"] is not None:
        raise ValueError("Require uncached Edit prefill")
    originals = tuple(loss_record["projection_inputs"][name] for name in QKV_NAMES)
    for tensor in originals[1:]:
        assert_bytes(tensor, originals[0], stride=True)
    if originals[0].ndim != 3:
        raise AssertionError("Require common flattened BLH projection input")
    value = originals[0] if input_dtype == "original" else originals[0].bfloat16()
    prepared = tuple(clone_strided(value, "cpu") for _ in range(1 if input_layout == "shared" else 3))
    constants = {name: bound[name] for name in ("rotary_emb", "segments", "key_valid", "attention_mask")}
    if (constants["rotary_emb"] is None or not constants["segments"] or
            not any(s[2] for s in constants["segments"]) or
            not any(not s[2] for s in constants["segments"]) or
            constants["segments"][-1][1] >= value.shape[1] or constants["key_valid"] is None):
        raise ValueError("Require complex RoPE, text/image prefixes, target and key-valid mask")
    inputs = tree(constants, lambda t: clone_strided(t, device))
    projections = ("q", "k", "v", "out") if phase == "attention" else ("q", "k", "v")
    selected = [projection_target(block, network, index, name, "lora") for name in projections]
    parameters = [item for _, params in selected for item in params]
    if len(parameters) != (8 if phase == "attention" else 6):
        raise AssertionError("Incomplete real FP32 LoRA parameter coverage")
    q_forward, k_forward, v_forward = (item[0] for item in selected[:3])

    def forward(*leaves):
        q_input, k_input, v_input = (leaves[0],) * 3 if input_layout == "shared" else leaves
        query, key, value = q_forward(q_input), k_forward(k_input), v_forward(v_input)
        if phase == "qkv":
            return query, key, value
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query = apply_rotary_emb_qwen(attn.norm_q(query).to(value.dtype), inputs["rotary_emb"], use_real=False)
        key = apply_rotary_emb_qwen(attn.norm_k(key).to(value.dtype), inputs["rotary_emb"], use_real=False)
        hidden = segmented_attention(query, key, value, segments=inputs["segments"],
                                     key_valid=inputs["key_valid"], attention_mask=inputs["attention_mask"],
                                     attention_backend=attn.processor._attention_backend,
                                     parallel_config=attn.processor._parallel_config)
        hidden = hidden.flatten(2, 3).type_as(query)
        return attn.to_out[1](selected[3][0](hidden)) if phase == "attention" else hidden

    names = QKV_NAMES if phase == "qkv" else ("projection_out" if phase == "attention" else "attention_flat",)
    sources = tuple(loss_record["outputs"][name] for name in names)
    cotangents = tuple(loss_record["cotangents"][name] for name in names)
    metadata = {"phase": phase, "boundaries": list(names), "input_layout": input_layout,
                "input_dtype": str(value.dtype), "original_input_dtype": str(originals[0].dtype),
                "original_input_stride": list(originals[0].stride()), "qkv_inputs_bytes_shape_dtype_stride_equal": True,
                "leaf_shapes": [list(t.shape) for t in prepared], "leaf_strides": [list(t.stride()) for t in prepared],
                "source_strides": [list(t.stride()) for t in sources],
                "cotangent_strides": [list(t.stride()) for t in cotangents],
                "gradient_targets": [name for name, _ in parameters], "phase_nf4_modules": len(projections),
                "norm_class": "RMSNorm", "rope_use_real": False, "optimizer_updates": 0}
    return prepared, forward, parameters, sources, cotangents, metadata, constants


def joint_vjp(forward, prepared, cotangents, parameters, phase, backend, *, device="cuda"):
    """Differentiate shared/independent input leaves and actual FP32 LoRA weights."""
    import torch
    from diffusers.models.attention_dispatch import attention_backend
    leaves = tuple(clone_strided(t, device, leaf=True) for t in prepared)
    if any(t.stride() != source.stride() for t, source in zip(leaves, prepared, strict=True)):
        raise AssertionError("Input stride changed")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    targets = (*leaves, *(p for _, p in parameters))
    with attention_backend(backend), torch.autocast(device, dtype=torch.bfloat16):
        outputs = forward(*leaves)
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        if len(outputs) != (3 if phase == "qkv" else 1) or any(t.dtype != torch.bfloat16 for t in outputs):
            raise AssertionError("Incomplete BF16 output coverage")
        gradients = torch.autograd.grad(outputs, targets, grad_outputs=tuple(t.to(device) for t in cotangents))
    for gradient, target in zip(gradients, targets, strict=True):
        if gradient.shape != target.shape or gradient.dtype != target.dtype:
            raise AssertionError("Gradient shape/dtype changed")
    names = (tuple(f"output_{n}" for n in ("query", "key", "value")) if phase == "qkv" else ("output",))
    gradient_names = (("hidden_gradient",) if len(leaves) == 1 else
                      ("query_input_gradient", "key_input_gradient", "value_input_gradient"))
    record = {name: clone_strided(t, "cpu") for name, t in
              zip((*names, *gradient_names, *(n for n, _ in parameters)), (*outputs, *gradients), strict=True)}
    if device == "cuda":
        torch.cuda.synchronize()
    summary = {"elapsed_seconds": time.monotonic() - started, "calls": 1,
               "dtypes": {n: str(t.dtype) for n, t in record.items()},
               "finite": {n: bool(torch.isfinite(t).all()) for n, t in record.items()},
               "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
               "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else 0}
    return record, summary
