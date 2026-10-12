"""Eager input preparation and isolated callables for Qwen NF4 diagnostics."""
from __future__ import annotations


def _stage_forward(block, phase):
    """Keep the real block signature, but compile only the selected stage."""
    def forward(hidden_states, modulation, rotary_emb=None, attention_mask=None,
                target_token_mask=None, layer_cache=None, kv_cache_mode=None,
                cache_write_slice=None, segments=None, key_valid=None):
        if phase in ("norm1-entry", "norm2-entry"):
            import torch
            position = 0 if phase == "norm1-entry" else 1
            norm = block.img_norm1 if position == 0 else block.img_norm2
            modulated, _gate = block._modulate(
                norm.forward(hidden_states), modulation.chunk(2, dim=-1)[position], target_token_mask)
            # The actual NF4 Linear wrapper casts the modulated input to BF16.
            return modulated.to(torch.bfloat16)
        if phase in ("norm1-to-q", "norm1-to-k", "norm1-to-v"):
            modulated, _gate = block._modulate(
                block.img_norm1(hidden_states), modulation.chunk(2, dim=-1)[0], target_token_mask)
            # The patched projection owns NF4 casts and FP32 adapter parameters;
            # adapter rank compute follows BF16 autocast/base-output precision.
            return getattr(block.attn, "to_" + phase[-1])(modulated)
        if phase == "norm1":
            return block.img_norm1.forward(hidden_states)
        if phase == "norm2":
            return block.img_norm2.forward(hidden_states)
        if phase == "mlp":
            return block.img_mlp.forward(hidden_states)
        if phase == "mlp-suffix":
            modulated, gate2 = block._modulate(
                block.img_norm2(hidden_states), modulation.chunk(2, dim=-1)[1], target_token_mask)
            return hidden_states + gate2.tanh() * block.img_mlp(modulated)
        attention_input = hidden_states
        if phase in ("attention-prefix", "attention-residual"):
            attention_input, gate1 = block._modulate(
                block.img_norm1(hidden_states), modulation.chunk(2, dim=-1)[0], target_token_mask)
        attention = block.attn.forward(
            hidden_states=attention_input, attention_mask=attention_mask,
            rotary_emb=rotary_emb, layer_cache=layer_cache, kv_cache_mode=kv_cache_mode,
            cache_write_slice=cache_write_slice, segments=segments, key_valid=key_valid)
        if phase == "attention-residual":
            return hidden_states + gate1.tanh() * attention
        return attention
    return forward


def _prepared_hidden(block, inputs, phase):
    """Follow transformer_qwenimage21.py's real residual ordering, outside compile."""
    hidden = inputs["hidden_states"]
    mod1, mod2 = inputs["modulation"].chunk(2, dim=-1)
    target_mask = inputs["target_token_mask"]
    modulated, gate1 = block._modulate(block.img_norm1(hidden), mod1, target_mask)
    if phase == "attention":
        return modulated
    attention = block.attn.forward(
        hidden_states=modulated, attention_mask=inputs["attention_mask"],
        rotary_emb=inputs["rotary_emb"], layer_cache=inputs["layer_cache"],
        kv_cache_mode=inputs["kv_cache_mode"], cache_write_slice=inputs["cache_write_slice"],
        segments=inputs["segments"], key_valid=inputs["key_valid"])
    after_attention = hidden + gate1.tanh() * attention
    if phase in ("norm2", "norm2-entry", "mlp-suffix"):
        return after_attention
    return block._modulate(block.img_norm2(after_attention), mod2, target_mask)[0]


def prepare_stage(block, captured, phase, *, backend, tree, device="cuda"):
    """Preserve original CPU args/kwargs and retain the original bound artifact.

    Only the stage bound hidden input changes. No model is copied and no CPU IO
    preparation is performed by the compiled callable; connected stages retain
    their selected norm/modulation and residual operations inside that callable.
    ``device`` permits a small CPU fixture; production probes use CUDA.
    """
    if phase not in ("block", "norm1", "norm2", "norm1-entry", "norm2-entry", "attention", "mlp",
                     "attention-prefix", "attention-residual", "mlp-suffix",
                     "norm1-to-q", "norm1-to-k", "norm1-to-v"):
        raise ValueError(f"Unknown Qwen diagnostic phase: {phase}")
    modules = {"block": "block.forward", "norm1": "block.img_norm1.forward",
               "norm2": "block.img_norm2.forward",
               "norm1-entry": "img_norm1 -> frozen mod1 -> BF16 NF4 projection input",
               "norm2-entry": "img_norm2 -> frozen mod2 -> BF16 NF4 projection input",
               "attention": "block.attn.forward", "mlp": "block.img_mlp.forward",
               "attention-prefix": "img_norm1 -> frozen mod1 -> attention",
               "attention-residual": "img_norm1 -> frozen mod1 -> attention -> gate1 residual",
               "mlp-suffix": "img_norm2 -> frozen mod2 -> img_mlp -> gate2 residual"}
    origins = {
        "block": "Original full-model captured block hidden_states",
        "norm1": "Original full-model captured block hidden_states",
        "norm1-entry": "Original full-model captured block hidden_states",
        "attention-prefix": "Original full-model captured block hidden_states",
        "attention-residual": "Original full-model captured block hidden_states",
        "norm2": "eager no_grad: norm1/mod1 -> attention/gate1 residual",
        "norm2-entry": "eager no_grad: norm1/mod1 -> attention/gate1 residual",
        "attention": "eager no_grad: norm1(hidden) -> mod1/target_token_mask",
        "mlp": "eager no_grad: norm1/mod1 -> attention/gate1 residual -> norm2/mod2",
        "mlp-suffix": "eager no_grad: norm1/mod1 -> attention/gate1 residual",
    }
    for projection in ("q", "k", "v"):
        connected = "norm1-to-" + projection
        modules[connected] = "img_norm1 -> frozen mod1 -> real NF4 to_" + projection + " (FP32 adapter parameters)"
        origins[connected] = "Original full-model captured block hidden_states"
    metadata = {"phase": phase, "module": modules[phase], "input_origin": origins[phase],
                "preparation_backend": backend, "preparation_compiled": False,
                "prepared_input": "Original captured CPU tensor"}
    if phase == "block":
        return captured, block.forward, metadata
    stage = {**captured, "origin_bound": captured["bound"], "bound": dict(captured["bound"])}
    if phase in ("attention", "norm2", "norm2-entry", "mlp", "mlp-suffix"):
        import torch
        from diffusers.models.attention_dispatch import attention_backend
        inputs = tree(captured["bound"], lambda tensor: tensor.detach().to(device).clone())
        with attention_backend(backend), torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
            prepared = _prepared_hidden(block, inputs, phase)
        stage["bound"]["hidden_states"] = prepared.detach().cpu().clone()
        metadata["prepared_input"] = "Shared detached CPU clone, reloaded as independent hidden leaf per VJP"
    return stage, _stage_forward(block, phase), metadata


def phase_parameters(network, index, phase):
    """Restrict substage VJPs to their adapter group; norm1 has no adapters."""
    if phase in ("norm1", "norm2", "norm1-entry", "norm2-entry"):
        return []
    projection = "to_" + phase[-1] if phase in ("norm1-to-q", "norm1-to-k", "norm1-to-v") else None
    if projection:
        prefix = f"lora_unet_transformer_blocks_{index}_attn_{projection}."
        selected = [(name, parameter) for name, parameter in network.named_parameters() if name.startswith(prefix)]
        expected = {prefix + "lora_down.weight", prefix + "lora_up.weight"}
        if {name for name, _ in selected} != expected or not all(p.requires_grad for _, p in selected):
            raise AssertionError(f"Missing exact trainable {phase} adapter up/down coverage")
        return selected
    component = {"attention": "attn", "attention-prefix": "attn", "attention-residual": "attn",
                 "mlp": "img_mlp", "mlp-suffix": "img_mlp"}[phase]
    prefix = f"lora_unet_transformer_blocks_{index}_{component}_"
    selected = [(name, parameter) for name, parameter in network.named_parameters() if name.startswith(prefix)]
    if not selected or not all(parameter.requires_grad for _, parameter in selected):
        raise AssertionError(f"Missing trainable {phase} adapter parameters for block {index}")
    groups = {}
    for name, _ in selected:
        group, direction = name.rsplit(".", 2)[0], ".".join(name.rsplit(".", 2)[1:])
        groups.setdefault(group, set()).add(direction)
    expected = {"lora_down.weight", "lora_up.weight"}
    for group, directions in groups.items():
        if directions != expected:
            raise AssertionError(f"Incomplete {phase} adapter up/down coverage: {group}: {directions}")
    if phase in ("attention-prefix", "attention-residual", "mlp-suffix"):
        projections = ("to_q", "to_k", "to_v", "to_out_0") if component == "attn" else ("gate_layer", "proj", "out")
        expected_groups = {prefix + projection for projection in projections}
        if set(groups) != expected_groups:
            raise AssertionError(f"Unexpected {phase} adapter targets: {sorted(groups)}")
    return selected


def phase_nf4_modules(packed, phase):
    """Count only packed Linear modules executed by the selected stage."""
    if phase == "block":
        return len(packed)
    if phase in ("norm1", "norm2", "norm1-entry", "norm2-entry"):
        return 0
    if phase in ("norm1-to-q", "norm1-to-k", "norm1-to-v"):
        return sum(name == "attn.to_" + phase[-1] for name in packed)
    component = {"attention": "attn", "attention-prefix": "attn", "attention-residual": "attn",
                 "mlp": "img_mlp", "mlp-suffix": "img_mlp"}[phase]
    return sum(name.startswith(component + ".") for name in packed)
