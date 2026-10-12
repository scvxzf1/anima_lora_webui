"""Temporary first-forward hooks for the real step-zero full-loss adjoint."""
from __future__ import annotations

from contextlib import contextmanager

from scripts.quantization.qwen_nf4_attention_core_stages import clone_strided


@contextmanager
def capture_attention_adjoint(model, indices, *, capture_projection_outputs=False):
    """Capture CPU outputs/cotangents, ignoring non-reentrant recomputation.

    No graph tensors are retained. Hooks on the first graph's tensors receive
    the actual backward adjoint even when checkpointing recomputes the block.
    """
    from diffusers.models.transformers import transformer_qwenimage21 as implementation

    records = {index: {"outputs": {}, "cotangents": {}} for index in indices}
    if capture_projection_outputs:
        for record in records.values():
            record["projection_inputs"] = {}
    handles, tensor_handles, seen = [], [], set()
    active = None
    rotary_calls = 0
    original_rotary = implementation.apply_rotary_emb_qwen

    def save(index, name, tensor):
        record = records[index]
        if name in record["outputs"]:
            raise AssertionError(f"Duplicate first-forward capture: block {index}/{name}")
        if not tensor.requires_grad:
            raise AssertionError(f"Capture is not graph-recording: block {index}/{name}")
        record["outputs"][name] = clone_strided(tensor, "cpu")

        def gradient_hook(gradient):
            if name in record["cotangents"]:
                raise AssertionError(f"Duplicate adjoint: block {index}/{name}")
            record["cotangents"][name] = clone_strided(gradient, "cpu")

        tensor_handles.append(tensor.register_hook(gradient_hook))

    def enter(index):
        def hook(_block, _args):
            nonlocal active, rotary_calls
            active = None if index in seen else index
            rotary_calls = 0
            seen.add(index)
        return hook

    def leave(index):
        def hook(_block, _args, output):
            nonlocal active
            if active == index:
                save(index, "block_out", output)
            active = None
        return hook

    def norm_hook(name):
        def hook(_module, _args, output):
            if active is not None:
                save(active, name, output)
        return hook

    def attention_hook(module, args):
        if active is not None:
            tensor = args[0]
            if tensor.ndim != 3:
                raise AssertionError("Expected flattened BLH attention output")
            # Register on the actual flattened tensor, not an unused view.
            save(active, "attention_flat", tensor)

    def projection_hook(name):
        def hook(_module, args, output):
            if active is not None:
                records[active]["projection_inputs"][name] = clone_strided(args[0], "cpu")
                save(active, name, output)
        return hook

    def rotary(*args, **kwargs):
        nonlocal rotary_calls
        output = original_rotary(*args, **kwargs)
        if active is not None:
            if rotary_calls >= 2:
                raise AssertionError("Expected exactly Q/K complex RoPE calls")
            save(active, ("rope_query", "rope_key")[rotary_calls], output)
            rotary_calls += 1
        return output

    try:
        implementation.apply_rotary_emb_qwen = rotary
        for index in indices:
            block = model.transformer_blocks[index]
            handles.extend((block.register_forward_pre_hook(enter(index)),
                            block.register_forward_hook(leave(index)),
                            block.attn.norm_q.register_forward_hook(norm_hook("norm_query")),
                            block.attn.norm_k.register_forward_hook(norm_hook("norm_key")),
                            block.attn.to_out[0].register_forward_pre_hook(attention_hook)))
            if capture_projection_outputs:
                for name, module in (("query", block.attn.to_q), ("key", block.attn.to_k),
                                     ("value", block.attn.to_v), ("out", block.attn.to_out[0])):
                    handles.append(module.register_forward_hook(projection_hook(f"projection_{name}")))
        yield records
    finally:
        implementation.apply_rotary_emb_qwen = original_rotary
        for handle in (*handles, *tensor_handles):
            handle.remove()


def phase_adjoint(record, phase, heads):
    """Composite phases share the cotangent at their true final output."""
    names = (("norm_query", "norm_key") if phase == "qk-norm" else
             ("rope_query", "rope_key") if phase in ("rope", "qk-rope") else
             ("attention_flat",))
    outputs = tuple(record["outputs"][name] for name in names)
    cotangents = tuple(record["cotangents"][name] for name in names)
    if names == ("attention_flat",):
        outputs = tuple(clone_strided(t.unflatten(-1, (heads, -1)), "cpu") for t in outputs)
        cotangents = tuple(clone_strided(t.unflatten(-1, (heads, -1)), "cpu") for t in cotangents)
    return outputs, cotangents


def capture_full_loss(model, network, indices, backend, *, adapter, packed, output,
                      capture_projection_outputs=False):
    """One checkpointed swap-24 BF16 edit forward/backward, without an update."""
    import torch
    from diffusers.models.attention_dispatch import attention_backend
    from library.models.qwen_image_2_1.block_swap import enable_qwen_image_2_1_block_swap
    from scripts.quantization.probe_qwen_nf4_block_precision import (
        _adapter_state, _assert_states, _digest, _packed_state,
    )
    from scripts.quantization.qwen_nf4_compile_acceptance import _forward, edit_inputs

    offloader = None
    try:
        model.enable_gradient_checkpointing()
        network.enable_gradient_checkpointing()
        enable_qwen_image_2_1_block_swap(model, 24, torch.device("cuda"),
                                       restore_mode="foreach", transfer_dtype="bf16")
        offloader = model.offloader
        model.move_to_device_except_swap_blocks(torch.device("cuda"))
        model.prepare_block_swap_before_forward(free_cache=False)
        with attention_backend(backend), capture_attention_adjoint(
            model, indices, capture_projection_outputs=capture_projection_outputs,
        ) as records:
            latent, result = _forward(model, edit_inputs()[0])
            loss = result.float().square().mean()
            loss.backward()
            if latent.grad is None:
                raise AssertionError("Missing full-loss latent gradient")
            artifact = {"blocks": records, "loss": loss.detach().cpu(),
                        "output": clone_strided(result, "cpu"),
                        "gradleaf": clone_strided(latent.grad, "cpu"),
                        "inputs": edit_inputs()[0]}
        expected = {"block_out", "norm_query", "norm_key", "rope_query", "rope_key", "attention_flat"}
        if capture_projection_outputs:
            expected.update(f"projection_{name}" for name in ("query", "key", "value", "out"))
        for index, record in records.items():
            if set(record["outputs"]) != expected or set(record["cotangents"]) != expected:
                raise AssertionError(f"Missing first-forward outputs/adjoints for block {index}")
        metadata = {"source": "full-loss", "actual_loss": float(loss.detach()),
                    "input_source": "edit_inputs()[0]", "initial_state": True,
                    "adjoint": "true full-network step0 output.float().square().mean()",
                    "optimizer_updates": 0, "checkpoint": "full non-reentrant",
                    "blocks_to_swap": 24, "restore_mode": "foreach", "transfer_dtype": "bf16",
                    "adapter_sha256_before": _digest(adapter),
                    "packed_sha256_before": {i: _digest(p) for i, p in packed.items()}}
    finally:
        try:
            if offloader is not None:
                for index in list(offloader.futures):
                    offloader._wait_blocks_move(index, phase="full_loss_capture_cleanup")
                offloader.restore_blocks_to_device(model.transformer_blocks, torch.device("cuda"))
        finally:
            if offloader is not None:
                offloader.thread_pool.shutdown(wait=True)
            model.disable_gradient_checkpointing()
            # Core preparation never calls block.forward; disable swap dispatch
            # as well so the closed executor cannot accidentally be scheduled.
            if offloader is not None:
                model.blocks_to_swap = 0
            for parameter in (*model.parameters(), *network.parameters()):
                parameter.grad = None
    for index in indices:
        _assert_states(model.transformer_blocks[index], network, packed[index], adapter)
    metadata.update(state_unchanged=True, adapter_sha256_after=_digest(_adapter_state(network)),
                    packed_sha256_after={i: _digest(_packed_state(model.transformer_blocks[i])) for i in indices})
    artifact["metadata"] = metadata
    torch.save(artifact, output / "full_loss_capture.pt")
    return records, metadata
