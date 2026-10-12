"""Shared real-Inductor Qwen NF4 edit-training acceptance machinery."""
from __future__ import annotations

import gc
import time

import torch
from torch._dynamo.utils import counters

from library.models.qwen_image_2_1.block_swap import enable_qwen_image_2_1_block_swap
from library.models.qwen_image_2_1.compile import compile_qwen_image_2_1_blocks
from library.models.qwen_image_2_1.family import forward_for_loss
from library.quantization import BF16AutocastLinear4bit
from networks.lora_anima.factory import create_network


def tiny_model():
    from diffusers import QwenImage21Transformer2DModel
    return QwenImage21Transformer2DModel(num_layers=4, num_attention_heads=1,
        attention_head_dim=128, context_in_dim=4096, mlp_ratio=1).bfloat16()


def tiny_loader(path):
    from library.quantization import load_mixed_nf4_into
    from library.models.qwen_image_2_1.weights import _materialize_qwen_derived_tensors
    with torch.device("meta"):
        model = tiny_model()
    load_mixed_nf4_into(model, path, family="qwen_image_2_1", dtype=torch.bfloat16, device="cpu")
    _materialize_qwen_derived_tensors(model)
    return model


def edit_inputs():
    generator = torch.Generator().manual_seed(171)
    result = []
    for width, length in ((2, 3), (4, 5)):
        result.append((torch.randn(1, 64, 2, width, generator=generator).bfloat16(),
            torch.randn(1, length, 4096, generator=generator).bfloat16(),
            torch.randn(1, 64, 2, 2, generator=generator).bfloat16()))
    return result


def _forward(model, inputs, grad=True):
    x, text, reference = (v.cuda() for v in inputs)
    x.requires_grad_(grad)
    mask = torch.ones(text.shape[:2], dtype=torch.bool, device="cuda")
    slots = torch.zeros_like(mask)
    slots[:, 0] = True
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = forward_for_loss(model, x, text, mask, torch.tensor([.5], device="cuda"),
            reference_latents=reference, image_slot_mask=slots)
    return x, output


def _snapshot_packed(model):
    return [(m, m.weight, m.weight.quant_state, m.weight.detach().cpu().clone())
        for m in model.modules() if isinstance(m, BF16AutocastLinear4bit)]


def _check_packed(model, packed):
    assert packed, "No packed NF4 modules were exercised"
    model.offloader.restore_blocks_to_device(model.transformer_blocks, torch.device("cuda"))
    torch.cuda.synchronize()
    for module, weight, state, payload in packed:
        assert module.weight is weight and weight.quant_state is state
        assert not weight.requires_grad and weight.grad is None
        assert weight.device.type == state.absmax.device.type == "cuda"
        if module.bias is not None:
            assert module.bias.device.type == "cuda"
        torch.testing.assert_close(weight.detach().cpu(), payload, rtol=0, atol=0)


def _finite_nonzero(tensor, name):
    assert torch.isfinite(tensor).all(), name
    assert tensor.count_nonzero(), name


def _attach_network(model, rank):
    network = create_network(1., rank, rank, None, [], model, model_family="qwen_image_2_1",
        unet_target_replace_modules=["QwenImage21TransformerBlock"])
    network.apply_to([], model, apply_text_encoder=False, apply_unet=True)
    network.to(device="cuda", dtype=torch.float32)
    with torch.no_grad():
        for name, parameter in network.named_parameters():
            if name.endswith("lora_up.weight"):
                parameter.normal_(0, .01)
    assert any("mlp" in name for name, _ in network.named_parameters())
    assert any("attn" in name for name, _ in network.named_parameters())
    model.enable_gradient_checkpointing()
    network.enable_gradient_checkpointing()
    return network


def _track_compiled_blocks(model, count):
    calls = [0] * count

    def tracked(index, forward):
        def call(*args, **kwargs):
            calls[index] += 1
            return forward(*args, **kwargs)
        return call

    adapter = model._qwen_image_2_1_block_swap_adapter
    for index in range(count):
        adapter.inner_forwards[index] = tracked(index, adapter.inner_forwards[index])
    return calls


def _training_records(model, network, compiled):
    optimizer = torch.optim.SGD(network.parameters(), lr=.001)
    records = []
    for index, inputs in enumerate(edit_inputs()):
        optimizer.zero_grad(set_to_none=True)
        x, output = _forward(model, inputs)
        loss = output.float().square().mean()
        loss.backward()
        _finite_nonzero(x.grad, "latent gradient")
        gradients = {}
        for name, parameter in network.named_parameters():
            assert parameter.grad is not None, name
            _finite_nonzero(parameter.grad, name)
            gradients[name] = parameter.grad.detach().cpu().clone()
        before_step = [parameter.detach().clone() for parameter in network.parameters()]
        optimizer.step()
        assert any(not torch.equal(before, parameter.detach())
            for before, parameter in zip(before_step, network.parameters())), "No optimizer update"
        del before_step
        records.append({"output": output.detach().cpu(), "loss": loss.detach().cpu(),
            "input_gradient": x.grad.detach().cpu(), "gradients": gradients,
            "adapter": {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}})
        if index == 0:
            model.switch_block_swap_for_inference()
            with torch.no_grad():
                records[-1]["inference"] = _forward(model, inputs, False)[1].cpu()
            model.switch_block_swap_for_training()
        torch.cuda.synchronize()
        print(f"{'compiled' if compiled else 'eager'}: step {index + 1} loss={loss.item():.8g}", flush=True)
    return records


def run_case(loader, *, compiled, count, restore_mode, scope, rank=4, dynamic_seq: bool = True):
    """Load/run/release one model; outputs stay on CPU for sequential A/B."""
    torch.manual_seed(41)
    torch._dynamo.reset()
    counters.clear()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model = loader().requires_grad_(False).train()
    network = packed = None
    try:
        assert getattr(model, "_mixed_nf4_family", None) == "qwen_image_2_1", "Expected Qwen NF4 checkpoint"
        packed = _snapshot_packed(model)
        assert packed, "Expected packed NF4 modules"
        print(f"{'compiled' if compiled else 'eager'}: loaded {len(model.transformer_blocks)} blocks", flush=True)
        enable_qwen_image_2_1_block_swap(model, count, torch.device("cuda"),
            restore_mode=restore_mode, transfer_dtype="bf16")
        model.move_to_device_except_swap_blocks(torch.device("cuda"))
        model.prepare_block_swap_before_forward(free_cache=False)
        network = _attach_network(model, rank)
        initial_adapter = {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}
        compiled_count = compile_qwen_image_2_1_blocks(model, mode=None,
            dynamic_seq=dynamic_seq, scope=scope) if compiled else 0
        block_calls = _track_compiled_blocks(model, compiled_count)
        records = _training_records(model, network, compiled)
        _check_packed(model, packed)
        graph_count = int(counters["stats"]["unique_graphs"])
        graph_breaks = dict(counters["graph_break"])
        if compiled:
            assert graph_count > 0, "No real Dynamo/Inductor graphs were compiled"
            assert compiled_count > 0 and all(n >= 2 for n in block_calls), "Missing compiled block coverage"
            assert not graph_breaks, f"NF4 block graph breaks: {graph_breaks}"
        return {"records": records, "unique_graphs": graph_count,
            "initial_adapter": initial_adapter, "compiled_block_calls": block_calls,
            "graph_breaks": graph_breaks, "nf4_modules": len(packed),
            "inductor_counters": dict(counters["inductor"]), "compiled_blocks": compiled_count,
            "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()}
    finally:
        offloader = getattr(model, "offloader", None)
        if offloader is not None:
            for index in list(offloader.futures):
                offloader._wait_blocks_move(index, phase="acceptance_cleanup")
            offloader.thread_pool.shutdown(wait=True)
        del network, model, packed
        gc.collect()
        torch.cuda.empty_cache()


def compare_runs(eager, compiled, *, report_path=None):
    import json
    metrics, failures = {}, []
    torch.testing.assert_close(eager["initial_adapter"], compiled["initial_adapter"], rtol=0, atol=0)
    assert len(eager["records"]) == len(compiled["records"]) == 2, "Incomplete training comparison"
    for index, (expected, actual) in enumerate(zip(eager["records"], compiled["records"])):
        assert expected.keys() == actual.keys(), "Missing acceptance outputs"
        for field in expected:
            if isinstance(expected[field], dict):
                assert expected[field].keys() == actual[field].keys(), "Missing adapter comparison"
            pairs = expected[field].items() if isinstance(expected[field], dict) else [(field, expected[field])]
            for name, reference in pairs:
                key = f"step{index}/{field}/{name}"
                value = actual[field][name] if isinstance(actual[field], dict) else actual[field]
                tolerance = 2e-4 if field in ("gradients", "adapter") else 2e-3
                try:
                    torch.testing.assert_close(value, reference, rtol=2e-2, atol=tolerance)
                except AssertionError as error:
                    failures.append({"key": key, "error": str(error)})
                delta = value.float() - reference.float()
                relative = float(delta.norm() / reference.float().norm().clamp_min(1e-12))
                metrics[key] = {"max_abs": float(delta.abs().max()), "relative_l2": relative}
                if relative > .01:
                    failures.append({"key": key, "error": f"relative L2 exceeds 1%: {relative}"})
    report = {"metrics": metrics, "failures": failures,
        "runs": {k: {field: value for field, value in v.items() if field not in ("records", "initial_adapter")}
            for k, v in (("eager", eager), ("compiled", compiled))}}
    if report_path is not None:
        report_path.write_text(json.dumps(report, indent=2))
    worst = sorted(metrics.items(), key=lambda item: item[1]["relative_l2"], reverse=True)[:5]
    print(json.dumps({"unique_graphs": compiled["unique_graphs"], "failures": len(failures), "worst": worst}))
    assert not failures, f"{len(failures)} numerical checks failed: {failures[:5]}"
    return metrics
