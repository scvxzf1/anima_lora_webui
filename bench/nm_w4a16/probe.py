"""Future synthetic first-order LoRA probe. No model, optimizer, or clock changes."""

import gc
import time

import torch
from torch import nn
from torch.nn import functional as F

from library.runtime.nm_w4a16.linear import FrozenNMLinear


def _relative(actual, expected):
    denominator = expected.float().norm()
    if denominator == 0:
        return {"relative_l2": None, "absolute_l2": float(actual.float().norm())}
    return {"relative_l2": float((actual.float() - expected.float()).norm() / denominator)}


def _make_source(args):
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    base = nn.Linear(args.in_features, args.out_features, bias=False, dtype=torch.bfloat16)
    with torch.no_grad():
        base.weight.copy_(torch.randn(base.weight.shape, generator=generator) / args.in_features**0.5)
    base.requires_grad_(False)
    data = {
        "x": torch.randn(args.tokens, args.in_features, generator=generator).bfloat16(),
        "g": (torch.randn(args.tokens, args.out_features, generator=generator) / args.out_features**0.5).bfloat16(),
        "a": (torch.randn(args.rank, args.in_features, generator=generator) / args.in_features**0.5).bfloat16(),
        "b": (torch.randn(args.out_features, args.rank, generator=generator) / args.rank**0.5).bfloat16(),
    }
    return base, data


def _iteration(layer, x, a, b, g):
    x.grad = a.grad = b.grad = None
    output = layer(x) + F.linear(F.linear(x, a), b)
    output.backward(g)
    return output


def _run_case(args, config, source, data, *, candidate):
    device = torch.device(args.device)
    torch.cuda.synchronize(device)
    initial_bytes = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    if candidate:
        layer = FrozenNMLinear.from_linear(source, config=config).to(device)
        if config.cache_sparse:
            layer.prepare_sparse()
    else:
        layer = nn.Linear(args.in_features, args.out_features, bias=False, dtype=torch.bfloat16, device=device)
        with torch.no_grad():
            layer.weight.copy_(source.weight)
        layer.requires_grad_(False)
    x = data["x"].to(device).requires_grad_()
    a = data["a"].to(device).requires_grad_()
    b = data["b"].to(device).requires_grad_()
    g = data["g"].to(device)
    torch.cuda.synchronize(device)
    preparation_ms = (time.perf_counter() - start) * 1000
    preparation_peak = torch.cuda.max_memory_allocated(device) - initial_bytes
    for _ in range(args.warmup):
        _iteration(layer, x, a, b, g)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    timings = []
    for _ in range(args.iterations):
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start_event.record()
        output = _iteration(layer, x, a, b, g)
        end_event.record()
        end_event.synchronize()
        timings.append(start_event.elapsed_time(end_event))
    tensors = {"output": output.detach().cpu(), "input_grad": x.grad.cpu(),
               "lora_a_grad": a.grad.cpu(), "lora_b_grad": b.grad.cpu()}
    weight_bytes = source.weight.numel() * source.weight.element_size()
    residency = layer.resident_bytes() if candidate else {"resident_bytes": weight_bytes}
    report = {
        "forward_and_backward_ms_mean": sum(timings) / len(timings),
        "forward_and_backward_ms_samples": timings,
        "preparation_ms_excluded_from_iterations": preparation_ms,
        "preparation_incremental_peak_allocated_bytes": preparation_peak,
        "iteration_incremental_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) - initial_bytes,
        "weight_residency": residency,
        "decode_and_compress_in_iteration": candidate and not config.cache_sparse,
    }
    del layer, x, a, b, g, output
    gc.collect()
    torch.cuda.empty_cache()
    return report, tensors


def run_probe(args, config):
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("select an explicit CUDA index")
    with torch.cuda.device(device):
        if torch.cuda.get_device_capability(device) < (8, 0):
            raise ValueError("this BF16 experiment requires SM80 or newer")
        source, data = _make_source(args)
        baseline, expected = _run_case(args, config, source, data, candidate=False)
        candidate, actual = _run_case(args, config, source, data, candidate=True)
        return {
            "status": "synthetic_probe_not_training_or_thermal_acceptance",
            "baseline": "dense_bf16",
            "candidate": f"2:4_uniform_int4_bf16_{config.backend}",
            "device": {"name": torch.cuda.get_device_name(device), "index": device.index,
                       "capability": torch.cuda.get_device_capability(device)},
            "configuration": vars(args),
            "bf16": baseline, "nm_w4a16": candidate,
            "speedup_vs_bf16": baseline["forward_and_backward_ms_mean"] / candidate["forward_and_backward_ms_mean"],
            "errors_vs_bf16": {name: _relative(actual[name], expected[name]) for name in actual},
            "limitations": ["fixed baseline-first order, not a sustained thermal benchmark",
                            "no checkpoint recomputation, optimizer, offload, or real DiT quality test",
                            "staged cuSPARSELt is not fused sparse W4A16"],
        }
