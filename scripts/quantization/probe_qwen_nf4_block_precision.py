"""Diagnostic full-width Qwen NF4 block VJPs; never enables a production gate.

Capture the first acceptance Edit input with a no-grad full-model forward, then
run one block and one VJP graph at a time. No optimizer, swap, or training loop.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Existing local Qwen Image 2.1 NF4 .safetensors")
    parser.add_argument("--output-dir", default="output/tests/qwen_nf4_block_precision",
                        help="Parent for a new timestamp directory; existing results are never overwritten")
    parser.add_argument("--blocks", nargs="+", type=int, default=[0, 5, 19, 31])
    parser.add_argument("--phase", choices=("block", "norm1", "norm2", "norm1-entry", "norm2-entry", "attention", "mlp",
                                          "attention-prefix", "attention-residual", "mlp-suffix",
                                          "norm1-to-q", "norm1-to-k", "norm1-to-v"), default="block",
                        help="Isolated stage; sub-block phases require --no-modulation-gradient")
    parser.add_argument("--preset", choices=("casts", "unfused", "division"), default="casts")
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint", action=argparse.BooleanOptionalAction, default=False,
                        help="Non-reentrant checkpoint around each isolated block, never the full model")
    parser.add_argument("--modulation-gradient", action=argparse.BooleanOptionalAction, default=True,
                        help="Extra diagnostic VJP; disable to match frozen timestep/modulation training")
    parser.add_argument("--native-layer-norm", action=argparse.BooleanOptionalAction, default=False,
                        help="Diagnostic only: preserve native non-affine image LayerNorm forward/backward kernels")
    parser.add_argument("--native-layer-norm-targets", nargs="+", choices=("img_norm1", "img_norm2"),
                        default=["img_norm1", "img_norm2"], help="Image norms to patch when --native-layer-norm is enabled")
    parser.add_argument("--attention-backend", choices=("native", "_native_math"), default="native")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--edit-input-index", type=int, choices=(0, 1), default=0,
                        help="Fixed acceptance Edit input to capture; index 1 is not the post-step adapter state")
    return parser


def _options(preset):
    options = {"emulate_precision_casts": True}
    if preset == "unfused":
        options.update(epilogue_fusion=False, max_fusion_size=1)
    elif preset == "division":
        options["eager_numerics.division_rounding"] = True
    return options


def _tree(value, transform):
    import torch
    if isinstance(value, torch.Tensor):
        return transform(value)
    if isinstance(value, dict):
        return {key: _tree(item, transform) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_tree(item, transform) for item in value)
    if isinstance(value, list):
        return [_tree(item, transform) for item in value]
    if value is None or isinstance(value, (str, int, float, bool, slice)):
        return value
    raise TypeError(f"Unsupported block input type: {type(value).__qualname__}")


def _shape_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, dict):
        return {key: _shape_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_shape_tree(item) for item in value]
    return str(value) if isinstance(value, slice) else value


def _content(value):
    """Hash raw tensor bytes (including BF16) and every QuantState attribute."""
    import torch
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "sha256": hashlib.sha256(raw).hexdigest()}
    if isinstance(value, dict):
        return {str(key): _content(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_content(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (torch.dtype, torch.Size)):
        return str(value)
    if hasattr(value, "__dict__"):
        return {"type": type(value).__qualname__, "attributes": _content(vars(value))}
    raise TypeError(f"Cannot hash state value of type {type(value).__qualname__}")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _packed_state(block):
    from library.quantization import BF16AutocastLinear4bit
    result = {}
    for name, module in block.named_modules():
        if not isinstance(module, BF16AutocastLinear4bit):
            continue
        weight = module.weight
        if weight.requires_grad or weight.grad is not None:
            raise AssertionError(f"Packed NF4 weight is not frozen: {name}")
        if weight.device.type != "cuda" or weight.quant_state is None:
            raise AssertionError(f"Expected resident CUDA NF4 state: {name}")
        result[name] = {"packed": _content(weight), "quant_state": _content(weight.quant_state)}
    if not result:
        raise AssertionError("Selected block has no packed NF4 modules")
    return result


def _adapter_state(network):
    return {name: _content(value) for name, value in network.state_dict().items()}


def _assert_states(block, network, packed, adapter):
    actual_packed = _packed_state(block)
    if actual_packed != packed:
        changed = sorted(name for name in packed.keys() | actual_packed.keys()
                         if packed.get(name) != actual_packed.get(name))
        raise AssertionError(f"Packed NF4 payload or QuantState content changed: {changed}")
    if _adapter_state(network) != adapter:
        raise AssertionError("Adapter state changed without an optimizer step")
    if any(parameter.grad is not None for parameter in network.parameters()):
        raise AssertionError("autograd.grad unexpectedly populated adapter .grad")


def _capture(model, indices, backend, *, input_index=0):
    import torch
    from diffusers.models.attention_dispatch import attention_backend
    from scripts.quantization.qwen_nf4_compile_acceptance import _forward, edit_inputs
    captured, handles = {}, []

    def hook(index):
        def save(block, args, kwargs):
            if index in captured:
                raise AssertionError(f"Block {index} captured more than once")
            signature = inspect.signature(block.forward)
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            def copy(tensor):
                return tensor.detach().cpu().clone()

            captured[index] = {"args": _tree(args, copy), "kwargs": _tree(kwargs, copy),
                               "bound": _tree(dict(bound.arguments), copy)}
        return save

    try:
        for index in indices:
            handles.append(model.transformer_blocks[index].register_forward_pre_hook(hook(index), with_kwargs=True))
        with attention_backend(backend), torch.no_grad():
            _forward(model, edit_inputs()[input_index], grad=False)
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
        model.disable_gradient_checkpointing()
    if set(captured) != set(indices):
        raise AssertionError(f"Missing captured blocks: {sorted(set(indices) - set(captured))}")
    return captured


def _parameters(network, index, phase="block"):
    if phase != "block":
        from scripts.quantization.qwen_nf4_block_stages import phase_parameters
        return phase_parameters(network, index, phase)
    prefix = f"lora_unet_transformer_blocks_{index}_"
    selected = [(name, parameter) for name, parameter in network.named_parameters() if name.startswith(prefix)]
    if not selected or not all(parameter.requires_grad for _, parameter in selected):
        raise AssertionError(f"Missing trainable adapter parameters for block {index}")
    for component in ("attn", "mlp"):
        for direction in ("lora_down.weight", "lora_up.weight"):
            if not any(component in name and name.endswith(direction) for name, _ in selected):
                raise AssertionError(f"Block {index} lacks {component} {direction} coverage")
    return selected


def _clear_graphs():
    import torch
    gc.collect()
    torch.cuda.empty_cache()


def _vjp(forward, captured, parameters, cotangent, *, backend, checkpoint, calls,
         modulation_gradient=True, fp32_output=False, output_dtype=None):
    import torch
    from diffusers.models.attention_dispatch import attention_backend
    from torch.utils.checkpoint import checkpoint as checkpoint_fn
    kwargs = _tree(captured["bound"], lambda tensor: tensor.detach().to("cuda").clone())
    hidden, modulation = kwargs["hidden_states"], kwargs["modulation"]
    hidden.requires_grad_(True)
    modulation.requires_grad_(modulation_gradient)
    leaves = [hidden, *([modulation] if modulation_gradient else []), *(parameter for _, parameter in parameters)]
    names = ["hidden_gradient", *(["modulation_gradient"] if modulation_gradient else []),
             *(name for name, _ in parameters)]
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()

    def invoke(**inputs):
        calls[0] += 1  # Outside the compiled callable, including checkpoint recomputation.
        return forward(**inputs)

    with attention_backend(backend), torch.autocast("cuda", dtype=torch.bfloat16):
        output = checkpoint_fn(invoke, use_reentrant=False, **kwargs) if checkpoint else invoke(**kwargs)
        expected_dtype = output_dtype or (torch.float32 if fp32_output else torch.bfloat16)
        if not isinstance(output, torch.Tensor) or output.dtype != expected_dtype:
            raise AssertionError(f"Expected single {expected_dtype} output")
        gradients = torch.autograd.grad(output, leaves,
                                       grad_outputs=cotangent.to(device="cuda", dtype=output.dtype))
    record = {"output": output.detach().cpu().clone()}
    record.update({name: gradient.detach().cpu().clone() for name, gradient in zip(names, gradients)})
    torch.cuda.synchronize()
    summary = {"elapsed_seconds": time.monotonic() - started, "calls": calls[0],
               "output_dtype": str(output.dtype),
               "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
               "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
               "finite": {name: bool(torch.isfinite(tensor).all()) for name, tensor in record.items()}}
    del kwargs, hidden, modulation, leaves, output, gradients
    return record, summary


def _compare(reference, actual):
    import torch
    metrics, failures = {}, []
    if reference.keys() != actual.keys():
        raise AssertionError("Incomplete output/gradient comparison")
    for name, expected in reference.items():
        value = actual[name]
        atol = 2e-4 if name.startswith("lora_") else 2e-3
        delta = value.float() - expected.float()
        relative = float(delta.norm() / expected.float().norm().clamp_min(1e-12))
        mismatch = value.contiguous().reshape(-1).view(torch.uint8) != expected.contiguous().reshape(-1).view(torch.uint8)
        unequal = int(mismatch.reshape(value.numel(), value.element_size()).any(dim=1).sum())
        close_error = None
        try:
            torch.testing.assert_close(value, expected, rtol=2e-2, atol=atol)
        except AssertionError as error:
            close_error = str(error)
            failures.append({"key": name, "error": close_error})
        if not torch.isfinite(value).all() or not torch.isfinite(expected).all():
            failures.append({"key": name, "error": "Nonfinite comparison tensor"})
        if not torch.isfinite(torch.tensor(relative)) or relative > .01:
            failures.append({"key": name, "error": f"relative L2 exceeds 1% or is nonfinite: {relative}"})
        metrics[name] = {"max_abs": float(delta.abs().max()), "relative_l2": relative,
                         "bitwise_unequal": unequal, "numel": value.numel(), "rtol": .02,
                         "atol": atol, "assert_close_error": close_error}
    return {"passed": not failures, "metrics": metrics, "failures": failures}


def _counter_snapshot():
    from torch._dynamo.utils import counters
    return {"unique_graphs": int(counters["stats"]["unique_graphs"]),
            "graph_breaks": dict(counters["graph_break"]),
            "inductor_counters": dict(counters["inductor"])}


def _probe_block(model, network, captured, index, args, output, adapter, report):
    import torch
    from torch._dynamo.utils import counters
    from scripts.quantization.qwen_nf4_block_stages import phase_nf4_modules, prepare_stage
    block = model.transformer_blocks[index]
    parameters = _parameters(network, index, args.phase)
    packed = _packed_state(block)
    _assert_states(block, network, packed, adapter)
    captured, stage_forward, stage = prepare_stage(block, captured, args.phase,
                                                   backend=args.attention_backend, tree=_tree)
    generator = torch.Generator(device="cpu").manual_seed(172)
    output_dtype = None
    if args.phase in ("norm1-to-q", "norm1-to-k", "norm1-to-v"):
        from diffusers.models.attention_dispatch import attention_backend
        inputs = _tree(captured["bound"], lambda tensor: tensor.detach().to("cuda").clone())
        with attention_backend(args.attention_backend), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            stage_output = stage_forward(**inputs)
        output_dtype = stage_output.dtype
        cotangent = torch.randn(stage_output.shape, generator=generator).to(output_dtype)
        stage["cotangent_origin"] = "Independent seed172 same-shape stage cotangent; not a real training loss VJP"
        stage["output"] = _shape_tree(stage_output)
        del inputs, stage_output
    else:
        cotangent = torch.randn(captured["bound"]["hidden_states"].shape, generator=generator).bfloat16()
    torch.save({"captured": captured, "cotangent": cotangent}, output / f"block-{index}-inputs.pt")
    result = {"inputs": _shape_tree(captured), "lora_parameters": [name for name, _ in parameters],
              "stage": stage, "nf4_snapshot_scope": "whole_block",
              "phase_nf4_modules": phase_nf4_modules(packed, args.phase),
              "nf4_modules": len(packed), "packed_before": packed,
              "packed_sha256": _digest(packed), "adapter_sha256": _digest(adapter), "runs": {}}
    if output_dtype is not None:
        result["source_bound_sha256"] = _digest(_tree(captured["origin_bound"], _content))
        result["cotangent_sha256"] = _digest(_content(cotangent))
    report["blocks"][str(index)] = result
    report["current_block"] = index
    _write_result(output, report)
    records = {}
    for label in ("eager", "repeat_eager", "compiled"):
        report["current_run"] = label
        _write_result(output, report)
        _clear_graphs()
        _assert_states(block, network, packed, adapter)
        torch._dynamo.reset()
        counters.clear()
        calls = [0]
        forward = stage_forward
        if label == "compiled":
            forward = torch.compile(forward, backend="inductor", fullgraph=True,
                                    dynamic=args.dynamic, options=_options(args.preset))
        record, summary = _vjp(forward, captured, parameters, cotangent,
                               backend=args.attention_backend, checkpoint=args.checkpoint, calls=calls,
                               modulation_gradient=args.modulation_gradient,
                               fp32_output=args.phase in ("norm1", "norm2"), output_dtype=output_dtype)
        summary.update(_counter_snapshot())
        result["runs"][label] = summary
        _write_result(output, report)
        records[label] = record
        torch.save(record, output / f"block-{index}-{label}.pt")
        del forward, record
        _clear_graphs()
        _assert_states(block, network, packed, adapter)
        if label == "compiled":
            if summary["unique_graphs"] <= 0 or calls[0] <= 0 or summary["graph_breaks"]:
                raise AssertionError(f"Missing real fullgraph Inductor coverage: {summary}")
        print(f"block {index} {label}: peak={summary['peak_cuda_bytes']} bytes calls={calls[0]}", flush=True)
        if label == "repeat_eager":
            result["baseline"] = _compare(records["eager"], records["repeat_eager"])
            result["compiled_attribution_valid"] = result["baseline"]["passed"]
            _write_result(output, report)
            print(f"block {index}: eager repeat baseline passed={result['baseline']['passed']}", flush=True)
    result["compiled_comparison"] = _compare(records["eager"], records["compiled"])
    result["compiled_attribution_valid"] = result["baseline"]["passed"]
    result["packed_after"] = _packed_state(block)
    result["adapter_unchanged"] = _adapter_state(network) == adapter
    result["passed"] = result["baseline"]["passed"] and result["compiled_comparison"]["passed"]
    for comparison in ("baseline", "compiled_comparison"):
        worst = sorted(result[comparison]["metrics"].items(), key=lambda item: item[1]["relative_l2"], reverse=True)[:5]
        print(json.dumps({"block": index, "comparison": comparison, "passed": result[comparison]["passed"],
                          "compiled_attribution_valid": result["compiled_attribution_valid"], "worst": worst}), flush=True)
    del records
    return result


def _write_result(output, report):
    (output / "result.json").write_text(json.dumps(report, indent=2, default=str) + "\n")


def _run(args, output, report):
    import bitsandbytes
    import diffusers
    import torch
    from library.models.qwen_image_2_1.weights import load_qwen_image_2_1_transformer
    from scripts.quantization.qwen_nf4_compile_acceptance import _attach_network
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This full-width diagnostic requires a BF16-capable CUDA GPU")
    if args.rank <= 0 or len(set(args.blocks)) != len(args.blocks) or any(index < 0 or index >= 32 for index in args.blocks):
        raise ValueError("Require rank > 0 and unique block indices in 0..31")
    options = _options(args.preset)
    from torch._inductor import config
    available = config.shallow_copy_dict()
    for key in options:
        if key not in available:
            raise RuntimeError(f"Installed PyTorch does not support diagnostic option {key}")
    report["versions"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                          "diffusers": diffusers.__version__, "bitsandbytes": bitsandbytes.__version__}
    report["gpu"] = torch.cuda.get_device_name()
    report["compile_options"] = options
    torch.manual_seed(41)
    torch.cuda.reset_peak_memory_stats()
    model = load_qwen_image_2_1_transformer(str(Path(args.model).expanduser()),
                                          dtype=torch.bfloat16, device="cuda").requires_grad_(False).train()
    if getattr(model, "_mixed_nf4_family", None) != "qwen_image_2_1":
        raise ValueError("Require an existing mixed-NF4 Qwen Image 2.1 checkpoint")
    width = model.transformer_blocks[0].attn.to_q.in_features
    mlp_width = model.transformer_blocks[0].img_mlp.gate_layer.out_features
    if len(model.transformer_blocks) != 32 or width != 4096 or mlp_width != 12288:
        raise ValueError(f"Require full 32-block width4096/mlp12288 model; got {len(model.transformer_blocks)}/{width}/{mlp_width}")
    report["model_geometry"] = {"blocks": len(model.transformer_blocks), "width": width, "mlp_width": mlp_width}
    network = _attach_network(model, args.rank)
    report["native_layer_norm"] = {"enabled": False, "patch_count": 0}
    if args.native_layer_norm:
        from scripts.quantization.qwen_nf4_native_layer_norm import patch_native_layer_norm
        report["native_layer_norm"] = patch_native_layer_norm(model, norm_names=args.native_layer_norm_targets)
    adapter = _adapter_state(network)
    torch.save({key: value.detach().cpu().clone() for key, value in network.state_dict().items()}, output / "initial_adapter.pt")
    captured = _capture(model, args.blocks, args.attention_backend, input_index=args.edit_input_index)
    report["capture_peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    report["capture_peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
    report["adapter_sha256"] = _digest(adapter)
    _clear_graphs()
    for index in args.blocks:
        _probe_block(model, network, captured[index], index, args, output, adapter, report)
        _write_result(output, report)
    report["status"] = "passed" if all(result["passed"] for result in report["blocks"].values()) else "failed"
    del network, model, captured
    _clear_graphs()


def main():
    parser = _parser()
    args = parser.parse_args()
    if args.phase != "block" and args.modulation_gradient:
        parser.error("Sub-block phases require --no-modulation-gradient (modulation is frozen)")
    output = Path(args.output_dir).expanduser() / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "arguments": vars(args), "blocks": {},
              "diagnostic_only": True, "seeds": {"adapter": 41, "edit_inputs": 171, "cotangent": 172},
              "edit_input": {"index": args.edit_input_index,
                             "shape_source": f"scripts.quantization.qwen_nf4_compile_acceptance.edit_inputs()[{args.edit_input_index}]",
                             "state": "fixed acceptance input; not the step-1-updated adapter state"},
              "vjp": "autograd.grad(output, [hidden, modulation, *block_lora], grad_outputs=BF16_cotangent)",
              "limits": "One isolated block graph at a time; full base remains resident; no optimizer or full-model backward"}
    if args.phase != "block":
        report["vjp"] = "autograd.grad(stage_output, [prepared_hidden, *phase_lora], grad_outputs=BF16_cotangent)"
    if args.phase in ("norm1-to-q", "norm1-to-k", "norm1-to-v"):
        report["vjp"] = "Stage-only autograd.grad(real_projection_output, [original_hidden, projection_down, projection_up], independent same-shape cotangent); not training loss"
    elif not args.modulation_gradient and args.phase == "block":
        report["vjp"] = "autograd.grad(output, [hidden, *block_lora], grad_outputs=BF16_cotangent)"
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Diagnostic artifacts: {output}", flush=True)
    try:
        path = Path(args.model).expanduser()
        if not path.is_file() or path.suffix != ".safetensors":
            raise ValueError(f"Require an existing local .safetensors checkpoint: {path}")
        _run(args, output, report)
    except BaseException as error:
        report.update(status="error", error={"type": type(error).__qualname__, "message": str(error),
                                             "traceback": traceback.format_exc()})
        if "torch" in sys.modules:
            report["error_counters"] = _counter_snapshot()
        _write_result(output, report)
        raise
    _write_result(output, report)
    print(json.dumps({"status": report["status"], "result": str(output / "result.json")}), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
