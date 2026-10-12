"""Isolated real NF4/LoRA Q/K/V/out projection VJPs; diagnostic only, no update."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.quantization.probe_qwen_nf4_block_precision import (  # noqa: E402
    _adapter_state, _assert_states, _clear_graphs, _compare,
    _counter_snapshot, _digest, _packed_state, _tree, _write_result,
)
from scripts.quantization.qwen_nf4_attention_core_stages import clone_strided  # noqa: E402

BOUNDARIES = {"q": "projection_query", "k": "projection_key",
              "v": "projection_value", "out": "projection_out"}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Existing local mixed-NF4 Qwen 2.1 .safetensors")
    parser.add_argument("--blocks", nargs="+", type=int, default=[4])
    parser.add_argument("--projections", nargs="+", choices=tuple(BOUNDARIES), default=list(BOUNDARIES))
    parser.add_argument("--cotangent-source", choices=("full-loss", "random"), default="full-loss")
    parser.add_argument("--input-dtype", choices=("original", "bf16"), default="original")
    parser.add_argument("--forward-mode", choices=("lora", "base"), default="lora")
    parser.add_argument("--output-dir", default="output/tests/qwen_nf4_projection")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--attention-backend", choices=("native", "_native_math"), default="native")
    return parser


def projection_target(block, network, index, projection, mode):
    """Use patched module forward, or the adapter's saved BF16 NF4 forward."""
    import torch
    module = block.attn.to_out[0] if projection == "out" else getattr(block.attn, f"to_{projection}")
    suffix = "to_out_0" if projection == "out" else f"to_{projection}"
    prefix = f"lora_unet_transformer_blocks_{index}_attn_{suffix}."
    selected = [(name, parameter) for name, parameter in network.named_parameters()
                if name.startswith(prefix)]
    if (len(selected) != 2 or {name.removeprefix(prefix) for name, _ in selected} !=
            {"lora_down.weight", "lora_up.weight"} or
            any(not p.requires_grad or p.dtype != torch.float32 for _, p in selected)):
        raise AssertionError(f"Expected exactly two FP32 trainable LoRA directions: {prefix}")
    adapter = getattr(module.forward, "__self__", None)
    if adapter is None or not hasattr(adapter, "org_forward"):
        raise AssertionError(f"Projection is not monkey-patched by an actual LoRA: {prefix}")
    if adapter.org_forward.__self__ is not module:
        raise AssertionError("Saved LoRA base forward does not belong to the projection")
    return (module.forward, selected) if mode == "lora" else (adapter.org_forward, [])


def projection_vjp(forward, prepared, cotangent, parameters, *, device="cuda"):
    """Clone an independent strided input leaf; differentiate actual FP32 weights."""
    import torch
    leaf = clone_strided(prepared, device, leaf=True)
    if leaf.stride() != prepared.stride():
        raise AssertionError("Projection input stride changed")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with torch.autocast(device, dtype=torch.bfloat16):
        output = forward(leaf)
        if output.dtype != torch.bfloat16:
            raise AssertionError("Expected BF16 projection output")
        gradients = torch.autograd.grad(output, [leaf, *(p for _, p in parameters)],
                                        grad_outputs=cotangent.to(device))
    names = ["hidden_gradient", *(name for name, _ in parameters)]
    if gradients[0].dtype != prepared.dtype or any(g.dtype != torch.float32 for g in gradients[1:]):
        raise AssertionError("Unexpected input/LoRA gradient dtype")
    record = {"output": output.detach().cpu().clone()}
    record.update({name: gradient.detach().cpu().clone()
                   for name, gradient in zip(names, gradients, strict=True)})
    if device == "cuda":
        torch.cuda.synchronize()
    return record, {"elapsed_seconds": time.monotonic() - started, "calls": 1,
                    "dtypes": {name: str(t.dtype) for name, t in record.items()},
                    "finite": {name: bool(torch.isfinite(t).all()) for name, t in record.items()},
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else 0,
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if device == "cuda" else 0}


def _probe(block, network, index, projection, captured, args, output, adapter, packed, report):
    import torch
    from torch._dynamo.utils import counters
    report.update(current_block=index, current_projection=projection, current_run="preparation")
    result = {"status": "running", "runs": {}}
    report["blocks"][str(index)][projection] = result
    _write_result(output, report)
    forward, parameters = projection_target(block, network, index, projection, args.forward_mode)
    name = BOUNDARIES[projection]
    original = captured["projection_inputs"][name]
    prepared = original if args.input_dtype == "original" else original.bfloat16()
    source_output = captured["outputs"][name]
    cotangent = captured["cotangents"][name]
    if args.cotangent_source == "random":
        generator = torch.Generator(device="cpu").manual_seed(172)
        cotangent = torch.randn(source_output.shape, generator=generator).bfloat16()
    metadata = {"boundary": name, "input_source": "real full-network step0 edit_inputs()[0]",
                "input_dtype": str(prepared.dtype), "original_input_dtype": str(original.dtype),
                "input_stride": list(prepared.stride()), "original_input_stride": list(original.stride()),
                "source_output_dtype": str(source_output.dtype),
                "source_output_stride": list(source_output.stride()),
                "cotangent_dtype": str(cotangent.dtype), "cotangent_stride": list(cotangent.stride()),
                "forward_mode": args.forward_mode, "cotangent_source": args.cotangent_source,
                "gradient_targets": ["hidden_gradient", *(n for n, _ in parameters)],
                "initial_state": True, "optimizer_updates": 0,
                "base_only_counterfactual": args.forward_mode == "base"}
    if args.forward_mode == "lora":
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            actual = forward(clone_strided(prepared, "cuda"))
        if actual.dtype != source_output.dtype or not torch.equal(
            actual.detach().cpu().contiguous().view(torch.uint8),
            source_output.contiguous().view(torch.uint8),
        ):
            raise AssertionError(f"Block {index}/{projection}: source output changed after input preparation")
        metadata["source_outputs_bitwise_equal"] = True
    stem = f"block-{index}-{projection}"
    torch.save({"prepared": prepared, "cotangent": cotangent, "source_output": source_output,
                "metadata": metadata}, output / f"{stem}-inputs.pt")
    result.update(stage=metadata, adapter_sha256_before=_digest(adapter), packed_sha256_before=_digest(packed))
    records = {}
    for label in ("eager", "repeat_eager", "compiled"):
        report["current_run"] = label
        _write_result(output, report)
        _clear_graphs()
        _assert_states(block, network, packed, adapter)
        torch._dynamo.reset()
        counters.clear()
        call = forward if label != "compiled" else torch.compile(
            forward, backend="inductor", fullgraph=True, dynamic=True,
            options={"emulate_precision_casts": True})
        record, summary = projection_vjp(call, prepared, cotangent, parameters)
        summary.update(_counter_snapshot())
        summary["coverage_valid"] = all(summary["finite"].values()) and (
            label != "compiled" or summary["unique_graphs"] > 0 and not summary["graph_breaks"])
        records[label] = record
        result["runs"][label] = summary
        torch.save(record, output / f"{stem}-{label}.pt")
        _assert_states(block, network, packed, adapter)
        _write_result(output, report)
        print(json.dumps({"block": index, "projection": projection, "run": label, **summary}), flush=True)
        del call
    result["baseline"] = _compare(records["eager"], records["repeat_eager"])
    result["baseline_bitwise_stable"] = all(
        m["bitwise_unequal"] == 0 for m in result["baseline"]["metrics"].values())
    result["compiled_comparison"] = _compare(records["eager"], records["compiled"])
    result["adapter_sha256_after"] = _digest(_adapter_state(network))
    result["packed_sha256_after"] = _digest(_packed_state(block))
    result["passed"] = (result["baseline"]["passed"] and result["baseline_bitwise_stable"]
                        and result["compiled_comparison"]["passed"]
                        and all(r["coverage_valid"] for r in result["runs"].values()))
    result["status"] = "passed" if result["passed"] else "failed"
    _write_result(output, report)
    _clear_graphs()


def _run(args, output, report):
    import bitsandbytes
    import diffusers
    import torch
    from library.models.qwen_image_2_1.weights import load_qwen_image_2_1_transformer
    from library.quantization import BF16AutocastLinear4bit, inspect_mixed_nf4
    from scripts.quantization.qwen_nf4_attention_core_capture import capture_full_loss
    from scripts.quantization.qwen_nf4_compile_acceptance import _attach_network
    path = Path(args.model).expanduser()
    if not path.is_file() or path.suffix != ".safetensors":
        raise ValueError(f"Require existing local .safetensors: {path}")
    if inspect_mixed_nf4(path, family="qwen_image_2_1") is None:
        raise ValueError("Require mixed-NF4 Qwen 2.1 checkpoint")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Require BF16-capable CUDA")
    if args.rank <= 0 or len(set(args.blocks)) != len(args.blocks) or any(i not in range(32) for i in args.blocks):
        raise ValueError("Require rank > 0 and unique block indices in 0..31")
    if len(set(args.projections)) != len(args.projections):
        raise ValueError("Require unique projections")
    from torch._inductor import config
    if "emulate_precision_casts" not in config.shallow_copy_dict():
        raise RuntimeError("Inductor emulate_precision_casts is unavailable")
    report["versions"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                          "diffusers": diffusers.__version__, "bitsandbytes": bitsandbytes.__version__}
    report["gpu"] = torch.cuda.get_device_name()
    torch.manual_seed(41)
    model = load_qwen_image_2_1_transformer(str(path), dtype=torch.bfloat16,
                                          device="cuda").requires_grad_(False).train()
    blocks = model.transformer_blocks
    if len(blocks) != 32 or any((b.attn.to_q.in_features, b.img_mlp.gate_layer.out_features,
                               b.attn.heads, b.attn.norm_q.weight.numel()) !=
                              (4096, 12288, 32, 128) for b in blocks):
        raise ValueError("Require full 32-block width4096/mlp12288/32x128 geometry")
    nf4 = [m for m in model.modules() if isinstance(m, BF16AutocastLinear4bit)]
    if not nf4 or any(m.compute_dtype != torch.bfloat16 for m in nf4):
        raise ValueError("Require NF4 with BF16 compute")
    for index in args.blocks:
        for projection in args.projections:
            module = (blocks[index].attn.to_out[0] if projection == "out" else
                      getattr(blocks[index].attn, f"to_{projection}"))
            if not isinstance(module, BF16AutocastLinear4bit):
                raise ValueError(f"Require actual NF4 projection: block {index}/{projection}")
    network = _attach_network(model, args.rank)
    adapter = _adapter_state(network)
    packed = {i: _packed_state(blocks[i]) for i in args.blocks}
    torch.save(_tree(network.state_dict(), lambda t: t.detach().cpu().clone()), output / "initial_adapter.pt")
    report["current_run"] = "full_loss_capture"
    _write_result(output, report)
    captured, report["full_loss_capture"] = capture_full_loss(
        model, network, args.blocks, args.attention_backend, adapter=adapter, packed=packed,
        output=output, capture_projection_outputs=True)
    for index in args.blocks:
        report["blocks"][str(index)] = {}
        for projection in args.projections:
            _probe(blocks[index], network, index, projection, captured[index], args, output,
                   adapter, packed[index], report)
    results = [r for projections in report["blocks"].values() for r in projections.values()]
    expected = len(args.blocks) * len(args.projections)
    report["coverage"] = {"expected_projections": expected, "completed_projections": len(results)}
    report["status"] = "passed" if len(results) == expected and all(r["passed"] for r in results) else "failed"
    del network, model, captured
    _clear_graphs()


def main():
    args = _parser().parse_args()
    output = Path(args.output_dir).expanduser() / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "arguments": vars(args), "blocks": {}, "diagnostic_only": True,
              "seeds": {"adapter": 41, "edit_inputs": 171, "cotangent": 172},
              "compile": {"backend": "inductor", "fullgraph": True, "dynamic": True,
                          "options": {"emulate_precision_casts": True}},
              "vjp": "projection output wrt independent input plus actual LoRA down/up FP32 weights",
              "limits": "No optimizer; base-only changes forward; random cot uses real full-loss inputs",
              "current_block": None, "current_projection": None}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Diagnostic artifacts: {output}", flush=True)
    try:
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
