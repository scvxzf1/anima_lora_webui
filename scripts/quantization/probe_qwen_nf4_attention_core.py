"""Diagnostic isolated QK RMSNorm, complex RoPE and segmented SDPA first-order VJPs.

Full-width NF4 base capture is eager/no_grad; optional full-loss cotangents come
from one checkpointed step0 backward without an optimizer. Core graphs execute
no NF4 or LoRA projections and do not enable a production compile gate.
"""
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
    _adapter_state, _assert_states, _capture, _clear_graphs, _compare,
    _counter_snapshot, _digest, _packed_state, _tree, _write_result,
)
from scripts.quantization.qwen_nf4_attention_core_stages import (  # noqa: E402
    clone_strided, prepare_core,
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Existing local Qwen 2.1 mixed-NF4 .safetensors")
    parser.add_argument("--output-dir", default="output/tests/qwen_nf4_attention_core",
                        help="Parent for unique timestamp directory; old artifacts are preserved")
    parser.add_argument("--blocks", nargs="+", type=int, default=[4])
    parser.add_argument("--phases", nargs="+", choices=("qk-norm", "rope", "sdpa", "qk-rope", "core"),
                        default=["qk-norm", "rope", "sdpa"])
    parser.add_argument("--edit-input-index", choices=(0, 1), type=int, default=0)
    parser.add_argument("--cotangent-source", choices=("random", "full-loss"), default="random",
                        help="Full-loss uses the true full-network step0 loss adjoint (input index 0 only)")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--attention-backend", choices=("native", "_native_math"), default="native")
    return parser


def _vjp(forward, prepared, cotangents, phase, backend):
    import torch
    from diffusers.models.attention_dispatch import attention_backend
    leaves = tuple(clone_strided(tensor, "cuda", leaf=True) for tensor in prepared)
    if any(t.stride() != source.stride() for t, source in zip(leaves, prepared)):
        raise AssertionError("Prepared input stride changed")
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    with attention_backend(backend), torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = forward(*leaves)
        outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        if any(output.dtype != torch.bfloat16 for output in outputs):
            raise AssertionError("Expected BF16 core outputs")
        gradients = torch.autograd.grad(
            outputs, leaves, grad_outputs=tuple(t.to("cuda") for t in cotangents))
    output_names = ("output_attention",) if phase in ("sdpa", "core") else ("output_query", "output_key")
    gradient_names = ("query_gradient", "key_gradient", "value_gradient")[:len(leaves)]
    record = {name: tensor.detach().cpu().clone() for name, tensor in
              zip((*output_names, *gradient_names), (*outputs, *gradients))}
    if len(record) != len(output_names) + len(leaves):
        raise AssertionError("Incomplete leaf/output coverage")
    torch.cuda.synchronize()
    return record, {"elapsed_seconds": time.monotonic() - started, "calls": 1,
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
                    "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "finite": {name: bool(torch.isfinite(t).all()) for name, t in record.items()}}


def _probe_phase(block, network, captured, index, phase, args, output, adapter, packed, report,
                 loss_record=None):
    import torch
    from torch._dynamo.utils import counters
    report.update(current_block=index, current_phase=phase, current_run="preparation")
    result = {"status": "running", "runs": {}, "packed_sha256_before": _digest(packed),
              "adapter_sha256_before": _digest(adapter)}
    report["blocks"][str(index)][phase] = result
    _write_result(output, report)
    _assert_states(block, network, packed, adapter)
    prepared, stage_forward, metadata, constants = prepare_core(
        block, captured, phase, backend=args.attention_backend, tree=_tree)
    if loss_record is None:
        generator = torch.Generator(device="cpu").manual_seed(172)
        shapes = [prepared[0].shape] if phase in ("sdpa", "core") else [t.shape for t in prepared]
        cotangents = tuple(torch.randn(shape, generator=generator).bfloat16() for shape in shapes)
    else:
        from diffusers.models.attention_dispatch import attention_backend
        from scripts.quantization.qwen_nf4_attention_core_capture import phase_adjoint
        source_outputs, cotangents = phase_adjoint(loss_record, phase, block.attn.heads)
        with attention_backend(args.attention_backend), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            actual_outputs = stage_forward(*(clone_strided(t, "cuda") for t in prepared))
        actual_outputs = actual_outputs if isinstance(actual_outputs, tuple) else (actual_outputs,)
        for actual, expected in zip(actual_outputs, source_outputs, strict=True):
            if actual.dtype != expected.dtype or not torch.equal(
                actual.detach().cpu().contiguous().view(torch.uint8),
                expected.contiguous().view(torch.uint8),
            ):
                raise AssertionError(f"Block {index}/{phase}: prepared output differs from real full-loss source")
        metadata["source_outputs_bitwise_equal"] = True
    metadata["cotangent_source"] = args.cotangent_source
    if loss_record is not None:
        metadata["initial_state"] = True
        metadata["input_source"] = "edit_inputs()[0]; full-network step0 loss adjoint"
    stem = f"block-{index}-{phase}"
    torch.save({"prepared": prepared, "cotangents": cotangents, "constants": constants,
                "captured": captured, "metadata": metadata}, output / f"{stem}-inputs.pt")
    result["stage"] = metadata
    records = {}
    for label in ("eager", "repeat_eager", "compiled"):
        report["current_run"] = label
        _write_result(output, report)
        _clear_graphs()
        _assert_states(block, network, packed, adapter)
        torch._dynamo.reset()
        counters.clear()
        forward = stage_forward
        if label == "compiled":
            forward = torch.compile(forward, backend="inductor", fullgraph=True, dynamic=True,
                                    options={"emulate_precision_casts": True})
        record, summary = _vjp(forward, prepared, cotangents, phase, args.attention_backend)
        summary.update(_counter_snapshot())
        summary["coverage_valid"] = all(summary["finite"].values()) and (
            label != "compiled" or (summary["unique_graphs"] > 0 and not summary["graph_breaks"]))
        result["runs"][label] = summary
        records[label] = record
        torch.save(record, output / f"{stem}-{label}.pt")
        _assert_states(block, network, packed, adapter)
        if label == "repeat_eager":
            result["baseline"] = _compare(records["eager"], record)
            result["baseline_bitwise_stable"] = all(
                metric["bitwise_unequal"] == 0 for metric in result["baseline"]["metrics"].values())
        _write_result(output, report)
        print(json.dumps({"block": index, "phase": phase, "run": label, **summary}), flush=True)
        del forward
    result["compiled_comparison"] = _compare(records["eager"], records["compiled"])
    result["packed_sha256_after"] = _digest(_packed_state(block))
    result["adapter_sha256_after"] = _digest(_adapter_state(network))
    result["compiled_attribution_valid"] = result["baseline"]["passed"] and result["baseline_bitwise_stable"]
    result["passed"] = (result["compiled_attribution_valid"] and result["compiled_comparison"]["passed"]
                        and all(run["coverage_valid"] for run in result["runs"].values()))
    result["status"] = "passed" if result["passed"] else "failed"
    _write_result(output, report)
    _clear_graphs()


def _run(args, output, report):
    if args.cotangent_source == "full-loss" and args.edit_input_index != 0:
        raise ValueError("full-loss cotangents require --edit-input-index 0 (initial step0 state)")
    import bitsandbytes
    import diffusers
    import torch
    from library.models.qwen_image_2_1.weights import load_qwen_image_2_1_transformer
    from library.quantization import BF16AutocastLinear4bit, inspect_mixed_nf4
    from scripts.quantization.qwen_nf4_compile_acceptance import _attach_network
    path = Path(args.model).expanduser()
    if not path.is_file() or path.suffix != ".safetensors":
        raise ValueError(f"Require existing local .safetensors: {path}")
    if inspect_mixed_nf4(path, family="qwen_image_2_1") is None:
        raise ValueError("Require an existing mixed-NF4 Qwen 2.1 checkpoint")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Require BF16-capable CUDA")
    if args.rank <= 0 or len(set(args.blocks)) != len(args.blocks) or any(i not in range(32) for i in args.blocks):
        raise ValueError("Require rank > 0 and unique block indices in 0..31")
    if len(set(args.phases)) != len(args.phases):
        raise ValueError("Require unique phases")
    from torch._inductor import config
    if "emulate_precision_casts" not in config.shallow_copy_dict():
        raise RuntimeError("Inductor emulate_precision_casts option is unavailable")
    report["versions"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                          "diffusers": diffusers.__version__, "bitsandbytes": bitsandbytes.__version__}
    report["gpu"] = torch.cuda.get_device_name()
    torch.manual_seed(41)
    model = load_qwen_image_2_1_transformer(str(path), dtype=torch.bfloat16,
                                          device="cuda").requires_grad_(False).train()
    blocks = model.transformer_blocks
    geometry = (len(blocks), blocks[0].attn.to_q.in_features, blocks[0].img_mlp.gate_layer.out_features)
    if geometry != (32, 4096, 12288) or any(
        (b.attn.to_q.in_features, b.img_mlp.gate_layer.out_features,
         b.attn.heads, b.attn.norm_q.weight.numel()) != (4096, 12288, 32, 128) for b in blocks
    ):
        raise ValueError(f"Require full 32-block width4096/mlp12288/32x128 geometry: {geometry}")
    report["model_geometry"] = {"blocks": 32, "width": 4096, "mlp_width": 12288, "heads": 32, "head_dim": 128}
    nf4_modules = [m for m in model.modules() if isinstance(m, BF16AutocastLinear4bit)]
    if not nf4_modules or any(m.compute_dtype != torch.bfloat16 for m in nf4_modules):
        raise ValueError("Require actual NF4 modules with BF16 compute throughout the base")
    report["nf4_compute_dtype"] = "torch.bfloat16"
    network = _attach_network(model, args.rank)
    adapter = _adapter_state(network)
    packed = {index: _packed_state(blocks[index]) for index in args.blocks}
    torch.save({name: t.detach().cpu().clone() for name, t in network.state_dict().items()},
               output / "initial_adapter.pt")
    report["current_phase"] = "full_model_capture"
    _write_result(output, report)
    torch.cuda.reset_peak_memory_stats()
    captured = _capture(model, args.blocks, args.attention_backend, input_index=args.edit_input_index)
    report["capture_peak_cuda_bytes"] = torch.cuda.max_memory_allocated()
    loss_records = {}
    if args.cotangent_source == "full-loss":
        from scripts.quantization.qwen_nf4_attention_core_capture import capture_full_loss
        report["current_phase"] = "full_loss_adjoint_capture"
        _write_result(output, report)
        loss_records, report["full_loss_capture"] = capture_full_loss(
            model, network, args.blocks, args.attention_backend,
            adapter=adapter, packed=packed, output=output)
    for index in args.blocks:
        _assert_states(blocks[index], network, packed[index], adapter)
        report["blocks"][str(index)] = {}
        for phase in args.phases:
            _probe_phase(blocks[index], network, captured[index], index, phase,
                         args, output, adapter, packed[index], report, loss_records.get(index))
    expected = len(args.blocks) * len(args.phases)
    results = [result for phases in report["blocks"].values() for result in phases.values()]
    report["coverage"] = {"expected_phases": expected, "completed_phases": len(results)}
    report["status"] = "passed" if len(results) == expected and all(r["passed"] for r in results) else "failed"
    del network, model, captured
    _clear_graphs()


def main():
    args = _parser().parse_args()
    output = Path(args.output_dir).expanduser() / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "arguments": vars(args), "blocks": {}, "diagnostic_only": True,
              "seeds": {"adapter": 41, "edit_inputs": 171,
                        "cotangent": 172 if args.cotangent_source == "random" else None},
              "compile": {"backend": "inductor", "fullgraph": True, "dynamic": True,
                          "options": {"emulate_precision_casts": True}},
              "vjp": "autograd.grad(core_outputs, independent prepared Q/K[/V] leaves, BF16 cotangents)",
              "limits": "Core graphs contain no NF4 projections, output projection or LoRA gradients; no optimizer",
              "current_block": None, "current_phase": "initialization"}
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
