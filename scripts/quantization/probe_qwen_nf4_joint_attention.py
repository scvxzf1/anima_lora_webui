"""Connected NF4/production LoRA Qwen attention VJP diagnostic, no optimizer.

Cotangents come from synthetic Edit output.float().square().mean(), not the
flow-matching training objective. Isolated results do not establish whole-model
compile acceptance or enable a production gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.quantization.probe_qwen_nf4_block_precision import (  # noqa: E402
    _adapter_state, _assert_states, _capture, _clear_graphs, _compare,
    _counter_snapshot, _digest, _packed_state, _tree, _write_result,
)
from scripts.quantization.qwen_nf4_joint_attention_stages import (  # noqa: E402
    PHASES, assert_bytes, clone_strided, joint_vjp, prepare_joint,
)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Existing local mixed-NF4 Qwen 2.1 .safetensors")
    parser.add_argument("--blocks", nargs="+", type=int, default=[4])
    parser.add_argument("--phases", nargs="+", choices=PHASES, default=list(PHASES))
    parser.add_argument("--input-layout", choices=("shared", "independent"), default="shared")
    parser.add_argument("--input-dtype", choices=("original", "bf16"), default="original")
    parser.add_argument("--cotangent-source", choices=("full-loss", "random"), default="full-loss")
    parser.add_argument("--attention-backend", choices=("native", "_native_math"), default="native")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--output-dir", default="output/tests/qwen_nf4_joint_attention")
    return parser


def _probe(block, network, index, captured, loss_record, phase, args, output, adapter, packed, report):
    import torch
    from diffusers.models.attention_dispatch import attention_backend
    from torch._dynamo.utils import counters
    report.update(current_block=index, current_phase=phase, current_run="preparation")
    result = {"status": "running", "runs": {}, "adapter_sha256_before": _digest(adapter),
              "packed_sha256_before": _digest(packed)}
    report["blocks"][str(index)][phase] = result
    _write_result(output, report)
    _assert_states(block, network, packed, adapter)
    prepared, forward, parameters, sources, cotangents, metadata, constants = prepare_joint(
        block, network, index, captured, loss_record, phase, tree=_tree,
        input_layout=args.input_layout, input_dtype=args.input_dtype)
    with attention_backend(args.attention_backend), torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = forward(*(clone_strided(t, "cuda") for t in prepared))
    actual = actual if isinstance(actual, tuple) else (actual,)
    for value, source in zip(actual, sources, strict=True):
        assert_bytes(value, source)
    metadata.update(source_outputs_bitwise_equal=True, cotangent_source=args.cotangent_source,
                    input_source="synthetic Edit full-network initial step0 projection inputs",
                    adjoint_objective="output.float().square().mean()", initial_state=True)
    if args.cotangent_source == "random":
        generator = torch.Generator(device="cpu").manual_seed(172)
        cotangents = tuple(torch.randn(t.shape, generator=generator).to(t.dtype) for t in sources)
    stem = f"block-{index}-{phase}"
    torch.save({"prepared": prepared, "original_qkv_inputs": loss_record["projection_inputs"],
                "source_outputs": sources, "cotangents": cotangents, "constants": constants,
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
        call = forward if label != "compiled" else torch.compile(
            forward, backend="inductor", fullgraph=True, dynamic=True,
            options={"emulate_precision_casts": True})
        record, summary = joint_vjp(call, prepared, cotangents, parameters, phase, args.attention_backend)
        summary.update(_counter_snapshot())
        summary["coverage_valid"] = all(summary["finite"].values()) and (
            label != "compiled" or summary["unique_graphs"] > 0 and not summary["graph_breaks"])
        records[label] = record
        result["runs"][label] = summary
        torch.save(record, output / f"{stem}-{label}.pt")
        _assert_states(block, network, packed, adapter)
        if label == "repeat_eager":
            result["baseline"] = _compare(records["eager"], record)
            result["baseline_bitwise_stable"] = all(
                m["bitwise_unequal"] == 0 for m in result["baseline"]["metrics"].values())
        _write_result(output, report)
        print(json.dumps({"block": index, "phase": phase, "run": label, **summary}), flush=True)
        del call
    result["compiled_comparison"] = _compare(records["eager"], records["compiled"])
    result.update(adapter_sha256_after=_digest(_adapter_state(network)),
                  packed_sha256_after=_digest(_packed_state(block)))
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
        raise ValueError("Require rank > 0 and unique blocks in 0..31")
    if len(set(args.phases)) != len(args.phases):
        raise ValueError("Require unique phases")
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
        projections = [blocks[index].attn.to_q, blocks[index].attn.to_k, blocks[index].attn.to_v]
        if "attention" in args.phases:
            projections.append(blocks[index].attn.to_out[0])
        if any(not isinstance(m, BF16AutocastLinear4bit) for m in projections):
            raise ValueError(f"Require actual NF4 phase projections: block {index}")
    network = _attach_network(model, args.rank)
    adapter = _adapter_state(network)
    packed = {i: _packed_state(blocks[i]) for i in args.blocks}
    torch.save(_tree(network.state_dict(), lambda t: clone_strided(t, "cpu")), output / "initial_adapter.pt")
    report["current_run"] = "no_grad_bound_capture"
    _write_result(output, report)
    captured = _capture(model, args.blocks, args.attention_backend, input_index=0)
    report["current_run"] = "synthetic_edit_adjoint_capture"
    _write_result(output, report)
    loss_records, report["full_loss_capture"] = capture_full_loss(
        model, network, args.blocks, args.attention_backend, adapter=adapter, packed=packed,
        output=output, capture_projection_outputs=True)
    report["full_loss_capture"].update(objective_kind="synthetic Edit diagnostic, not flow matching",
                                       whole_model_acceptance=False)
    for index in args.blocks:
        report["blocks"][str(index)] = {}
        for phase in args.phases:
            _probe(blocks[index], network, index, captured[index], loss_records[index], phase,
                   args, output, adapter, packed[index], report)
    results = [r for phases in report["blocks"].values() for r in phases.values()]
    expected = len(args.blocks) * len(args.phases)
    report["coverage"] = {"expected_phases": expected, "completed_phases": len(results)}
    report["status"] = "passed" if len(results) == expected and all(r["passed"] for r in results) else "failed"
    del network, model, captured, loss_records
    _clear_graphs()


def main():
    args = _parser().parse_args()
    output = Path(args.output_dir).expanduser() / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "arguments": vars(args), "blocks": {}, "diagnostic_only": True,
              "seeds": {"adapter": 41, "edit_inputs": 171, "cotangent": 172},
              "compile": {"backend": "inductor", "fullgraph": True, "dynamic": True,
                          "options": {"emulate_precision_casts": True}},
              "vjp": "connected real QKV[/core/out] wrt shared or independent input and FP32 LoRA down/up",
              "limits": "Synthetic Edit square-mean diagnostic cotangent; no flow-matching objective, optimizer or whole-model claim",
              "capture": {"checkpoint": "full non-reentrant", "blocks_to_swap": 24, "input_index": 0},
              "current_block": None, "current_phase": None}
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
