"""Sequential synthetic Edit eager/Inductor NF4 diagnostic; not production training."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import tomllib
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", help="TOML containing pretrained_model_name_or_path")
    source.add_argument("--model", help="Existing local Qwen Image 2.1 NF4 checkpoint")
    parser.add_argument("--output-dir", default="output/tests/qwen_nf4_compile_20261005")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--restore-mode", choices=("foreach", "slab"), default="foreach")
    parser.add_argument("--scope", choices=("all", "resident"), default="all")
    parser.add_argument("--dynamic", action=argparse.BooleanOptionalAction, default=True,
        help="Use dynamic sequence compilation (default: enabled)")
    parser.add_argument("--persistent-reductions", action=argparse.BooleanOptionalAction, default=True,
        help="Enable Inductor Triton persistent reductions for compilation and VJPs (default: enabled)")
    parser.add_argument("--native-layer-norm", action=argparse.BooleanOptionalAction, default=False,
        help="Patch both runs with diagnostic native LayerNorm first-order VJPs (default: disabled)")
    parser.add_argument("--native-layer-norm-targets", nargs="+", choices=("img_norm1", "img_norm2"),
        default=("img_norm1", "img_norm2"), help="Unique image norms to patch (default: both)")
    parser.add_argument("--native-silu", action=argparse.BooleanOptionalAction, default=False,
        help="Patch both runs with diagnostic native SiLU first-order VJPs (default: disabled)")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=False,
        help="Enable strict diagnostic determinism; use a fresh process for cuBLAS workspace settings")
    args = parser.parse_args()
    if len(set(args.native_layer_norm_targets)) != len(args.native_layer_norm_targets):
        parser.error("--native-layer-norm-targets must be unique")
    model_path = args.model or tomllib.loads(Path(args.config).read_text())["pretrained_model_name_or_path"]
    output = Path(args.output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True, exist_ok=False)
    _run_probe(args, model_path, output)


def _run_probe(args, model_path, output):
    added_workspace = args.deterministic and "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    if added_workspace:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    try:
        return _run_probe_with_environment(args, model_path, output)
    finally:
        if added_workspace:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)


def _run_probe_with_environment(args, model_path, output):
    import torch
    import torch._inductor.config as inductor_config
    from library.models.qwen_image_2_1.weights import load_qwen_image_2_1_transformer
    from scripts.quantization.qwen_nf4_compile_acceptance import compare_runs, run_case
    native_patches = {}
    silu_patches = {}
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    label = "eager"

    def loader():
        model = load_qwen_image_2_1_transformer(model_path, dtype=torch.bfloat16, device="cpu")
        patch = {"enabled": False, "diagnostic_only": True, "patch_count": 0,
            "norm_names": list(args.native_layer_norm_targets)}
        if args.native_layer_norm:
            from scripts.quantization.qwen_nf4_native_layer_norm import patch_native_layer_norm
            patch = patch_native_layer_norm(model, norm_names=tuple(args.native_layer_norm_targets))
        native_patches[label] = patch
        silu_patch = {"enabled": False, "diagnostic_only": True, "patch_count": 0}
        if args.native_silu:
            from scripts.quantization.qwen_nf4_native_silu import patch_native_silu
            silu_patch = patch_native_silu(model)
        silu_patches[label] = silu_patch
        return model

    results = {}
    report = {"model": model_path, "scope": args.scope, "count": args.count,
        "restore_mode": args.restore_mode, "rank": 16, "dynamic_seq": args.dynamic,
        "persistent_reductions": args.persistent_reductions,
        "native_layer_norm": args.native_layer_norm,
        "native_layer_norm_targets": list(args.native_layer_norm_targets),
        "native_silu": args.native_silu, "deterministic_requested": args.deterministic,
        "loss": "output.float().square().mean()", "synthetic_edit_fixture": True,
        "production_gate": "not approved", "torch": torch.__version__,
        "cuda": torch.version.cuda, "python": sys.version, "status": "failed"}
    comparison_path = output / "comparison.json"
    try:
        if args.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=False)
        report["determinism"] = {
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cuda_initialized_before_probe": torch.cuda.is_initialized(),
            "algorithms": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        }
        # Keep the patch through lazy first trace, backward compilation and VJPs.
        # The production compiler still supplies NF4 emulate_precision_casts=True.
        with inductor_config.patch({"triton.persistent_reductions": args.persistent_reductions}):
            report["inductor_config"] = {"triton.persistent_reductions": inductor_config.triton.persistent_reductions}
            for label in ("eager", "compiled"):
                result = run_case(loader, compiled=label == "compiled", count=args.count,
                    restore_mode=args.restore_mode, scope=args.scope, rank=16, dynamic_seq=args.dynamic)
                result["diagnostic_config"] = {"dynamic_seq": args.dynamic,
                    "persistent_reductions": args.persistent_reductions,
                    "native_layer_norm": native_patches[label], "native_silu": silu_patches[label]}
                torch.save(result, output / f"{label}.pt")
                results[label] = result
        report["metrics"] = compare_runs(results["eager"], results["compiled"], report_path=comparison_path)
        report["status"] = "passed"
    except BaseException as error:
        report["error"] = str(error)
        report["error_type"] = type(error).__name__
        report["traceback"] = traceback.format_exc()
        # Structural comparison errors occur before compare_runs writes its report.
        if len(results) == 2 and not comparison_path.exists():
            comparison_path.write_text(json.dumps({"status": "failed",
                "error": str(error), "error_type": type(error).__name__,
                "traceback": report["traceback"]}, indent=2))
        raise
    finally:
        if args.deterministic:
            torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
        report["native_patches"] = native_patches
        report["native_silu_patches"] = silu_patches
        report["runs"] = {k: {field: value for field, value in v.items()
            if field not in ("records", "initial_adapter")} for k, v in results.items()}
        (output / "result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"status": report["status"], "result": str(output / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
