"""Compare captured Linear weights against the same values computed in FP32."""

import argparse
import hashlib
import json
from pathlib import Path

from safetensors.torch import load_file
import torch

from bench.adaptive_runtime.capture import capture_path, read_manifest
from library.training.adaptive_runtime.precision import calibrate_unit, preferred_candidate
from library.training.auto_block_swap.process import write_result


def calibrate(directory, unit, candidate):
    weights = capture_path(directory, unit["weights"])
    if weights.stat().st_size > 512 * 1024**2:
        raise ValueError("A single captured unit exceeds the 512MiB calibration budget")
    state = load_file(str(weights))
    if (set(state) not in ({"weight"}, {"weight", "bias"}) or state["weight"].ndim != 2
            or state["weight"].numel() == 0
            or not all(t.is_floating_point() and torch.isfinite(t).all() for t in state.values())):
        raise ValueError("Invalid captured Linear state")
    out_features, in_features = state["weight"].shape
    layer = torch.nn.Linear(in_features, out_features, bias="bias" in state, device="meta")
    layer.load_state_dict(state, strict=True, assign=True)
    layer.requires_grad_(False)
    cases, hashes = [], []
    for item in unit["cases"]:
        path = capture_path(directory, item["file"])
        if path.stat().st_size > 64 * 1024**2:
            raise ValueError("Captured activation exceeds the 64MiB case budget")
        tensors = load_file(str(path))
        value = tensors.get("input")
        if (set(tensors) != {"input"} or value.ndim != 2 or value.numel() == 0
                or value.shape[1] != in_features
                or value.dtype != torch.float32 or not torch.isfinite(value).all()):
            raise ValueError("Invalid captured Linear input")
        cases.append(((value,), {}))
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    result = calibrate_unit(layer, cases, device="cuda", candidate=candidate)
    return {**result, "name": unit["name"], "input_sha256": hashes,
            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "scope": "frozen_linear_output_and_input_vjp_only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = read_manifest(args.capture)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    prop = torch.cuda.get_device_properties(0)
    candidate = args.candidate
    if candidate == "auto":
        candidate = preferred_candidate(torch.cuda.get_device_capability())
    if candidate == "bf16" and prop.major < 8:
        raise ValueError("Native BF16 calibration requires Ampere+")
    report = {"status": "running", "candidate": candidate, "gpu": prop.name,
              "uuid": str(prop.uuid), "full_model_calibrated": False,
              "low_precision_tested": candidate != "fp32",
              "reference": "captured_weight_values_promoted_to_fp32_not_original_fp32_checkpoint",
              "source": manifest["source"], "units": []}
    write_result(args.output, report)
    try:
        for unit in manifest["units"]:
            report["units"].append(calibrate(args.capture, unit, candidate))
            write_result(args.output, report)
        report["local_plan"] = {u["name"]: u["selected"] for u in report["units"]}
        report["status"] = "local_calibration_complete"
    except Exception as exc:
        report.update(status="cuda_oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "error",
                      error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_result(args.output, report)
        print(json.dumps({k: report.get(k) for k in ("status", "candidate", "local_plan", "error")}, indent=2))


if __name__ == "__main__":
    main()
