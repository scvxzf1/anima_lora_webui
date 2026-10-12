"""Private native FP16 execution of the unchanged six-case canonical replay loop."""
from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from pathlib import Path

import torch

from bench.adaptive_runtime import anima_validation_updates as updates
from bench.adaptive_runtime import compare_training as comparison
from bench.adaptive_runtime.replay import contract, tensor_digest

PRECISION = "native-fp16-amp"
GPU_UUID = "GPU-806debd9-d254-da57-d515-ccae15b4ac6a"
DOMAIN = "canonical_fp32_pre_boundary_cast"
POLICY = {"base_linear_storage": "torch.float16", "adapter_master": "torch.float32",
          "autocast": "torch.float16", "residual_guards": 29, "swap": 20,
          "loss_scale": 1024, "attention": "torch", "checkpoint": "full", "compile": False,
          "input_gradient_domain": DOMAIN, "mixture_domain": "fixed_fp32_latent_noise_sigma",
          "target": "(noise-latents).float()",
          "noise_scope": "fixed_canonical_noise_not_production_rng_sampling"}


def validate_native_args(args, original):
    safe = copy.copy(args)
    safe.precision = "fp16-islands"
    original(safe)
    if (args.precision != PRECISION or args.swap != 20 or args.loss_scale != 1024
            or args.steps != 6 or not args.validation_states or not args.replay_reference
            or args.record_replay or args.fp32_pattern or args.fp16_only_pattern):
        raise ValueError("Native acceptance requires the single frozen FP16 AMP configuration")


def boundary_tensors(noisy, sigma, prompt):
    if noisy.dtype != torch.float32 or not noisy.is_leaf or not noisy.requires_grad:
        raise ValueError("Canonical FP32 noisy leaf required before the model boundary cast")
    return {"noisy": noisy.to(torch.float16), "sigma": sigma.to(torch.float16),
            "prompt": prompt.to(torch.float16),
            "padding_mask": torch.zeros(noisy.shape[0], 1, *noisy.shape[-2:],
                                        device=noisy.device, dtype=torch.float16)}


def describe(tensors):
    return {name: {"dtype": str(value.dtype), "shape": list(value.shape),
                   "sha256": tensor_digest({name: value})} for name, value in tensors.items()}


def snapshot(model, network):
    linears = {name: str(module.weight.dtype) for name, module in model.named_modules()
               if type(module) is torch.nn.Linear}
    guards = {f"blocks.{i}": bool(block.fp32_residual) for i, block in enumerate(model.blocks)}
    guards["final_layer"] = bool(model.final_layer.fp32_residual)
    params = [p for p in network.parameters() if p.requires_grad]
    if (len(linears) != 377 or set(linears.values()) != {"torch.float16"}
            or len(guards) != 29 or not all(guards.values()) or not params
            or any(p.dtype != torch.float32 for p in params)):
        raise ValueError("Native storage/adapter/residual guard contract changed")
    return {"linears": linears, "guards": guards, "adapter_dtype": "torch.float32"}


@contextmanager
def observe_first_case(model, records):
    modules = dict(model.named_modules())
    names = ["blocks.0.mlp.layer1", "blocks.0.mlp.layer2"]
    names += [name for name, module in modules.items()
              if name.startswith("blocks.0.self_attn.") and type(module) is torch.nn.Linear][:2]
    if len(names) != 4:
        raise ValueError("Expected four block 0 Linear observations")
    handles = []
    final = model.final_layer
    original = final._fp32_project
    def project(x):
        linear = torch.nn.functional.linear
        def observed(inputs, weight, bias=None):
            output = linear(inputs, weight, bias)
            records.setdefault("final_layer._fp32_project", {
                "input_dtype": str(inputs.dtype), "weight_compute_dtype": str(weight.dtype),
                "output_dtype": str(output.dtype),
                "internal_autocast_enabled": torch.is_autocast_enabled(inputs.device.type)})
            return output
        torch.nn.functional.linear = observed
        try:
            return original(x)
        finally:
            torch.nn.functional.linear = linear
    try:
        for name in names:
            def hook(module, inputs, output, name=name):
                records.setdefault(name, {"input_dtype": str(inputs[0].dtype),
                    "output_dtype": str(output.dtype),
                    "autocast_enabled": torch.is_autocast_enabled(inputs[0].device.type)})
            handles.append(modules[name].register_forward_hook(hook))
        final._fp32_project = project
        yield
    finally:
        for handle in handles:
            handle.remove()
        final._fp32_project = original


def build_native(args, report, resources):
    from bench.adaptive_runtime import probe_anima_train as probe
    device = torch.device("cuda:0")
    model = probe.load_anima_model(device=device, dit_path=str(args.weights), attn_mode="torch",
                                   loading_device="cpu", dit_weight_dtype=torch.bfloat16)
    resources.append(model)
    model.requires_grad_(False)
    model.half()
    cfg = probe.LoRANetworkCfg.from_kwargs({}, network_dim=4, network_alpha=4,
                                           neuron_dropout=None, module_class=probe.LoRAModule)
    network = probe.LoRANetwork(text_encoders=[], unet=model, cfg=cfg, multiplier=1.0)
    network.apply_to(text_encoders=[], unet=model, apply_text_encoder=False, apply_unet=True)
    network.to(device=device, dtype=torch.float32)
    model.enable_gradient_checkpointing()
    model.enable_fp32_residual()
    report.update(precision_manifest=None, fp32_modules=[], precision_profile_resolved=True,
                  compute_dtype="native_fp16_autocast", timestep_compute_dtype="fp16_boundary",
                  runtime_input_scope="canonical_fp32_replay_with_native_fp16_model_boundary",
                  base_storage_dtype="fp16")
    report["native_execution"]["dtype_snapshot"] = snapshot(model, network)
    model.enable_block_swap(20, device, transfer_dtype=None, restore_mode="foreach")
    model.move_to_device_except_swap_blocks(device)
    model.switch_block_swap_for_training()
    model.train()
    network.train()
    return model, network


def run_worker(args, execution):
    from bench.adaptive_runtime import probe_anima_train as probe
    originals = {name: getattr(probe, name) for name in ("validate_args", "build", "forward_prediction")}
    active_report = {}
    def build(args, report, resources):
        active_report.update(report=report)
        report["native_execution"] = {**execution, "policy": POLICY.copy(), "boundaries": [],
            "signature_semantics": "canonical_source_replay_comparison_contract",
            "observation_scope": "first_case_initial_sigma_0.2_forward_only", "observations": {}}
        if report["uuid"] != GPU_UUID:
            raise ValueError("Wrong native GPU UUID")
        return build_native(args, report, resources)
    def forward(model, noisy, sigma, prompt, precision):
        if precision != PRECISION:
            raise ValueError("Native precision label required")
        report = active_report["report"]
        native = report["native_execution"]
        tensors = boundary_tensors(noisy, sigma, prompt)
        native["boundaries"].append({"step": report["step"], "case_id": report["case_id"],
            "canonical": describe({"noisy": noisy, "sigma": sigma, "prompt": prompt}),
            "actual": describe(tensors), "input_gradient_domain": DOMAIN})
        with torch.autocast(noisy.device.type, dtype=torch.float16):
            if report["step"] == 1:
                with observe_first_case(model, native["observations"]):
                    return model(tensors["noisy"], tensors["sigma"], tensors["prompt"],
                                 padding_mask=tensors["padding_mask"])
            return model(tensors["noisy"], tensors["sigma"], tensors["prompt"],
                         padding_mask=tensors["padding_mask"])
    try:
        probe.validate_args = lambda value: validate_native_args(value, originals["validate_args"])
        probe.build = build
        probe.forward_prediction = forward
        return probe.run(args)
    finally:
        for name, original in originals.items():
            setattr(probe, name, original)


def validate_native_result(worker, execution):
    native = worker.get("native_execution", {})
    if (worker.get("status") != "ok" or worker.get("precision") != PRECISION
            or worker.get("uuid") != GPU_UUID or worker.get("swap") != 20
            or worker.get("loss_scale") != 1024 or worker.get("replay_mode") != "replay"
            or native.get("policy") != POLICY
            or any(native.get(k) != v for k, v in execution.items())
            or native.get("signature_semantics") != "canonical_source_replay_comparison_contract"
            or native.get("observation_scope") != "first_case_initial_sigma_0.2_forward_only"):
        raise ValueError("Native execution/worker provenance mismatch")
    signature = worker.get("signature", {})
    if signature.get("precision") != PRECISION or signature.get("loss_scale") != 1024:
        raise ValueError("Native signature must retain its actual label and scale")
    steps, boundaries = worker.get("updates", []), native.get("boundaries", [])
    if len(steps) != 6 or len(boundaries) != 6:
        raise ValueError("Native worker must complete six unique cases")
    for i, ((label, sigma), step, boundary) in enumerate(zip(updates.CASES, steps, boundaries, strict=True), 1):
        if (step.get("step") != i or step.get("case_id") != updates.case_id(label, sigma)
                or step.get("state_label") != label or step.get("effective_update") is not True
                or step.get("loss_scale_before") != 1024 or step.get("loss_scale_after") != 1024
                or boundary.get("step") != i or boundary.get("case_id") != step["case_id"]
                or boundary.get("input_gradient_domain") != DOMAIN):
            raise ValueError("Native case/scale/update contract changed")
    snap = native.get("dtype_snapshot", {})
    if (len(snap.get("linears", {})) != 377 or set(snap["linears"].values()) != {"torch.float16"}
            or len(snap.get("guards", {})) != 29 or not all(snap["guards"].values())
            or snap.get("adapter_dtype") != "torch.float32"):
        raise ValueError("Native dtype inventory invalid")
    obs = native.get("observations", {})
    linears = {k: v for k, v in obs.items() if k.startswith("blocks.0.")}
    final = obs.get("final_layer._fp32_project", {})
    if (len(linears) != 4 or any(v.get("output_dtype") != "torch.float16" or v.get("autocast_enabled") is not True for v in linears.values())
            or any(final.get(k) != "torch.float32" for k in ("input_dtype", "weight_compute_dtype", "output_dtype"))
            or final.get("internal_autocast_enabled") is not False):
        raise ValueError("Native first-case actual compute observations invalid")


def compare_native(reference_dir, worker_dir, execution):
    ref, rw = comparison.read_capture(Path(reference_dir) / "training-capture")
    trial, worker, _ = updates._read(worker_dir)
    capture = json.loads((Path(worker_dir) / "training-capture/manifest.json").read_text())
    validate_native_result(worker, execution)
    if (contract(ref["signature"]) != contract(capture["signature"])
            or rw["uuid"] != worker["uuid"] or ref["initial_adapter_sha256"] != capture["initial_adapter_sha256"]
            or capture.get("schema") != "adaptive_training_capture_v1" or capture.get("steps") != 6):
        raise ValueError("Native canonical comparison contract differs")
    cases = []
    for i, (a, b, boundary) in enumerate(zip(ref["cases"], capture["cases"], worker["native_execution"]["boundaries"], strict=True)):
        if a["replay"] != b.get("replay") or a["sigma"] != b.get("sigma"):
            raise ValueError("Native replay case differs")
        from safetensors.torch import load_file
        replay_case = json.loads((Path(reference_dir) / "training-replay/manifest.json").read_text())["cases"][i]
        tensors = load_file(str(Path(reference_dir) / "training-replay" / replay_case["file"]))
        canonical = {"noisy": tensors["input.noisy"], "sigma": tensors["input.sigma"], "prompt": tensors["input.prompt.0"]}
        actual = {k: v.half() for k, v in canonical.items()}
        actual["padding_mask"] = torch.zeros(canonical["noisy"].shape[0], 1, *canonical["noisy"].shape[-2:], dtype=torch.float16)
        if boundary["canonical"] != describe(canonical) or boundary["actual"] != describe(actual):
            raise ValueError("Native boundary differs from canonical replay tensors")
        metrics = comparison.compare_case(comparison.load_case(Path(reference_dir) / "training-capture", a, ref["expected_gradients"]),
                                          comparison.load_case(Path(worker_dir) / "training-capture", b, capture["expected_gradients"]))
        cases.append({"step": i + 1, "sigma": a["sigma"], **metrics})
    return {"status": "compared", "cases": cases,
            "within_experimental_tolerances": all(c["within_experimental_tolerances"] for c in cases)}
