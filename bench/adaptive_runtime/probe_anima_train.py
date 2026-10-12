"""Disposable full-Anima short-training precision probe on immutable real caches."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from fnmatch import fnmatchcase
import json
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace
import time
import traceback
from uuid import UUID

import numpy as np
from safetensors import safe_open
import torch

from bench.adaptive_runtime.anima_inputs import load_anima_inputs, sha256_file
from bench.adaptive_runtime import anima_backward_counterfactual as backward_counterfactual
from bench.adaptive_runtime import anima_block_boundaries as block_boundaries
from bench.adaptive_runtime import anima_block_vjp_capture as block_vjp
from bench.adaptive_runtime import anima_update_comparison as update_comparison
from bench.adaptive_runtime import anima_validation_probe as validation_probe
from bench.adaptive_runtime.anima_validation_state import StateBundleWriter
from bench.adaptive_runtime.anima_validation_updates import ValidationUpdateCapture
from bench.adaptive_runtime.anima_local_vjp import LocalVJPCapture
from bench.adaptive_runtime.replay import MODE, TrainingReplay
from bench.adaptive_runtime.scaling import backward_unscaled, make_scaler
from bench.adaptive_runtime.training_capture import TrainingCapture
from library.anima.weights import load_anima_model
from library.training.adaptive_runtime.training_precision import install_training_precision
from library.training.auto_block_swap.process import write_result
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.network import LoRANetwork
from networks.lora_modules.lora import LoRAModule


SEED = 20260930


class NonfiniteTrainingError(RuntimeError):
    pass


def block_vjp_indices_for(args):
    raw = getattr(args, "block_vjp_indices", None)
    if raw is None:
        return (14,)
    if not getattr(args, "capture_block_vjp", False):
        raise ValueError("--block-vjp-indices requires --capture-block-vjp")
    if raw not in ("14", "15,17,27"):
        raise ValueError("Block VJP indices must be 14 or 15,17,27")
    return block_vjp.validate_block_indices(tuple(int(value) for value in raw.split(",")))


def validate_args(args):
    block_vjp_indices_for(args)
    validation_states = getattr(args, "validation_states", None)
    export_states = getattr(args, "export_validation_states", None)
    if validation_states and export_states:
        raise ValueError("Choose state export or validation, not both")
    if args.output.name != "result.json":
        raise ValueError("--output must be a worker-directory result.json")
    if (not validation_states and (args.steps < 1 or args.steps > 5)
            or validation_states and args.steps != 6 or args.swap < 0):
        raise ValueError("steps must be 1..5 and swap nonnegative")
    if validation_states:
        validation_probe.validate_validation_args(args)
    if export_states and (args.steps != 1 or args.single_sigma != 0.5
            or args.precision != "fp32-reference" or not args.record_replay
            or not args.capture_update or args.replay_reference
            or args.capture_local_vjp or args.capture_block_boundaries or args.capture_block_vjp
            or args.backward_counterfactual_suite or args.block_vjp_indices is not None
            or args.local_vjp_profile != "linear" or args.base_backward_mode != "native"
            or getattr(args, "resume", False)):
        raise ValueError("State export requires one FP32 replay/update step at sigma 0.5")
    if not math.isfinite(args.loss_scale) or args.loss_scale < 1:
        raise ValueError("loss scale must be finite and >= 1")
    if args.precision != "fp16-islands" and (args.fp32_pattern or getattr(args, "fp16_only_pattern", ())
                                               or args.loss_scale != 1):
        raise ValueError("FP32 patterns/scaling require fp16-islands")
    if getattr(args, "fp16_only_pattern", ()) and args.fp32_pattern:
        raise ValueError("Choose FP16 targets or FP32 patterns, not both")
    if args.record_replay and args.precision != "fp32-reference":
        raise ValueError("Replay recording requires FP32 reference")
    if args.replay_reference and args.precision == "fp32-reference":
        raise ValueError("Replay candidate cannot be FP32 reference")
    if args.record_replay and args.replay_reference:
        raise ValueError("Choose recording or replay, not both")
    if getattr(args, "single_sigma", None) is not None:
        if args.steps != 1 or not math.isfinite(args.single_sigma) or not 0 < args.single_sigma < 1:
            raise ValueError("--single-sigma requires one step and finite sigma in (0,1)")
    if getattr(args, "capture_update", False) and not validation_states and (args.steps != 1
            or getattr(args, "single_sigma", None) != 0.5
            or not (args.record_replay or args.replay_reference)):
        raise ValueError("Update capture requires one replay-record/replay step at sigma 0.5")
    if getattr(args, "capture_local_vjp", False) and (args.steps != 1 or not
            (args.record_replay or args.replay_reference) or getattr(args, "single_sigma", None) != 0.5):
        raise ValueError("Local VJP capture requires one replay-record/replay step at sigma 0.5")
    if getattr(args, "capture_block_boundaries", False) and (
            args.steps != 1 or getattr(args, "single_sigma", None) != 0.5
            or not (args.record_replay or args.replay_reference)
            or getattr(args, "capture_local_vjp", False)
            or getattr(args, "backward_counterfactual_suite", False)
            or args.precision == "bf16"):
        raise ValueError("Block boundaries require one replay step at sigma 0.5")
    if getattr(args, "capture_block_vjp", False) and (
            args.steps != 1 or getattr(args, "single_sigma", None) != 0.5
            or not (args.record_replay or args.replay_reference)
            or getattr(args, "capture_local_vjp", False)
            or getattr(args, "capture_block_boundaries", False)
            or getattr(args, "backward_counterfactual_suite", False)
            or not (args.precision == "fp32-reference" and args.record_replay
                    and args.loss_scale == 1
                    or args.precision == "fp16-islands" and args.replay_reference
                    and tuple(getattr(args, "fp16_only_pattern", ())) == ("blocks.*.mlp.*",)
                    and not args.fp32_pattern and args.loss_scale == 1024)):
        raise ValueError("Block VJP requires reference or MLP-only FP16 replay at sigma 0.5")
    if getattr(args, "local_vjp_profile", "linear") != "linear" and not getattr(args, "capture_local_vjp", False):
        raise ValueError("MLP path profile requires local VJP capture")
    suite = getattr(args, "backward_counterfactual_suite", False)
    mode = getattr(args, "base_backward_mode", "native")
    if mode not in {"native", "fp32"} or (mode == "fp32" and not suite):
        raise ValueError("FP32 base backward requires counterfactual suite")
    if suite and (args.steps != 1 or getattr(args, "single_sigma", None) != 0.5
                  or not (args.record_replay or args.replay_reference)
                  or getattr(args, "capture_local_vjp", False) or args.precision == "bf16"):
        raise ValueError("Counterfactual suite requires one replay step at sigma 0.5 without local capture")
    if suite and args.precision == "fp32-reference" and mode != "native":
        raise ValueError("FP32 reference requires native base backward")
    if suite and args.precision == "fp16-islands" and (
            args.fp32_pattern or tuple(getattr(args, "fp16_only_pattern", ())) != ("blocks.*.mlp.*",)
            or args.loss_scale != 1024):
        raise ValueError("Counterfactual candidate requires MLP-only FP16 and loss scale 1024")
    if not args.weights.is_file() or not args.inputs.is_file():
        raise ValueError("Existing weights and prepared inputs are required")


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def device_uuid(value):
    return "GPU-" + str(UUID(str(value).removeprefix("GPU-")))


def checkpoint_source_dtypes(path):
    counts = {}
    with safe_open(str(path), framework="pt", device="cpu") as source:
        for key in source.keys():
            dtype = source.get_slice(key).get_dtype()
            counts[dtype] = counts.get(dtype, 0) + 1
    if not counts or set(counts) != {"BF16"}:
        raise ValueError("Anima comparison probe requires an exclusively BF16 source checkpoint")
    return counts


def fp32_complement(model, patterns):
    names = [name for name, module in model.named_modules() if type(module) is torch.nn.Linear]
    targets = {name for name in names if any(fnmatchcase(name, pattern) for pattern in patterns)}
    if not targets or any(not any(fnmatchcase(name, pattern) for name in names)
                          for pattern in patterns):
        raise ValueError("FP16 targets must match actual Anima Linear modules")
    return [name for name in names if name not in targets]


def build(args, report, resources=None):
    device = torch.device("cuda:0")
    model = load_anima_model(device=device, dit_path=str(args.weights), attn_mode="torch",
                             loading_device="cpu", dit_weight_dtype=torch.bfloat16)
    if resources is not None:
        resources.append(model)
    model.requires_grad_(False)
    if args.swap > model.num_blocks - 2:
        raise ValueError("swap exceeds Anima num_blocks - 2")
    manifest = None
    if args.precision == "fp32-reference":
        model.float()
    elif args.precision == "fp16-islands":
        patterns = args.fp32_pattern
        if getattr(args, "fp16_only_pattern", ()):
            patterns = fp32_complement(model, args.fp16_only_pattern)
        manifest = install_training_precision(
            model, SimpleNamespace(adaptive_fp32_modules=patterns,
                                   adaptive_loss_scale=args.loss_scale, model_family="anima"),
            model_family="anima",
        )
        if getattr(args, "backward_counterfactual_suite", False) and args.base_backward_mode == "fp32":
            backward_counterfactual.install(model, manifest)
    report["precision_manifest"] = manifest
    report["precision_profile_resolved"] = True
    report["fp32_modules"] = ([] if manifest is None else
                               [k for k, v in manifest["assignments"].items() if v == "fp32"])
    cfg = LoRANetworkCfg.from_kwargs({}, network_dim=4, network_alpha=4,
                                    neuron_dropout=None, module_class=LoRAModule)
    network = LoRANetwork(text_encoders=[], unet=model, cfg=cfg, multiplier=1.0)
    network.apply_to(text_encoders=[], unet=model, apply_text_encoder=False, apply_unet=True)
    network.to(device=device, dtype=torch.float32)
    if not network.unet_loras or any(p.dtype != torch.float32 for p in network.parameters()
                                     if p.requires_grad):
        raise ValueError("Expected trainable FP32 Anima LoRA parameters")
    model.enable_gradient_checkpointing()
    if args.precision == "fp16-islands":
        model.enable_fp32_residual()
    report["stage"] = "weight_placement"
    write_result(args.output, report)
    if args.swap:
        model.enable_block_swap(args.swap, device, transfer_dtype=None, restore_mode="foreach")
        model.move_to_device_except_swap_blocks(device)
        model.switch_block_swap_for_training()
    else:
        model.to(device)
    model.train()
    network.train()
    return model, network


def signature_for(args, report):
    signature = {"family": "anima", "weights_sha256": report["weights_sha256"],
            "probe_sha256": sha256_file(Path(__file__)),
            "reference_weight_scope": "bf16_source_values_promoted_to_fp32",
            "inputs_sha256": report["inputs_sha256"], "rank": 4, "alpha": 4,
            "seed": SEED, "sigmas": ([args.single_sigma] if getattr(args, "single_sigma", None) is not None else
                                     [0.2 + 0.6 * i / max(1, args.steps - 1) for i in range(args.steps)]),
            "steps": args.steps, "backend": "torch", "checkpoint": "full",
            "latent_padding_mask": "zeros_B1HW",
            "timestep_policy": "bf16_cast_at_model_boundary_only_for_bf16",
            "precision": args.precision, "fp32_modules": report["fp32_modules"],
            "loss_scale": args.loss_scale}
    if args.record_replay or args.replay_reference:
        signature["comparison_mode"] = MODE
    if getattr(args, "capture_update", False):
        signature["capture_update"] = True
        signature["update_source_sha256"] = sha256_file(Path(update_comparison.__file__))
    if getattr(args, "capture_local_vjp", False):
        signature["local_vjp_source_sha256"] = sha256_file(Path(__file__).with_name("anima_local_vjp.py"))
        profile = getattr(args, "local_vjp_profile", "linear")
        signature["local_vjp_profile"] = profile
        if profile == "mlp-path":
            signature["mlp_path_source_sha256"] = sha256_file(Path(__file__).with_name("anima_mlp_path.py"))
    if getattr(args, "backward_counterfactual_suite", False):
        signature["backward_counterfactual_scope"] = backward_counterfactual.SCOPE
        signature["backward_counterfactual_helper_sha256"] = sha256_file(
            Path(backward_counterfactual.__file__))
    if getattr(args, "capture_block_boundaries", False):
        signature["block_boundary_scope"] = block_boundaries.SCOPE
        signature["block_boundary_helper_sha256"] = sha256_file(Path(block_boundaries.__file__))
    if getattr(args, "capture_block_vjp", False):
        indices = block_vjp_indices_for(args)
        signature["block_vjp_scope"] = block_vjp.scope_for(indices)
        signature["block_vjp_indices"] = list(indices)
        signature["block_vjp_helper_sha256"] = sha256_file(Path(block_vjp.__file__))
    return signature


def _optimizer_step(optimizer, scaler, params):
    before = [float(optimizer.state[p].get("step", 0)) for p in params]
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    after = [float(optimizer.state[p].get("step", 0)) for p in params]
    if any(end != start + 1 for start, end in zip(before, after, strict=True)):
        raise NonfiniteTrainingError("AdamW step skipped or incomplete")
    return previous_scale, scaler.get_scale()


def forward_prediction(model, noisy, sigma, prompt, precision):
    # AdaLN uses the sinusoidal input directly with autocast disabled.
    model_sigma = sigma.to(torch.bfloat16) if precision == "bf16" else sigma
    context = (torch.autocast(noisy.device.type, dtype=torch.bfloat16)
               if precision == "bf16" else nullcontext())
    padding_mask = noisy.new_zeros(noisy.shape[0], 1, *noisy.shape[-2:])
    with context:
        return model(noisy, model_sigma, prompt, padding_mask=padding_mask)


def clip_finite_gradients(params):
    norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=False)
    if not bool(torch.isfinite(norm)):
        raise NonfiniteTrainingError("Nonfinite Anima gradient norm")
    return norm


def train_steps(args, model, network, inputs, report, resources):
    device = inputs["latents"].device
    params = [p for p in network.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable Anima LoRA parameters")
    initial = [p.detach().cpu().clone() for p in params]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    state_writer = None
    if getattr(args, "export_validation_states", None):
        state_writer = StateBundleWriter(args.export_validation_states,
                                         identity=validation_probe.state_identity(report),
                                         uuid=report["uuid"])
        state_writer.record("initial", network, optimizer, device)
    scaler = make_scaler(args.loss_scale, device=device.type)
    signature = signature_for(args, report)
    report["signature"] = signature
    update_capture = None
    if getattr(args, "capture_update", False):
        update_capture = update_comparison.UpdateCapture(
            args.output.parent / "update-capture", network=network,
            signature=signature, uuid=report["uuid"])
        resources.append(update_capture)
    capture = TrainingCapture(args.output.parent / "training-capture", network=network,
                              signature=signature, steps=args.steps)
    if getattr(args, "backward_counterfactual_suite", False):
        capture.manifest["counterfactual_mode"] = args.base_backward_mode
        capture.manifest["counterfactual_helper_sha256"] = signature[
            "backward_counterfactual_helper_sha256"]
        capture._write()
    resources.append(capture)
    replay = None
    if args.record_replay or args.replay_reference:
        replay = TrainingReplay(args.output.parent / "training-replay", signature=signature,
                                uuid=report["uuid"], steps=args.steps,
                                source=args.replay_reference)
        resources.append(replay)
    for index in range(args.steps):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        optimizer.zero_grad(set_to_none=True)
        sigma_value = signature["sigmas"][index]
        sigma = torch.tensor([sigma_value], device=device, dtype=torch.float32)
        noisy = ((1 - sigma) * inputs["latents"] + sigma * inputs["noise"]).detach().requires_grad_(True)
        target = (inputs["noise"] - inputs["latents"]).float()
        prompt = inputs["prompt"]
        replay_evidence = None
        if replay is not None:
            noisy, target, prompts, sigma, replay_evidence = replay.prepare(
                network, noisy=noisy, target=target, prompts=[prompt], sigma=sigma, step=index + 1)
            prompt = prompts[0]
        report.update(stage="forward", step=index + 1, sigma=sigma_value)
        write_result(args.output, report)
        started = time.perf_counter()
        local = None
        boundary = None
        block_capture = None
        if getattr(args, "capture_local_vjp", False):
            local = LocalVJPCapture(args.output.parent / "local-vjp", model, network,
                                    signature=signature, uuid=report["uuid"], scale=args.loss_scale,
                                    replay=replay_evidence)
        if getattr(args, "capture_block_boundaries", False):
            boundary = block_boundaries.BlockBoundaryCapture(
                args.output.parent / "block-boundaries", model, signature=signature,
                uuid=report["uuid"], scale=args.loss_scale, replay=replay_evidence)
        if getattr(args, "capture_block_vjp", False):
            indices = block_vjp_indices_for(args)
            capture_type = (block_vjp.BlockVJPCapture if indices == (14,)
                            else block_vjp.MultiBlockVJPCapture)
            block_capture = capture_type(args.output.parent / "block-vjp", model, network,
                                         signature=signature, uuid=report["uuid"],
                                         scale=args.loss_scale, replay=replay_evidence)
        try:
            prediction = forward_prediction(model, noisy, sigma, prompt, args.precision)
            if not isinstance(prediction, torch.Tensor) or prediction.shape != noisy.shape:
                raise ValueError("Anima prediction must match 5D latent shape")
            loss = (prediction.float() - target).square().mean()
            if not bool(torch.isfinite(loss)):
                raise NonfiniteTrainingError("Nonfinite Anima loss")
            report["stage"] = "backward"
            write_result(args.output, report)
            backward_unscaled(loss, noisy, optimizer, scaler)
            if local is not None:
                local.close()
            if boundary is not None:
                boundary.close()
            if block_capture is not None:
                block_capture.close()
        finally:
            if local is not None:
                local.restore()
            if boundary is not None:
                boundary.restore()
            if block_capture is not None:
                block_capture.restore()
        if (noisy.grad is None or not bool(torch.isfinite(noisy.grad).all())
                or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params)):
            raise NonfiniteTrainingError("Missing or nonfinite Anima input/adapter gradients")
        capture.capture(network=network, prediction=prediction, noisy=noisy,
                        step=index + 1, sigma=sigma_value, replay=replay_evidence)
        if update_capture is not None:
            update_capture.before(network=network, optimizer=optimizer, step=index + 1,
                                  sigma=sigma_value, replay=replay_evidence,
                                  training_case=capture.manifest["cases"][-1], clip_max_norm=1.0)
        norm = clip_finite_gradients(params)
        report["stage"] = "optimizer"
        write_result(args.output, report)
        scale_before, scale_after = _optimizer_step(optimizer, scaler, params)
        if update_capture is not None:
            update_capture.after(network=network, optimizer=optimizer, gradient_norm=norm)
        if state_writer is not None:
            state_writer.record("nonzero", network, optimizer, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        report["updates"].append({"step": index + 1, "loss": float(loss.detach()),
                                  "gradient_norm": float(norm), "loss_scale_before": scale_before,
                                  "loss_scale_after": scale_after,
                                  "adapter_max_delta": max(float((p.detach().cpu() - old).abs().max())
                                                           for p, old in zip(params, initial, strict=True)),
                                  "seconds": time.perf_counter() - started,
                                  "peak_allocated": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
                                  "peak_reserved": torch.cuda.max_memory_reserved() if device.type == "cuda" else 0})
        write_result(args.output, report)
    report["adapter_max_delta"] = report["updates"][-1]["adapter_max_delta"]
    if report["adapter_max_delta"] <= 0:
        raise RuntimeError("No Anima LoRA update")
    if state_writer is not None:
        state_writer.finish({"warmup_inputs_sha256": report["inputs_sha256"],
                             "warmup_signature": signature,
                             "training_case_sha256": capture.manifest["cases"][0]["sha256"],
                             "update_case_sha256": update_capture.manifest["cases"][0]["sha256"]})
        report["validation_state_bundle"] = str(state_writer.directory)


def close_resources(resources):
    model = next((value for value in resources if hasattr(value, "offloader")), None)
    for resource in reversed(resources):
        if hasattr(resource, "close"):
            resource.close()
    offloader = getattr(model, "offloader", None)
    if offloader is not None:
        for index in list(offloader.futures):
            offloader._wait_blocks_move(index, phase="probe_close")
        offloader.flush_profile_events(blocking=True)


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "stage": "validate", "scope":
              "full_anima_lora_real_cached_inputs_disposable_not_production_certificate",
              "production_ready": False, "full_model_calibrated": False,
              "precision_calibrated": False,
              "disposable_probe": True, "precision_profile_resolved": False,
              "precision": args.precision, "swap": args.swap, "steps": args.steps,
              "loss_scale": args.loss_scale, "updates": [], "adapter_saved": False,
              "checkpoint_saved": False, "replay_mode": "record" if args.record_replay
              else "replay" if args.replay_reference else None, "pid": os.getpid()}
    if getattr(args, "backward_counterfactual_suite", False):
        report.update(counterfactual_mode=args.base_backward_mode,
                      backward_counterfactual_scope=backward_counterfactual.SCOPE,
                      counterfactual_helper_sha256=sha256_file(Path(backward_counterfactual.__file__)))
    resources, model = [], None
    write_result(args.output, report)
    try:
        validate_args(args)
        report["source_weight_dtypes"] = checkpoint_source_dtypes(args.weights)
        seed_all()
        properties = torch.cuda.get_device_properties(0)
        report.update(gpu=properties.name, uuid=device_uuid(properties.uuid), torch=str(torch.__version__))
        if args.precision == "bf16" and torch.cuda.get_device_capability(0)[0] < 8:
            raise ValueError("BF16 requires native Ampere+ capability")
        inputs, digest = load_anima_inputs(args.inputs, device="cuda:0", dtype=torch.float32)
        report.update(inputs_sha256=digest, weights_sha256=sha256_file(args.weights),
                      input_shapes={k: list(v.shape) for k, v in inputs.items()},
                      input_source=json.loads(args.inputs.with_suffix(".json").read_text()),
                      base_load_dtype="bf16", adapter_master_dtype="fp32",
                      timestep_compute_dtype="bf16" if args.precision == "bf16" else "fp32",
                      runtime_input_scope="fixed_fp32_replay_not_production_input_rounding",
                      compute_dtype="bf16_autocast" if args.precision == "bf16" else
                                    "fp16_fp32_islands" if args.precision == "fp16-islands" else "fp32")
        report["stage"] = "model_load"
        write_result(args.output, report)
        model, network = build(args, report, resources)
        if getattr(args, "validation_states", None):
            validation_probe.train_validation(args, model, network, inputs, report, resources)
        else:
            train_steps(args, model, network, inputs, report, resources)
        report["status"] = "ok"
    except BaseException as exc:
        traceback.print_exc()
        report.update(status="cuda_oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "error",
                      failure_kind="nonfinite" if isinstance(exc, NonfiniteTrainingError) else
                                   "cuda_oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "runtime",
                      error_kind="nonfinite" if isinstance(exc, NonfiniteTrainingError) else
                                 "cuda_oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "runtime",
                      error_type=type(exc).__name__, error=str(exc))
    finally:
        try:
            close_resources(resources)
        except Exception as exc:
            report.update(status="error", stage="cleanup", failure_kind="cleanup", error_kind="cleanup",
                          error_type=type(exc).__name__, error=str(exc))
        write_result(args.output, report)
        if report["status"] == "ok":
            try:
                for resource in resources:
                    if isinstance(resource, (update_comparison.UpdateCapture, ValidationUpdateCapture)):
                        resource.bind_result(args.output.parent)
            except Exception as exc:
                report.update(status="error", stage="bind_result", failure_kind="cleanup",
                              error_kind="cleanup", error_type=type(exc).__name__, error=str(exc))
                write_result(args.output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32-reference", "fp16-islands", "bf16"), required=True)
    parser.add_argument("--swap", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--loss-scale", type=float, default=1.0)
    parser.add_argument("--fp32-pattern", action="append", default=[])
    parser.add_argument("--fp16-only-pattern", action="append", default=[])
    parser.add_argument("--capture-local-vjp", action="store_true")
    parser.add_argument("--capture-block-boundaries", action="store_true")
    parser.add_argument("--capture-block-vjp", action="store_true")
    parser.add_argument("--capture-update", action="store_true")
    states = parser.add_mutually_exclusive_group()
    states.add_argument("--export-validation-states", type=Path)
    states.add_argument("--validation-states", type=Path)
    parser.add_argument("--block-vjp-indices")
    parser.add_argument("--local-vjp-profile", choices=("linear", "mlp-path"), default="linear")
    parser.add_argument("--single-sigma", type=float)
    parser.add_argument("--backward-counterfactual-suite", action="store_true")
    parser.add_argument("--base-backward-mode", choices=("native", "fp32"), default="native")
    replay = parser.add_mutually_exclusive_group()
    replay.add_argument("--record-replay", action="store_true")
    replay.add_argument("--replay-reference", type=Path)
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
