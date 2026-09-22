"""Full pretrained Z-Image DiT + production LoRA, synthetic cached inputs.

The FP16-islands mode is an explicit experimental candidate, not a calibrated
production policy. This probe does not relax training compatibility checks.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from fnmatch import fnmatchcase
import json
import os
from pathlib import Path
import time

import torch
from bench.adaptive_runtime.inputs import load_z_image_inputs
from bench.adaptive_runtime.capture import LinearCapture
from bench.adaptive_runtime.checkpoint import save_checkpoint
from bench.adaptive_runtime.z_image_checkpoint import resume_if_requested, signature_for
from bench.adaptive_runtime.training_capture import TrainingCapture
from bench.adaptive_runtime.scaling import backward_unscaled, make_scaler, validate_scaling
from bench.adaptive_runtime.replay import MODE, TrainingReplay, validate_replay

from library.models.z_image.attention_backend import prepare_z_image_attention
from library.models.z_image.block_swap import enable_z_image_block_swap
from library.models.z_image.family import forward_for_loss
from library.models.z_image.lora_targets import z_image_target_kwargs
from library.models.z_image.weights import load_z_image_transformer
from library.training.adaptive_runtime.islands import install_precision_islands
from library.training.adaptive_runtime.finite_trace import FiniteTrace
from library.training.auto_block_swap.process import write_result
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.network import LoRANetwork
from networks.lora_modules.lora import LoRAModule


class NonfiniteTrainingError(RuntimeError):
    pass


def build(args, device, report=None):
    mixed = args.precision == "fp16-islands"
    reference = args.precision == "fp32-reference"
    dtype = torch.float32 if mixed or reference else torch.bfloat16
    if args.precision == "bf16" and torch.cuda.get_device_capability()[0] < 8:
        raise ValueError("BF16 probe requires native BF16; use islands on Turing")
    model = load_z_image_transformer(str(args.weights), dtype=torch.bfloat16, device="cpu")
    model.requires_grad_(False)
    assignments = {}
    if reference:
        model.float()
    if mixed:
        assignments = {name: "fp16" for name, module in model.named_modules()
                       if type(module) is torch.nn.Linear}
        for name in args.fp32_module:
            if name not in assignments:
                raise ValueError(f"Unknown Linear precision island: {name}")
            assignments[name] = "fp32"
        for pattern in args.fp32_pattern:
            matches = [name for name in assignments if fnmatchcase(name, pattern)]
            if not matches:
                raise ValueError(f"FP32 pattern matched no Linear: {pattern}")
            for name in matches:
                assignments[name] = "fp32"
        install_precision_islands(model, assignments)
        # Checkpoint weights are BF16; promote only the non-Linear state rather
        # than materializing all 6B weights in FP32 before downcasting them.
        for module in model.modules():
            if type(module) is torch.nn.Linear:
                continue
            for parameter in module.parameters(recurse=False):
                if parameter.is_floating_point():
                    parameter.data = parameter.data.float()
            for name, buffer in module.named_buffers(recurse=False):
                if buffer.is_floating_point():
                    module._buffers[name] = buffer.float()
    if report is not None:
        report.update(island_count=len(assignments), precision_profile_resolved=True,
                      fp32_modules=[name for name, value in assignments.items() if value == "fp32"])
        write_result(args.output, report)
    prepare_z_image_attention(model, "torch", dtype=dtype)
    cfg = LoRANetworkCfg.from_kwargs(z_image_target_kwargs(), network_dim=4,
                                    network_alpha=4, neuron_dropout=None,
                                    module_class=LoRAModule)
    network = LoRANetwork(text_encoders=[], unet=model, cfg=cfg, multiplier=1.0)
    network.apply_to(text_encoders=[], unet=model, apply_text_encoder=False, apply_unet=True)
    network.to(device=device, dtype=torch.float32)
    model.enable_gradient_checkpointing()
    if args.swap:
        enable_z_image_block_swap(model, args.swap, device, restore_mode="foreach")
        model.move_to_device_except_swap_blocks(device)
        model.prepare_block_swap_before_forward()
    else:
        model.to(device)
    model.train()
    network.train()
    return model, network, dtype, assignments


def train_updates(model, network, dtype, args, report, captures):
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-4)
    scaler = make_scaler(args.loss_scale)
    side = args.resolution // 8
    if args.inputs:
        cached, digest = load_z_image_inputs(
            args.inputs, resolution=args.resolution, prompt_dim=model.config.cap_feat_dim,
            device="cuda", dtype=dtype,
        )
        latents, noise = cached["latents"], cached["noise"]
        prompts = [cached["prompt"]]
        report.update(scope="full_pretrained_dit_lora_real_cached_inputs",
                      inputs_sha256=digest)
    else:
        latents = torch.randn(1, 16, 1, side, side, device="cuda", dtype=dtype)
        noise = torch.randn_like(latents)
        prompts = [torch.randn(32, model.config.cap_feat_dim, device="cuda", dtype=dtype) * 0.02]
    params = [p for p in network.parameters() if p.requires_grad]
    initial = [p.detach().cpu().clone() for p in params]
    signature = signature_for(args, report) if args.checkpoint_every_step or args.capture_training else None
    if signature is not None and args.loss_scale != 1:
        signature["loss_scale"] = args.loss_scale
    replay = None
    if args.record_replay or args.replay_reference:
        signature["comparison_mode"] = MODE
        replay = TrainingReplay(args.output.parent / "training-replay", signature=signature,
                                uuid=report["uuid"], steps=args.steps, source=args.replay_reference)
    if args.capture_training:
        captures.append(TrainingCapture(args.output.parent / "training-capture", network=network,
                                        signature=signature, steps=args.steps))
    if replay is not None:
        captures.append(replay)
    checkpoint_scaler = scaler if args.scaled_checkpoint else None
    initial = resume_if_requested(args, report, signature, network, optimizer, params, initial,
                                  scaler=checkpoint_scaler)
    for step in range(len(report["updates"]), args.steps):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        sigma = torch.tensor([0.2 + 0.6 * step / max(1, args.steps - 1)], device="cuda")
        noisy = ((1 - sigma) * latents + sigma * noise).to(dtype).requires_grad_(True)
        target = (noise - latents).float()
        replay_evidence = None
        step_prompts = prompts
        if replay is not None:
            noisy, target, step_prompts, sigma, replay_evidence = replay.prepare(
                network, noisy=noisy, target=target, prompts=prompts, sigma=sigma, step=step + 1)
        report["stage"] = "forward"
        report["sigma"] = 0.2 + 0.6 * step / max(1, args.steps - 1)
        write_result(args.output, report)
        context = torch.autocast("cuda", dtype=dtype) if dtype == torch.bfloat16 else nullcontext()
        with context:
            prediction = forward_for_loss(model, noisy, step_prompts, sigma)
        loss = (prediction.float() - target).square().mean()
        if not torch.isfinite(loss):
            raise NonfiniteTrainingError("Nonfinite full-model loss")
        report["stage"] = "backward"
        write_result(args.output, report)
        backward_unscaled(loss, noisy, optimizer, scaler)
        gradients = [p.grad for p in params if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise NonfiniteTrainingError("Missing or nonfinite LoRA gradients")
        if captures:
            captures[0].capture(network=network, prediction=prediction, noisy=noisy,
                                step=step + 1, sigma=report["sigma"], replay=replay_evidence)
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        report.update(stage="optimizer", optimizer_started=True)
        write_result(args.output, report)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        report["updates"].append({"step": step + 1, "loss": float(loss.detach()),
                                  "gradient_norm": float(norm),
                                  "seconds": time.perf_counter() - start,
                                  "swap": args.swap,
                                  "peak_allocated": torch.cuda.max_memory_allocated()})
        del prediction, loss, noisy
        if args.checkpoint_every_step:
            report["committed_checkpoint"] = save_checkpoint(
                args.output.parent / "checkpoints" / f"step-{step + 1:06d}.pt",
                network=network, optimizer=optimizer, initial=initial, updates=report["updates"],
                signature=signature, device="cuda", scaler=checkpoint_scaler,
            )
        write_result(args.output, report)
    report["adapter_max_delta"] = max(float((p.detach().cpu() - old).abs().max())
                                      for p, old in zip(params, initial, strict=True))
    if report["adapter_max_delta"] == 0:
        raise RuntimeError("Optimizer did not update adapter weights")
    report["adapter_saved"] = not args.disposable_probe
    if not args.disposable_probe:
        network.save_weights(str(args.output.with_suffix(".safetensors")), torch.float32, {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp16-islands", "fp32-reference"), required=True)
    parser.add_argument("--swap", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--trace-numerics", action="store_true")
    parser.add_argument("--fp32-module", action="append", default=[])
    parser.add_argument("--fp32-pattern", action="append", default=[])
    parser.add_argument("--structured-worker", action="store_true")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--capture-linear", action="append", default=[])
    parser.add_argument("--checkpoint-every-step", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--memory-limit-gib", type=float)
    parser.add_argument("--disposable-probe", action="store_true")
    parser.add_argument("--capture-training", action="store_true")
    parser.add_argument("--loss-scale", type=float, default=1.0)
    parser.add_argument("--scaled-checkpoint", action="store_true")
    replay_args = parser.add_mutually_exclusive_group()
    replay_args.add_argument("--record-replay", action="store_true")
    replay_args.add_argument("--replay-reference", type=Path)
    args = parser.parse_args()
    try:
        validate_scaling(args)
        validate_replay(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.steps < 1 or args.resolution < 32 or args.resolution % 16:
        parser.error("Positive steps and resolution divisible by 16 (>=32) required")
    if args.checkpoint_every_step and not args.inputs:
        parser.error("Checkpoint requires immutable real cached inputs")
    if args.resume and (not args.checkpoint_every_step or args.capture_linear):
        parser.error("Resume requires checkpoint-every-step and cannot collect activations")
    if args.disposable_probe and (args.resume or args.checkpoint_every_step):
        parser.error("Disposable preflight cannot save or resume training checkpoints")
    if args.capture_training and (args.resume or not args.inputs):
        parser.error("Training comparison requires fresh immutable real inputs")
    if args.precision != "fp16-islands" and (args.fp32_module or args.fp32_pattern):
        parser.error("FP32 patterns/modules require fp16-islands")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260921)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    prop = torch.cuda.get_device_properties(0)
    if args.memory_limit_gib is not None:
        fraction = args.memory_limit_gib * 1024**3 / prop.total_memory
        if not 0 < fraction <= 1:
            parser.error("Memory limit must be positive and no larger than device memory")
        torch.cuda.set_per_process_memory_fraction(fraction)
    report = {"status": "running", "scope": "full_pretrained_dit_lora_synthetic_inputs",
              "pid": os.getpid(),
              "precision_calibrated": False, "precision": args.precision,
              "gpu": prop.name, "uuid": str(prop.uuid), "torch": torch.__version__,
              "swap": args.swap, "resolution": args.resolution, "updates": [],
              "fp32_modules": args.fp32_module,
              "requested_fp32_patterns": args.fp32_pattern, "precision_profile_resolved": False,
              "optimizer_started": False, "memory_limit_gib": args.memory_limit_gib,
              "disposable_probe": args.disposable_probe,
              "loss_scale": args.loss_scale,
              "replay_mode": "record" if args.record_replay else "replay" if args.replay_reference else None,
              "stage": "model_load"}
    write_result(args.output, report)
    trace = None
    capture = None
    training_captures = []
    try:
        model, network, dtype, assignments = build(args, torch.device("cuda:0"), report)
        report.update(island_count=len(assignments), lora_modules=len(network.unet_loras))
        report["fp32_modules"] = [name for name, dtype in assignments.items() if dtype == "fp32"]
        if args.trace_numerics:
            trace = FiniteTrace(model, report)
        if args.capture_linear:
            capture = LinearCapture(model, args.capture_linear, args.output.parent / "capture",
                                    report, max_cases=args.steps)
        train_updates(model, network, dtype, args, report, training_captures)
        report["status"] = "ok"
    except Exception as exc:
        status = "cuda_oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "error"
        report.update(status=status, error_type=type(exc).__name__, error=str(exc))
        report["failure_kind"] = "nonfinite" if isinstance(exc, NonfiniteTrainingError) else "runtime"
        if not args.structured_worker:
            raise
    finally:
        if trace is not None:
            trace.close()
        if capture is not None:
            capture.close()
        for training_capture in training_captures:
            training_capture.close()
        write_result(args.output, report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
