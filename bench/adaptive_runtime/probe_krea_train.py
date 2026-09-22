"""Full Krea DiT/LoRA probe with immutable real inputs and incremental evidence."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from fnmatch import fnmatchcase
import json
import os
from pathlib import Path
import time

import torch

from bench.adaptive_runtime.inputs import load_krea_inputs
from bench.adaptive_runtime.capture import LinearCapture
from bench.adaptive_runtime.checkpoint import read_checkpoint, restore_checkpoint, save_checkpoint
from bench.adaptive_runtime.training_capture import TrainingCapture
from bench.adaptive_runtime.scaling import backward_unscaled, make_scaler, validate_scaling
from bench.adaptive_runtime.reference_storage import promote_placed_reference, validate_reference_storage
from bench.adaptive_runtime.replay import MODE, TrainingReplay, validate_replay
from library.models.krea2_raw.attention_backend import prepare_krea2_attention
from library.models.krea2_raw.family import Krea2TextEmbedding, forward_for_loss
from library.models.krea2_raw.lora_targets import krea2_target_kwargs
from library.models.krea2_raw.quantize import inspect_nf4_checkpoint
from library.models.krea2_raw.weights import load_krea2_dit
from library.training.adaptive_runtime.finite_trace import FiniteTrace
from library.training.adaptive_runtime.islands import install_precision_islands
from library.training.auto_block_swap.process import write_result
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.network import LoRANetwork
from networks.lora_modules.lora import LoRAModule


class NonfiniteTrainingError(RuntimeError):
    pass


def build(args, report):
    device = torch.device("cuda:0")
    mixed = args.precision == "fp16-islands"
    reference = args.precision == "fp32-reference"
    dtype = torch.float32 if mixed or reference else torch.bfloat16
    if args.precision == "bf16" and torch.cuda.get_device_capability()[0] < 8:
        raise ValueError("BF16/NF4 probe requires native BF16; use nonquantized islands on Turing")
    if (mixed or reference) and inspect_nf4_checkpoint(str(args.weights)).is_nf4:
        raise ValueError("Precision islands/reference require nonquantized base weights, not NF4")
    model = load_krea2_dit(args.weights, device="cpu", dtype=torch.bfloat16, eval=False)
    model.requires_grad_(False)
    if not 0 <= args.swap <= len(model.blocks) - 2:
        raise ValueError("Swap outside Krea model bounds")
    assignments = {}
    compact_reference = getattr(args, "reference_bf16_storage", False)
    if reference and not compact_reference:
        report.update(substage="fp32_reference_conversion")
        write_result(args.output, report)
        # Free resident CPU payload before expanding the remaining base to FP32.
        for block in model.blocks[:len(model.blocks) - args.swap]:
            block.to(device=device, dtype=torch.float32)
        model.float()
    if mixed:
        assignments = {n: "fp16" for n, m in model.named_modules() if type(m) is torch.nn.Linear}
        for pattern in args.fp32_pattern:
            matches = [n for n in assignments if fnmatchcase(n, pattern)]
            if not matches:
                raise ValueError(f"Unknown FP32 Linear pattern: {pattern}")
            assignments.update({n: "fp32" for n in matches})
        install_precision_islands(model, assignments)
        for module in model.modules():
            if type(module) is torch.nn.Linear:
                continue
            for parameter in module.parameters(recurse=False):
                if parameter.is_floating_point():
                    parameter.data = parameter.data.float()
            for name, buffer in module.named_buffers(recurse=False):
                if buffer.is_floating_point():
                    module._buffers[name] = buffer.float()
    report["fp32_modules"] = [n for n, d in assignments.items() if d == "fp32"]
    report["island_count"] = len(assignments)
    prepare_krea2_attention(model, "torch", dtype=dtype)
    cfg = LoRANetworkCfg.from_kwargs(krea2_target_kwargs(), network_dim=4,
                                    network_alpha=4, neuron_dropout=None, module_class=LoRAModule)
    network = LoRANetwork(text_encoders=[], unet=model, cfg=cfg, multiplier=1.0)
    network.apply_to(text_encoders=[], unet=model, apply_text_encoder=False, apply_unet=True)
    network.to(device=device, dtype=torch.float32)
    report["lora_modules"] = len(network.unet_loras)
    model.enable_gradient_checkpointing()
    report.update(stage="model_load", substage="weight_placement")
    write_result(args.output, report)
    if args.swap:
        model.enable_block_swap(args.swap, device, restore_mode="foreach")
        model.move_to_device_except_swap_blocks(device)
        # Release resident CPU payload before master capture, avoiding a full
        # duplicated host model while keeping the intended GPU residency.
        for block in model.blocks[:len(model.blocks) - args.swap]:
            block.to(device)
        model.switch_block_swap_for_training()
    else:
        model.to(device)
    if compact_reference:
        model._reference_execution_guard = promote_placed_reference(model, report)
    model.train()
    network.train()
    return model, network, dtype


def updates(args, model, network, dtype, report, captures):
    cached, digest = load_krea_inputs(args.inputs, resolution=args.resolution,
                                      device="cuda", dtype=dtype)
    report["inputs_sha256"] = digest
    text = Krea2TextEmbedding(cached["hidden"], cached["mask"])
    params = [p for p in network.parameters() if p.requires_grad]
    initial = [p.detach().cpu().clone() for p in params]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    scaler = make_scaler(args.loss_scale)
    stat = args.weights.stat()
    signature = {"inputs_sha256": digest, "weights": str(args.weights.resolve()),
                 "weight_bytes": stat.st_size, "weight_mtime_ns": stat.st_mtime_ns,
                 "precision": args.precision, "fp32_modules": report["fp32_modules"],
                 "steps": args.steps, "resolution": args.resolution, "rank": 4, "alpha": 4,
                 "learning_rate": 1e-4, "torch": str(torch.__version__), "scheduler": None}
    if args.loss_scale != 1:
        signature["loss_scale"] = args.loss_scale
    replay = None
    if getattr(args, "record_replay", False) or getattr(args, "replay_reference", None):
        signature["comparison_mode"] = MODE
        replay = TrainingReplay(args.output.parent / "training-replay", signature=signature,
                                uuid=report["uuid"], steps=args.steps, source=args.replay_reference)
    if args.capture_training:
        captures.append(TrainingCapture(args.output.parent / "training-capture", network=network,
                                        signature=signature, steps=args.steps))
    if replay is not None:
        captures.append(replay)
    if args.resume:
        payload = read_checkpoint(args.resume, signature=signature)
        if payload["step"] > args.steps:
            raise ValueError("Checkpoint is beyond the requested training length")
        report.update(stage="resume", optimizer_started=True,
                      committed_checkpoint=str(args.resume.resolve()), resumed_from=str(args.resume))
        write_result(args.output, report)
        initial, report["updates"] = restore_checkpoint(
            payload, network=network, optimizer=optimizer, params=params, device="cuda",
            scaler=scaler if args.scaled_checkpoint else None,
        )
        del payload
    for index in range(len(report["updates"]), args.steps):
        optimizer.zero_grad(set_to_none=True)
        sigma = torch.tensor([0.2 + 0.6 * index / max(1, args.steps - 1)], device="cuda")
        noisy = ((1 - sigma) * cached["latents"] + sigma * cached["noise"]).to(dtype)
        noisy.requires_grad_(True)
        target = (cached["noise"] - cached["latents"]).float()
        step_text = text
        replay_evidence = None
        if replay is not None:
            noisy, target, prompts, sigma, replay_evidence = replay.prepare(
                network, noisy=noisy, target=target, prompts=[text.hiddens, text.mask],
                sigma=sigma, step=index + 1)
            step_text = Krea2TextEmbedding(*prompts)
        started = time.perf_counter()
        report["stage"] = "forward"
        report["sigma"] = 0.2 + 0.6 * index / max(1, args.steps - 1)
        write_result(args.output, report)
        context = torch.autocast("cuda", dtype=dtype) if dtype == torch.bfloat16 else nullcontext()
        with context:
            prediction = forward_for_loss(model, noisy, step_text, sigma)
        loss = (prediction.float() - target).square().mean()
        if not torch.isfinite(loss):
            raise NonfiniteTrainingError("Nonfinite Krea loss")
        report["stage"] = "backward"
        write_result(args.output, report)
        backward_unscaled(loss, noisy, optimizer, scaler)
        gradients = [p.grad for p in params if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise NonfiniteTrainingError("Missing or nonfinite Krea adapter gradients")
        if captures:
            captures[0].capture(network=network, prediction=prediction, noisy=noisy,
                                step=index + 1, sigma=report["sigma"], replay=replay_evidence)
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        report.update(stage="optimizer", optimizer_started=True)
        write_result(args.output, report)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        report["updates"].append({"step": index + 1, "loss": float(loss.detach()),
                                  "gradient_norm": float(norm),
                                  "seconds": time.perf_counter() - started,
                                  "swap": args.swap,
                                  "peak_allocated": torch.cuda.max_memory_allocated(),
                                  "peak_reserved": torch.cuda.max_memory_reserved()})
        del prediction, loss, noisy
        if args.checkpoint_every_step:
            report["committed_checkpoint"] = save_checkpoint(
                args.output.parent / "checkpoints" / f"step-{index + 1:06d}.pt",
                network=network, optimizer=optimizer, initial=initial, updates=report["updates"],
                signature=signature, device="cuda",
                scaler=scaler if args.scaled_checkpoint else None,
            )
        write_result(args.output, report)
    report["adapter_max_delta"] = max(float((p.detach().cpu() - old).abs().max())
                                      for p, old in zip(params, initial, strict=True))
    if report["adapter_max_delta"] == 0:
        raise RuntimeError("No adapter update")
    report["adapter_saved"] = not args.disposable_probe
    if not args.disposable_probe:
        network.save_weights(str(args.output.with_suffix(".safetensors")), torch.float32, {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("bf16", "fp16-islands", "fp32-reference"), required=True)
    parser.add_argument("--swap", type=int, default=20)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--fp32-pattern", action="append", default=[])
    parser.add_argument("--memory-limit-gib", type=float)
    parser.add_argument("--checkpoint-every-step", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--structured-worker", action="store_true")
    parser.add_argument("--capture-linear", action="append", default=[])
    parser.add_argument("--disposable-probe", action="store_true")
    parser.add_argument("--capture-training", action="store_true")
    parser.add_argument("--loss-scale", type=float, default=1.0)
    parser.add_argument("--scaled-checkpoint", action="store_true")
    parser.add_argument("--reference-bf16-storage", action="store_true")
    replay_args = parser.add_mutually_exclusive_group()
    replay_args.add_argument("--record-replay", action="store_true")
    replay_args.add_argument("--replay-reference", type=Path)
    args = parser.parse_args()
    try:
        validate_scaling(args)
        validate_reference_storage(args)
        validate_replay(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.steps < 1 or args.resolution < 32 or args.resolution % 16:
        parser.error("Positive steps and resolution divisible by 16 (>=32) required")
    if args.resume and not args.checkpoint_every_step:
        parser.error("Resume requires checkpoint-every-step")
    if args.disposable_probe and (args.resume or args.checkpoint_every_step):
        parser.error("Disposable preflight cannot save or resume training checkpoints")
    if args.capture_training and args.resume:
        parser.error("Training comparison requires a fresh initial adapter")
    if args.precision != "fp16-islands" and args.fp32_pattern:
        parser.error("FP32 patterns require fp16-islands")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20260921)
    torch.backends.cuda.matmul.allow_tf32 = False
    prop = torch.cuda.get_device_properties(0)
    if args.memory_limit_gib is not None:
        fraction = args.memory_limit_gib * 1024**3 / prop.total_memory
        if not 0 < fraction <= 1:
            parser.error("Memory limit must be positive and no larger than device memory")
        torch.cuda.set_per_process_memory_fraction(fraction)
    report = {"status": "running", "stage": "model_load", "pid": os.getpid(),
              "gpu": prop.name, "uuid": str(prop.uuid), "torch": torch.__version__,
              "scope": "full_pretrained_krea_lora_real_cached_inputs", "weights": str(args.weights),
              "precision": args.precision, "precision_calibrated": False,
              "swap": args.swap, "resolution": args.resolution, "updates": [],
              "optimizer_started": False, "memory_limit_gib": args.memory_limit_gib}
    report["disposable_probe"] = args.disposable_probe
    report["loss_scale"] = args.loss_scale
    report["replay_mode"] = "record" if args.record_replay else "replay" if args.replay_reference else None
    write_result(args.output, report)
    trace = None
    capture = None
    training_captures = []
    model = None
    try:
        model, network, dtype = build(args, report)
        trace = FiniteTrace(model, report)
        if args.capture_linear:
            if args.resume:
                raise ValueError("Activation capture requires a fresh non-resumed probe")
            capture = LinearCapture(model, args.capture_linear, args.output.parent / "capture",
                                    report, max_cases=args.steps)
        updates(args, model, network, dtype, report, training_captures)
        report["status"] = "ok"
    except Exception as exc:
        report.update(status="cuda_oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "error",
                      error_type=type(exc).__name__, error=str(exc),
                      failure_kind="nonfinite" if isinstance(exc, NonfiniteTrainingError) else "runtime")
        if not args.structured_worker:
            raise
    finally:
        guard = getattr(model, "_reference_execution_guard", None)
        if guard is not None:
            guard.close()
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
