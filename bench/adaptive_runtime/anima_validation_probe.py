"""Restored-state six-case Anima validation using the disposable probe math."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import torch

from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime.anima_validation_state import StateBundle
from bench.adaptive_runtime.anima_validation_updates import CASES, SIGMAS, ValidationUpdateCapture, case_id
from bench.adaptive_runtime.replay import TrainingReplay
from bench.adaptive_runtime.scaling import backward_unscaled, make_scaler
from bench.adaptive_runtime.training_capture import TrainingCapture
from library.training.auto_block_swap.process import write_result


SOURCE_FILES = ("bench/adaptive_runtime/probe_anima_train.py",
                "bench/adaptive_runtime/anima_validation_evidence.py",
                "bench/adaptive_runtime/anima_validation_probe.py",
                "bench/adaptive_runtime/anima_validation_state.py",
                "bench/adaptive_runtime/anima_validation_updates.py",
                "bench/adaptive_runtime/anima_update_comparison.py",
                "bench/adaptive_runtime/anima_inputs.py", "bench/adaptive_runtime/replay.py",
                "bench/adaptive_runtime/training_capture.py", "bench/adaptive_runtime/scaling.py",
                "library/anima/models.py", "library/anima/weights.py",
                "library/training/adaptive_runtime/precision.py",
                "library/training/adaptive_runtime/profile_search.py",
                "library/training/adaptive_runtime/islands.py",
                "library/training/adaptive_runtime/training_precision.py",
                "networks/lora_anima/network.py", "networks/lora_modules/lora.py",
                "networks/attention_dispatch.py", "networks/attn_fuse.py")


def source_code_sha256():
    """One input-independent identity for the numerical probe and its state helpers."""
    root = Path(__file__).resolve().parents[2]
    sources = {name: sha256_file(root / name) for name in SOURCE_FILES}
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


def state_identity(report):
    return {"family": "anima", "weights_sha256": report["weights_sha256"],
            "source_code_sha256": source_code_sha256(), "backend": "torch", "rank": 4,
            "alpha": 4, "optimizer": "torch.optim.AdamW", "scheduler": "none"}


def validation_signature(args, report):
    from bench.adaptive_runtime.probe_anima_train import signature_for

    signature = signature_for(args, report)
    signature["sigmas"] = list(SIGMAS)
    signature["steps"] = 6
    signature["case_schedule"] = [{"case_id": case_id(label, sigma), "state_label": label,
                                   "sigma": sigma} for label, sigma in CASES]
    signature["validation_source_code_sha256"] = source_code_sha256()
    signature["validation_state_identity"] = state_identity(report)
    return signature


def validate_validation_args(args):
    if (args.steps != 6 or args.single_sigma is not None or not args.capture_update
            or not (args.record_replay or args.replay_reference)
            or args.capture_local_vjp or args.capture_block_boundaries or args.capture_block_vjp
            or args.backward_counterfactual_suite or args.block_vjp_indices is not None
            or args.local_vjp_profile != "linear" or args.base_backward_mode != "native"
            or getattr(args, "resume", False)):
        raise ValueError("Validation requires six replay/update cases without local diagnostics or resume")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _forward_loss(probe, args, model, noisy, target, prompt, sigma):
    prediction = probe.forward_prediction(model, noisy, sigma, prompt, args.precision)
    if not isinstance(prediction, torch.Tensor) or prediction.shape != noisy.shape:
        raise ValueError("Anima prediction must match 5D latent shape")
    loss = (prediction.float() - target).square().mean()
    if not bool(torch.isfinite(loss)):
        raise probe.NonfiniteTrainingError("Nonfinite Anima loss")
    return prediction, loss


def train_validation(args, model, network, inputs, report, resources):
    from bench.adaptive_runtime import probe_anima_train as probe

    device = inputs["latents"].device
    params = [p for p in network.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable Anima LoRA parameters")
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    signature = validation_signature(args, report)
    report["signature"] = signature
    bundle = StateBundle(args.validation_states, identity=state_identity(report), uuid=report["uuid"])
    capture = TrainingCapture(args.output.parent / "training-capture", network=network,
                              signature=signature, steps=6)
    resources.append(capture)
    replay = TrainingReplay(args.output.parent / "training-replay", signature=signature,
                            uuid=report["uuid"], steps=6, source=args.replay_reference)
    resources.append(replay)
    updates = ValidationUpdateCapture(args.output.parent / "update-capture", network=network,
                                      signature=signature, uuid=report["uuid"])
    resources.append(updates)
    for index, (label, sigma_value) in enumerate(CASES, start=1):
        evidence = bundle.restore(label, network, optimizer, device)
        scaler = make_scaler(args.loss_scale, device=device.type)
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sigma = torch.tensor([sigma_value], device=device, dtype=torch.float32)
        noisy = ((1 - sigma) * inputs["latents"] + sigma * inputs["noise"]).detach().requires_grad_(True)
        target = (inputs["noise"] - inputs["latents"]).float()
        prompt = inputs["prompt"]
        noisy, target, prompts, sigma, replay_evidence = replay.prepare(
            network, noisy=noisy, target=target, prompts=[prompt], sigma=sigma, step=index)
        prompt = prompts[0]
        report.update(stage="forward", step=index, sigma=sigma_value,
                      case_id=case_id(label, sigma_value), state_label=label,
                      prestate_evidence=evidence)
        write_result(args.output, report)
        _sync(device)
        started = time.perf_counter()
        prediction, loss = _forward_loss(probe, args, model, noisy, target, prompt, sigma)
        _sync(device)
        forward_seconds = time.perf_counter() - started
        io_started = time.perf_counter()
        report["stage"] = "backward"
        write_result(args.output, report)
        report_io_seconds = time.perf_counter() - io_started
        started = time.perf_counter()
        backward_unscaled(loss, noisy, optimizer, scaler)
        _sync(device)
        backward_seconds = time.perf_counter() - started
        if (noisy.grad is None or not bool(torch.isfinite(noisy.grad).all())
                or any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in params)):
            raise probe.NonfiniteTrainingError("Missing or nonfinite Anima input/adapter gradients")
        io_started = time.perf_counter()
        capture.capture(network=network, prediction=prediction, noisy=noisy, step=index,
                        sigma=sigma_value, replay=replay_evidence)
        updates.before(network=network, optimizer=optimizer, step=index,
                       case_id=case_id(label, sigma_value), state_label=label, sigma=sigma_value,
                       replay=replay_evidence, training_case=capture.manifest["cases"][-1],
                       state_evidence=evidence, clip_max_norm=1.0)
        _sync(device)
        capture_io_seconds = time.perf_counter() - io_started + report_io_seconds
        before_params = [p.detach().clone() for p in params]
        io_started = time.perf_counter()
        report["stage"] = "optimizer"
        write_result(args.output, report)
        capture_io_seconds += time.perf_counter() - io_started
        step_started = time.perf_counter()
        norm = probe.clip_finite_gradients(params)
        before, after = probe._optimizer_step(optimizer, scaler, params)
        _sync(device)
        update_seconds = time.perf_counter() - step_started
        io_started = time.perf_counter()
        updates.after(network=network, optimizer=optimizer, gradient_norm=norm)
        capture_io_seconds += time.perf_counter() - io_started
        case = updates.manifest["cases"][-1]
        max_delta = max(float((p.detach() - old).abs().max())
                        for p, old in zip(params, before_params, strict=True))
        record = {"step": index, "case_id": case_id(label, sigma_value), "state_label": label,
                  "prestate_evidence": evidence, "loss": float(loss.detach()),
                  "gradient_norm": float(norm), "loss_scale_before": before,
                  "loss_scale_after": after, "update_sha256": case["sha256"],
                  "effective_update": max_delta > 0,
                  "adapter_max_delta": max_delta,
                  "compute_seconds": forward_seconds + backward_seconds + update_seconds,
                  "capture_io_seconds": capture_io_seconds,
                  "seconds": forward_seconds + backward_seconds + update_seconds + capture_io_seconds,
                  "peak_allocated": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                  "peak_reserved": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0}
        report["updates"].append(record)
        write_result(args.output, report)
    report["adapter_max_delta"] = max(case["adapter_max_delta"] for case in report["updates"])
    if report["adapter_max_delta"] <= 0:
        raise RuntimeError("No Anima LoRA update")
