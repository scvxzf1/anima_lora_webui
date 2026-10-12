"""Run one mode per fresh process: python -m bench.convrot_90hx.runner."""

from __future__ import annotations
import argparse
import json
import math
import time
import traceback
from pathlib import Path
import torch
from safetensors.torch import save_file
from .inputs import load_input
from .manifest import (
    ROOT,
    GPU_UUID,
    file_hash,
    object_hash,
    software_manifest,
    validate_device,
    write_json,
)
from .model import build_model, drain, placement
from .telemetry import Telemetry

DEFAULT_MODEL = (
    "/home/scv/pac/models/diffusion_models/anima/anima-base-v1.0-sha.safetensors"
)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=["prepare", "run"])
    cli.add_argument(
        "--kind",
        choices=["smoke", "performance", "numerical", "profile", "latency"],
        default="smoke",
    )
    cli.add_argument("--mode", choices=["bf16", "w8a16", "w8a8"], default="bf16")
    cli.add_argument("--out-dir", type=Path, required=True)
    cli.add_argument("--model-path", type=Path, default=Path(DEFAULT_MODEL))
    cli.add_argument(
        "--cache-dir", type=Path, default=ROOT / "output/ab-anima-20261002/cache"
    )
    cli.add_argument(
        "--image-dir", type=Path, default=ROOT / "output/ab-anima-20261002/images-1024"
    )
    cli.add_argument("--swap", type=int, default=12)
    cli.add_argument("--warmup", type=int, default=10)
    cli.add_argument("--steps", type=int, default=3)
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--cache-index", type=int, default=0)
    cli.add_argument("--timestep", type=float)
    cli.add_argument("--expected-gpu-uuid", default=GPU_UUID)
    cli.add_argument(
        "--forward-policy",
        choices=["torch", "triton_quant", "triton_gemm", "triton_both"],
        default="torch",
    )
    cli.add_argument(
        "--backward-policy", choices=["fp32", "bf16", "int8"], default="fp32"
    )
    return cli


def effective_config(args):
    if args.steps < 1 or args.warmup < 0 or args.swap < 0:
        raise ValueError("steps>=1, warmup>=0, swap>=0 required")
    return {
        "mode": args.mode,
        "forward_policy": args.forward_policy,
        "backward_policy": args.backward_policy,
        "kind": args.kind,
        "rank": 32,
        "alpha": 32,
        "lora_dtype": "float32",
        "base_dtype": "bfloat16",
        "attention": "flash",
        "checkpoint": "full",
        "swap": args.swap,
        "transfer_dtype": "bf16",
        "restore_mode": "foreach",
        "compile": False,
        "batch": 1,
        "accumulation": 1,
        "optimizer": "AdamW",
        "lr": 2e-5,
        "weight_decay": 0.01,
        "warmup": args.warmup,
        "steps": args.steps,
        "seed": args.seed,
        "scope": "all",
        "group_size": 256,
        "rht": "dense",
        "hadamard": "sylvester",
        "free_base": args.mode != "bf16",
        "int8_backend": "int_mm",
        "ste_tf32": False,
        "matmul_precision": "highest",
        "numerical_samples_independent": args.kind == "numerical",
    }


def save_named(path, tensors):
    save_file({n: t.detach().cpu().contiguous() for n, t in tensors.items()}, str(path))


def step(model, network, optimizer, batch, swap, diagnostics=True):
    if swap:
        model.prepare_block_swap_before_forward()
    optimizer.zero_grad(set_to_none=True)
    output = model(
        batch["x"],
        batch["timesteps"],
        batch["context"],
        padding_mask=batch["padding_mask"],
    )
    loss = torch.nn.functional.mse_loss(output.float(), batch["target"].float())
    loss.backward()
    if diagnostics:
        grads = [
            p.grad
            for p in network.parameters()
            if p.requires_grad and p.grad is not None
        ]
        finite_grad = (
            torch.stack([torch.isfinite(g).all() for g in grads]).all()
            if grads
            else loss.new_tensor(False, dtype=torch.bool)
        )
    optimizer.step()
    if diagnostics:
        finite_param = torch.stack(
            [torch.isfinite(p).all() for p in network.parameters() if p.requires_grad]
        ).all()
        check = torch.stack([torch.isfinite(loss), finite_grad, finite_param])
    else:
        check = None
    return output, loss.detach(), check


def execute(args, config, batch, manifest):
    model, network, initial, metadata = build_model(
        args.model_path,
        args.mode,
        args.swap,
        args.seed,
        args.forward_policy,
        args.backward_policy,
    )
    manifest["model"] = metadata
    manifest["placement_after_apply"] = placement(model, network)
    manifest["loading_peak_bytes"] = torch.cuda.max_memory_allocated()
    write_json(args.out_dir / "manifest.json", manifest)
    save_named(args.out_dir / "initial.safetensors", initial)
    params = {n: p for n, p in network.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(list(params.values()), lr=2e-5, weight_decay=0.01)
    numerical = args.kind == "numerical"
    warmup = 0 if numerical else args.warmup
    history, checks, raw = [], [], []
    runtime_shapes, shape_hooks = {}, []

    def capture_shape(name):
        def capture(_module, inputs):
            if (
                name not in runtime_shapes
                and inputs
                and isinstance(inputs[0], torch.Tensor)
            ):
                tensor = inputs[0]
                runtime_shapes[name] = {
                    "input_shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "weight_shape": list(_module.weight.shape),
                }

        return capture

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            shape_hooks.append(module.register_forward_pre_hook(capture_shape(name)))
    for _ in range(warmup):
        _, loss, finite = step(model, network, optimizer, batch, args.swap)
        history.append(loss)
        checks.append(finite)
        for hook in shape_hooks:
            hook.remove()
        shape_hooks.clear()
    drain(model)
    torch.cuda.reset_peak_memory_stats()
    profiler = None
    if args.kind == "profile":
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        )
        profiler.__enter__()
    window_start = time.perf_counter()
    for index in range(args.steps):
        if numerical:
            torch.manual_seed(args.seed)
            with torch.no_grad():
                for name, param in params.items():
                    param.copy_(initial[name].to(param.device))
            # A new optimizer for every replay sample; no moment/state carry-over.
            optimizer = torch.optim.AdamW(
                list(params.values()), lr=2e-5, weight_decay=0.01
            )
        start = time.perf_counter()
        cuda_start, cuda_end = None, None
        if args.kind == "latency":
            cuda_start = torch.cuda.Event(enable_timing=True)
            cuda_end = torch.cuda.Event(enable_timing=True)
            cuda_start.record()
        output, loss, finite = step(
            model,
            network,
            optimizer,
            batch,
            args.swap,
            diagnostics=args.kind not in {"performance", "latency"},
        )
        if cuda_end is not None:
            cuda_end.record()
            drain(model)
        raw.append(
            {"step": index + 1, "host_submission_sec": time.perf_counter() - start}
        )
        if cuda_end is not None:
            raw[-1]["synchronized_wall_sec"] = raw[-1]["host_submission_sec"]
            raw[-1]["cuda_event_ms"] = cuda_start.elapsed_time(cuda_end)
        for hook in shape_hooks:
            hook.remove()
        shape_hooks.clear()
        history.append(loss)
        if finite is not None:
            checks.append(finite)
        if numerical:
            prefix = args.out_dir / f"sample-{index:03d}"
            gradients = {n: p.grad for n, p in params.items() if p.grad is not None}
            save_named(Path(str(prefix) + "-grads.safetensors"), gradients)
            save_named(Path(str(prefix) + "-params.safetensors"), params)
            save_named(
                Path(str(prefix) + "-updates.safetensors"),
                {n: p.detach().cpu() - initial[n] for n, p in params.items()},
            )
            save_named(Path(str(prefix) + "-output.safetensors"), {"output": output})
            write_json(
                Path(str(prefix) + "-missing-grads.json"),
                sorted(set(params) - set(gradients)),
            )
        if profiler:
            profiler.step()
    drain(model)
    elapsed = time.perf_counter() - window_start
    if profiler:
        profiler.__exit__(None, None, None)
        profiler.export_chrome_trace(str(args.out_dir / "profile.json"))
        (args.out_dir / "profile-table.txt").write_text(
            profiler.key_averages().table(sort_by="self_device_time_total", row_limit=50)
        )
    losses = torch.stack(history).float().cpu().tolist()
    finite_checks = torch.stack(checks).cpu().tolist() if checks else []
    for index, record in enumerate(raw):
        record.update(
            loss=losses[warmup + index],
            finite=None
            if args.kind in {"performance", "latency"}
            else finite_checks[warmup + index],
        )
    (args.out_dir / "steps.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in raw)
    )
    allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    summary = {
        "status": "PASS" if all(all(c) for c in finite_checks) else "FAIL",
        "finite_checks_including_warmup": finite_checks,
        "window_sec": elapsed,
        "steps_per_sec": args.steps / elapsed,
        "wall_sec_per_step": elapsed / args.steps,
        "timing_valid_for_ranking": args.kind == "performance",
        "warmup_steps": warmup,
        "step_latency_percentiles": None,
        "timing_note": "host_submission_sec is asynchronous submission, not full step latency; boundary drains CPU offload futures then all CUDA streams",
        "allocated_bytes": allocated,
        "reserved_bytes": reserved,
        "allocated_gib": allocated / 2**30,
        "reserved_gib": reserved / 2**30,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "placement_after_steps": placement(model, network),
    }
    summary["runtime_linear_shapes"] = runtime_shapes
    if args.kind == "latency":
        times = torch.tensor([r["synchronized_wall_sec"] for r in raw])
        summary["step_latency_percentiles"] = {
            "p50": float(times.quantile(0.5)),
            "p95": float(times.quantile(0.95)),
            "scope": "per-step synchronized diagnostic; not pipelined throughput",
        }
    summary["all_losses_finite"] = all(math.isfinite(x) for x in losses)
    summary["final_gradients_finite"] = all(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for p in params.values()
    )
    summary["final_updates_finite"] = all(
        bool(torch.isfinite(p.detach().cpu() - initial[n]).all())
        for n, p in params.items()
    )
    if not summary["all_losses_finite"] or not summary["final_gradients_finite"]:
        summary["status"] = "FAIL"
    if not summary["final_updates_finite"]:
        summary["status"] = "FAIL"
    save_named(args.out_dir / "adapter.safetensors", network.state_dict())
    write_json(args.out_dir / "result.json", summary)
    if summary["status"] != "PASS":
        raise RuntimeError("loss/gradient/parameter NaN or Inf")
    return summary


def main():
    args = parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    telemetry = None
    try:
        config = effective_config(args)
        tensors, input_manifest = load_input(
            args.cache_dir, args.image_dir, args.cache_index, args.seed, args.timestep
        )
        manifest = software_manifest()
        manifest.update(
            config=config,
            config_hash=object_hash(config),
            inputs=input_manifest,
            model_path=str(args.model_path.resolve()),
            model_sha256=file_hash(args.model_path),
            quality_gates="not applied; numerical budgets require separate frozen reference assessment",
        )
        save_named(args.out_dir / "input.safetensors", tensors)
        write_json(args.out_dir / "config.json", config)
        write_json(args.out_dir / "manifest.json", manifest)
        if args.command == "prepare":
            print(json.dumps({"status": "PREPARED", "out_dir": str(args.out_dir)}))
            return
        manifest["device"] = validate_device(args.expected_gpu_uuid)
        telemetry = Telemetry(args.out_dir / "telemetry.jsonl", args.expected_gpu_uuid)
        telemetry.start()
        torch.cuda.reset_peak_memory_stats()
        result = execute(
            args, config, {k: v.to("cuda:0") for k, v in tensors.items()}, manifest
        )
        print(
            json.dumps({"status": result["status"], "window_sec": result["window_sec"]})
        )
    except Exception as exc:
        write_json(
            args.out_dir / "failure.json",
            {"status": "FAIL", "error": str(exc), "traceback": traceback.format_exc()},
        )
        raise
    finally:
        if telemetry is not None:
            telemetry.stop()


if __name__ == "__main__":
    main()
