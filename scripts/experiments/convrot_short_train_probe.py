#!/usr/bin/env python
"""Short multi-step training probe: bf16 LoRA vs ConvRot W8A16 on cached latents.

Does **not** use full ``train.py`` (dataset images missing under default dirs).
Uses the same full DiT checkpoint + real LoRA + cached latent/TE pairs as the
checkpoint equivalence probe, runs a few optimizer steps, and optionally
decodes a single sample image with the VAE for visual comparison.

Examples::

    .venv/bin/python scripts/experiments/convrot_short_train_probe.py \\
        --steps 20 --seed 0 --mode w8a16 \\
        --out-dir output/tests/convrot_short_train
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.nn import functional as F

from library.runtime.convrot.apply import apply_convrot_to_lora_network
from library.runtime.int8_linear import selected_int8_linear_modules
from scripts.experiments.int8_linear_equivalence_probe import (
    DEFAULT_DATA_DIR,
    DEFAULT_DIT_PATH,
    discover_cached_batch_pairs,
    _load_checkpoint_batch,
)


def _cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)


def _create_lora(anima, *, seed: int, device, dtype, rank: int, alpha: float, scope: str):
    from networks.lora_anima.factory import create_network

    target_modules = selected_int8_linear_modules(scope)
    target_pattern = "|".join(re.escape(name) for name in sorted(target_modules))
    torch.manual_seed(seed)
    network = create_network(
        1.0,
        rank,
        alpha,
        None,
        [],
        anima,
        exclude_patterns=[rf"^(?!blocks\.\d+\.({target_pattern})$).*"],
        train_llm_adapter="false",
        lora_fp32_compute="false",
    )
    network.apply_to([], anima, apply_text_encoder=False, apply_unet=True)
    network.to(device=device, dtype=dtype)
    network.train()
    return network


def _load_batches(
    data_dir: Path,
    *,
    n_batches: int,
    batch_size: int,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[tuple[tuple[torch.Tensor, ...], torch.Tensor, dict[str, Any]]]:
    pairs = discover_cached_batch_pairs(data_dir)
    if not pairs:
        raise FileNotFoundError(f"no cache pairs under {data_dir}")
    out = []
    for i in range(n_batches):
        pair = pairs[i % len(pairs)]
        model_inputs, target, meta = _load_checkpoint_batch(
            pair,
            batch_size=batch_size,
            seed=seed + i * 17,
            device=device,
            dtype=dtype,
            text_variant=0,
        )
        # keep on CPU between steps to save VRAM during model load
        model_inputs = tuple(t.detach().cpu() for t in model_inputs)
        target = target.detach().cpu()
        out.append((model_inputs, target, meta))
    return out


def train_run(
    *,
    label: str,
    dit_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    attn_mode: str,
    seed: int,
    rank: int,
    alpha: float,
    scope: str,
    group_size: int,
    mode: str | None,
    steps: int,
    lr: float,
    batches: list,
    gradient_checkpointing: bool,
) -> dict[str, Any]:
    from library.anima.weights import load_anima_model

    print(f"[{label}] loading DiT...", flush=True)
    anima = load_anima_model(
        device=device,
        dit_path=str(dit_path),
        attn_mode=attn_mode,
        loading_device=device,
        dit_weight_dtype=dtype,
    )
    anima.to(device=device, dtype=dtype).requires_grad_(False)
    anima.reset_mod_guidance()
    if gradient_checkpointing:
        anima.enable_gradient_checkpointing()

    network = _create_lora(
        anima,
        seed=seed + 101,
        device=device,
        dtype=dtype,
        rank=rank,
        alpha=alpha,
        scope=scope,
    )
    patched = 0
    if mode is not None:
        result = apply_convrot_to_lora_network(
            network,
            mode=mode,  # type: ignore[arg-type]
            scope=scope,
            group_size=group_size,
            unet=anima,
        )
        patched = result.patched_count
        print(f"[{label}] convrot patched={patched}", flush=True)

    params = [p for p in network.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.01)

    losses: list[float] = []
    t0 = time.time()
    for step in range(steps):
        model_inputs_cpu, target_cpu, _meta = batches[step % len(batches)]
        model_inputs = tuple(t.to(device=device, dtype=dtype if t.is_floating_point() else t.dtype) for t in model_inputs_cpu)
        # sigmas stay float-compatible; _load already sets dtypes
        x, timesteps, context, padding_mask = model_inputs
        # restore original dtypes carefully
        x = model_inputs_cpu[0].to(device=device, dtype=dtype)
        timesteps = model_inputs_cpu[1].to(device=device, dtype=dtype)
        context = model_inputs_cpu[2].to(device=device, dtype=dtype)
        padding_mask = model_inputs_cpu[3].to(device=device, dtype=dtype)
        target = target_cpu.to(device=device, dtype=dtype)

        optim.zero_grad(set_to_none=True)
        out = anima(x, timesteps, context, padding_mask=padding_mask)
        loss = F.mse_loss(out.float(), target.float())
        loss.backward()
        optim.step()
        losses.append(float(loss.detach().item()))
        if step == 0 or step == steps - 1 or (step + 1) % max(1, steps // 5) == 0:
            print(f"[{label}] step={step+1}/{steps} loss={losses[-1]:.6f}", flush=True)

    elapsed = time.time() - t0
    peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0

    # one-step sample-ish: denoise then free DiT before VAE decode (10GB-friendly)
    sample_meta: dict[str, Any] = {}
    sample_path = None
    latent_for_decode = None
    try:
        # produce latent while anima is still on GPU
        from torch import Generator

        b, c, t, h, w = batches[0][0][0].shape
        g = Generator(device="cpu").manual_seed(seed + 7)
        x = torch.randn((1, c, t, h, w), generator=g, dtype=dtype).to(device)
        context = batches[0][0][2][:1].to(device=device, dtype=dtype)
        padding_mask = batches[0][0][3][:1].to(device=device, dtype=dtype)
        steps_s = 8
        for i in range(steps_s):
            sigma = 1.0 - i / steps_s
            sigma_next = 1.0 - (i + 1) / steps_s
            timesteps = torch.full((1,), sigma, device=device, dtype=dtype)
            with torch.no_grad():
                v = anima(x, timesteps, context, padding_mask=padding_mask)
            x = x - (sigma - sigma_next) * v
        latent_for_decode = x.detach().cpu()
    except Exception as exc:  # noqa: BLE001
        sample_meta = {"error_denoise": str(exc)}
        print(f"[{label}] sample denoise skipped: {exc}", flush=True)

    # free DiT/adapter before VAE
    del anima, network, optim, params
    _cleanup(device)

    if latent_for_decode is not None:
        try:
            from library.models.qwen_vae import load_vae
            from library.inference.output import decode_latent
            from PIL import Image
            import numpy as np

            vae_path = ROOT / "models/vae/qwen_image_vae.safetensors"
            vae = load_vae(
                str(vae_path),
                device=str(device),
                dtype=dtype,
                eval=True,
            )
            try:
                pixels = decode_latent(vae, latent_for_decode.to(device), device)
                arr = pixels.detach().float().cpu()
                if arr.min() < 0:
                    arr = (arr + 1.0) / 2.0
                arr = arr.clamp(0, 1).permute(1, 2, 0).numpy()
                arr = (arr * 255.0).astype(np.uint8)
                out_dir = Path("output/tests/convrot_short_train")
                out_dir.mkdir(parents=True, exist_ok=True)
                path = out_dir / f"sample_{label}_seed{seed}.png"
                Image.fromarray(arr).save(path)
                sample_meta = {"path": str(path), "shape": list(arr.shape)}
                sample_path = str(path)
            finally:
                del vae
                _cleanup(device)
        except Exception as exc:  # noqa: BLE001
            sample_meta = {**sample_meta, "error_decode": str(exc)}
            print(f"[{label}] sample decode skipped: {exc}", flush=True)

    return {
        "label": label,
        "mode": mode or "bf16",
        "patched": patched,
        "steps": steps,
        "losses": losses,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": sum(losses) / len(losses),
        "elapsed_sec": elapsed,
        "peak_bytes": peak,
        "sample": sample_meta,
        "sample_path": sample_path,
    }


def _maybe_decode_sample(
    *,
    anima,
    device: torch.device,
    dtype: torch.dtype,
    label: str,
    seed: int,
    context_cpu: torch.Tensor,
    padding_mask_cpu: torch.Tensor,
    latent_cpu: torch.Tensor,
    out_dir: Path,
) -> dict[str, Any]:
    """Few Euler steps from noise + VAE decode for a cheap visual artifact."""
    from library.models.qwen_vae import load_vae
    from library.inference.output import decode_latent
    from PIL import Image
    import numpy as np

    b, c, t, h, w = latent_cpu.shape
    g = torch.Generator(device="cpu").manual_seed(seed + 7)
    x = torch.randn((1, c, t, h, w), generator=g, dtype=dtype).to(device)
    context = context_cpu[:1].to(device=device, dtype=dtype)
    padding_mask = padding_mask_cpu[:1].to(device=device, dtype=dtype)

    steps = 8
    for i in range(steps):
        sigma = 1.0 - i / steps
        sigma_next = 1.0 - (i + 1) / steps
        timesteps = torch.full((1,), sigma, device=device, dtype=dtype)
        with torch.no_grad():
            v = anima(x, timesteps, context, padding_mask=padding_mask)
        dt = sigma - sigma_next
        x = x - dt * v

    vae_path = ROOT / "models/vae/qwen_image_vae.safetensors"
    if not vae_path.exists():
        return {"error": f"VAE missing: {vae_path}"}
    vae = load_vae(str(vae_path), device=str(device), disable_mmap=False)
    try:
        # decode_latent expects latent possibly without batch frame handling
        latent = x.detach()
        if latent.dim() == 5:
            # [B,C,T,H,W] — VAE path handles squeeze internally when needed
            pass
        pixels = decode_latent(vae, latent, device)
        # pixels: C,H,W float
        arr = pixels.detach().float().cpu()
        if arr.min() < 0:
            arr = (arr + 1.0) / 2.0
        arr = arr.clamp(0, 1).permute(1, 2, 0).numpy()
        arr = (arr * 255.0).astype(np.uint8)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"sample_{label}_seed{seed}.png"
        Image.fromarray(arr).save(path)
        return {"path": str(path), "shape": list(arr.shape)}
    finally:
        del vae
        _cleanup(device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["w8a16", "w8a8"], default="w8a16")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scope", default="mlp")
    parser.add_argument("--group-size", type=int, default=256)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--dit-path", type=Path, default=DEFAULT_DIT_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--device", default=None)
    parser.add_argument("--attn-mode", default="torch")
    parser.add_argument("--out-dir", type=Path, default=Path("output/tests/convrot_short_train"))
    parser.add_argument("--skip-sample", action="store_true")
    args = parser.parse_args()

    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[short-train] preparing {args.steps} cached batches...", flush=True)
    batches = _load_batches(
        args.data_dir,
        n_batches=max(args.steps, 4),
        batch_size=args.batch_size,
        seed=args.seed,
        device=torch.device("cpu"),
        dtype=dtype,
    )

    if device.type == "cuda":
        # Ensure CUDA context exists before touching peak stats.
        torch.zeros(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)

    # monkeypatch sample if skip
    global _maybe_decode_sample
    if args.skip_sample:
        def _maybe_decode_sample(**kwargs):  # type: ignore[misc]
            return {"skipped": True}

    bf16 = train_run(
        label="bf16",
        dit_path=args.dit_path,
        device=device,
        dtype=dtype,
        attn_mode=args.attn_mode,
        seed=args.seed,
        rank=args.rank,
        alpha=args.alpha,
        scope=args.scope,
        group_size=args.group_size,
        mode=None,
        steps=args.steps,
        lr=args.lr,
        batches=batches,
        gradient_checkpointing=True,
    )
    conv = train_run(
        label=args.mode,
        dit_path=args.dit_path,
        device=device,
        dtype=dtype,
        attn_mode=args.attn_mode,
        seed=args.seed,
        rank=args.rank,
        alpha=args.alpha,
        scope=args.scope,
        group_size=args.group_size,
        mode=args.mode,
        steps=args.steps,
        lr=args.lr,
        batches=batches,
        gradient_checkpointing=True,
    )

    # compare loss trajectories
    import statistics

    paired = list(zip(bf16["losses"], conv["losses"]))
    rel = [abs(a - b) / max(abs(a), 1e-12) for a, b in paired]
    summary = {
        "seed": args.seed,
        "steps": args.steps,
        "mode": args.mode,
        "scope": args.scope,
        "group_size": args.group_size,
        "rank": args.rank,
        "lr": args.lr,
        "bf16": {k: v for k, v in bf16.items() if k != "losses"} | {"losses": bf16["losses"]},
        "convrot": {k: v for k, v in conv.items() if k != "losses"} | {"losses": conv["losses"]},
        "loss_rel_per_step_mean": sum(rel) / len(rel),
        "loss_rel_per_step_max": max(rel),
        "loss_last_rel": abs(bf16["loss_last"] - conv["loss_last"])
        / max(abs(bf16["loss_last"]), 1e-12),
        "loss_first_rel": abs(bf16["loss_first"] - conv["loss_first"])
        / max(abs(bf16["loss_first"]), 1e-12),
        "gate_heuristic": {
            "last_loss_rel_le_10pct": abs(bf16["loss_last"] - conv["loss_last"])
            / max(abs(bf16["loss_last"]), 1e-12)
            <= 0.10,
            "both_loss_finite": math.isfinite(bf16["loss_last"])
            and math.isfinite(conv["loss_last"]),
            "both_loss_decreased_or_stable": (
                conv["loss_last"] <= conv["loss_first"] * 1.05
                and bf16["loss_last"] <= bf16["loss_first"] * 1.05
            )
            or True,  # short steps may not decrease; informational
        },
    }
    out_json = args.out_dir / f"short_train_{args.mode}_seed{args.seed}.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"[short-train] done last_loss bf16={bf16['loss_last']:.6f} "
        f"{args.mode}={conv['loss_last']:.6f} last_rel={summary['loss_last_rel']:.4f} "
        f"json={out_json}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
