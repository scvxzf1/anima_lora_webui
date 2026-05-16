#!/usr/bin/env python
"""Probe FEI (Frequency-Energy Index) on Anima trajectories.

FeRA (paper 2511.17979v1, arXiv) proposes routing LoRA experts on the
DoG-banded energy of the current latent ``z_t``, normalized to a
3-simplex. The paper's "3 bands optimal" and "clean low→high migration"
claims are on DDPM UNets at fixed pixel resolutions with kernel scale
``κ = min(H,W)/128``. Anima is flow-matching with constant-token-bucket
DiT — both the schedule and the kernel scale differ. This script
verifies the FEI premise on Anima before designing a 3-band HydraLoRA
around it.

For each (bucket, prompt, seed):

  - sample ``z_T ~ N(0, I)`` at the bucket's latent shape
  - iterate denoising; at each step, BEFORE the model forward, compute::

        LP_low(z)  = GaussianBlur(z, σ_low)    # σ_low > σ_mid (bigger blur)
        LP_mid(z)  = GaussianBlur(z, σ_mid)
        band_low   = LP_low(z)
        band_mid   = LP_mid(z) − LP_low(z)
        band_high  = z − LP_mid(z)
        E_k        = ||band_k||²               # summed over C, H, W
        e_k        = E_k / Σ E                 # FEI[t]

    σ_low, σ_mid scale with ``min(H_lat, W_lat)`` so the same band
    semantics carry across buckets (FeRA's κ-rule, in latent space).

Outputs (``bench/fera/results/<ts>[-<label>]/``):

  - ``result.json``      standard envelope
  - ``fei_per_step.csv`` long: bucket, prompt_idx, seed, step, sigma,
                         e_low, e_mid, e_high
  - ``fei_traces.png``   one panel per bucket; stack-area of mean FEI
                         across (prompt, seed)

What to look for
----------------
  - Clean monotonic ``e_low ↓``, ``e_mid → peak``, ``e_high ↑`` over
    decreasing σ → 3-band split is sensible; HydraLoRA-FEI is viable.
  - ``e_mid`` collapsed (≈0 or fused with a neighbor) → 2 bands suffice.
  - Aspect-dependent FEI shapes → bucket-aware σ_k needed; can't ship
    one scale.
  - No migration / FEI roughly flat → flow-matching velocity target's
    spectral order differs from DDPM ``x_0``; FeRA's premise doesn't
    transfer without modification.

Usage
-----
    uv run python bench/fera/probe_fei.py \\
        --buckets 1024x1024,832x1248,1248x832 \\
        --n_prompts 4 --n_seeds 2 --infer_steps 28 \\
        --guidance_scale 4.0 --label fera-pilot
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from library.anima import weights as anima_utils  # noqa: E402
from library.inference import sampling as inference_utils  # noqa: E402
from library.inference.adapters import clear_hydra_sigma, set_hydra_sigma  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fera-probe")

DEFAULT_PROMPTS = [
    "1girl, solo, looking at viewer, blue hair, school uniform, classroom",
    "scenery, mountain, lake, reflection, sunset, no humans",
    "1boy, solo, armor, sword, dramatic lighting, fantasy",
    "cat, fluffy, sitting indoors, simple background",
]


# ---------------------------------------------------------------- DoG / FEI


def _gaussian_kernel_1d(
    sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    half = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian along (H, W) with reflect padding. ``x`` is ``[B, C, H, W]``."""
    if sigma <= 0:
        return x
    k1 = _gaussian_kernel_1d(sigma, x.device, x.dtype)
    K = k1.numel()
    pad = K // 2
    C = x.shape[1]
    kw = k1.view(1, 1, 1, K).expand(C, 1, 1, K).contiguous()
    kh = k1.view(1, 1, K, 1).expand(C, 1, K, 1).contiguous()
    x = F.pad(x, (pad, pad, 0, 0), mode="reflect")
    x = F.conv2d(x, kw, groups=C)
    x = F.pad(x, (0, 0, pad, pad), mode="reflect")
    x = F.conv2d(x, kh, groups=C)
    return x


def fei_3band(z: torch.Tensor, sigma_low: float, sigma_mid: float) -> torch.Tensor:
    """``z`` is ``[B, C, H, W]``. Returns ``[B, 3]`` on the simplex.

    Three bands by stacked Gaussian low-pass (Laplacian pyramid style):

        band_low  = LP(z, σ_low)               (σ_low > σ_mid)
        band_mid  = LP(z, σ_mid) − LP(z, σ_low)
        band_high = z − LP(z, σ_mid)

    Energy summed over (C, H, W), then normalized.
    """
    z = z.float()
    lp_low = gaussian_blur_2d(z, sigma_low)
    lp_mid = gaussian_blur_2d(z, sigma_mid)
    b_low = lp_low
    b_mid = lp_mid - lp_low
    b_high = z - lp_mid
    e_low = b_low.pow(2).flatten(1).sum(-1)
    e_mid = b_mid.pow(2).flatten(1).sum(-1)
    e_high = b_high.pow(2).flatten(1).sum(-1)
    energies = torch.stack([e_low, e_mid, e_high], dim=-1)
    return energies / energies.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def pick_dog_sigmas(
    h_lat: int, w_lat: int, low_div: float, mid_div: float
) -> tuple[float, float]:
    """``σ_low = D / low_div``, ``σ_mid = D / mid_div`` with ``D = min(H_lat, W_lat)``.

    Requires ``low_div < mid_div`` so ``σ_low > σ_mid`` (stronger blur for
    the low band — keeps only the lowest frequencies).
    """
    D = float(min(h_lat, w_lat))
    return D / float(low_div), D / float(mid_div)


# -------------------------------------------------------- Text encoding


def _setup_text_strategies(text_encoder_path: str) -> None:
    from library.anima import strategy as strategy_anima, text_strategies as ts

    ts.TokenizeStrategy.set_strategy(
        strategy_anima.AnimaTokenizeStrategy(
            qwen3_path=text_encoder_path,
            t5_tokenizer_path=None,
            qwen3_max_length=512,
            t5_max_length=512,
        )
    )
    ts.TextEncodingStrategy.set_strategy(strategy_anima.AnimaTextEncodingStrategy())


@torch.no_grad()
def _encode_prompt(anima, text_encoder, prompt: str, device: torch.device) -> torch.Tensor:
    """Tokenize → encode → ``anima._preprocess_text_embeds`` → pad to 512.

    Mirrors ``scripts/dcw/trajectory.py::encode_uncond_embed`` but kept
    local so the bench has no cross-bench dependency.
    """
    from library.anima import text_strategies as ts

    tok = ts.TokenizeStrategy.get_strategy()
    enc = ts.TextEncodingStrategy.get_strategy()
    tokens = tok.tokenize(prompt)
    embed = enc.encode_tokens(tok, [text_encoder], tokens)
    crossattn, _ = anima._preprocess_text_embeds(
        source_hidden_states=embed[0].to(anima.device),
        target_input_ids=embed[2].to(anima.device),
        target_attention_mask=embed[3].to(anima.device),
        source_attention_mask=embed[1].to(anima.device),
    )
    crossattn[~embed[3].bool()] = 0
    if crossattn.shape[1] < 512:
        crossattn = torch.nn.functional.pad(
            crossattn, (0, 0, 0, 512 - crossattn.shape[1])
        )
    return crossattn.to(device, dtype=torch.bfloat16)


# ------------------------------------------------------------ Trajectory


@torch.no_grad()
def run_trajectory(
    anima,
    embed: torch.Tensor,
    embed_uncond: torch.Tensor | None,
    sigmas: torch.Tensor,
    h_lat: int,
    w_lat: int,
    seed: int,
    device: torch.device,
    *,
    cfg_scale: float,
    sigma_low: float,
    sigma_mid: float,
) -> np.ndarray:
    """Run one denoising trajectory; return ``[n_steps, 3]`` FEI.

    FEI is captured on ``z_t`` BEFORE each model forward — i.e. the state
    the model is about to denoise from, the same signal a FeRA-style
    router would observe.
    """
    n_steps = sigmas.numel() - 1
    latent_c = 16  # Anima.LATENT_CHANNELS — pinned by load_anima_model's dit_config
    g = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(
        (1, latent_c, 1, h_lat, w_lat), generator=g, dtype=torch.bfloat16
    ).to(device)
    pad = torch.zeros(1, 1, h_lat, w_lat, dtype=torch.bfloat16, device=device)
    timesteps = sigmas[:-1].to(device, dtype=torch.bfloat16)
    fei_traj = np.zeros((n_steps, 3), dtype=np.float32)
    sigmas_d = sigmas.to(device)

    try:
        for i in range(n_steps):
            # FEI on z_t (pre-forward). Squeeze the singleton temporal dim.
            z2d = latents.squeeze(2)
            fei = fei_3band(z2d, sigma_low, sigma_mid)
            fei_traj[i] = fei[0].cpu().numpy()

            t = timesteps[i].expand(latents.shape[0])
            set_hydra_sigma(anima, t)

            if cfg_scale == 1.0 or embed_uncond is None:
                v = anima(latents, t, embed, padding_mask=pad)
            else:
                # Batched CFG: [uncond, cond] in one forward, then combine.
                B = latents.shape[0]
                e_u = (
                    embed_uncond
                    if embed_uncond.shape[0] == B
                    else embed_uncond.expand(B, -1, -1).contiguous()
                )
                x2 = torch.cat([latents, latents], dim=0)
                t2 = torch.cat([t, t], dim=0)
                e2 = torch.cat([e_u, embed], dim=0)
                p2 = torch.cat([pad, pad], dim=0)
                v_pair = anima(x2, t2, e2, padding_mask=p2)
                v_u, v_c = v_pair[:B], v_pair[B:]
                v = v_u + cfg_scale * (v_c - v_u)

            latents = inference_utils.step(latents, v, sigmas_d, i).to(latents.dtype)
    finally:
        clear_hydra_sigma(anima)

    return fei_traj


# ------------------------------------------------------------------ CLI


def _parse_buckets(s: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        w, h = tok.lower().split("x")
        out.append((int(w), int(h)))
    return out


class _TextEncoderArgs:
    """Minimal arg shim so ``load_text_encoder`` can be called without the full inference parser."""

    def __init__(self, text_encoder: str) -> None:
        self.text_encoder = text_encoder
        self.lora_weight = None
        self.lora_multiplier = None


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dit", default="models/diffusion_models/anima-base-v1.0.safetensors"
    )
    p.add_argument(
        "--text_encoder",
        default="models/text_encoders/qwen_3_06b_base.safetensors",
    )
    p.add_argument(
        "--buckets",
        default="1024x1024,832x1248,1248x832",
        help="Comma-sep WxH list (pixel). Default: square + 1.5-ratio mirror pair.",
    )
    p.add_argument(
        "--prompts",
        nargs="*",
        default=None,
        help="Override prompt list. Defaults to a 4-prompt generic set.",
    )
    p.add_argument("--n_prompts", type=int, default=4)
    p.add_argument("--n_seeds", type=int, default=2)
    p.add_argument("--seed_base", type=int, default=1234)
    p.add_argument("--infer_steps", type=int, default=28)
    p.add_argument("--flow_shift", type=float, default=1.0)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument(
        "--negative_prompt",
        default="worst quality, low quality, score_1, score_2, score_3",
    )
    p.add_argument("--attn_mode", default="flash")
    p.add_argument(
        "--compile",
        action="store_true",
        help="Wrap the DiT with torch.compile after load. One compile pass per "
        "bucket shape (3 shapes for the default bucket list); amortizes over "
        "n_prompts × n_seeds × infer_steps forwards per shape.",
    )
    p.add_argument(
        "--dog_low_div",
        type=float,
        default=8.0,
        help="σ_low = min(H_lat, W_lat) / dog_low_div (larger blur → low band)",
    )
    p.add_argument(
        "--dog_mid_div",
        type=float,
        default=32.0,
        help="σ_mid = min(H_lat, W_lat) / dog_mid_div",
    )
    p.add_argument("--label", default=None)
    args = p.parse_args()

    if args.dog_low_div >= args.dog_mid_div:
        raise SystemExit(
            f"--dog_low_div ({args.dog_low_div}) must be < --dog_mid_div "
            f"({args.dog_mid_div}) so σ_low > σ_mid (stronger blur for low band)."
        )

    buckets = _parse_buckets(args.buckets)
    prompts = (args.prompts if args.prompts else DEFAULT_PROMPTS)[: args.n_prompts]
    if not prompts:
        raise SystemExit("no prompts after --n_prompts truncation")

    out_dir = make_run_dir("fera", label=args.label)
    log.info(f"output → {out_dir}")
    log.info(
        f"buckets={buckets}, prompts={len(prompts)}, seeds={args.n_seeds}, "
        f"steps={args.infer_steps}, CFG={args.guidance_scale}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    log.info("loading DiT…")
    anima = anima_utils.load_anima_model(
        device=device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        split_attn=False,
        loading_device=device,
        dit_weight_dtype=dtype,
    )
    anima.to(device, dtype=dtype).eval().requires_grad_(False)
    anima.reset_mod_guidance()

    if args.compile:
        log.info("torch.compile(anima) — first forward per bucket will pay compile cost")
        anima = torch.compile(anima)

    # Transient TE block — encode all prompts (+ optional uncond), then free.
    _setup_text_strategies(args.text_encoder)
    from library.inference.models import load_text_encoder

    log.info("loading text encoder (transient)…")
    te = load_text_encoder(_TextEncoderArgs(args.text_encoder), dtype=dtype, device=device)
    te.eval()

    embeds_cond: list[torch.Tensor] = []
    for prompt in prompts:
        embeds_cond.append(_encode_prompt(anima, te, prompt, device))
    embed_uncond: torch.Tensor | None = None
    if args.guidance_scale != 1.0:
        embed_uncond = _encode_prompt(anima, te, args.negative_prompt, device)

    del te
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log.info("text encoder released")

    _, sigmas = inference_utils.get_timesteps_sigmas(
        args.infer_steps, args.flow_shift, device
    )
    sigmas = sigmas.cpu()
    n_steps = int(sigmas.numel() - 1)

    rows: list[dict] = []
    per_bucket_traces: dict[tuple[int, int], list[np.ndarray]] = {b: [] for b in buckets}
    per_bucket_sigmas_dog: dict[tuple[int, int], tuple[float, float]] = {}

    for (w_pix, h_pix) in buckets:
        h_lat, w_lat = h_pix // 8, w_pix // 8
        sigma_low, sigma_mid = pick_dog_sigmas(
            h_lat, w_lat, args.dog_low_div, args.dog_mid_div
        )
        per_bucket_sigmas_dog[(w_pix, h_pix)] = (sigma_low, sigma_mid)
        log.info(
            f"bucket {w_pix}x{h_pix} (latent {w_lat}x{h_lat}): "
            f"σ_low={sigma_low:.2f}, σ_mid={sigma_mid:.2f}"
        )

        for pi, embed in enumerate(embeds_cond):
            for si in range(args.n_seeds):
                seed = args.seed_base + 1000 * pi + si
                fei_traj = run_trajectory(
                    anima,
                    embed,
                    embed_uncond,
                    sigmas,
                    h_lat,
                    w_lat,
                    seed,
                    device,
                    cfg_scale=args.guidance_scale,
                    sigma_low=sigma_low,
                    sigma_mid=sigma_mid,
                )
                per_bucket_traces[(w_pix, h_pix)].append(fei_traj)
                for i in range(n_steps):
                    rows.append(
                        {
                            "bucket_w": w_pix,
                            "bucket_h": h_pix,
                            "prompt_idx": pi,
                            "seed": seed,
                            "step": i,
                            "sigma": float(sigmas[i]),
                            "e_low": float(fei_traj[i, 0]),
                            "e_mid": float(fei_traj[i, 1]),
                            "e_high": float(fei_traj[i, 2]),
                        }
                    )

    csv_path = out_dir / "fei_per_step.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"wrote {csv_path} ({len(rows)} rows)")

    artifacts: list[str] = [csv_path.name]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(buckets)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True, squeeze=False)
        sigma_x = sigmas[:-1].numpy()
        for ax, (w_pix, h_pix) in zip(axes[0], buckets):
            traces = np.stack(per_bucket_traces[(w_pix, h_pix)], axis=0)  # [N, T, 3]
            mean_fei = traces.mean(axis=0)  # [T, 3]
            ax.stackplot(
                sigma_x,
                mean_fei.T,
                labels=["low", "mid", "high"],
                colors=["#3b82f6", "#10b981", "#ef4444"],
                alpha=0.85,
            )
            sl, sm = per_bucket_sigmas_dog[(w_pix, h_pix)]
            ax.set_title(
                f"{w_pix}×{h_pix}  (latent {w_pix // 8}×{h_pix // 8})\n"
                f"σ_low={sl:.1f}, σ_mid={sm:.1f}"
            )
            ax.set_xlabel("σ  (denoising →)")
            ax.set_ylim(0, 1)
            ax.set_xlim(float(sigma_x.max()), 0.0)
        axes[0, 0].set_ylabel("FEI (simplex)")
        axes[0, -1].legend(loc="upper right", fontsize=8)
        fig.suptitle(
            f"FEI(z_t) — Anima  (CFG={args.guidance_scale}, "
            f"steps={args.infer_steps}, prompts={len(prompts)}×seeds={args.n_seeds})"
        )
        fig.tight_layout()
        png = out_dir / "fei_traces.png"
        fig.savefig(png, dpi=120)
        plt.close(fig)
        artifacts.append(png.name)
        log.info(f"wrote {png}")
    except Exception as e:
        log.warning(f"plot failed (continuing): {e}")

    metrics = {
        "n_buckets": len(buckets),
        "n_prompts": len(prompts),
        "n_seeds": args.n_seeds,
        "n_steps": n_steps,
        "dog_sigma_per_bucket": {
            f"{wp}x{hp}": list(per_bucket_sigmas_dog[(wp, hp)]) for (wp, hp) in buckets
        },
        "fei_at_first_step": {
            f"{wp}x{hp}": np.stack(per_bucket_traces[(wp, hp)], 0)
            .mean(0)[0]
            .tolist()
            for (wp, hp) in buckets
        },
        "fei_at_last_step": {
            f"{wp}x{hp}": np.stack(per_bucket_traces[(wp, hp)], 0)
            .mean(0)[-1]
            .tolist()
            for (wp, hp) in buckets
        },
    }
    write_result(
        out_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        artifacts=artifacts,
        label=args.label,
        device=device,
    )
    log.info("done")


if __name__ == "__main__":
    main()
