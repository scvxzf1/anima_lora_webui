# FM Variance-Reduction Headroom

> **2026-05-14 errata.** The headline `λ_global = −0.996 ± 0.002` and
> `ρ² ≈ 0.999` from `results/20260514-1300-tlora-vs-base/` were measured
> with `trainable_dit = anima-tlora-0509-12_merged.safetensors` (a
> near-converged adapter) vs `frozen_dit = anima-preview3-base.safetensors`.
> In that regime `u_pred ≈ u_pred^L` is true by construction, so the
> readings are an *upper bound for the near-converged regime*, not a
> training-time prediction. **Live training disagrees**: the first run
> with `vr/lambda_ema` logging (`output/logs/lora_default_20260514-1921/`,
> step ~586) shows `vr/lambda_ema` walks from −0.89 → −0.72, mean −0.75 —
> ~30% off the bench asymptote. Two consequences:
>
> 1. **Don't cite `λ ≈ −1` as a training-time fact.** `docs/experimental/
>    vr_loss.md`'s "λ_ema settles at −0.996 → fixed-λ=−1 should match"
>    and the (now archived) `archive/proposals/hf_residual_adapter.md`'s
>    "`λ → −1` collapse" both extrapolated from this bench past its scope.
> 2. **ρ² is loss-level; the optimizer cares about gradient variance.**
>    With Var(y) small in the near-converged regime, "99.99% of that
>    variance" is a tiny absolute number against actual SGD noise. The
>    gradient-variance probe in vr_loss.md Open Q #2 remains unaddressed.
>
> What the bench *did* validate: ε-pairing produces real structural
> correlation (null ρ² ≈ 0.024 → 0.99 is not an artifact of "both
> forwards see similar inputs"). VR-loss v1.5 still trains and the
> eyeball A/B held; the bench just doesn't justify the strength of the
> conclusions that were drawn from it.

Diagnostic for whether the AsymFlow §5.2 control-variate trick (Chen et al.,
arXiv:2605.12964) has measurable headroom on Anima's *standard latent*
flow-matching loss. The trick replaces the FM target

    Y = (x_0 − x̂_0) / σ_t

with the variance-reduced

    Y_VR = Y + λ·Z,   Z = (x_0^L − x̂_0^L) / σ_t

where `x_0^L` is a "simple" auxiliary target paired with `x_0`, `x̂_0^L` is
the same (frozen) model's prediction on `x_t^L := α_t·x_0^L + σ_t·ε` using
**the same `ε`** as `x_t`. With optimal `λ* = −Cov(Y, Z)/Var(Z)`, the
variance reduces by a factor of `(1 − ρ²)` where ρ = Corr(Y, Z).

If `ρ² ≈ 0`, the control variate is dead — Anima's FM loss can't be
variance-reduced this way and Stage 1 (full training A/B) is a waste. If
`ρ² ≳ 0.3`, the trick has real headroom and is worth wiring into the
training loop.

## Choice of `x_0^L`

This bench uses **FEI-aligned latent low-pass**:

    x_0^L = gaussian_blur_2d(x_0, σ_low),   σ_low = min(H_lat, W_lat) / fei_sigma_low_div

i.e. the same low-pass kernel that defines the model's existing FEI router
(`library/runtime/fei.py::compute_fei_2band`). Two reasons:

1. The model is *already* internally aware of this band split via the FEI
   router (HydraLoRA / FeRA). The variance-reduction control variate inherits
   the same inductive bias instead of inventing a new one.
2. The 2-band FEI gives a free per-sample diagnostic axis: does VR work
   better on low-frequency-dominated samples (e_low ≫ e_high) than on
   high-frequency ones? That tells us whether the trick is content-dependent.

Default `fei_sigma_low_div = 4.0` matches the live training default
(`configs/gui-methods/fera.toml`).

## What it measures

For each `(x_0, t)` pair we fix the clean latent and timestep, then sample
`N` noise vectors `ε_i`. The model runs `2N` forward passes (`x̂_0` and
`x̂_0^L`). From the `N` residuals we compute:

| metric | meaning |
|--------|---------|
| `rho_sq_global` | Corr(Y, Z)² aggregated over all elements at fixed (x_0,t). Variance reduction with a single global λ. |
| `rho_sq_per_elem` | Corr(Y, Z)² averaged element-wise. Upper bound on variance reduction if λ is allowed to vary per latent position. |
| `lambda_global` | optimal single λ. Should be close to 1 if (x_0 − x̂_0) ≈ (x_0^L − x̂_0^L) on the high-freq orthogonal complement. |
| `var_Y / var_Y_per_pixel` | irreducible noise floor of the *standard* FM target — for context. |

Aggregations: mean / median across `(x_0, t)` pairs, plus per-`t`-bucket
and per-FEI-band breakdown.

**Pass criterion**: if `rho_sq_global` is consistently ≳ 0.3 across t-bands
and FEI bands, Stage 1 is worth doing. If it's <0.1, the trick is dead on
Anima's latent space and you should not pursue the training A/B.

## Usage

```bash
uv run python bench/fm_vr_headroom/run_bench.py \
    --dit models/diffusion_models/anima-base-v1.0.safetensors \
    --data_dir post_image_dataset/lora \
    --num_samples 16 --num_timesteps 8 --num_noise 64 \
    --bucket 1024x1024 \
    --label first-probe
```

`--bucket WxH` filters cached latents to a single bucket so all forwards
have the same shape. Pick whatever bucket has enough cached samples in your
`post_image_dataset/lora`.

For a smoke run that finishes in seconds:

```bash
uv run python bench/fm_vr_headroom/run_bench.py \
    --dit models/diffusion_models/anima-base-v1.0.safetensors \
    --data_dir post_image_dataset/lora \
    --num_samples 2 --num_timesteps 3 --num_noise 8 \
    --label smoke
```

## Output

Standard bench envelope (`result.json`) + `per_sample_t.csv` (one row per
`(sample, t)` measurement) + `summary.json` (aggregated read).

## Followup

If headroom is present: Stage 1 = A/B LoRA training run with the VR loss
plumbed into `train.py`. Note `[[project_fm_val_loss_uninformative]]` —
quality verdict needs HPSv3/VQA scoring, not val FM curve alone.
