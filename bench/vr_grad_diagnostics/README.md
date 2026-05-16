# VR-Loss Gradient-Level Diagnostics

Answers `docs/experimental/vr_loss.md` Open Question #2:

> **Gradient-level diagnostics** — the headroom bench measures loss-level
> ρ² (0.9999), but the optimizer cares about `Var[g]` and
> `cos(g_vr, g_full-batch)`.

The fm-vr-headroom bench measured `ρ²` on the *loss* — how much per-sample
loss variance is recoverable. That's necessary but not sufficient: the
optimizer only cares about the variance and direction of `∂L/∂θ`. This
bench measures the latter, offline, at a fixed checkpoint θ.

## What it computes

Given a base DiT + a LoRA-family adapter, for K minibatches at fixed θ:

```
y_k = u_pred(x_t_k, t_k; θ) − target_k                 # standard FM residual
z_k = u_pred^L(x_t_k^L, t_k; θ_base) − target^L_k      # control variate residual
g_std_k = ∂‖y_k‖² / ∂θ_adapter                          # standard FM gradient
g_vr_k  = ∂‖y_k + λ·z_k‖² / ∂θ_adapter                  # VR-loss gradient
```

Then we aggregate over K samples:

| Metric | What it tells you |
|---|---|
| `gradient_magnitude_ratio_vr_over_std = ‖g_vr_ref‖ / ‖g_std_ref‖` | How much smaller (or larger) the VR gradient is in absolute scale. A converged adapter with `λ ≈ −1` collapses `L_vr ≈ ‖x_0^H‖²` (mostly θ-independent) and this ratio can be ≪ 1 — *that's not variance reduction, it's signal collapse*. Read every other variance number through this filter. |
| `cov_sq_ratio_vr_over_std = (Var[g_vr]/‖g_vr_ref‖²) / (Var[g_std]/‖g_std_ref‖²)` | **Load-bearing.** Squared coefficient of variation, units-free. `< 1` means VR is *relatively* less noisy per unit gradient signal; `> 1` means VR's gradient is noisier in proportion to what reaches the optimizer. This is what survives the magnitude collapse. |
| `var_ratio_vr_over_std_raw = Var[g_vr] / Var[g_std]` | The naïve loss-level analog. Reported for completeness but **misleading by itself** when gradient magnitude differs between the two streams — and it usually does. Always pair with `gradient_magnitude_ratio`. |
| `cos(g_std_ref, g_vr_ref)` | Do VR and standard FM agree on direction *in expectation*? Near 1.0 = VR is approximately a denoised standard gradient; near 0 = VR redirects the optimizer toward a different objective (the high-frequency residual `x_0^H`, per `docs/experimental/vr_loss.md:184-199`). |
| `mean_k cos(g_vr_k, g_vr_ref) − mean_k cos(g_std_k, g_std_ref)` | **The optimizer's question.** If positive, single-batch VR gradients point closer to the true VR direction than single-batch standard gradients point to the true standard direction — i.e. VR is genuinely useful per-step. If ≤ 0, the +40% step cost of VR buys nothing the optimizer can use. |
| `mean_k cos(g_vr_k, g_std_ref)` | Cross-cosine. Independent sanity check on the redirection story. |

`g_*_ref` = mean over K minibatches at fixed θ. As K → ∞ this is the true
(full-batch) gradient.

> **Pitfall — raw variance lies under magnitude collapse.** If VR ends up
> with gradients 16× smaller than standard FM (which is what happens at
> a converged adapter with `λ ≈ −1`), squared magnitudes differ by 256×.
> "VR has 50× less variance" then means "VR has 5× MORE relative noise"
> — opposite story. Always read `gradient_magnitude_ratio` and `cov_sq_*`
> together before claiming VR helped.

## Why this is the right shape

- **Offline, fixed θ.** `g_largebatch_reference` in the Open Question
  is the true full-batch gradient at a specific θ. Approximating it
  online (EMA across training) mixes gradients from moving θ and gives
  a biased estimate of nothing in particular.
- **Real adapter, real training loop.** The bench loads an actual LoRA
  checkpoint, constructs the LoRA network the same way `train.py` does,
  and differentiates against the adapter's parameters only. The gradient
  surface is what training actually sees.
- **Two backward passes per batch.** Naive: separately backward `L_std`
  and `L_vr`. Cheaper: backward `L_std`, then backward the cross-term
  `2λ·⟨z, y⟩` once with `retain_graph=True`. `g_vr = g_std + cross`. The
  script uses the cheaper version.

## Why not a synthetic noise model

You might ask: "isn't `Var[g_vr]/Var[g_std]` just `1 − ρ²` from the
headroom bench?" — no. `1 − ρ²` is the variance reduction on the **scalar
loss estimator**. The gradient is the loss differentiated through the
chain `(y, z) → u_pred → x_t → θ_adapter`. `z` is detached, but it shares
its input `x_t` with `y`, so `Cov(z, ∂y/∂θ) ≠ 0` and the loss-level ρ²
does not transfer directly to gradient space. That's exactly why this
bench exists.

## Usage

```bash
# Minimal — base DiT + a LoRA-family adapter
uv run python bench/vr_grad_diagnostics/run_bench.py \
    --dit models/diffusion_models/anima-base-v1.0.safetensors \
    --adapter output/ckpt/<your-vr-trained-lora>.safetensors \
    --data_dir post_image_dataset/lora \
    --num_batches 24 \
    --label vr-trained-mid

# Pair with the "fixed-λ=−1 vs learned-EMA" question
uv run python bench/vr_grad_diagnostics/run_bench.py ... --lambda_mode fixed --lambda_value -1.0   --label fixed-lam-neg1
uv run python bench/vr_grad_diagnostics/run_bench.py ... --lambda_mode fixed --lambda_value -0.72  --label fixed-lam-live
uv run python bench/vr_grad_diagnostics/run_bench.py ... --lambda_mode online                       --label online-ema
```

`--lambda_mode online` reproduces the live EMA — a single λ_ema is built
from `λ_batch = −Cov(y, z) / Var(z)` across the K batches and used uniformly
(matches `library/training/losses.py::_flow_matching_vr_loss` at steady
state). `--lambda_mode fixed` pins λ to `--lambda_value` and is the cheap
ablation knob the doc calls out (`vr_loss.md` Open Q #1).

Smoke run (~minutes):

```bash
uv run python bench/vr_grad_diagnostics/run_bench.py \
    --dit models/diffusion_models/anima-base-v1.0.safetensors \
    --adapter output/ckpt/<your-lora>.safetensors \
    --num_batches 4 --num_timesteps 2 --label smoke
```

## Output

Standard envelope (`result.json`) + `per_batch.csv` (one row per
minibatch with per-batch `‖g_std‖`, `‖g_vr‖`, `cos(g_*_k, g_*_ref)`) +
`summary.json` (aggregated read).

## Pre-conditions & gotchas

- **Adapter required.** The gradient surface needs trainable params.
  Pass a LoRA / OrthoLoRA / T-LoRA / HydraLoRA / FeRA / ReFT checkpoint.
  If the checkpoint was trained with VR loss, this bench measures how VR
  did during *its own* training-time regime; if it was trained with
  standard FM, it measures whether VR *would have* helped from that
  starting point.
- **Memory.** Each gradient is `num_adapter_params × 4 bytes`. At
  LoRA-r=32 over the full module list (~80M params) that's ~320 MB per
  gradient. We keep `g_std_k` and `g_vr_k` on CPU per batch, so peak
  CPU RAM ≈ `K × 2 × 320 MB`. At K=24, ~15 GB. Reduce K or pass a
  smaller adapter if you don't have the RAM.
- **Block swap.** Disable `blocks_to_swap` — the adapter forward must
  hold parameters on-device for autograd.

## Pass criteria (suggested)

This is a measurement, not a gate, but for decision-making:

- `cov_sq_ratio_vr_over_std < 0.7`: VR is substantially reducing
  *relative* gradient noise. Composes with the loss-level claim.
- `direction_lift_vr_minus_std > 0.05`: single-batch VR gradients are
  **directionally cleaner** than single-batch standard gradients. This
  is the result that justifies the +40% step cost.
- `gradient_magnitude_ratio` reasonably close to 1 (say `0.5–2`). Far
  below 0.5 = VR loss has collapsed; the gradient is mostly noise
  around a θ-independent target. Far above 2 = VR is amplifying
  gradient magnitude relative to standard, which is also suspicious.

Either of the cosine/CoV criteria failing while `ρ²_loss ≈ 0.999`
(from fm_vr_headroom) → **loss-level variance doesn't transfer to
gradient space**. The headroom bench was measuring a quantity the
optimizer cannot use. Reconsider v1's continued existence.

## References

- `docs/experimental/vr_loss.md` Open Q #2 — the question this bench
  answers.
- `bench/fm_vr_headroom/` — the loss-level companion; if you see
  contradictory readings, that's the interesting case.
- `[[project_vr_loss_status]]` — bench-setup-artifact erratum on the
  −0.996 asymptote; informs choice of `--lambda_value`.
