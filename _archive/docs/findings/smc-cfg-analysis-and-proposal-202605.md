# SMC-CFG on Anima: |e| analysis & adaptive-k proposal

状态：历史分析与提案
适用版本：2026-05，当前使用说明见 `docs/methods/smc_cfg.md`
原始路径：`bench/smc_cfg/analysis_and_proposal.md`（删除前提交 `f2c5f72f^`）

Status: live. Findings below are from the 4-prompt sweep
(`results/20260518-1014-cfg-sweep-4p/`, n=4 prompts × 4 CFGs × 28 steps =
448 measured `|e|` rows). The earlier n=2 quick run
(`results/20260518-1007-quick/`) agreed qualitatively; numbers came in a touch
milder at n=4 (mean k/|e| at CFG=4 with k=0.02: 0.86 → 0.80, fraction
saturated: 32 % → 30 %).

## TL;DR

- The CFG-Ctrl paper's SMC-CFG (Wang et al., arXiv:2603.03281) injects a
  per-element switching term `Δe = −k · sign(s_t)` (we use `tanh`) into the
  cond/uncond combine. The intent is a **small refinement** of CFG — `|Δe| ≪ |e|`.
- On Anima at production CFG=4 with our default `k=0.02`, the mean `k/|e|`
  ratio is **0.80** and `k ≥ |e|` at **30 % of denoising steps**. The
  controller's energy budget is roughly the same as the cond/uncond signal it
  is supposed to refine.
- At paper-best `k=0.1`, mean `k/|e| = 4.0` and `k ≥ |e|` at **93 % of steps**.
  Paper's `k` is ~14× too large for Anima's velocity field.
- Increasing CFG *worsens* the regime: `|e|` actually **shrinks monotonically**
  with CFG on Anima (0.0385 → 0.0377 → 0.0354 → 0.0333 at CFG ∈ {2, 4, 6, 8}).
  Mean `k/|e|` at `k=0.02` climbs accordingly (0.73 → 0.80 → 0.95 → 1.00).
  At CFG=8 we're literally at break-even on average.
- **The per-step profile localizes the problem**: `|e|` collapses from ≈0.16
  at σ=1.0 to a plateau of 0.018–0.020 over steps 16–22 (σ ≈ 0.18–0.43). It's
  this mid-late plateau where `k=0.02` is in the noise-injection regime.
  Early steps (where `|e|` is large) are the *only* place a constant `k=0.02`
  acts as a small refinement.
- **Proposal**: replace dimensional scalar `k` with dimensionless
  `α ∈ (0, 1]`, computing `k_t = α · |e_t|.mean()` per step. Self-scaling
  across model / CFG / σ / sample; single hyperparameter; trivially fits the
  paper's stability framing as a gain-scheduled extension.
- σ-scheduled and FEI-scheduled variants are alternatives but mostly subsumed
  by the adaptive form (which auto-tracks `|e_t|` in real time).

## What we measured

`bench/smc_cfg/measure_error_magnitude.py` — in-process reverse-denoise that
captures `|e| = |v_cond − v_uncond|` per step and derives the SMC sliding
surface `|s_t|` (element-wise, `s = (e − e_prev) + λ · e_prev`, λ=5.0) offline.
Does not run SMC — `k/|e|` is a strict upper bound on the controller's
per-element influence, so we can characterize whether SMC is operating as a
refinement or as a noise source without ever applying it.

- Prompts: 4 from `post_image_dataset/lora/` at 1024² (shuffle_seed=0):
    - `12941930` — booru-style explicit 1boy1girl original (heavy tag list)
    - `9273775` — booru-style explicit 1boy1girl original w/ accessories
    - `10347680` — booru-style explicit multi-character go-toubun no hanayome
    - `11126524` — booru-style sensitive 1girl hatsune miku (clean character)
- CFG sweep: {2, 4, 6, 8}
- Steps: 28, `flow_shift=1.0`, `negative_prompt=""`
- Adapter: `output/ckpt/anima_chimera_chimera.safetensors` (latest LoRA at run
  time)
- One noise seed per prompt (shared across CFGs so the magnitude→CFG curve
  isn't seed-confounded)

## Findings (n=4 sweep)

Per-CFG aggregates over 112 (4 prompts × 28 steps) `|e|`-per-step rows:

|  CFG  | `|e|`_p10 | `|e|`_p50 | `|e|`_p90 | k=0.02 mean k/`|e|` | k=0.02 frac k≥`|e|` | k=0.1 mean k/`|e|` | k=0.1 frac k≥`|e|` |
|------:|----------:|----------:|----------:|--------------------:|--------------------:|-------------------:|-------------------:|
|  2.0  |   0.0170  |   0.0269  |   0.0733  |             0.73    |            22 %     |            3.63    |           95 %     |
|  4.0  |   0.0158  |   0.0246  |   0.0669  |           **0.80**  |          **30 %**   |            4.01    |           93 %     |
|  6.0  |   0.0111  |   0.0220  |   0.0699  |             0.95    |            38 %     |            4.74    |           93 %     |
|  8.0  |   0.0102  |   0.0213  |   0.0667  |           **1.00**  |          **41 %**   |            5.01    |           93 %     |

### Per-step `|e|` profile at CFG=4 (mean across 4 prompts)

```
step   sigma     |e|       step   sigma     |e|
   0  1.0000   0.1569         14  0.5000   0.0215
   1  0.9643   0.1491         15  0.4643   0.0208
   2  0.9286   0.0767         16  0.4286   0.0202  ┐
   3  0.8929   0.0571         17  0.3929   0.0192  │
   4  0.8571   0.0498         18  0.3571   0.0186  │ plateau where
   5  0.8214   0.0464         19  0.3214   0.0182  │ k=0.02 ≈ |e|
   6  0.7857   0.0394         20  0.2857   0.0179  │ (steps 16-22)
   7  0.7500   0.0344         21  0.2500   0.0178  │
   8  0.7143   0.0309         22  0.2143   0.0178  ┘
   9  0.6786   0.0292         23  0.1786   0.0182
  10  0.6429   0.0274         24  0.1429   0.0188
  11  0.6071   0.0262         25  0.1071   0.0205
  12  0.5714   0.0242         26  0.0714   0.0236
  13  0.5357   0.0231         27  0.0357   0.0310
```

Key shape facts:

- `|e|` is large only at steps 0–1 (σ ≥ 0.96), then collapses ~3× in two
  steps, decays monotonically through step ~22, and ticks back up over the
  last 5 steps as the trajectory enters the detail-refinement regime. The
  late uptick is `|e|.mean()` recovering to the same scale as early steps,
  but it doesn't drop the saturated-step fraction much because by then σ is
  small enough that the controller's contribution to `x` is also small.
- `k=0.02` is in-band (`k/|e| < 0.3`) *only* at the first two steps. At every
  σ ≤ 0.85 the controller already starts shouldering the signal.
- `|e|` *shrinks monotonically* as CFG grows. The harder steering converges
  the next-step cond and uncond more — the linear extrapolation eats the
  divergence that the paper's "high-CFG nonlinearity" hypothesis was built
  on for SD3.5/Flux/Qwen. So the standard escape hatch ("SMC shines at high
  CFG") inverts on Anima.
- Per-step variance is wide: p10 vs p90 is ~4–6×. A single constant `k`
  cannot be in-band across the trajectory by construction: at peak `|e|`
  (early σ) any reasonable `k` is negligible; at trough `|e|` (the σ≈0.2–0.4
  plateau) the same `k` swamps the signal.
- `|e|_max = 0.217` is identical across all CFGs because step 0 runs on the
  same seeded `x_T ~ N(0, I)` and CFG combination hasn't compounded yet.

### Per-prompt variability at CFG=4

Each prompt's `|e|` across 28 steps (all four prompts run with their own seed
shared across CFGs, so columns are directly comparable):

|        prompt (stem)          |  mean  |   p10   |   p50   |   p90   |
|-------------------------------|-------:|--------:|--------:|--------:|
| 12941930 (heavy explicit, long tag list) | 0.0313 | 0.0129 | 0.0167 | 0.0384 |
|  9273775 (explicit)                      | 0.0399 | 0.0234 | 0.0269 | 0.0672 |
| 10347680 (multi-character)               | 0.0380 | 0.0165 | 0.0249 | 0.0500 |
| 11126524 (clean Miku)                    | 0.0414 | 0.0187 | 0.0258 | 0.0740 |

- Prompt 12941930 is a systematic ~25 % low outlier across the whole
  trajectory. Plausible (untested) story: very long, dense tag lists tighten
  cond/uncond agreement — more constraints → less room for divergence. n=4 is
  too small to commit to that hypothesis.
- **Per-step CV (std/mean across the 4 prompts at each step) sits at ~18–22 %
  through most of the trajectory** (step 0 is 28 % from seed × prompt
  coupling, late steps drop to 12–16 %). So at any given σ, individual
  prompts wander ±20 % around the population mean.

What this implies for the proposals:

- **A is unaffected**. `k_t = α · |e_t|.mean()` is computed inside each
  inference call from *that call's* actual `|e_t|`, so it picks the right
  scale per-prompt for free.
- **B is weakened**. A k(σ) fit on a population would be off by ±20 % for any
  individual prompt at any step. Not catastrophic, but a structural
  shortfall vs A.
- **C loses its main motivator**. Per-prompt variability is precisely what C
  was supposed to capture. A already eats it. C would only beat A if FEI
  captured *within-element spatial* variation, which it doesn't (FEI is a
  per-batch-element scalar). C is now strictly dominated for this purpose.

## Re-reading the drift symptom

The original `sign()`-mode SMC produced visible texture noise + composition
drift on Anima at CFG=4. We patched that with `tanh(s/ε)` boundary-layer
smoothing (classical SMC textbook fix, Edwards & Spurgeon 1998) and the noise
floor dropped. Re-reading those symptoms now:

- Tanh kills the high-frequency *chattering* (rapid sign flips on near-zero
  elements). That's the spatial-frequency component of the disturbance.
- Tanh does *not* change the controller's **energy budget**: `|Δe| ≤ k` per
  element, and our `k=0.02` is comparable to `|e|` itself across the σ≈0.2–0.4
  plateau (steps 16–22). So the perturbation to the velocity field still has
  order-of-magnitude `|e|` energy injected at every step in the regime where
  the diffusion is most visually sensitive (early-detail). With or without
  tanh, that is the kind of constant-energy nudge that produces compositional
  drift over 28 steps.

So the drift wasn't really a chattering symptom — it was the controller doing
exactly what a too-large `k` does: pushing the trajectory off course by an
amount comparable to its actual correction signal. Tanh smoothed the *texture*
but couldn't address the magnitude.

## Why the regime mismatch (hypothesis)

The paper's experiments are on SD3.5 / Flux / Qwen-Image at CFG ≥ 7. These
models presumably have a larger per-element `|cond − uncond|` velocity (deeper
text conditioning, less-decoupled CFG branches, or just different model
scale → output mapping). Their `k=0.1` lives in the `k/|e| ≪ 1` regime by
construction.

Anima's `|e|` per element is small (~0.024 median). Whether this is because
(a) the velocity field is intrinsically smoother, (b) our LoRA narrows the
cond/uncond gap, or (c) the chimera adapter's text conditioning is
quantitatively weaker than SD3.5's is not the question we need to answer here.
The dimensional `k` simply does not transfer.

## Proposals

### A. Adaptive `k_t = α · |e_t|.mean()` *(recommended starting point)*

Replace the dimensional scalar `k` with a dimensionless gain `α ∈ (0, 1]`.
At each step:

    k_t = α · |e_t|.mean()
    Δe  = −k_t · tanh(s / ε)

The controller's per-element influence is then bounded by `α` *relative to
the current cond/uncond divergence*. By construction:

- `α = 0.2`: controller can never inject more than 20 % of the local error
  magnitude per element, regardless of model / CFG / σ / sample.
- `α = 1.0`: maximal — reproduces the worst-case bound where `|Δe| ≈ |e|`.

Properties:

- **Self-scaling**. The 14× regime gap between Anima and the paper's models
  disappears — α picks the operating point in dimensionless units.
- **Stable across CFG**. `|e|` shifts with CFG; `k_t` follows it. No
  per-CFG retuning.
- **Cheap**. One extra mean per step; negligible vs the DiT forwards.
- **Compatible with the paper's stability framing**. The CFG-Ctrl framework
  (Sec 3.3) already admits time-varying gain `K_t` (the Weight-Scheduler
  variant). `K_t = α · |e_t|.mean()` is just a state-dependent gain
  schedule — the Lyapunov argument re-derives identically because `k_t > 0`
  is preserved.

Code surface: ~3 lines in `library/inference/smc_cfg.py` plus a new
`--smc_cfg_alpha` flag in `inference.py`. The fixed `--smc_cfg_k` path stays
for paper reproducibility.

### B. σ-scheduled `k(σ)`

Fit `|e|_baseline(σ)` once from a one-shot bench run, set
`k(σ) = α · |e|_baseline(σ)`. Captures the ~4.5× p10→p90 spread.

- Pro: no per-step reduction; lookup is free.
- Con: needs per-model recalibration (re-fit after every LoRA / Spectrum / DCW
  change); off by ±20 % per prompt at any step (see *Per-prompt variability*).
- Verdict: **subsumed by A**, which tracks `|e_t|` directly and absorbs both
  the σ-trend and the per-prompt variability. Only worth considering if A's
  per-step reduction shows up in profiling.

### C. FEI-scheduled `k(z_t)`

The FEI 2-band simplex (`library/runtime/fei.py::compute_fei_2band`) is
already computed per step for Hydra/FeRA routing. Hypothesis:
`e_high(z_t)` co-varies with `|e|` — early high-noise steps have both
high-frequency latent energy and a large cond/uncond gap.

- Parameterization: `k_t = α · (a + b · e_high(z_t))` with `(a, b)` fit from
  one-shot data.
- Pro: captures per-sample variability that σ-only misses; reuses an existing
  compute.
- Con: empirically fit; adds two hyperparameters; FEI's *strongest*
  discriminative signal is at low/mid σ (per `project_fera_probe_2band_decision`),
  which is *not* where `|e|` is largest. Likely a weak proxy for what we
  actually need.
- **The per-prompt-variability argument that motivated C is fully absorbed
  by A** (which tracks live `|e_t|` per inference call). For C to add value
  it would need to model per-element *spatial* variation that the
  `|e_t|.mean()` reduction discards — but FEI itself is a per-batch-element
  scalar, so it can't help with that either.
- Verdict: **strictly dominated by A** for this purpose. Skip.

## Open questions

1. **Does A actually improve output quality at CFG=4?**
   The bench so far only characterizes `|e|`. It does not establish that SMC
   (even at the right `k`) improves output vs vanilla CFG on Anima at
   production CFG. A small A/B with α ∈ {0.05, 0.1, 0.2, 0.3} via
   `bench/smc_cfg/compare_cfg.py` is the gating experiment.
2. **Does SMC have *any* operating regime on Anima where it's a clean win?**
   `|e|` doesn't grow with CFG here (unlike paper). So the high-CFG escape
   hatch may not exist for us. If A at α=0.2 is a wash even at CFG=8, the
   honest conclusion is "shelve at production CFG".
3. **Lyapunov re-derivation under state-dependent k_t.**
   The CFG-Ctrl paper proves finite-time convergence under fixed `k > 0`.
   `k_t = α · |e_t|.mean()` is positive and Lipschitz in `|e|`, so finite-time
   convergence should carry through, but the bound on convergence time changes
   from `T_c ≤ V(s_0) / k` to something involving the trajectory of `|e_t|`.
   Not load-bearing for shipping the controller; worth a footnote.

## Next steps (in order)

1. ~~Refresh the table with the 4-prompt sweep.~~ Done — see *Findings (n=4 sweep)* above.
2. Wire A in `library/inference/smc_cfg.py`:
   - Add `alpha: float | None` to `SMCCFGState.__init__`.
   - In `combine()`: if `alpha` is set, compute `k_t = alpha * e.abs().mean()`
     in place of `self.k`.
   - Add `--smc_cfg_alpha` to `inference.py`'s SMC group; mutually exclusive
     with `--smc_cfg_k`.
3. A/B at α ∈ {0.05, 0.1, 0.2, 0.3} × CFG ∈ {4, 8} via
   `bench/smc_cfg/compare_cfg.py`. Look at composition drift, fine-texture
   noise, and saturation behavior.
4. Outcome gate:
   - If α=0.2 at CFG=4 produces a perceptual win: ship adaptive SMC as the
     default, document `--smc_cfg_alpha 0.2` as the production setting.
   - If wash or regression: shelve SMC-CFG as not-fit-for-Anima at production
     CFG. Keep code path alive for users at CFG ≥ 6 with `α=0.1` (still
     in-band per the |e| sweep).

## File map

| File | Role |
|------|------|
| `library/inference/smc_cfg.py`            | `SMCCFGState` — sliding-surface combine. Tanh boundary layer with auto `ε`. Currently fixed-k. |
| `inference.py` (search `--smc_cfg`)       | CLI surface: `--smc_cfg`, `--smc_cfg_lambda`, `--smc_cfg_k`, `--smc_cfg_eps`. |
| `library/inference/generation.py`         | Calls `smc_cfg.combine(...)` from the per-step cond/uncond combine in `generate_body` and `generate_body_tiled`. |
| `bench/smc_cfg/compare_cfg.py`            | Existing A/B harness (vanilla vs SMC, pixel divergence). Subprocess-based. |
| `bench/smc_cfg/measure_error_magnitude.py` | This study's source. In-process reverse-denoise, captures `|e|` per step + SMC stats offline. |
| `library/runtime/fei.py`                  | `compute_fei_2band` — used by proposal C. |
