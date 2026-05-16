# Postfix-Inversion Bridge

Diagnostic for whether a closed-form linear map `PE_feature → postfix s-vector`
gives a useful warm-start for postfix-tail inversion — and as a side product,
how much of the PE feature space actually reaches the postfix manifold.

Motivated by the Procrustes lift in *Asymmetric Flow Models* (Chen et al.,
arXiv:2605.12964) §5: a patch-wise linear alignment served as initialization
for a much harder finetune. Here we ask the same question one level up —
can a linear bridge between two of Anima's existing spaces (PE image features
and the SVD-orthogonalized postfix tail) replace, or at least warm-start,
the gradient-descent inversion in
`scripts/inversion/invert_postfix_tail.py`?

The interpretability payoff is independent of the warm-start question: the
SVD of the fit map tells you which PE-feature directions have a representation
in postfix space and which don't. Compare to the AnisoAlign measurement on
PE↔T5 (most directions disjoint) — the postfix tail is a different target
manifold and may be more or less compatible.

## Data prerequisites

Two per-image artifacts, paired by stem:

| Tensor | Source | Path pattern |
|--------|--------|--------------|
| `s` ∈ R^K | `scripts/inversion/invert_postfix_tail.py` | `{out_dir}/s/{stem}_s.safetensors` |
| `image_features` ∈ R^(T, 1024) | `make preprocess-pe` | `post_image_dataset/lora/{stem}_anima_pe.safetensors` |

`s` shape `K` must be consistent across the pool — the bench fails loud on
mixed `K`. Filter your `--s_glob` accordingly.

Populate inverted `s` vectors first:

```bash
uv run python scripts/inversion/invert_postfix_tail.py \
    --dit models/diffusion_models/anima-base-v1.0.safetensors \
    --image_dir post_image_dataset/lora \
    --num_images 128 --shuffle --seed 0 \
    --K 48 --basis svd_te \
    --basis_path output/probes/postfix_tail/svd_basis_K48.pt \
    --steps 100 --lr 0.01 --grad_accum 4 \
    --output_dir output/probes/postfix_tail/bridge_pool_v0
```

Then run the bench against that pool. `N ≥ 4·K` is the minimum for the
held-out fit to be remotely meaningful (with the default ridge λ); `N ≥ 32·K`
is where the singular-value diagnostic becomes trustworthy.

## Usage

```bash
uv run python bench/postfix_inversion_bridge/run_bench.py \
    --s_glob 'output/probes/postfix_tail/bridge_pool_v0/s/*_s.safetensors' \
    --pe_dir post_image_dataset/lora \
    --test_frac 0.2 --seed 0 \
    --label first-pool
```

Writes the standard `bench/_common.py` envelope plus three artifacts:

- `singular_values.csv` — `(rank, sigma, cumulative_variance)` of the fit map.
  Read as: "the first `r` PE-feature directions carry `cumulative_variance[r]`
  of the postfix manifold's reachable variance."
- `predictions.csv` — per held-out image: `stem, l2_err_pred, l2_err_zero, cosine, warm_gain`.
- `bridges.safetensors` — fit matrices and centering means so the linear
  bridge is reusable (e.g. as a warm-start in `invert_postfix_tail.py`).

## What the metrics mean

- **`test_r2`** (ridge bridge) — fraction of held-out `s`-variance explained
  by a linear function of PE features. Negative → bridge is worse than
  predicting the train mean.
- **`mean_warm_gain`** — fraction of the gradient distance `||s||` that the
  bridge already covers at init. Zero means "no better than starting from
  zero." Positive means "warm-start saves this fraction of the inversion's
  work."
- **`top_k_var_explained[r]`** — diagnostic only. If the first ~5 singular
  values already account for ~80% of the variance, PE and postfix spaces
  share a low-dim aligned subspace. If the curve is near-flat, the bridge
  is high-rank and there's no compact aligned subspace.

A flat curve + low `mean_warm_gain` recapitulates the AnisoAlign result and
says "the linear bridge is dead here, gradient inversion is the only path."
A curve with structure + positive gain says "fit this once, use it for free."
