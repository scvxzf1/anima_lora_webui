# Soft Tokens — per-layer × per-t learnable text tokens (SoftREPA)

Per-layer, time-indexed soft tokens in T5-compatible space. DiT is frozen. ~1M trainable params at default config (n_layers=10, K=4, D=1024, n_t_buckets=100). Each of the first `n_layers` DiT blocks gets its own learned (K, D) token bank plus a per-(t-bucket, layer) D-vector offset, spliced into `crossattn_emb` for that block alone. Trained with plain FM.

Reference: Lee et al., *Aligning Text to Image in Diffusion Models is Easier Than You Think* (arXiv:2503.08250, NeurIPS 2025) — "SoftREPA". The base recipe adopts only the parameterization (per-layer × per-t soft tokens), trained under plain FM; the paper's InfoNCE contrastive objective was originally skipped because at Anima's training batch size (B=1) there are no in-batch negatives, and the paper itself reported SD3 FID regression at paper-strength contrastive. An **optional, B=1-adapted contrastive objective** is now available (off by default) — it builds negatives by swapping a cached text embedding off disk instead of using batch peers. See §"Contrastive objective" below and `docs/proposal/soft_tokens_contrastive.md`.

## Quick start

```bash
make exp-soft-tokens                    # default preset
python tasks.py exp-soft-tokens         # cross-platform
```

v1 is **training-only**. `inference.py` cannot load these checkpoints — `create_network_from_weights(for_inference=True)` raises `NotImplementedError`. Wire up the per-step block hook inside the denoising loop if early training looks promising.

## What it is

For each block `k ∈ [0, n_layers)`, the cross-attention input is replaced by a layer-specific variant:

```
s^(k, t)         = tokens[k] + t_offsets[bucket(t), k]      # shape (K, D)
crossattn_emb_k  = splice(crossattn_emb, s^(k, t))
block_k(x, ..., crossattn_emb_k)                            # original block, modified text input
```

`tokens ∈ ℝ^(n_layers × K × D)` is the base bank; `t_offsets ∈ ℝ^(n_t_buckets × n_layers × D)` is a per-(bucket, layer) D-vector broadcast across the K-token axis. Zero-init on `t_offsets` means at step 0 the layer banks reduce to their base values — no time conditioning until gradients learn it.

```
                       Soft Tokens
              ┌─────────────────────────────┐
              │  DiT Block 0                │
              │  ┌─────────────┐            │
crossattn ────┼─►│ +s^(0,t)    │  cross    │
   (B,S,D)    │  │   spliced   │  attn ──► │ ──► x'
              │  └─────────────┘            │
              ├─────────────────────────────┤
              │  DiT Block 1                │
              │  ┌─────────────┐            │
crossattn ────┼─►│ +s^(1,t)    │  cross    │
   (B,S,D)    │  │   spliced   │  attn ──► │ ──► x''
              │  └─────────────┘            │
              ├─────────────────────────────┤
              │  ...                        │
              ├─────────────────────────────┤
              │  DiT Block (n_layers..N-1)  │  no splice — block sees
              │  cross-attn (unmodified)    │  the original crossattn
              └─────────────────────────────┘
```

The crossattn passed in is unchanged across blocks (Anima is **not** joint-stream MM-DiT — text features don't evolve through blocks). Each block independently sees a different splice; no strip/re-prepend dance.

## Parameter count

```
n_layers · K · D       (base tokens)
+ n_t_buckets · n_layers · D   (t-offsets, broadcast across K)
```

Defaults: 10 · 4 · 1024 + 100 · 10 · 1024 ≈ 41k + 1.05M ≈ **1.05M params**. 30–60× lighter than a typical LoRA.

## Implementation map

| File | Role |
|------|------|
| `networks/methods/soft_tokens.py` | `SoftTokensNetwork` — per-(layer, t) token bank, splice hook, save/load. |
| `apply_to(text_encoders, unet)` | Walks `unet.blocks[:n_layers]`, replaces each `block.forward` with a wrapper that splices `s^(k, t)` into `crossattn_emb` before calling the original (ReFT-pattern monkey-patch). |
| `append_postfix(crossattn_emb, seqlens, timesteps)` | Receives `timesteps` from `train.py`'s existing per-step hook; computes `(n_layers, B, K, D)` step-scoped tokens and caches them on the network. **Returns `crossattn_emb` unchanged** — splicing happens inside the block hooks. |
| `_make_block_hook(layer_idx, org_forward)` | Closure that reads the cached step tokens at `layer_idx`, splices into `crossattn_emb`, calls the original block forward. |
| `SoftTokensMethodAdapter` (same file) | Contrastive extra-forward driver: stashes `neg_crossattn_emb` in `prime_for_forward`, runs the `k` negative forwards + InfoNCE in `extra_forwards`, surfaces metrics. Auto-resolved by `resolve_adapters` when `_contrastive_target_weight > 0`. |
| `contrastive_loss(...)` / `step_contrastive_warmup(...)` | InfoNCE over the negatives (with optional jaccard penalty) + the warmup gate. |
| `library/datasets/base.py` | `setup_contrastive_negatives` / `_load_te_for_stem` — negative TE sourcing + `neg_crossattn_emb` / `neg_jaccard` on the example. |
| `library/datasets/identity_pairs.py` | `IdentityPairSampler.hard_negative` / `shuffled` / `tag_jaccard` — negative policy. |
| `library/training/losses.py::_soft_tokens_contrastive_loss` | Applies the warmup-gated `λ_con` to the adapter's InfoNCE scalar. |
| `configs/methods/soft_tokens.toml` | Default config (splice_position=front_of_padding, lr=1e-3, 4 epochs; plain FM, contrastive off). |
| `configs/gui-methods/soft_tokens.toml` | Sibling for `make lora-gui GUI_PRESETS=soft_tokens`. |
| `scripts/experimental_tasks/training.py::cmd_soft_tokens` | Task entry-point. |
| `tasks.py` `exp-soft-tokens` | Make/CLI registration. |

## Splice position

Two options, mirroring postfix:

| Mode | Where | Trade-off |
|---|---|---|
| `end_of_sequence` (default) | overwrite the K tail slots `[S-K, S)` of the zero-padding region | Static splice index → maximally compile-friendly. Caption-position-agnostic. Preserves the strongest front-of-padding attention sinks intact. |
| `front_of_padding` | place K tokens at `[seqlens[i], seqlens[i]+K)` per sample (`scatter`) | Caption-position-aware. Displaces the strongest sinks. Per-sample variable indices via the cached `crossattn_seqlens`. |

Toggle via `network_args = ["splice_position=front_of_padding"]`. The choice is metadata-tagged (`ss_splice_position`) so checkpoints round-trip with the right splice mode.

Anima's text-encoder padding invariant (zero-padded positions act as cross-attention sinks) means writing into the padded tail is *not* a no-op — those slots receive attention mass and the soft tokens get exposure to every spatial query. See the "Text encoder padding" note in the root CLAUDE.md.

## Why a separate module from `postfix.py`

Postfix splices **once** at the cached adapter output (training-time and inference-time, in `train.py:762` and `library/inference/generation.py`). Soft tokens splice **per-block** via a monkey-patched `Block.forward`. Different surface entirely — keeping them separate avoids muddying the postfix abstraction. Both modules expose `append_postfix(...)` so `train.py`'s existing per-step trainer hook routes timesteps to either family without code changes.

## Why no slot-collapse

The existing postfix module logs an aggressive guard against K-slot permutation symmetry collapse (`anima_postfix.safetensors` was effectively K=1 due to zero-init + symmetric splice — see the postfix module docstring and the `slot_embed_init_std` knob). Soft tokens **structurally avoid** this: tokens at different `(k, t)` pairs are consumed at different positions in the network and gradients differ from step 1, so no symmetry to break.

> **Removed: bank-axis dispersive regularizer (2026-05-22).** Earlier versions
> shipped an optional parameter-space dispersive regularizer (Wang & He,
> *Diffuse and Disperse*, arXiv:2506.09027) over the bank's `K` and
> `n_t_buckets` axes, meant to guard against slot collapse and under-sampled
> bucket degeneracy. It was removed after it showed no effect worth keeping —
> soft tokens already **structurally avoid** slot collapse (see "Why no
> slot-collapse" above: different `(k, t)` pairs are consumed at different
> positions, so gradients differ from step 1 and there's no symmetry to break).
> The repr-space variant was separately probed and found redundant
> ([[project_soft_tokens_contrastive_phase0]]). Plain FM is now the baseline;
> the only optional add-on is the contrastive objective below.

## Contrastive objective (optional, B=1-adapted SoftREPA InfoNCE)

A revival of SoftREPA's contrastive objective, off by default. It is **data-conditioned** and **needs negatives** — it sharpens prompt-following by making the *matched* text explain the anchor's latent better than *mismatched* text does. Full design + phasing: `docs/proposal/soft_tokens_contrastive.md`.

The B=1 trick (no batch peers): a negative is a **different stem's cached text embedding** (`{stem}_anima_te.safetensors`, the post-LLM-adapter `crossattn_emb`) swapped off disk — the same cached-feature-swap precedent the IP-Adapter identity pairs use, but swapping the TE feature instead of the PE feature. Each step runs the primary forward (matched text = the positive) plus `k` extra DiT forwards with the negative text spliced through the same soft tokens; the logit of a forward is its negative flow-matching error against the shared velocity target:

```
ℓ_*           = -‖v_* − v_target‖² / τ      (mean over C·H·W; logit = neg FM error)
L_contrastive = -log( exp(ℓ_pos) / Σ_{pos, neg_1..k} exp(ℓ_*) )
L_total       = L_FM + λ_con · L_contrastive                         (post-warmup)
```

Only `crossattn_emb` differs across the forwards, so the gradient isolates text-conditioning. Each negative is one extra full DiT forward — `k=4` ≈ 5× step time — so keep `k ∈ {1, 2}`.

**Negative modes** (`contrastive_negative_mode`):

| Mode | Sourcing | Notes |
|---|---|---|
| `shuffled` (default) | an unrelated image (no character/copyright overlap) | The Phase-1 go/no-go negative. |
| `jaccard` | shuffled sourcing + per-negative logit down-weight `ℓ_neg −= α·s` | `s` = caption tag-overlap (character ∪ copyright ∪ artist) Jaccard; a near-miss negative pulls less gradient. Cheap middle path — no new sampler. `α = contrastive_jaccard_alpha`. |
| `hard` | a same-artist / **different-character** sibling (style-matched, content-different) | Cancels style-induced velocity similarity so the only axis left to win on is content. Falls back to `shuffled` for orphan/untagged artists — on the current dataset Phase 0 measured the strict pool at ~29% coverage, so ~71% of steps degrade to shuffled. |

Negative grouping comes from the shared caption index (`make caption-index` → `post_image_dataset/captions/caption_index.json`), reusing `IdentityPairSampler` (`hard_negative` / `shuffled` / `tag_jaccard`). The index path is not a user knob.

**Wiring.** Negatives are sourced in `library/datasets/base.py::setup_contrastive_negatives` / `_load_te_for_stem` (surfaced as `neg_crossattn_emb` `(B, k, S, D)` on **train steps only** — validation FM-MSE stays a clean baseline). The `k` extra forwards + InfoNCE live in `SoftTokensMethodAdapter.extra_forwards` + `SoftTokensNetwork.contrastive_loss`; the warmup-gated weight is composed in `library/training/losses.py::_soft_tokens_contrastive_loss` (active iff `_contrastive_target_weight > 0`). `step_contrastive_warmup` holds `λ_con` at 0 for the first `warmup_ratio` of steps. The objective leaves **no learned parameters** — a trained checkpoint is bit-identical whether or not contrastive was on, and inference ignores it entirely.

**Config knobs** (`network_args`, all off-by-default-safe):

| Knob | Default | Meaning |
|---|---|---|
| `contrastive_weight` | `0.0` | λ_con; `0` = bit-identical to plain FM (dataset stops producing negatives → no extra forwards). |
| `contrastive_k` | `1` | negatives per step → `(k+1)×` forward cost. |
| `contrastive_negative_mode` | `shuffled` | `shuffled` \| `jaccard` \| `hard`. |
| `contrastive_jaccard_alpha` | `1.0` | logit penalty for `jaccard` (sweep 0.5–2.0). |
| `contrastive_tau` | `0.5` | InfoNCE temperature. |
| `contrastive_warmup_ratio` | `0.1` | hold λ_con at 0 for the first 10% of steps. |

TensorBoard signals: `reg/soft_tokens_contrastive` (raw InfoNCE), `_weighted`, `_lambda_live` (warmup gate), `soft_tokens/contrastive_acc` (positive beats every negative) and `soft_tokens/contrastive_logit_gap`.

## Compatibility

| Component | Compat | Notes |
|---|---|---|
| Training loop | ✅ | `train.py` already passes `timesteps=...` into `append_postfix` (legacy `cond-timestep` postfix mode); soft tokens piggyback on the same hook. |
| Standard inference | ❌ v1 | `for_inference=True` raises `NotImplementedError`. Per-step block hook would need to fire inside `library/inference/generation.py::generate_body`. |
| Spectrum inference | ❌ v1 | Same blocker as standard inference — Spectrum's actual-step forwards would need the per-step hook too. |
| `torch.compile` (`_run_blocks`) | ✅ | `end_of_sequence` keeps `crossattn_emb` shape static; the cached `_step_layer_tokens` is read as a runtime tensor with static shape. `front_of_padding` uses `scatter` with dynamic per-sample indices but static buffer shape — also compile-clean. |
| `blocks_to_swap` | ❌ method-forced 0 | The hook captures each `Block` by reference at `apply_to()` time; a swapped block is a different object instance, so the hook would fire on the wrong tensor. |
| `gradient_checkpointing` | ✅ | The hook is the outermost wrapper; the original `forward` (which itself runs `checkpoint(_forward, ...)`) is called underneath, and the spliced `crossattn_emb` is part of the saved input graph. |
| Modulation guidance | ✅ orthogonal | Modulation = per-block AdaLN path; soft tokens = K/V input path per block. |
| T-LoRA / OrthoLoRA / ReFT | n/a | Soft tokens freeze the DiT; LoRA-family methods are not stacked in this config. |

## Evaluation

What to measure to know if this is doing anything:

1. **`|t_offsets|` at convergence as a function of bucket**: flat/near-zero → time conditioning collapsed (the per-layer base tokens absorbed everything; SoftREPA's `use_dc_t=True` won't be load-bearing). Curve should grow away from zero, ideally with structure across t.
2. **Per-layer token norm**: `‖tokens[k]‖` should differ across `k`. If they converge to a single shared bank, we're effectively running a single-layer postfix and the per-layer parameterization is dead weight.
3. **Held-out prompt-following**: this is the load-bearing question. The existing DCW v4 calibrator targets the same axis (text-image alignment, prompt-following) but at inference time. If soft tokens move the same metrics, they're a training-time alternative. If not, they're parameter overhead.
4. **Anatomy / style breakdown**: REPA helped anatomy on Anima but broke anime style (vision-encoder photo-prior leak). Soft tokens have no external visual prior, so the failure mode shouldn't recur — but they also can't reproduce the anatomy gain. The plausible win is text alignment, not structural quality. If anatomy *also* improves, that's a surprise worth tracking.
5. **Splice position A/B**: `end_of_sequence` vs `front_of_padding`. Front-of-padding displaces the strongest sinks and might give the tokens more attention mass at the cost of disturbing what the pretrained model relies on. Worth a short bench before committing to a default.

## Hyperparameters worth sweeping

| Knob | Default | Range to try | Why |
|---|---|---|---|
| `n_layers` | 10 | 5, 10, 14, 28 | SoftREPA used 5/24 layers on SD3. 10/28 is proportional. Going to 28 (all blocks) doubles params and tests whether deep-block tokens do anything. |
| `network_dim` (K) | 4 | 1, 4, 8, 16 | SoftREPA used m=4 on SD3. K=1 collapses to "per-layer prefix vector" — clean ablation. |
| `n_t_buckets` | 100 | 0 (disable t-cond), 20, 100 | Setting `t_offsets.weight.requires_grad_(False)` is a clean ablation for whether time conditioning is load-bearing. |
| `init_std` | 0.02 | 0.0, 0.02, 0.1 | Zero-init = strict identity at step 0 (block sees zeroed padding tail). 0.02 = small perturbation. 0.1 = aggressive. |
| `splice_position` | `end_of_sequence` | both | See §"Splice position" above. |
| `learning_rate` | 1e-3 | 1e-4 to 5e-3 | Soft tokens are tiny + zero-inited offsets; high LR is fine. |
