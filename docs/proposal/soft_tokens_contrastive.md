# Soft Tokens — contrastive objective via cached-TE hard negatives

Status: **proposal** (2026-05-22). Supersedes the archived scratch note
`_archive/soft_tokens_contrastive/soft_token_idea.md` (2026-05-08), which was
written against the **since-removed** in-batch InfoNCE path. Builds on
`docs/experimental/soft_tokens.md`; reuses the identity-pairing machinery shipped
for IP-Adapter (`docs/proposal/ip-adapter-identity-pairs.md`,
`library/datasets/identity_pairs.py`).

## TL;DR

Soft tokens today train under **plain FM** alone. An earlier parameter-space
dispersive regularizer (the negative-free substitute for SoftREPA's InfoNCE, used
because at Anima's `B=1` the paper's `torch.roll(text, shifts=j)` in-batch
negatives don't exist) was **removed 2026-05-22** — it showed no effect worth
keeping, and soft tokens structurally avoid slot collapse anyway (see
[[project_soft_tokens_contrastive_phase0]] and the removal note in
`docs/experimental/soft_tokens.md`).

This proposes the **other** lever: revive a contrastive objective, but build its
negatives the way the IP-Adapter identity-pair work just proved out — by
**swapping a cached feature off disk** (no batch peers, no live encoder). For
soft tokens the swapped feature is the cached **text embedding**
(`{stem}_anima_te.safetensors`) instead of IP-Adapter's PE feature, and
`IdentityPairSampler` + `caption_index.json` supply the grouping to make those
negatives *hard* (same artist, different content) instead of style-confounded.

Contrastive sharpens prompt-following (a data-conditioned objective). It is
gate-driven: the Phase 1 go/no-go (λ_con ∈ {0, >0}, eval on prompt-following not
FM-MSE) decides whether it earns its `(k+1)×` step cost over the plain-FM
baseline.

## Background — what changed since the scratch note

The 2026-05-08 note debated negative-sampling *policy* for a contrastive path that
existed at the time (`soft_tokens.py:606`, `torch.roll(text, shifts=j, dims=0)`
over batch peers). Two things invalidate its framing:

1. **The contrastive path was removed** in commit 20ec6b1 (2026-05-19) and
   replaced by the parameter-space dispersive loss. There is no live InfoNCE term
   to retune.
2. **`torch.roll` negatives were always dead at `B=1`.** `batch_size=1` is fixed
   (`configs/base.toml:80`); rolling a 1-row batch yields the same sample. The
   note's options (a)/(b)/(c) all presupposed batch peers that never existed.

So the question is no longer "which negative-sampling policy" — it's "can we
construct negatives *at all* at B=1, and is the resulting contrastive term
load-bearing." The IP-Adapter work answered the first half.

## Dispersive removed — contrastive is the sole optional add-on

The parameter-space dispersive regularizer was **removed 2026-05-22**. It was a
negative-free, parameter-space repulsion (over the bank's `K` and `n_t_buckets`
axes, adapted from Wang & He, *Diffuse and Disperse*, arXiv:2506.09027) meant to
fix K-slot / bucket collapse. It was dropped because (a) soft tokens already
structurally avoid slot collapse — different `(k, t)` pairs are consumed at
different positions, so gradients differ from step 1 — and (b) no result showed
it improving samples. Plain FM is now the baseline; contrastive (this proposal)
is the only optional add-on, behind its own warmup-gated weight in
`library/training/losses.py`. The repr-space variant of dispersive was probed
separately and also found redundant — recorded below for the record.

## Variant (b) — representation-space dispersive (negative-free): PROBED → REDUNDANT

A tempting third lever, given Phase 0 FAILED the hard-negative gate: keep the
dispersive but move it from **parameter space** (raw `tokens`) to the
**representation space the model discriminates the soft tokens in** — the frozen
cross-attention K projection (`net.blocks.{k}.cross_attn.{k_proj,k_norm}`). The
Diffuse-and-Disperse loss was originally a *representation*-space regularizer;
the shipped soft-tokens version disperses raw bank coords instead. Variant (b)
would disperse `k_norm(k_proj · token)` over the K axis — still negative-free,
still B=1-safe (K axis), still no extra forward, but measuring "apartness" where
the model actually competes the tokens for attention mass.

**Probed and shelved 2026-05-22.** `bench/soft_tokens_contrastive/repr_dispersive_probe.py`
simulates the slot-collapse failure mode (K slots init'd near a shared per-layer
direction, param |cos|≈1.0), then runs the *same* bounded dispersive form in
param space vs. repr space, logging both cosines. Result across {K-only, K+V} ×
{k_norm, no-k_norm} × {cosine_sq, normalized_pdist}: **param-space dispersion
already separates the tokens in K representation space** — the param arm leaves
repr |cos| ≈ 0.08 (vs. the repr arm's 0.00; leakage +0.08, below the +0.10 wire
threshold), and under the weaker `normalized_pdist` form both arms plateau at the
*same* repr |cos| ≈ 0.33 (leakage ≈ 0). The mechanism: `k_proj` is high-rank
(eff_rank ≈ 618/1024) and well-conditioned (top-1 singular energy 3.7%), so it
approximately preserves angular geometry — there is no projection-induced
re-correlation for (b) to undo. **Don't wire it**; the shipped parameter-space
dispersive already covers the representation space. (If a future base model
shipped a low-rank/ill-conditioned cross-attn K projection, re-run the probe —
the verdict is `k_proj`-spectrum-dependent.)

## The reframe — negatives by cached-TE swap

IP-Adapter's distinct-pair training proved you don't need batch peers to get a
contrast partner: you load a **different stem's cached feature** off disk and feed
it into the forward, decoupled from the VAE target. For soft tokens the cached
feature to swap is the **text embedding** (`{stem}_anima_te.safetensors`, the
post-LLM-adapter `crossattn_emb`).

A contrastive step at B=1 becomes:

```
x_t, ε, t        ← anchor sample (its noised latent, the FM target)
v_pos            = DiT(x_t, crossattn_emb=anchor_TE,  soft_tokens(t))     # matched
v_neg^j          = DiT(x_t, crossattn_emb=neg_TE^j,   soft_tokens(t))     # j = 1..k
ℓ_*              = -‖v_* − v_target‖² / τ                                 # logit = neg FM loss
L_contrastive    = -log( exp(ℓ_pos) / Σ_*∈{pos, neg^1..k} exp(ℓ_*) )
L_total          = L_FM + λ_con · L_contrastive                            # post-warmup
```

The soft tokens get gradient to make the **matched** text explain the anchor's
latent better than mismatched text does — i.e. to sharpen the cross-attention's
text discrimination, which is exactly the prompt-following axis (eval criterion #3
in `soft_tokens.md`). `v_target` is the same flow-matching target for every
forward; only `crossattn_emb` changes, so the gradient isolates text-conditioning.

`caption_index.json` lets us pick `neg_TE^j` on the **right axis**: a same-artist,
different-character negative cancels the style-induced velocity similarity, so the
only axis left to win on is content (the scratch note's option (c), now sourced by
the existing sampler instead of a new batch sampler).

## Design

### Negative sourcing (reuse `IdentityPairSampler`)

`library/datasets/identity_pairs.py::IdentityPairSampler` already has the two ends
of the spectrum from the IP-Adapter work:

- `resolve()` — a *positive* (same character → franchise → artist back-off).
- `shuffled()` — an *unrelated* negative (no character/copyright overlap).

Add one method, `hard_negative(target_stem, rng)`: from `groups["artist"]` pick a
same-artist stem whose `character` tags are **disjoint** from the target's →
style-matched, content-different (option (c)). Fall back to `shuffled()` for
orphan artists (the note's orphan policy, answered: fall back, don't skip). This
keeps all negative modes behind one sampler, unit-tested like
`tests/test_identity_pairs.py`.

### Dataset hook (mirror `setup_identity_pairs`, swap TE not PE)

Mirror the IP-Adapter hook in `library/datasets/base.py`: where
`setup_identity_pairs()` / `_load_ip_features_for_stem()` resolve a distinct
reference's nested **PE** cache, add `setup_contrastive_negatives()` /
`_load_te_for_stem()` resolving a negative's nested **TE** cache. `__getitem__`
stacks `neg_crossattn_emb` of shape `(k, S, D)` into the example. Requires
`cache_text_encoder_outputs=true` (already the soft_tokens default,
`cache_llm_adapter_outputs=true`).

### Loss

Add `contrastive_loss(v_pos, v_neg, v_target)` to `SoftTokensNetwork` and compose
it in `library/training/losses.py` as `_soft_tokens_contrastive_loss`, behind a
warmup gate (`step_contrastive_warmup`). The extra `k` forwards run inside the
trainer step with the same `(x_t, ε, t)` and the same spliced soft tokens; only
`crossattn_emb` differs.

### The cheap middle path first (Jaccard soft-weighting)

Before `hard_negative`, try the note's ~10-line middle path: source negatives with
`shuffled()` (random), then **down-weight** each negative's logit by its caption
tag-overlap Jaccard `s` against the anchor (`ℓ_neg -= α·s`). Negatives that share
artist+content become less-surprising mismatches and pull less gradient, capturing
most of the style-confound control with no new sampler. If the cheap variant moves
prompt-following, the hard-negative version is worth building; if not, it probably
isn't either.

### Config knobs (`configs/methods/soft_tokens.toml`)

| Knob | Default | Meaning |
|---|---|---|
| `contrastive_weight` | `0.0` | λ_con; `0` = bit-identical to today |
| `contrastive_k` | `1` | negatives per step → `(k+1)×` forward cost |
| `contrastive_negative_mode` | `shuffled` | `shuffled` \| `jaccard` \| `hard` |
| `contrastive_jaccard_alpha` | `1.0` | logit penalty for `jaccard` mode (sweep 0.5–2.0) |
| `contrastive_tau` | `0.5` | InfoNCE temperature |
| `contrastive_warmup_ratio` | `0.1` | hold at 0 for first 10% (lets plain FM shape the bank first) |

Defaults to off, so existing soft_tokens runs are unchanged until opted in.

## Phasing — gates, cheapest-first

- **Phase 0 — dataset structure (no training, runnable today). DONE 2026-05-22 →
  FAIL on the strict gate.** `bench/soft_tokens_contrastive/negative_audit.py`
  (`results/<ts>-phase0/{negative_audit.md,result.json}`): per stem, does a **same-artist /
  different-character** sibling exist? The note's kill threshold was <~50%
  coverage. Result: **strict (genuine, both sides character-tagged) = 29.0%**
  (755/2600) — below the 50% line. The intersection is essentially
  *capped by character coverage* (31.5%): 755/819 ≈ 92% of character-tagged
  images do find a strict negative, and where they exist the pool is deep
  (median 22, max 72) and clean (96.8% have a different-franchise option). The
  remaining 69.4% land in a *lenient* bucket where one side is untagged, so
  `hard` mode there is indistinguishable from a same-artist random pick.
  **Implication:** don't build the `hard` sampler yet — the strict pool is a
  *tagging* floor, not a structural ceiling, but at 29% it's too sparse across
  the full set. Phase 1 should gate on `shuffled` (and possibly `jaccard`); only
  expand to `hard` if character tagging is extended or Phase 1 shows the win is
  style-confounded.
- **Phase 1 — is a contrastive term load-bearing at all? IMPLEMENTED
  2026-05-22 (`shuffled` only).** Wire
  `contrastive_mode=shuffled`, `k=1`, and A/B `contrastive_weight ∈ {0, >0}` for
  ≥1 epoch. Implementation: cached-TE negatives sourced in
  `library/datasets/base.py::setup_contrastive_negatives` / `_load_te_for_stem`
  (reusing `IdentityPairSampler.shuffled`), the `k` extra DiT forwards +
  InfoNCE in `networks/methods/soft_tokens.py`
  (`SoftTokensMethodAdapter.extra_forwards` + `SoftTokensNetwork.contrastive_loss`),
  warmup-gated weight composed in `library/training/losses.py`
  (`_soft_tokens_contrastive_loss`). Knobs in `configs/methods/soft_tokens.toml`
  (`contrastive_*`, off by default). `jaccard`/`hard` modes raise
  `NotImplementedError` (Phase 2). **Not yet trained/benched** — the A/B run is
  the open go/no-go. The note is explicit: if there's no measurable
  prompt-following gap,
  "changing the negative policy is rearranging dead weight." This is the real
  go/no-go. Confirm with the same DCW-v4 / prompt-following axis named in
  `soft_tokens.md` eval #3 (FM-MSE val deltas don't track quality on Anima —
  `project_fm_val_loss_uninformative`).
- **Phase 2 — negative quality.** Only if Phase 1 is positive *and* the win looks
  style-confounded (check token differentiation, eval #2). Add `jaccard` (cheap)
  first, then `hard` (the new sampler method). Ablate
  `{shuffled, jaccard, hard}` × `k ∈ {1, 2}`. Decide whether contrastive ships
  on by default or stays opt-in — update `configs/methods/soft_tokens.toml` +
  `docs/experimental/soft_tokens.md`.

## Costs to keep honest

- **Each negative = one extra full DiT forward.** Soft tokens' whole appeal is
  being a single forward on a frozen DiT; `k=4` → ~5× step time. Keep `k ∈ {1, 2}`
  through Phases 1–2.
- **SoftREPA's own SD3 FID regression** at paper-strength contrastive
  (`soft_tokens.md:5`). Phase 1's λ=0 vs λ>0 A/B is exactly the check that we
  aren't reproducing it.
- **B=1 false-negative risk.** Within an artist, captions are often correlated
  (same series/characters); a "different-content" negative may genuinely produce a
  similar velocity, corrupting (not softening) the signal. The Jaccard middle path
  degrades more gracefully here than hard sampling — another reason to try it first.

## Risks & open questions

- **Long-tail artists.** If the Phase-0 same-artist/different-character coverage is
  thin, `hard` mode silently falls back to `shuffled` for most steps — measure
  before building it.
- **Multiple / missing artist tags.** Same as the IP-Adapter sampler: define
  "same artist" as set-intersection ≥ 1; treat anonymous/untagged as
  exclude-from-hard (orphan → `shuffled` fallback).
- **Interaction with `n_t_buckets`.** At training scale ≪ dataset scale, a given
  t-bucket may see few artists; per-bucket contrastive signal could become
  artist-biased. Probably second-order; note and revisit if Phase 2 shows it.

## What this does NOT do

- Does not change `batch_size` or the dataloader's batching — negatives come from
  cached-TE swap, not batch peers (the same B=1-safe trick the IP-Adapter pairs
  use).
- Does not touch the soft-token parameterization, splice position, or block hook —
  this is purely an added training objective.
- Does not claim parity with the SoftREPA paper's reported numbers — it is a
  B=1-adapted reconstruction of the contrastive idea, gated on whether it earns
  its extra forwards.

## Reference points

- Module + contrastive loss: `networks/methods/soft_tokens.py`
  (`contrastive_loss`, `step_contrastive_warmup`, `SoftTokensMethodAdapter`),
  `docs/experimental/soft_tokens.md`
- Loss compose site: `library/training/losses.py` (`_soft_tokens_contrastive_loss`)
- Negative sourcing to reuse: `library/datasets/identity_pairs.py`
  (`IdentityPairSampler.resolve` / `.shuffled`), `tests/test_identity_pairs.py`
- Dataset-hook pattern to mirror: `library/datasets/base.py`
  (`setup_identity_pairs`, `_load_ip_features_for_stem`)
- Shared caption index: `post_image_dataset/captions/caption_index.json`
  (`make caption-index`, `preprocess/build_caption_index.py`)
- Sibling proposal (the cached-feature-swap precedent):
  `docs/proposal/ip-adapter-identity-pairs.md`
- Paper: Lee et al., *Aligning Text to Image in Diffusion Models is Easier Than
  You Think* (arXiv:2503.08250, NeurIPS 2025) — "SoftREPA"
- Superseded scratch note: `_archive/soft_tokens_contrastive/soft_token_idea.md`
