# postfix_residual (per-image inversion) — DirectEdit's ψ_src residual without an encoder

Sibling proposal to [`postfix_residual_for_directedit.md`](postfix_residual_for_directedit.md).
Same decomposition `ψ_src = T5(tags) + postfix(image)`, different supply
mechanism: instead of training an amortized `image → postfix` encoder, optimize
**K orthogonally-constrained tail tokens per image** against the frozen DiT.
The tags prefix stays frozen and editable; only the residual tail is solved
for.

## TL;DR

```
prefix = T5(tags)           # frozen, 256 tokens (or whatever the active length is)
tail   = Q @ S              # K=48 trainable slots, Q is a fixed orthogonal basis
ψ_src  = concat(prefix, tail)
optimize tail against L_FM(image, ψ_src)     # per-image, ~30–90 s on a 5060 Ti
                                              # (depending on K, steps, grad_accum)
ψ_tar  = concat(T5(tags_edited), tail)        # text edit; tail pinned across src→tar
```

No training pipeline, no encoder to design, no lane-overgrowth failure mode in
the amortized sense (the ortho parameterization caps capacity structurally),
and DirectEdit's spirit is preserved — this *is* inversion, just scoped.

## Why this exists alongside the encoder proposal

The companion proposal argues for an amortized
`image → postfix` encoder trained against FM loss + an auxiliary
`L_directedit_dry`. That's the right long-term shape, but it imports three
risks the per-image path sidesteps:

1. **Lane overgrowth.** An MLP fed pooled PE features can learn to encode
   tag-redundant content; ortho-basis × K=48 trainable scalars caps capacity
   structurally.
2. **PE-input re-validation of the slot-collapse fix.** The cond+ortho LN +
   maxabs fix in `project_postfix_ortho_ln_fix` was validated on pooled-T5
   input; PE features have different statistics
   (`project_pe_feature_diagnostics`) and the fix may need re-tuning. Per-image
   inversion bypasses the cond_mlp entirely — only the ortho parameterization
   has to port.
3. **FM-from-noise as the wrong supervision signal.** The companion proposal
   itself flags this: classical inversion is the hard problem DirectEdit was
   built to avoid. Per-image inversion *is* that hard problem, but solved
   directly per image with cheap optimization rather than amortized into an
   encoder that has to generalize. And — see "Roadmap" — the per-image
   solutions then become the *right* supervision target for the encoder
   version.

The honest cost: per-image inversion pays optimization at first-load time.
~30–90 s on a 5060 Ti with `--steps 100`, `--grad_accum 4`, `--blocks_to_swap
0` (estimate; needs measurement). Amortized encoder is ~0 ms after training.
For interactive DirectEdit this is the central tradeoff.

## What this is

- A per-image optimization that produces a `(K, D)` postfix tail bound to one
  source image, saved as a `.safetensors` sidecar (e.g.
  `{stem}_anima_postfix_tail.safetensors`).
- Loaded by DirectEdit alongside `{stem}_anima_te.safetensors` and spliced
  into the cross-attn sequence at inference time. Pinned across the src→tar
  edit; only the text prefix changes.
- Image-only by construction (the optimization target is the image; tags only
  set the frozen prefix).

## What this isn't

- **Not classical embedding inversion** (`archive/inversion/invert_embedding.py`).
  That optimizes the *entire* 512-length sequence (or the leading
  `active_length` of it) and yields a ψ that has no text-editable
  decomposition — the editing UI is gone. Per-image *tail* inversion freezes
  the prefix so the tags channel survives.
- **Not the amortized encoder** in `postfix_residual_for_directedit.md`. That
  produces `image → K×D` for any image in O(ms). This produces a tail for
  *one specific image* in O(seconds-to-minutes). The two compose
  (see Roadmap).
- **Not IP-Adapter.** Same conditioning path as the rest of the postfix
  family — appends into the cross-attn sequence, no parallel KV branch.

## Architecture

Reuse the existing `cond+ortho` postfix carrier structure, but **drop the
cond_mlp** (no image features feed it; the trainable parameters *are* the
tail). The ortho parameterization stays:

```
tail = ortho_basis @ S               # ortho_basis: (K, D) frozen
                                     # S:           (K, K) or (K,) trainable
```

Either the full `Q @ S` form (Stiefel-manifold-ish; S is the only trainable
piece, slots stay decorrelated by construction) or the simpler `Q · diag(s)`
form (s is a K-vector of per-slot scales, basis frozen). The simpler form has
fewer parameters (K vs K²) and is the natural first cut — measure
reconstruction ceiling against the richer parameterization in a later sweep.

Splice into the cross-attn sequence via the existing
`end_of_sequence` / `front_of_padding` paths in
`networks/methods/postfix.py` — no new splice strategy. The optimization
target is just the values of `S` (or `s`).

## How to dig up the archived inversion scripts

Per `CLAUDE.md`: "Embedding inversion (archived — run scripts directly under
`archive/inversion/`)". Everything you need to bootstrap this proposal lives
there. Map of what's relevant:

| Archived file | What to lift | What to change |
|---------------|--------------|----------------|
| `archive/inversion/invert_embedding.py` | Full optimization loop: cosine LR schedule, FM-loss forward, `--grad_accum`, `--timesteps_per_step`, `--sigma_sampling` (uniform/sigmoid), `--sigma_min` (P-GRAFT-analog low-σ skip), VAE-on-the-fly vs cached-latent modes, `--init_from_cache` (loads `_anima_te.safetensors` into the trainable embedding), `--blocks_to_swap` plumbing, `--verify` (generates an image from the inverted embedding after optimization) | Drop `--active_length` (we keep the full prefix frozen; only the *tail K slots* are trainable). Drop `--init_jitter_std` for the prefix region (still useful for the tail, since the ortho basis is the only symmetry breaker). Drop the Hungarian token-alignment aggregation (`--aggregate_by`) on v0 — single run suffices; revisit if multi-run averaging closes more gap. |
| `archive/inversion/invert_reference.py` | Reference-image inversion variant — has a different init path worth reading | Not needed for v0 but useful to compare init schemes |
| `archive/inversion/interpret_inversion.py` | Post-hoc analysis of inverted embeddings (token statistics, etc.) | Useful for the lane-overgrowth diagnostic in the validation plan |
| `docs/methods/invert.md` | Conceptual docs on embedding inversion | Read first — gives you the prior art's framing |

**Practical recovery path** (one option; pick whichever fits your style):

```bash
# 1. Read the prior art end-to-end first
cat archive/inversion/invert_embedding.py | less
cat docs/methods/invert.md

# 2. Spike a new script that reuses the optimization loop
cp archive/inversion/invert_embedding.py scripts/invert_postfix_tail.py
# edit: replace `embedding = nn.Parameter(...)` with the Q @ S form,
#       slice prefix vs tail, freeze prefix, only optimize S
#       (or factor the loop out of the archive script into
#       library/inference/inversion.py if you want to share with future
#       inversion work — the archive scripts already include
#       sys.path.insert(0, ...) hacks, so a clean home in library/ is
#       worth the small refactor up-front)
```

The archived scripts import from `library/` heavily already
(`library.anima.*`, `library.io.cache`, `library.runtime.device`,
`library.datasets.image_utils`), so they're not stale code — just not wired
into the `make` surface. Reviving the relevant primitives into
`library/inference/inversion.py` (or similar) and re-importing them from a
new `scripts/invert_postfix_tail.py` is the cleanest move. The archive copies
stay as reference.

## Optimization

```
L = L_FM (frozen DiT, ψ = concat(prefix, tail), random t per microbatch)
  + λ_zero · ‖tail‖²           # keep tail small — only encode what tags can't carry
  + λ_ortho · ‖S^T S − I‖_F²   # only if using a parameterization that doesn't enforce
                               # orthogonality structurally (i.e. plain S, not Q @ S)
```

Hyperparameters worth porting from `invert_embedding.py` defaults and then
sweeping:

- `--steps 100` (start; up to 500 if reconstruction stalls)
- `--lr 0.01`, `--lr_schedule cosine`
- `--grad_accum 4`, `--timesteps_per_step 1` (total 4 t samples per update)
- `--sigma_sampling uniform`, `--sigma_min 0.0` initially — the P-GRAFT-style
  `sigma_min > 0` skip is *more* relevant here than for full-embedding
  inversion because the tail is small and easily drowned out by the prefix
  at low σ. Sweep `sigma_min ∈ {0, 0.1, 0.2}`.
- `--init_jitter_std 0.149` for the tail's S init (same statistic the
  archived script uses for the active region).

What the archive script does that we should NOT do:

- `--active_length 256` style hard-zero of the tail region during
  optimization. We *want* the tail to fill that region with the ortho basis
  output; zeroing it defeats the purpose.

What we add that the archive doesn't have:

- **Frozen prefix** loaded from `{stem}_anima_te.safetensors` and not
  optimized. The archive's `--init_from_cache` initializes from cache but
  then optimizes the whole thing; here, only the tail slots are leaf
  parameters.
- **Ortho parameterization on the tail** (see Architecture).

## Failure modes

1. **FM val loss doesn't track quality** (`project_fm_val_loss_uninformative`).
   You'll see FM loss plummet but the dry-run reconstruction quality may not
   improve monotonically. **Early-stop on dry-run LPIPS / DreamSim**, not on
   FM. The validation plan below pins this.
2. **Tail saturates to a tag-redundant codebook.** If the prefix carries the
   tag content and the tail can also carry it, the tail will because FM
   doesn't penalize redundancy. λ_zero is the lever. Diagnostic: with
   `ψ = tail only` (zero out the prefix), reconstruction should *fail*. If
   it succeeds, λ_zero is too low.
3. **Ortho basis doesn't actually break slot symmetry under per-image
   optimization.** The companion proposal flagged that the LN + maxabs fix
   was validated on T5-input cond_mlp; per-image inversion doesn't go
   through cond_mlp at all, so the failure mode is structurally different —
   but it's worth running the same single-batch diagnostic
   (`project_postfix_slot_collapse`: check cross-image cos similarity of the
   tail values across different source images). Different images should
   converge to *different* tails. If they all converge to the same tail,
   something is wrong with the optimization (likely the ortho basis is
   getting zeroed and only the global mean of S is driving the loss).
4. **High per-image variance.** Two inversions of the same image with
   different seeds may produce visually-equivalent but structurally-different
   tails (the inversion problem is many-to-one). For DirectEdit's purposes
   this is fine as long as both reconstruct, but it complicates *distilling*
   into an encoder (see Roadmap) — the targets aren't unique. The archived
   `--aggregate_by` Hungarian alignment was designed for exactly this case;
   keep it in the back pocket for the distillation phase.

## Roadmap — per-image inversion as the encoder's supervision target

The strongest reason to do this before the encoder version is **the encoder
needs better supervision than FM loss**. Concretely:

1. **Phase A (this proposal).** Per-image invert tails for the
   training set. Validate the decomposition works in principle: dry-run
   reconstruction improves vs `T5(tags)` alone, edits stay coherent, tail
   doesn't overgrow.
2. **Phase B (companion proposal, but with a real target).** Train the
   amortized `image → tail` encoder against the **inverted tails from Phase
   A**, not against FM loss from noise. Loss is L2 (or cosine) between
   encoder output and the inverted target per image — i.e. distillation.
   This is dramatically better-conditioned than FM-from-noise because the
   targets are known-good (they already close the dry-run gap).
3. **Phase C (optional).** Mix: encoder predicts the initial tail, per-image
   inversion fine-tunes for K_refine steps starting from the prediction.
   Caps inference cost at "encoder forward + small refinement" while
   recovering the per-image fidelity headroom. Knob for users: "fast load"
   (encoder only) vs "fidelity" (refine).

Phase A is also the right place to **bound the achievable ceiling** of the
whole `ψ_src = T5(tags) + postfix(image)` decomposition. If per-image
inversion can't close the dry-run gap, the encoder version definitely can't,
and the proposal's whole framing needs rethinking before more engineering.

## Implementation steps

1. **Decide the parameterization.** `Q @ diag(s)` (K params) vs `Q @ S`
   (K² params) vs raw `S` + soft ortho penalty. Recommend `Q @ diag(s)` for
   v0 — fewest params, structural orthogonality, simplest checkpoint format.
2. **Lift the optimization loop** out of
   `archive/inversion/invert_embedding.py` into `library/inference/inversion.py`
   (or a fresh `library/inference/postfix_inversion.py`). Or, if "lift" feels
   premature, just copy the script into `scripts/invert_postfix_tail.py` and
   refactor when there's a second consumer.
3. **Splice path.** Reuse `networks/methods/postfix.py`'s
   `end_of_sequence` / `front_of_padding` mechanism for inference. The
   training script just needs to construct the same `crossattn_emb` layout
   the DiT forward expects.
4. **CLI** (`scripts/invert_postfix_tail.py`):

   ```
   --image <path> | --image_dir <path>      # source image(s)
   --dit ...                                # frozen DiT
   --te_cache_path <stem>_anima_te.safetensors   # prefix init
   --K 48                                   # tail length
   --ortho_basis svd_te | random            # same options as postfix.py
   --ortho_basis_seed 0
   --steps 100 --lr 0.01 --grad_accum 4
   --lambda_zero 0.01
   --sigma_min 0.0                          # consider 0.1
   --output_dir output/inversions/postfix_tails/
   --verify                                 # generate sample from inverted tail
   ```

5. **DirectEdit integration.** Add a `--postfix_tail <path>` flag to
   `scripts/edit.py` and the equivalent socket on the ComfyUI node
   (`custom_nodes/comfyui-anima-directedit/`). Loader splices the tail into
   `ψ_src` and `ψ_tar` identically (image-pinned across the edit).
6. **Bench.** New `bench/postfix_tail_inversion/` directory using
   `bench/_common.py`. Metrics mirror the companion proposal's validation
   plan plus inversion cost:
   - dry-run LPIPS / DreamSim (goal metric)
   - edit fidelity (10 curated edits)
   - tail-only reconstruction must fail (lane-overgrowth guard)
   - inversion wall-clock per image (cost metric — gates Phase B priority)

## Validation plan

| Question | How to answer |
|----------|---------------|
| Does dry-run reconstruction actually improve? | LPIPS / DreamSim on 20 held-out images, ψ_src = ψ_tar = T5(tags) + inverted_tail. Target: noticeably lower than `T5(tags)` alone. |
| Does the tail stay in its lane? | Tail-only reconstruction (`ψ = inverted_tail`, prefix zeroed) must fail. |
| Are edits still coherent? | 10 manually-curated edits (`sad→smile`, `day→night`, etc.) vs tag-only DirectEdit. |
| Per-image inversion cost? | Wall-clock at the default (`--steps 100 --grad_accum 4`) and at a "fast" preset (`--steps 30`). Decides Phase B / C urgency. |
| Are tails unique per image? | Cross-image cosine of inverted tails. If all images converge to the same tail, ortho basis isn't doing its job. |
| Multi-seed stability? | Re-invert the same image with 3 seeds. Functional cosine (probe with `--probe_functional` from the archived script) should be high even if raw tails differ. |
| Are tails *distillable*? | Phase B prerequisite — train a tiny encoder against the inverted targets on 200 images, check held-out L2 generalization. If high, the encoder version is viable; if not, per-image stays as-is. |

## Relationship to the other proposals

- **`postfix_residual_for_directedit.md`**: the amortized-encoder sibling.
  This proposal produces the supervision target the encoder ultimately wants.
  Run this first.
- **`img2emb_plan.md`**: a *replacement* for text conditioning, trained
  against FM from noise. Different goal (no text channel). This proposal's
  framework — invert with a frozen prefix + small ortho tail — could in
  principle be applied to img2emb too (invert the whole sequence as a
  prefix), but that's just classical inversion and `archive/inversion/`
  already does it.
- **`orthogonal_postfix.md`**: provides the ortho-basis machinery this
  proposal reuses. The slot-collapse fix that motivated cond+ortho's LN +
  maxabs path doesn't apply here (no cond_mlp), but the structural
  orthogonality of the basis itself does, and that's the part this proposal
  depends on.

## Why this is the right v0

- **No training infrastructure to build.** Single script, runs on one GPU,
  no dataloader, no preset, no checkpoint management beyond saving
  `.safetensors` per image.
- **Free slot-collapse fix.** Q @ S structurally orthogonal — no LN tuning,
  no maxabs path to debug.
- **No new failure mode invented.** Lane overgrowth still exists as a
  concept, but with K=48 constrained to an ortho basis and λ_zero on the
  scales, it's much harder to trigger than with an unconstrained MLP.
- **Bounds the ceiling.** If this doesn't work, neither does the encoder
  version. Cheap negative-result path.
- **Becomes the supervisor for the encoder.** If it does work, the
  Phase-B distillation path is dramatically better-conditioned than
  FM-from-noise training would be on its own.

Cost: per-image optimization at first-load time. For a tool meant to support
interactive editing, this is the central downside, and the Phase-C "encoder
warm start + small refinement" path is the natural mitigation. But until we
know the ceiling, paying that cost is the right tradeoff.
