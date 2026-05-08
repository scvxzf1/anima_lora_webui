# DirectEdit editing v3 — what changed in the implementation phase

Successor to [`docs/proposal/directedit_editing_v2.md`](../proposal/directedit_editing_v2.md).
v2 was the proposal; v3 documents what actually got built. Lives under
`docs/experimental/` (not `docs/proposal/`) because the components below
are wired and runnable, not speculative.

Companion: [`anima_tagger.md`](./anima_tagger.md) — the tagger architecture
and training procedure.

## What v2 picked

v2 surveyed five paths for the case-1 ψ_src problem (external image,
no recorded prompt) and recommended phasing in three steps:

* **v2.0** — train a custom tagger on Anima's caption distribution,
  drop wd-tagger
* **v2.1** — wire embedding inversion as an opt-in "premium" fallback
* **v2.2** — reconsider full img2emb only if v2.0 + v2.1 don't close the gap

This document covers **v2.0 implementation** (now phase v3.0 of the editing
pipeline) plus the integration glue that landed alongside it.

## What v3.0 actually built

Five concrete deliverables, all on disk:

### 1. `library/captioning/anima_tagger.py` — drop-in tagger

`AnimaTagger` mirrors `WDTagger`'s public surface (`predict`,
`predict_caption`) so the swap is import-only. See
[`anima_tagger.md`](./anima_tagger.md) for the full architecture.

Key fact for v3: **the inference wrapper is functional even before any
training** because the auto-detect in the edit driver falls back to
wd-tagger when the checkpoint is missing. Editing pipeline doesn't break
in either state.

### 2. `library/captioning/tag_rules.py` — caption normalization

Pure-Python port of the tag-rules semantics used to curate Anima's
training captions. Replacements (`questionable → sensitive`,
`&#039; → '`, alias rewrites), always-remove list, and clothing-base
dedup map (e.g. drop `bra` when `black bra` is present).

Used in three places:

* **Vocab build:** parse caption files into clean tag lists
* **Inference emit:** safety net before the tagger's output goes downstream
* **Inference snapshot:** the tagger's checkpoint dir holds a frozen copy
  of the rules used at vocab-build time, so the tagger's behavior doesn't
  drift when the source `tag_rules.yaml` is updated

### 3. `library/env.py` — minimal `.env` loader

50 lines, no python-dotenv dep. Reads `KEY=VALUE` lines from
`anima_lora/.env` into `os.environ` with no override of pre-existing
env vars. Called once at the top of `scripts/train_anima_tagger.py`
before argparse runs.

This is the indirection that hides the external corpus path from the
codebase. The repo references `CAPTION_CORPUS_DIR`; the user binds it
to whatever directory they keep crawled images in. No corpus-specific
naming in code, comments, or commit history.

### 4. `scripts/train_anima_tagger.py` — four-stage pipeline

| `--mode` | Purpose | Inputs | Outputs |
|---|---|---|---|
| `build_vocab` | Scan caption sources, intersect with tag taxonomy, snapshot rules | `$CAPTION_CORPUS_DIR/{retrieved,selected,...}` + `tag_cache.json` + `tag_rules.yaml` | `vocab.json`, `rules.yaml`, `dataset.json` |
| `build_features` | Encode each manifest image through frozen PE-Core, mean-pool, cache | `dataset.json` + PE-Core weights | `.cache/pooled-pe/{stem}.safetensors` |
| `train` | Train trunk + tag head + rating head on cached features | Cached features + manifest | `model.safetensors`, `config.json`, `train_history.json` |
| `calibrate` | Per-tag F1-optimal threshold sweep on val | Trained model + val features | `thresholds.safetensors` |

All four stages live in one script so the implementation stays in one
file; CLI selects via `--mode`.

### 5. `scripts/experimental_tasks/inference.py` — TAGGER env var

The `make exp-test-directedit` driver now respects `TAGGER ∈ {anima, wd, auto}`
(default `auto`). Auto-detects based on checkpoint presence. Fallback
on missing-checkpoint is loud (warns to stderr) but non-fatal.

```bash
make exp-test-directedit PROMPT='glasses'                       # auto
TAGGER=anima make exp-test-directedit PROMPT='glasses'          # force anima
TAGGER=wd make exp-test-directedit PROMPT='glasses'             # force wd
```

## Pipeline diagram (case-1, external image)

```
                  external image
                        │
                        ▼
           ┌──────────────────────────┐
           │  AnimaTagger.predict_caption()  │
           │   PIL → bucket-resize → PE-Core │
           │   → trunk → tag/rating heads    │
           │   → per-tag threshold filter    │
           │   → canonical caption format    │
           └────────────┬─────────────┘
                        │     ψ_src ≈ "1girl, smile, school uniform, ..."
                        ▼
           ┌──────────────────────────┐
           │  scripts/edit.py         │
           │   (DirectEdit invert+edit)      │
           │   ψ_src → invert; ψ_tar → edit  │
           └────────────┬─────────────┘
                        ▼
                  edited image
```

For case 3 (Anima-generated, prompt recorded) and case 2 (Anima-generated,
ComfyUI metadata): no tagging needed — the recorded ψ_src goes straight
into DirectEdit. v3 doesn't change those paths.

## What hasn't shipped yet (deferred from v2)

Two pieces from v2 are still pending:

### v2.1 — embedding inversion as a premium fallback

`archive/inversion/invert_embedding.py` already does per-image gradient
descent on ψ_src to minimize FM loss through the frozen DiT. Ground-truth
quality at the cost of minutes per image. Not yet wired into the edit
pipeline. Intended use: a `--ψ_src_mode invert` flag in `scripts/edit.py`
for users willing to wait for max fidelity.

Move-out-of-archive needed; otherwise no new code.

### v2.2 — img2emb

Defers per the v2 plan to "only if v3.0 (this) plus v2.1 don't cover the
use cases." Has its own design doc in
[`docs/proposal/img2emb_plan.md`](../proposal/img2emb_plan.md). v3.0 was
explicitly built first to avoid the failure mode the archived img2emb hit
— solving the cheap problem (Anima-distribution vocabulary) before
attempting the hard one (manifold-correct continuous embeddings).

### Other known gaps

* **No bench harness.** `bench/anima_tagger/` per the standard envelope is
  the next add. Will compare F1 against `WDTagger` on a shared held-out
  set, plus a downstream eyeball edit-success-rate on a fixed DirectEdit set.
* **No trained checkpoint shipped.** Vocab build, manifest, and partial
  feature cache are on disk. Training run pending.
* **Rating-class imbalance.** 0.6% `general` ratings will yield poor
  rating-head F1 on that class. Class-weighted CE partly compensates.
  If downstream consumers care about `general`-rating accuracy, oversample.

## How to validate the v3.0 ship

The single test that matters:

```bash
# After feature cache + train + calibrate complete:
TAGGER=anima make exp-test-directedit PROMPT='double peace'
TAGGER=wd    make exp-test-directedit PROMPT='double peace'
```

Compare both outputs visually for the same source image. If `TAGGER=anima`
produces edits that better isolate the requested change while preserving
non-edited regions, v3.0 is shipped.

If both fail, the issue is *not* tagger vocabulary and we move directly
to v2.1 (embedding inversion). If `TAGGER=wd` is competitive, the gain
isn't worth the training pipeline's existence — but the testing data
that motivated v2 in the first place suggests this won't happen.

## Files touched in this iteration

```
.env                                               +1 line  CAPTION_CORPUS_DIR=
library/env.py                                     +60      .env loader
library/captioning/__init__.py                     ~       expose AnimaTagger
library/captioning/tag_rules.py                    +110    rule loader/applier
library/captioning/anima_tagger.py                 +210    inference wrapper
library/captioning/anima_tagger_data.py            +220    manifest+cache+dataset
library/captioning/anima_tagger_model.py           +60     head architecture
scripts/train_anima_tagger.py                      +600    4-mode CLI
scripts/experimental_tasks/inference.py            ~30     TAGGER env var
docs/experimental/anima_tagger.md                  new     architecture+procedure
docs/experimental/directedit_editing_v3.md         new     this file
models/captioners/anima-tagger-v1/                 new     vocab+manifest+rules+(partial)cache
```

Total net: ~1,300 LoC of code + 2 docs. Reuses existing `library/vision/`
encoder + bucket utilities; doesn't touch DiT/training/inference paths.

## Why this approach was right (in hindsight)

The v2 proposal made the case for trying the cheap fix (vocabulary
distribution) before the speculative fix (full img2emb). Three things
during implementation reinforced that:

1. **gelcrawl already had the components we needed.** The tag taxonomy
   (`.tag_cache.json`), the rules YAML, and the canonical caption format
   were all already-debugged artifacts. We saved weeks of "what's the
   right output format" experimentation by adopting them.
2. **The architectural pattern was already validated.** `gelcrawl/classify.py`
   does frozen-encoder + MLP head + cached features for a different task
   (quality classifier). We reused the pattern.
3. **PE-Core caches existed.** The IP-Adapter pipeline already produces
   `{stem}_anima_pe.safetensors`. For the curated set we got cache
   coverage for free; for the larger crawled set we paid the build cost
   once.

If v3.0 ships and edit-success rate is good, v2.2 (img2emb) doesn't need
to happen — phase complete. If v3.0 ships and edit-success is bad on a
specific class of content, we have a much sharper problem statement to
hand to v2.2 ("img2emb specifically needs to handle X") than the
archived design started with.
