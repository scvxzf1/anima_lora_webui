# Anima Tagger — multi-label tagger trained on Anima's caption distribution

A small classifier that maps an image to a comma-separated tag string in
exactly the format Anima's training-time T5 saw. Used as the case-1 ψ_src
provider for DirectEdit (replacing wd-tagger when a checkpoint is present).

Status: **training-ready** (vocabulary + manifest + feature cache builders +
trainer + threshold calibrator + inference wrapper all wired). No trained
checkpoint shipped yet — first run pending.

## Why this exists

DirectEdit's invert/edit primitive is robust to ψ_src corruption — even
shuffled or tag-dropped source captions reconstruct the source image at
~99% pixel fidelity. But edit *leverage* (whether ψ_tar = ψ_src + edit-tag
actually applies the change) collapses when ψ_src is structurally far
from Anima's training-time embedding manifold. wd-tagger output is bad
enough at this to be the live blocker; see
[`docs/proposal/directedit_editing_v2.md`](../proposal/directedit_editing_v2.md)
for the diagnosis.

## Architecture

```
PIL image → PIL LANCZOS-resize to PE-Core bucket size
         → ToTensor + Normalize([0.5], [0.5])  (= [-1, 1])
         → frozen PE-Core-L14-336 → patch tokens [T, 1024]
         → mean-pool over T → feature [1024]
         ──────────────────────────────────────────────────  trunk (frozen)
         → LayerNorm + Linear(1024, 1024) + GELU + Dropout
         ────────────────────────────────────────────────── shared trunk_h
         ├→ Linear(1024, n_tags)       → tag_logits     ──── multi-label head
         └→ Linear(1024, 3)            → rating_logits  ──── 3-class rating head

Per-tag F1-calibrated threshold sweep at the end of training picks the
inference threshold for each output dimension.
```

Total trained params at default `n_tags=4965, d_hidden=1024`: **~6.1M**.
Frozen PE-Core trunk is loaded from existing `models/pe/PE-Core-L14-336.pt`
— shared with the IP-Adapter pipeline.

### Why a shared trunk for both heads

Rating prediction and tag prediction look at the same kinds of visual
content — lots of the rating signal is also expressible as tag co-occurrence.
A shared trunk gives the rating gradient a path into the same representation
the tag head reads from, at the cost of one extra Linear at the head split.
Empirically this is what gelcrawl's quality classifier does too
(`gelcrawl/classify.py`); we reuse that pattern.

### Why mean-pool over patch tokens

PE-Core's CLS token is contrastive-image-text trained — useful for retrieval,
not optimized for multi-label classification. Mean-pool over the patch
tokens gives a content-weighted summary; head capacity is enough that the
pooling choice doesn't bottleneck. Swap to attention pooling only if the
linear probe + MLP-head doesn't saturate at expected F1.

### Why a sqrt(neg/pos) BCE pos-weight

Anima's tag distribution has a heavy long-tail. Default BCE-with-logits
treats every tag-output identically, so common tags (1girl) dominate the
gradient. Inverse-frequency weights (`n_neg/n_pos`) over-correct and
explode rare-tag gradients. `sqrt(n_neg/n_pos)` is the standard middle
ground — softens the long-tail without overshoot.

## Files (under `library/captioning/`)

| File | Role |
|---|---|
| `tag_rules.py` | Load + apply the `tag_rules.yaml` semantics (replacements / always-remove / clothing dedup). Mirror of `gelcrawl/postprocess.py::dedup_tags_in_file`. |
| `anima_tagger_data.py` | `TaggerManifest` (loads `dataset.json`), `FeatureCacheBuilder` (PIL→PE-Core→pool→safetensors per stem), `CachedFeatureDataset` (in-memory tensors for the trainer), and `pil_resize_to_bucket` (LANCZOS→bucket pixel size). |
| `anima_tagger_model.py` | `AnimaTaggerConfig`, `AnimaTaggerHead` (the trunk + two heads). Imported by both trainer and inference wrapper. |
| `anima_tagger.py` | `AnimaTagger` — public inference class. Mirrors the `WDTagger` surface (`predict`, `predict_caption`). Loads checkpoint, encoder, vocab, thresholds, and rules from one directory. |

Trainer and CLI live at `scripts/train_anima_tagger.py`; the env loader
(`library/env.py`) reads paths from `anima_lora/.env`.

## Configuration via `.env`

External corpus paths are routed via `CAPTION_CORPUS_DIR`. Add to
`anima_lora/.env`:

```
CAPTION_CORPUS_DIR=/path/to/external/caption/corpus
```

The corpus is expected to contain:

| Path | What it is | How it's consumed |
|---|---|---|
| `<corpus>/retrieved/{artist}/{stem}.{webp,jpg,png,jpeg}` | Source images, paired with `.txt` captions | Training input (image) + label (caption). 12k+ images. |
| `<corpus>/retrieved/{artist}/{stem}.txt` | Booru-style caption per image, in Anima format (`rating, count, characters, copyrights, @artists, generals`) | Multi-hot label after `tag_rules` normalization |
| `<corpus>/retrieved/.tag_cache.json` | `tag → integer type id` (0=general, 1=artist, 3=copyright, 4=character, 5=metadata, 6=deprecated) | Vocab categorization + canonical-emit-slot routing |
| `<corpus>/tag_rules.yaml` | Replacements + always-remove + clothing dedup map | Applied at vocab-build time and snapshotted into the checkpoint dir for inference |
| `<corpus>/selected/` (optional) | Curated subset (already deduped) | Additional caption source; folded into the vocab scan |

`image_dataset/` (Anima's training set, 2.6k curated images already
preprocessed) is also scanned by default.

`CAPTION_CORPUS_DIR` is **not committed** — it's per-user. The trained
checkpoint snapshots `tag_rules.yaml` into `models/captioners/anima-tagger-v1/`
so inference has zero runtime dependency on the corpus dir.

## Training procedure

Four stages, each runnable independently:

```bash
# 1. Build the vocabulary + train/val split + per-stem dataset manifest.
python scripts/train_anima_tagger.py --mode build_vocab --min_freq 5

# 2. Encode every manifest image through PE-Core, mean-pool, cache to disk.
#    Idempotent — re-runnable on partial caches.
python scripts/train_anima_tagger.py --mode build_features

# 3. Train the head on cached features. ~50 MB total in VRAM, fast.
python scripts/train_anima_tagger.py --mode train --epochs 30

# 4. Sweep per-tag F1-optimal thresholds on the val split.
python scripts/train_anima_tagger.py --mode calibrate
```

Each mode's outputs go to `--out_dir` (default
`models/captioners/anima-tagger-v1/`). After all four:

```
models/captioners/anima-tagger-v1/
├── vocab.json              # tag list with categories + median emit position
├── rules.yaml              # snapshot of tag_rules.yaml at vocab-build time
├── dataset.json            # per-stem (image_path, multi_hot, rating) manifest
├── .cache/pooled-pe/       # per-stem [d_enc] safetensors
├── model.safetensors       # trained AnimaTaggerHead state_dict
├── config.json             # model + training metadata
├── thresholds.safetensors  # per-tag F1-calibrated thresholds + val_f1
└── train_history.json      # per-epoch loss + val metrics
```

The checkpoint dir is **fully self-contained** — moveable across machines,
no runtime dependency on `CAPTION_CORPUS_DIR`.

### Stage 1 outputs (this is what we have on disk now)

```
caption stems indexed:  12,951
unique tags seen:       12,939
vocab size (≥5):        4,965
dropped (low-freq):     7,974
cache hit rate:         98.83%   # tags categorized via .tag_cache.json
category counts:
  general    3,804
  character    722
  copyright    268
  artist        89
  deprecated    66
  count         15
  metadata       1
rating coverage:        99.94%
rating distribution:    explicit=8742  sensitive=4128  general=73
split:                  12,303 train / 648 val
trainable samples:      12,907   # captions with sibling images + valid rating
```

The 0.52% miss rate on tag categorization is dominated by apostrophe-vs-`&#039;`
variants in tag names (`grabbing another's breast`, `girls' frontline`); fall-back
to "general" is harmless. A future refinement could apply the
`&#039; → '` replacement to cache keys at load time.

### Stage 2 — feature cache

Each cached file is a single safetensors tensor `feature [d_enc=1024]` in
float32, ~4 KB per stem, ~50 MB total for the full set. Cold throughput on
a single GPU is ~5 img/s with PIL LANCZOS pre-resize to bucket size; full
build ~30–40 minutes. The pre-resize step is in
`pil_resize_to_bucket()` and matters — without it, multi-megapixel source
images get bilinear-resized inside the encoder, costing 2× throughput and
arguably some quality on severe downscales.

### Stage 3 — training knobs

| Flag | Default | Notes |
|---|---|---|
| `--epochs` | 30 | Cosine LR schedule from `--lr` to `--lr * 0.05` |
| `--batch_size` | 256 | Whole train set fits in one batch on most GPUs; small batch is just for SGD noise |
| `--lr` | 1e-3 | AdamW |
| `--weight_decay` | 0.01 | AdamW |
| `--d_hidden` | 1024 | Trunk hidden dim |
| `--dropout` | 0.1 | After GELU |
| `--lambda_rating` | 0.1 | Weight on rating CE relative to tag BCE |
| `--seed` | 42 | Permutation seed; eval split is set at vocab-build time |

All training/val tensors live in VRAM after one push (~50 MB) — no per-step
dataloader.

Loss formulation:
```
L = BCE_pos_weighted(tag_logits, multi_hot)
  + λ_rating · CE_class_weighted(rating_logits, rating_idx)

pos_weight = sqrt(n_neg / n_pos)   # per tag
class_weight = inv_freq normalized  # 3-class rating
```

### Stage 4 — calibration

`_calibrate_thresholds()` sweeps thresholds in [0.05, 0.95] step 0.05 per
tag and picks the F1-maximizing one on the val split. Tags with no positive
val examples or zero achievable F1 keep `default=0.5`. Tag-block size of
256 caps memory.

Calibrated thresholds are critical — a global 0.35 (wd-tagger's inheritance)
under-fires rare tags and over-fires common ones. Empirically tags spread
to thresholds in roughly [0.05, 0.65] with a long mode near 0.20.

## Inference

```python
from library.captioning import AnimaTagger
from PIL import Image

tagger = AnimaTagger("models/captioners/anima-tagger-v1")
caption = tagger.predict_caption(Image.open("foo.png"))
# → "sensitive, 1girl, hatsune miku, vocaloid, @some_artist, blue eyes, ..."

debug = tagger.predict(Image.open("foo.png"))
# → {"rating": "sensitive", "rating_scores": {...}, "scores": {...}, "kept": {...}}
```

The `predict_caption` emit pipeline:

1. PIL → bucket-resize → ToTensor+Normalize → frozen PE-Core → mean-pool → trunk → tag_logits + rating_logits
2. `sigmoid(tag_logits) ≥ thresholds` → kept tags; `argmax(rating_logits)` → rating
3. Slot kept tags by canonical category order: `rating, count, character, copyright, @artist, general`
4. Within each slot, sort by `median_pos` (tag's median position in training captions) — deterministic, mirrors how the corpus orders tags
5. Re-apply `tag_rules` as a safety net (the dedup rule was already enforced at training-data normalization, but the model could in principle predict both `bra` and `black bra`)
6. `_underscore_to_space` on every tag, comma-join

The per-tag thresholds were the missing piece in wd-tagger — the global
0.35 there leaks too many low-confidence false positives and drops too
many low-prevalence true positives.

## Wired-up touchpoints

`scripts/experimental_tasks/inference.py::cmd_test_directedit` chooses
between `WDTagger` and `AnimaTagger` via the `TAGGER` env var:

```bash
make exp-test-directedit PROMPT='glasses'                       # auto: anima if checkpoint present, else wd
TAGGER=anima make exp-test-directedit PROMPT='glasses'          # force anima
TAGGER=wd make exp-test-directedit PROMPT='glasses'             # force wd
```

The auto-detect checks for `models/captioners/anima-tagger-v1/model.safetensors`.
If the user requests `TAGGER=anima` but the checkpoint is missing, the driver
falls back to wd-tagger with a warning rather than crashing.

`scripts/edit.py` itself doesn't tag — it takes `--prompt_src` directly.
The driver script is the only place tag-source selection happens.

## Known limitations

1. **Tag-cache apostrophe miss.** Top-20 uncategorized tags all contain
   apostrophes (`grabbing another's breast` etc.) — the cache keys used
   `&#039;` instead. Affects 0.52% of tag occurrences, all fall back to
   "general" category. Cheap fix, not yet applied.
2. **Per-tag positives are thin for the long tail.** At `min_freq=5` and
   12k training images, each long-tail tag has 5–20 positives. Calibrated
   thresholds for those tags are noisier than for high-frequency ones.
   `--min_freq 10` is a knob to revisit if F1 disappoints.
3. **Heavy rating-class imbalance.** 8,742 explicit / 4,128 sensitive /
   73 general (0.6%). The rating head will struggle on `general`. Class-weighted
   CE compensates partially. If `general`-classification accuracy matters
   for downstream consumers, consider over-sampling at training time.
4. **No bench harness yet.** `bench/anima_tagger/` per the standard envelope
   (cf. `bench/_common.py::write_result`) is the next thing to add — should
   compare F1 vs `WDTagger` on a shared held-out set, plus a downstream
   "edit-success-rate" metric on a small DirectEdit set.

## Open design questions

1. **Should we also produce embedding output?** Right now `predict_caption`
   emits a string that gets re-tokenized by T5. We could add a head that
   directly produces `[K, D_t5]` continuous tokens — but that's the
   img2emb design (`docs/proposal/img2emb_plan.md`) and brings back the
   structural challenges that the proposal documents. Stick with tag-string
   output for v1.
2. **Should we use DINOv3 instead of PE-Core?** gelcrawl's `classify.py`
   uses DINOv3 ViT-L/16@224 and works well in this domain. PE-Core gives
   us a pre-existing cache (IP-Adapter pipeline). If F1 saturates and we
   suspect the trunk is the limit, swap encoders — `--encoder` flag
   already plumbs through.
3. **How aggressive should the dedup rule reapplication be at emit time?**
   Currently `apply_rules` runs over the full predicted tag set after slot
   ordering. If a user wants to keep both base and color variant ("show me
   *bra* AND *black bra*"), they'd need to opt out. Probably no one wants
   this — leave on by default.
