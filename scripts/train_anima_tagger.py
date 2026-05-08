"""Anima multi-label tagger — vocab build, train, calibrate, infer.

Companion to ``library/captioning/anima_tagger.py`` (inference wrapper). This
script is the source of truth for *training-time* logic; it produces a
self-contained checkpoint dir that the inference wrapper consumes with zero
runtime dependency on the external corpus that supplied the labels.

Subcommands (selected by ``--mode``):

* ``build_vocab`` — scan caption sources, intersect with the tag-taxonomy
  cache, snapshot ``tag_rules.yaml``, emit ``vocab.json`` plus a fixed
  train/val split and a per-stem ``dataset.json`` manifest.
* ``build_features`` — encode every manifest image through frozen PE-Core,
  mean-pool over patch tokens, write ``out_dir/.cache/pooled-pe/{stem}.safetensors``.
  Idempotent: skips entries already cached.
* ``train`` — train the multi-label head + 3-class rating head on cached
  PE features. (Wired in a follow-up step.)
* ``calibrate`` — sweep per-tag thresholds on the val split. (follow-up.)
* ``predict`` — single-image debug entry. (follow-up.)

The ``build_vocab`` mode is intentionally standalone so we can eyeball the
emitted vocabulary before committing to the trainer architecture.

External-corpus paths are resolved via the ``CAPTION_CORPUS_DIR`` env var
(typically set in ``anima_lora/.env``). The corpus directory is expected to
contain ``retrieved/`` (raw caption pool), ``selected/`` (curated subset),
``tag_rules.yaml`` (caption normalization rules), and ``.tag_cache.json``
(per-tag Booru-style category cache, indexed under ``retrieved/``). All of
these can be overridden individually by CLI flags.

Usage:

    # one-time: add CAPTION_CORPUS_DIR=/path/to/corpus to anima_lora/.env
    python scripts/train_anima_tagger.py \
        --mode build_vocab \
        --out_dir models/captioners/anima-tagger-v1 \
        --min_freq 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

# Make ``anima_lora/`` importable when this script is invoked as
# ``python scripts/train_anima_tagger.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library.captioning import tag_rules as tr  # noqa: E402
from library.env import load_dotenv  # noqa: E402
from library.log import setup_logging  # noqa: E402

# Pull CAPTION_CORPUS_DIR (and any other overrides) from anima_lora/.env
# before argparse builds defaults. CLI flags still win over env values.
load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)

# Booru-style tag-type integers from the corpus's tag-taxonomy cache.
TAG_TYPE_NAMES: Dict[int, str] = {
    0: "general",
    1: "artist",
    3: "copyright",
    4: "character",
    5: "metadata",
    6: "deprecated",
}

# Anima caption-format slot order. The inference emitter joins by this
# order; categories not in the list (``deprecated``, ``metadata``) are
# either filtered out or treated as ``general`` depending on context.
SLOT_ORDER: Tuple[str, ...] = (
    "rating",
    "count",
    "character",
    "copyright",
    "artist",
    "general",
)

# 3-class rating set (post-``questionable→sensitive`` collapse).
RATINGS: Tuple[str, ...] = ("general", "sensitive", "explicit")

# Count-tag detection. Matches ``1girl``, ``2girls``, ``1boy``, ``3others``,
# ``multiple_girls``, ``multiple_boys``. Underscores or spaces both fine.
_COUNT_RE = re.compile(
    r"^(?:\d+(?:girl|boy|other)s?|multiple[_ ](?:girls|boys|others))$"
)

# Image extensions we look for next to each .txt caption file. Order is
# preference; first hit wins.
_IMAGE_EXTS: Tuple[str, ...] = (".webp", ".jpg", ".jpeg", ".png")


def find_image_for_caption(caption_path: Path) -> Optional[Path]:
    """Return the sibling image file matching ``{stem}.<ext>``, or None."""
    for ext in _IMAGE_EXTS:
        candidate = caption_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def is_count_tag(tag: str) -> bool:
    return bool(_COUNT_RE.match(tag))


# ── Caption source discovery ──────────────────────────────────────────────


def find_caption_files(roots: Sequence[Path]) -> List[Path]:
    """Discover all ``.txt`` caption files under the given roots.

    Skips dotfiles and the ``tag_cache``/``hash_cache`` JSON sidecars.
    Returns a deduplicated list (by absolute path); a stem appearing under
    multiple roots is *not* deduped here — that's the caller's job (see
    :func:`build_caption_index`).
    """
    out: List[Path] = []
    for root in roots:
        if not root.exists():
            logger.warning("caption root %s does not exist — skipping", root)
            continue
        for p in root.rglob("*.txt"):
            if any(part.startswith(".") for part in p.parts):
                continue
            out.append(p)
    return out


def build_caption_index(
    paths: Iterable[Path],
    rules: tr.TagRules,
) -> Dict[str, Tuple[Path, Optional[Path], List[str]]]:
    """Map ``stem → (caption_path, image_path | None, parsed_tags)``.

    When a stem appears in multiple caption sources, the *first* path wins
    (caller controls precedence via root order). Stems whose sibling image
    file can't be found are still indexed (caption-only entries) so the
    coverage scan reflects what's *captioned*, not what's *trainable*; the
    image-required filter happens at manifest-build time.
    """
    index: Dict[str, Tuple[Path, Optional[Path], List[str]]] = {}
    for path in sorted(paths):
        stem = path.stem
        if stem in index:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("non-utf8 caption %s — skipped", path)
            continue
        tags = tr.parse_caption(content, rules)
        if not tags:
            continue
        image_path = find_image_for_caption(path)
        index[stem] = (path, image_path, tags)
    return index


# ── Categorization ────────────────────────────────────────────────────────


def load_tag_cache(path: Path) -> Dict[str, str]:
    """Load the corpus tag-taxonomy cache and map tag → category name."""
    with open(path) as f:
        raw = json.load(f)
    out: Dict[str, str] = {}
    for tag, type_id in raw.items():
        cat = TAG_TYPE_NAMES.get(int(type_id))
        if cat is not None:
            # Cache uses underscored tag names; the canonical caption format
            # writes them with spaces. Normalize to space form so lookups
            # against parsed captions hit.
            out[tag.replace("_", " ")] = cat
    return out


def categorize(
    tag: str,
    cache: Dict[str, str],
) -> str:
    """Return ``rating`` / ``count`` / ``character`` / ``copyright`` /
    ``artist`` / ``general`` / ``metadata`` / ``deprecated`` for ``tag``.

    Resolution order:

    1. Rating literals (``general``/``sensitive``/``explicit``) → ``rating``.
       Note ``general`` is *both* a rating value and a category name, so
       rating-tag membership is checked before any category lookup.
    2. ``@``-prefixed tags → ``artist``. Anima's caption format prefixes
       artists with ``@``; the underlying tag-cache key drops the ``@``,
       so cache lookups need the bare name.
    3. Count-tag regex → ``count`` (overrides ``general`` typing for
       ``1girl`` etc.).
    4. Cache lookup.
    5. Fallback: ``general``.
    """
    # Note: rating literals collide with the "general" category name. We
    # treat them as their own slot regardless of cache typing — the cache
    # doesn't actually carry rating values anyway (those come from a
    # separate corpus field, not the tag system).
    if tag in RATINGS:
        return "rating"
    if tag.startswith("@"):
        return "artist"
    if is_count_tag(tag):
        return "count"
    bare = tag[1:] if tag.startswith("@") else tag
    cat = cache.get(bare)
    if cat is None:
        return "general"
    return cat


# ── Vocab build ───────────────────────────────────────────────────────────


def build_vocab(
    caption_index: Dict[str, Tuple[Path, Optional[Path], List[str]]],
    tag_cache: Dict[str, str],
    min_freq: int,
) -> Dict:
    """Compute frequencies, categories, median emit positions; cut by min_freq."""
    freq: Counter = Counter()
    sum_pos: Dict[str, int] = defaultdict(int)
    pos_counts: Dict[str, int] = defaultdict(int)

    rating_freq: Counter = Counter()
    n_with_rating = 0

    for stem, (_, _, tags) in caption_index.items():
        # Pull rating off the front if present; everything else feeds the
        # multi-label vocab. Anima's format puts rating first, but be
        # defensive — scan the first few tags.
        rating_seen: Optional[str] = None
        for t in tags[:2]:
            if t in RATINGS:
                rating_seen = t
                break
        if rating_seen is not None:
            rating_freq[rating_seen] += 1
            n_with_rating += 1

        for i, tag in enumerate(tags):
            if tag in RATINGS:
                continue
            freq[tag] += 1
            sum_pos[tag] += i
            pos_counts[tag] += 1

    kept = sorted(
        (t for t, c in freq.items() if c >= min_freq),
        key=lambda t: (-freq[t], t),
    )
    dropped_lowfreq = sum(1 for c in freq.values() if c < min_freq)

    cat_buckets: Counter = Counter()
    cache_hits = 0
    for tag in kept:
        cat = categorize(tag, tag_cache)
        cat_buckets[cat] += 1
        bare = tag[1:] if tag.startswith("@") else tag
        if bare in tag_cache:
            cache_hits += 1

    tags_payload: List[Dict] = []
    for idx, tag in enumerate(kept):
        cat = categorize(tag, tag_cache)
        median_pos = sum_pos[tag] / max(pos_counts[tag], 1)
        tags_payload.append(
            {
                "name": tag,
                "index": idx,
                "category": cat,
                "freq": freq[tag],
                "median_pos": round(median_pos, 2),
            }
        )

    return {
        "tags": tags_payload,
        "ratings": list(RATINGS),
        "slot_order": list(SLOT_ORDER),
        "min_freq": min_freq,
        "n_captions_seen": len(caption_index),
        "n_unique_tags_seen": len(freq),
        "n_tags_kept": len(kept),
        "n_tags_dropped_lowfreq": dropped_lowfreq,
        "category_counts": dict(cat_buckets),
        "cache_hit_rate": round(cache_hits / max(len(kept), 1), 4),
        "rating_distribution": dict(rating_freq),
        "rating_coverage": round(n_with_rating / max(len(caption_index), 1), 4),
    }


def make_split(
    stems: Sequence[str],
    val_frac: float,
    seed: int,
) -> Dict[str, List[str]]:
    """Deterministic random split keyed by ``seed``."""
    rng = random.Random(seed)
    shuffled = list(stems)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_frac)))
    return {
        "val": sorted(shuffled[:n_val]),
        "train": sorted(shuffled[n_val:]),
        "seed": seed,
        "val_frac": val_frac,
    }


# ── Training manifest ─────────────────────────────────────────────────────


def build_manifest(
    caption_index: Dict[str, Tuple[Path, Optional[Path], List[str]]],
    vocab: Dict,
    split: Dict,
) -> Dict:
    """Compact dataset.json: per-stem image path, multi-hot indices, rating.

    Stems lacking a sibling image file are dropped from the manifest (the
    coverage scan in :func:`scan_cache_coverage` still counts them in vocab
    statistics — we just can't *train* on captions without pixels). The split
    is filtered to match.
    """
    tag_to_idx: Dict[str, int] = {t["name"]: t["index"] for t in vocab["tags"]}
    rating_to_idx: Dict[str, int] = {r: i for i, r in enumerate(vocab["ratings"])}

    stems: List[str] = []
    image_paths: List[str] = []
    tag_indices: List[List[int]] = []
    rating_indices: List[int] = []
    n_no_image = 0
    n_no_rating = 0
    n_no_tags = 0

    for stem in sorted(caption_index.keys()):
        _, image_path, tags = caption_index[stem]
        if image_path is None:
            n_no_image += 1
            continue
        rating_idx: Optional[int] = None
        for t in tags[:2]:
            if t in rating_to_idx:
                rating_idx = rating_to_idx[t]
                break
        if rating_idx is None:
            n_no_rating += 1
            continue
        idxs = sorted(
            tag_to_idx[t] for t in tags if t in tag_to_idx and t not in rating_to_idx
        )
        if not idxs:
            n_no_tags += 1
            continue
        stems.append(stem)
        image_paths.append(str(image_path.resolve()))
        tag_indices.append(idxs)
        rating_indices.append(rating_idx)

    kept = set(stems)
    filtered_split = {
        "val": [s for s in split["val"] if s in kept],
        "train": [s for s in split["train"] if s in kept],
        "seed": split["seed"],
        "val_frac": split["val_frac"],
    }

    return {
        "stems": stems,
        "image_paths": image_paths,
        "tag_indices": tag_indices,
        "rating_indices": rating_indices,
        "split": filtered_split,
        "n_tags": len(vocab["tags"]),
        "n_ratings": len(vocab["ratings"]),
        "dropped_no_image": n_no_image,
        "dropped_no_rating": n_no_rating,
        "dropped_no_invocab_tags": n_no_tags,
    }


# ── Coverage scan ─────────────────────────────────────────────────────────


def scan_cache_coverage(
    caption_index: Dict[str, Tuple[Path, Optional[Path], List[str]]],
    tag_cache: Dict[str, str],
) -> Dict:
    """How many caption tags lack a category in gelcrawl's cache?

    A high miss rate would mean ``categorize()`` is falling back to
    ``general`` for too many tags and we should run the gelbooru API fill-in
    pass before training. <5 % miss → safe to default-to-general.
    """
    seen: Counter = Counter()
    missing: Counter = Counter()
    for _, (_, _, tags) in caption_index.items():
        for tag in tags:
            if tag in RATINGS:
                continue
            seen[tag] += 1
            bare = tag[1:] if tag.startswith("@") else tag
            if (
                tag.startswith("@")
                or is_count_tag(tag)
                or bare in tag_cache
            ):
                continue
            missing[tag] += 1
    return {
        "n_unique_tags": len(seen),
        "n_unique_missing": len(missing),
        "n_total_tag_occurrences": sum(seen.values()),
        "n_missing_occurrences": sum(missing.values()),
        "missing_top20": missing.most_common(20),
    }


# ── CLI plumbing ──────────────────────────────────────────────────────────


def cmd_build_vocab(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rules_src = Path(args.rules)
    rules = tr.load_rules(rules_src)
    logger.info(
        "rules: %d replacements, %d remove, %d dedup base tags",
        len(rules.replacements),
        len(rules.remove),
        len(rules.dedup),
    )

    roots = [Path(r) for r in args.caption_roots]
    cap_paths = find_caption_files(roots)
    logger.info("scanning %d caption files across %d roots", len(cap_paths), len(roots))
    index = build_caption_index(cap_paths, rules)
    logger.info("kept %d unique stems with non-empty captions", len(index))

    tag_cache = load_tag_cache(Path(args.tag_cache))
    logger.info("loaded tag cache with %d entries", len(tag_cache))

    coverage = scan_cache_coverage(index, tag_cache)
    miss_rate = coverage["n_missing_occurrences"] / max(
        coverage["n_total_tag_occurrences"], 1
    )
    logger.info(
        "cache coverage: %d/%d unique tags categorized "
        "(%.2f%% of occurrences missing)",
        coverage["n_unique_tags"] - coverage["n_unique_missing"],
        coverage["n_unique_tags"],
        100 * miss_rate,
    )
    if coverage["missing_top20"]:
        logger.info("top-20 uncategorized tags (will fall back to 'general'):")
        for tag, n in coverage["missing_top20"]:
            logger.info("  %5d × %s", n, tag)

    vocab = build_vocab(index, tag_cache, min_freq=args.min_freq)
    vocab["caption_roots"] = [str(r.resolve()) for r in roots]
    vocab["tag_cache_path"] = str(Path(args.tag_cache).resolve())
    vocab["rules_source_path"] = str(rules_src.resolve())
    vocab["coverage"] = coverage

    split = make_split(
        sorted(index.keys()),
        val_frac=args.val_frac,
        seed=args.seed,
    )
    vocab["split"] = split

    # Write the vocab + split.
    vocab_path = out_dir / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    logger.info("wrote %s", vocab_path)

    # Snapshot the rules into the checkpoint dir so the inference wrapper
    # has zero runtime dependency on the source corpus.
    snap_path = out_dir / "rules.yaml"
    with open(snap_path, "w") as f:
        import yaml as _yaml

        _yaml.safe_dump(rules.to_dict(), f, sort_keys=False)
    logger.info("wrote %s", snap_path)

    # Build and persist the training manifest (drops captions without a
    # sibling image, without a rating tag, or with no in-vocab tags).
    manifest = build_manifest(index, vocab, split)
    manifest_path = out_dir / "dataset.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info(
        "wrote %s — %d trainable samples (dropped %d no_image, %d no_rating, "
        "%d no_invocab_tags)",
        manifest_path,
        len(manifest["stems"]),
        manifest["dropped_no_image"],
        manifest["dropped_no_rating"],
        manifest["dropped_no_invocab_tags"],
    )

    # Compact summary printout.
    print()
    print(f"  caption stems indexed:  {vocab['n_captions_seen']}")
    print(f"  unique tags seen:       {vocab['n_unique_tags_seen']}")
    print(f"  vocab size (≥{args.min_freq}):       {vocab['n_tags_kept']}")
    print(f"  dropped (low-freq):     {vocab['n_tags_dropped_lowfreq']}")
    print(f"  cache hit rate:         {vocab['cache_hit_rate']}")
    print(f"  category counts:")
    for cat, n in sorted(vocab["category_counts"].items(), key=lambda kv: -kv[1]):
        print(f"    {cat:<12} {n}")
    print(f"  rating coverage:        {vocab['rating_coverage']}")
    print(f"  rating distribution:    {vocab['rating_distribution']}")
    print(f"  split:                  {len(split['train'])} train / {len(split['val'])} val")
    print(f"  cache miss rate:        {miss_rate:.2%}")
    print(f"  trainable samples:      {len(manifest['stems'])}")
    print(
        f"    (dropped {manifest['dropped_no_image']} no-image, "
        f"{manifest['dropped_no_rating']} no-rating, "
        f"{manifest['dropped_no_invocab_tags']} no-invocab-tags)"
    )


def _corpus_default(rel: str) -> Optional[str]:
    """Resolve ``$CAPTION_CORPUS_DIR/<rel>`` for argparse defaults.

    Returns ``None`` when the env var is unset so argparse renders an
    explicit '(unset)' marker in --help instead of a misleading empty path.
    """
    root = os.environ.get("CAPTION_CORPUS_DIR")
    if not root:
        return None
    return str(Path(root) / rel)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Anima tagger trainer")
    p.add_argument(
        "--mode",
        choices=["build_vocab", "build_features", "train", "calibrate", "predict"],
        default="build_vocab",
    )
    p.add_argument(
        "--encoder",
        default="pe",
        help="Vision encoder registry name (passed to load_pe_encoder). "
        "Default: pe (PE-Core-L14-336).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device for build_features / train (default: cuda if available).",
    )

    # Vocab-build inputs. All three default to subpaths of
    # ``$CAPTION_CORPUS_DIR``; pass --caption_roots / --tag_cache / --rules
    # explicitly to override.
    raw_default = _corpus_default("retrieved")
    curated_default = _corpus_default("selected")
    p.add_argument(
        "--caption_roots",
        nargs="+",
        default=[d for d in (raw_default, curated_default, "image_dataset") if d],
        help="Directories to scan recursively for *.txt caption files. "
        "First-match-wins by stem when a duplicate appears across roots. "
        "Defaults: $CAPTION_CORPUS_DIR/retrieved + "
        "$CAPTION_CORPUS_DIR/selected + image_dataset/.",
    )
    p.add_argument(
        "--tag_cache",
        default=_corpus_default("retrieved/.tag_cache.json"),
        help="Tag-taxonomy JSON (tag → integer type ID). "
        "Default: $CAPTION_CORPUS_DIR/retrieved/.tag_cache.json.",
    )
    p.add_argument(
        "--rules",
        default=_corpus_default("tag_rules.yaml"),
        help="Caption-normalization rules (snapshotted into out_dir at "
        "build time). Default: $CAPTION_CORPUS_DIR/tag_rules.yaml.",
    )
    p.add_argument("--min_freq", type=int, default=5)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)

    # Train-mode knobs.
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--d_hidden", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument(
        "--lambda_rating",
        type=float,
        default=0.1,
        help="Weight on the rating CE loss relative to multi-label BCE.",
    )

    # Output.
    p.add_argument(
        "--out_dir",
        default="models/captioners/anima-tagger-v1",
    )

    args = p.parse_args()

    if args.mode == "build_vocab":
        missing = [
            name
            for name, val in (
                ("--tag_cache", args.tag_cache),
                ("--rules", args.rules),
            )
            if not val
        ]
        if missing or not args.caption_roots:
            raise SystemExit(
                "build_vocab needs CAPTION_CORPUS_DIR set in anima_lora/.env "
                f"(or {', '.join(missing) or '--caption_roots'} passed "
                "explicitly). Add a line like\n"
                "    CAPTION_CORPUS_DIR=/path/to/corpus\n"
                "to anima_lora/.env, or pass the paths via CLI flags."
            )

    return args


# ── Training ──────────────────────────────────────────────────────────────


def _pos_weight_sqrt(multi_hot: torch.Tensor) -> torch.Tensor:
    """``sqrt(n_neg / n_pos)`` per tag — softens BCE long-tail without overshoot."""
    n_pos = multi_hot.sum(dim=0).clamp_min(1.0)
    n_neg = multi_hot.shape[0] - n_pos
    return torch.sqrt(n_neg / n_pos)


def _rating_class_weights(rating_idx: torch.Tensor, n_ratings: int) -> torch.Tensor:
    """Inverse-frequency weights, normalized so mean weight = 1."""
    counts = torch.bincount(rating_idx, minlength=n_ratings).float().clamp_min(1.0)
    inv = 1.0 / counts
    inv = inv * n_ratings / inv.sum()
    return inv


@torch.no_grad()
def _eval_split(
    model: torch.nn.Module,
    feats: torch.Tensor,
    multi_hot: torch.Tensor,
    rating_idx: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Macro-F1 over tags + rating accuracy at the given threshold."""
    model.eval()
    tag_logits, rating_logits = model(feats)
    pred = (tag_logits.sigmoid() > threshold).float()
    tp = (pred * multi_hot).sum(dim=0)
    fp = (pred * (1 - multi_hot)).sum(dim=0)
    fn = ((1 - pred) * multi_hot).sum(dim=0)
    prec = tp / (tp + fp).clamp_min(1.0)
    rec = tp / (tp + fn).clamp_min(1.0)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-8)
    rating_pred = rating_logits.argmax(dim=-1)
    rating_acc = (rating_pred == rating_idx).float().mean().item()
    return {
        "macro_f1": f1.mean().item(),
        "macro_precision": prec.mean().item(),
        "macro_recall": rec.mean().item(),
        "rating_acc": rating_acc,
    }


def cmd_train(args: argparse.Namespace) -> None:
    from safetensors.torch import save_file as st_save

    from library.captioning.anima_tagger_data import (
        CachedFeatureDataset,
        TaggerManifest,
    )
    from library.captioning.anima_tagger_model import (
        AnimaTaggerConfig,
        AnimaTaggerHead,
    )

    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "dataset.json"
    cache_dir = out_dir / ".cache" / f"pooled-{args.encoder}"
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path} — run --mode build_vocab first.")
    if not cache_dir.exists():
        raise SystemExit(
            f"missing {cache_dir} — run --mode build_features first."
        )
    manifest = TaggerManifest.from_path(manifest_path)
    train_ds = CachedFeatureDataset(manifest, cache_dir, stems_subset=manifest.train_stems)
    val_ds = CachedFeatureDataset(manifest, cache_dir, stems_subset=manifest.val_stems)
    logger.info(
        "train: N=%d  val: N=%d  d_in=%d  n_tags=%d  n_ratings=%d",
        len(train_ds),
        len(val_ds),
        train_ds.d_in,
        train_ds.n_tags,
        train_ds.n_ratings,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = AnimaTaggerConfig(
        d_in=train_ds.d_in,
        n_tags=train_ds.n_tags,
        n_ratings=train_ds.n_ratings,
        d_hidden=args.d_hidden,
        dropout=args.dropout,
    )
    model = AnimaTaggerHead(cfg).to(device)

    # All training/val tensors fit in VRAM trivially (~50 MB) — push them
    # once instead of per-batch.
    train_feats = train_ds.features.to(device)
    train_mh = train_ds.multi_hot.to(device)
    train_rate = train_ds.rating_idx.to(device)
    val_feats = val_ds.features.to(device)
    val_mh = val_ds.multi_hot.to(device)
    val_rate = val_ds.rating_idx.to(device)

    pos_weight = _pos_weight_sqrt(train_mh).to(device)
    rating_w = _rating_class_weights(train_rate, train_ds.n_ratings).to(device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ce = torch.nn.CrossEntropyLoss(weight=rating_w)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    n_train = len(train_ds)
    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    best_f1 = -1.0
    best_state: Dict[str, torch.Tensor] = {}
    history: List[Dict[str, float]] = []

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, generator=rng)
        ep_loss = 0.0
        ep_tag_loss = 0.0
        ep_rate_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, args.batch_size):
            idx = perm[start : start + args.batch_size]
            feat = train_feats[idx]
            mh = train_mh[idx]
            rate = train_rate[idx]
            tag_logits, rating_logits = model(feat)
            l_tag = bce(tag_logits, mh)
            l_rate = ce(rating_logits, rate)
            loss = l_tag + args.lambda_rating * l_rate
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            ep_tag_loss += l_tag.item()
            ep_rate_loss += l_rate.item()
            n_batches += 1
        sched.step()
        avg_loss = ep_loss / max(n_batches, 1)
        avg_tag = ep_tag_loss / max(n_batches, 1)
        avg_rate = ep_rate_loss / max(n_batches, 1)
        val_metrics = _eval_split(model, val_feats, val_mh, val_rate)
        logger.info(
            "epoch %2d/%d  loss=%.4f (tag=%.4f rate=%.4f)  "
            "val_f1=%.4f  val_p=%.4f  val_r=%.4f  rate_acc=%.4f  lr=%.2e",
            epoch + 1,
            args.epochs,
            avg_loss,
            avg_tag,
            avg_rate,
            val_metrics["macro_f1"],
            val_metrics["macro_precision"],
            val_metrics["macro_recall"],
            val_metrics["rating_acc"],
            sched.get_last_lr()[0],
        )
        history.append({"epoch": epoch + 1, "loss": avg_loss, **val_metrics})
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if not best_state:
        raise SystemExit("no epochs ran — empty training set?")

    # Save best checkpoint + config.
    ckpt_path = out_dir / "model.safetensors"
    cfg_path = out_dir / "config.json"
    history_path = out_dir / "train_history.json"
    st_save(best_state, str(ckpt_path))
    with open(cfg_path, "w") as f:
        json.dump(
            {
                "model": cfg.to_dict(),
                "encoder": args.encoder,
                "d_in": train_ds.d_in,
                "best_val_macro_f1": best_f1,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "lambda_rating": args.lambda_rating,
                "seed": args.seed,
            },
            f,
            indent=2,
        )
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("wrote %s / %s / %s", ckpt_path, cfg_path, history_path)
    print(f"  best val macro_f1: {best_f1:.4f}")


# ── Per-tag F1 threshold calibration ──────────────────────────────────────


def _calibrate_thresholds(
    scores: torch.Tensor,        # [N, n_tags] sigmoid probabilities
    targets: torch.Tensor,       # [N, n_tags] multi-hot
    sweep: torch.Tensor,         # [K] candidate thresholds
    default: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-tag F1-optimal threshold sweep.

    Returns ``(thresholds[n_tags], best_f1[n_tags])``. Tags with no positives
    in the val split keep ``default`` (they can't be calibrated and the
    F1 sweep is degenerate — a 0.5 floor is harmless and keeps the head
    well-formed for inference). Same fallback for tags whose best
    achievable F1 is 0 (model never predicts them at any threshold).
    """
    n_tags = scores.shape[1]
    K = sweep.shape[0]
    best_thresh = torch.full((n_tags,), default)
    best_f1 = torch.zeros(n_tags)
    pos_count = targets.sum(dim=0)                              # [n_tags]
    has_pos = pos_count > 0
    # Process tag-blocks to keep memory bounded — the dense [N, n_tags, K]
    # tensor would be ~12k × 5k × 19 ≈ 1.1B floats which is too big.
    block_size = 256
    for start in range(0, n_tags, block_size):
        end = min(start + block_size, n_tags)
        s = scores[:, start:end]                                 # [N, b]
        t = targets[:, start:end]
        # [N, b, K] boolean
        pred = s.unsqueeze(-1) > sweep.view(1, 1, K)
        pred_f = pred.float()
        tp = (pred_f * t.unsqueeze(-1)).sum(dim=0)               # [b, K]
        fp = (pred_f * (1 - t).unsqueeze(-1)).sum(dim=0)
        fn = ((1 - pred_f) * t.unsqueeze(-1)).sum(dim=0)
        prec = tp / (tp + fp).clamp_min(1e-8)
        rec = tp / (tp + fn).clamp_min(1e-8)
        f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-8)       # [b, K]
        f1_best, k_best = f1.max(dim=-1)                          # [b]
        thresh_best = sweep[k_best]                               # [b]
        local_has_pos = has_pos[start:end]
        keep = local_has_pos & (f1_best > 0)
        best_f1[start:end] = torch.where(
            keep, f1_best, best_f1[start:end]
        )
        best_thresh[start:end] = torch.where(
            keep, thresh_best, best_thresh[start:end]
        )
    return best_thresh, best_f1


def cmd_calibrate(args: argparse.Namespace) -> None:
    from safetensors.torch import load_file as st_load
    from safetensors.torch import save_file as st_save

    from library.captioning.anima_tagger_data import (
        CachedFeatureDataset,
        TaggerManifest,
    )
    from library.captioning.anima_tagger_model import (
        AnimaTaggerConfig,
        AnimaTaggerHead,
    )

    out_dir = Path(args.out_dir)
    manifest = TaggerManifest.from_path(out_dir / "dataset.json")
    cache_dir = out_dir / ".cache" / f"pooled-{args.encoder}"
    val_ds = CachedFeatureDataset(manifest, cache_dir, stems_subset=manifest.val_stems)

    with open(out_dir / "config.json") as f:
        cfg_d = json.load(f)
    cfg = AnimaTaggerConfig.from_dict(cfg_d["model"])
    model = AnimaTaggerHead(cfg)
    state = st_load(str(out_dir / "model.safetensors"))
    model.load_state_dict(state)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device).eval()

    val_feats = val_ds.features.to(device)
    val_mh = val_ds.multi_hot.to(device)
    with torch.no_grad():
        tag_logits, _ = model(val_feats)
        scores = tag_logits.sigmoid().cpu()
    sweep = torch.linspace(0.05, 0.95, 19)
    thresh, f1 = _calibrate_thresholds(scores, val_mh.cpu(), sweep, default=0.5)

    st_save(
        {"thresholds": thresh, "val_f1": f1},
        str(out_dir / "thresholds.safetensors"),
    )
    n_active = int((f1 > 0).sum().item())
    macro_default = ((scores > 0.5).float() * val_mh.cpu()).sum() / max(
        val_mh.cpu().sum().item(), 1
    )  # rough at default
    logger.info(
        "calibrated %d/%d tags with non-zero F1 at sweep optimum",
        n_active,
        thresh.shape[0],
    )
    logger.info(
        "macro-F1 (calibrated) = %.4f  vs default 0.5 macro-F1 = %.4f",
        f1.mean().item(),
        _eval_split(model, val_feats, val_mh, val_ds.rating_idx.to(device))[
            "macro_f1"
        ],
    )
    print(f"  thresholds: {out_dir / 'thresholds.safetensors'}")
    print(f"  active tags (F1>0): {n_active} / {thresh.shape[0]}")
    print(f"  calibrated macro-F1: {f1.mean().item():.4f}")
    # Print a sample of low/mid/high thresholds for sanity.
    with open(out_dir / "vocab.json") as f:
        vocab = json.load(f)
    name_of = [t["name"] for t in vocab["tags"]]
    by_thresh = sorted(
        ((thresh[i].item(), f1[i].item(), name_of[i]) for i in range(thresh.shape[0])),
        key=lambda x: x[0],
    )
    print("  sample thresholds (lowest 5 / highest 5):")
    for t, fv, n in by_thresh[:5] + by_thresh[-5:]:
        print(f"    thresh={t:.2f}  f1={fv:.3f}  {n}")


def cmd_build_features(args: argparse.Namespace) -> None:
    from library.captioning.anima_tagger_data import (
        FeatureCacheBuilder,
        TaggerManifest,
    )

    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "dataset.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"missing {manifest_path} — run --mode build_vocab first."
        )
    manifest = TaggerManifest.from_path(manifest_path)
    cache_dir = out_dir / ".cache" / f"pooled-{args.encoder}"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(
        "build_features: %d manifest entries → %s (device=%s, encoder=%s)",
        len(manifest.stems),
        cache_dir,
        device,
        args.encoder,
    )
    builder = FeatureCacheBuilder(
        manifest=manifest,
        cache_dir=cache_dir,
        device=device,
        encoder_name=args.encoder,
    )
    n_new = builder.build()
    n_total = len(manifest.stems) - len(builder.missing_stems())
    print(f"  cache dir:        {cache_dir}")
    print(f"  newly encoded:    {n_new}")
    print(f"  cached / total:   {n_total} / {len(manifest.stems)}")


def main() -> None:
    args = parse_args()
    if args.mode == "build_vocab":
        cmd_build_vocab(args)
    elif args.mode == "build_features":
        cmd_build_features(args)
    elif args.mode == "train":
        cmd_train(args)
    elif args.mode == "calibrate":
        cmd_calibrate(args)
    else:
        raise SystemExit(
            f"--mode={args.mode!r} not yet wired in this script — see "
            f"the proposal task list."
        )


if __name__ == "__main__":
    main()
