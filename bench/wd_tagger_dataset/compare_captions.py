"""Compare wd-swinv2-tagger-v3 predictions against canonical .txt captions.

Samples N random images from a dataset directory, runs the tagger, and diffs
the predicted tag set against each image's sidecar caption. Reports per-image
false positives / false negatives plus aggregate Jaccard / precision / recall.

Tags that don't exist in the tagger's vocabulary (artists with ``@`` prefix,
niche copyright tags, etc.) are bucketed into ``oov_skipped`` rather than
counted as misses — the tagger physically cannot emit them, so dinging recall
for them is misleading. The headline ``mean_recall_in_vocab`` measures only
the tags that the tagger could in principle produce.

Usage::

    python bench/wd_tagger_dataset/compare_captions.py
    python bench/wd_tagger_dataset/compare_captions.py --n 50 --seed 7
    python bench/wd_tagger_dataset/compare_captions.py \\
        --dataset image_dataset --general_threshold 0.30

Drops ``result.json`` + ``per_image.csv`` into
``bench/wd_tagger_dataset/results/<YYYYMMDD-HHMM>[-<label>]/``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from bench._common import make_run_dir, write_result  # noqa: E402
from library.captioning.wd_tagger import WDTagger  # noqa: E402
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)

IMAGE_EXTS = (".webp", ".png", ".jpg", ".jpeg")
RATING_NAMES = {"general", "sensitive", "questionable", "explicit"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--dataset",
        default="image_dataset",
        help="Dataset dir relative to anima_lora/. Each image needs a .txt sidecar.",
    )
    p.add_argument("--n", type=int, default=20, help="Number of images to sample.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--general_threshold", type=float, default=0.35)
    p.add_argument("--character_threshold", type=float, default=0.85)
    p.add_argument(
        "--label",
        default=None,
        help="Optional run-dir suffix (e.g. 'thresh-030').",
    )
    return p.parse_args()


def _norm_tag(t: str) -> str:
    """Comparison form: spaces (no underscores), parens unescaped, lowercased, trimmed.

    Predicted tags arrive as ``looking_at_viewer``; canonical as
    ``looking at viewer``. We map both to space form so set ops match.
    """
    return (
        t.replace("\\(", "(")
        .replace("\\)", ")")
        .replace("_", " ")
        .strip()
        .lower()
    )


def _vocab_form(t: str) -> str:
    """Underscore form for tagger-vocab lookups."""
    return _norm_tag(t).replace(" ", "_")


def _split_canonical(text: str) -> list[str]:
    return [t for t in (s.strip() for s in text.split(",")) if t]


def _split_rating(tags: list[str]) -> tuple[str | None, list[str]]:
    if tags and _norm_tag(tags[0]) in RATING_NAMES:
        return _norm_tag(tags[0]), tags[1:]
    return None, tags


def _iter_images(root: Path) -> Iterable[Path]:
    for ext in IMAGE_EXTS:
        yield from root.glob(f"*{ext}")


def _read_caption(img_path: Path) -> str | None:
    txt = img_path.with_suffix(".txt")
    if not txt.is_file():
        return None
    return txt.read_text(encoding="utf-8").strip()


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    dataset = REPO_ROOT / args.dataset
    if not dataset.is_dir():
        logger.error("dataset dir not found: %s", dataset)
        sys.exit(1)

    candidates = [
        p for p in _iter_images(dataset) if p.with_suffix(".txt").is_file()
    ]
    if not candidates:
        logger.error("no <img> + <img>.txt pairs under %s", dataset)
        sys.exit(1)
    rng.shuffle(candidates)
    sampled = candidates[: args.n]
    logger.info(
        "sampled %d / %d images from %s (seed=%d)",
        len(sampled),
        len(candidates),
        dataset,
        args.seed,
    )

    tagger = WDTagger(
        general_threshold=args.general_threshold,
        character_threshold=args.character_threshold,
    )
    vocab: set[str] | None = None  # populated after first predict

    run_dir = make_run_dir("wd_tagger_dataset", label=args.label)
    rows_csv = run_dir / "per_image.csv"

    fp_counter: Counter[str] = Counter()
    fn_counter: Counter[str] = Counter()
    fn_oov_counter: Counter[str] = Counter()

    jaccards: list[float] = []
    precisions: list[float] = []
    recalls_in_vocab: list[float] = []
    raw_recalls: list[float] = []
    rating_correct = 0
    rating_seen = 0

    rows: list[dict] = []

    for i, img_path in enumerate(sampled, 1):
        caption = _read_caption(img_path)
        if caption is None:
            continue
        canonical_tags_raw = _split_canonical(caption)
        canonical_rating, canonical_tags_raw = _split_rating(canonical_tags_raw)
        canon_norm = [_norm_tag(t) for t in canonical_tags_raw]

        try:
            img = Image.open(img_path)
            rating, character, general = tagger.predict(img)
        except Exception as e:
            logger.warning("predict failed on %s: %s", img_path.name, e)
            continue

        if vocab is None:
            schema = tagger._schema
            assert schema is not None, "tagger schema missing post-predict"
            vocab = {n.lower() for n in schema.names}

        pred_rating = rating[0][0] if rating else None
        pred_tags_norm = [_norm_tag(n) for n, _ in (character + general)]

        canon_set = set(canon_norm)
        pred_set = set(pred_tags_norm)

        canon_oov = {t for t in canon_set if _vocab_form(t) not in vocab}
        canon_in_vocab = canon_set - canon_oov

        intersection = canon_in_vocab & pred_set
        recall_in_vocab = (
            len(intersection) / len(canon_in_vocab) if canon_in_vocab else 0.0
        )
        raw_recall = (
            len(canon_set & pred_set) / len(canon_set) if canon_set else 0.0
        )
        precision = len(intersection) / len(pred_set) if pred_set else 0.0
        union = canon_in_vocab | pred_set
        jaccard = len(intersection) / len(union) if union else 0.0

        if canonical_rating is not None:
            rating_seen += 1
            if pred_rating == canonical_rating:
                rating_correct += 1

        fn_in_vocab = canon_in_vocab - pred_set
        fp = pred_set - canon_in_vocab
        fn_counter.update(fn_in_vocab)
        fp_counter.update(fp)
        fn_oov_counter.update(canon_oov)

        jaccards.append(jaccard)
        precisions.append(precision)
        recalls_in_vocab.append(recall_in_vocab)
        raw_recalls.append(raw_recall)

        rows.append(
            {
                "image": img_path.name,
                "canonical_rating": canonical_rating or "",
                "predicted_rating": pred_rating or "",
                "rating_match": int(canonical_rating == pred_rating),
                "canon_total": len(canon_set),
                "canon_in_vocab": len(canon_in_vocab),
                "pred_total": len(pred_set),
                "intersection": len(intersection),
                "jaccard": f"{jaccard:.4f}",
                "precision": f"{precision:.4f}",
                "recall_in_vocab": f"{recall_in_vocab:.4f}",
                "raw_recall": f"{raw_recall:.4f}",
                "missed_in_vocab": "; ".join(sorted(fn_in_vocab)),
                "hallucinated": "; ".join(sorted(fp)),
                "oov_skipped": "; ".join(sorted(canon_oov)),
                "canon_caption": ", ".join(canon_norm),
                "pred_caption": ", ".join(pred_tags_norm),
            }
        )

        logger.info(
            "[%d/%d] %s rating=%s/%s J=%.3f P=%.3f R(vocab)=%.3f "
            "(canon=%d, in-vocab=%d, pred=%d)",
            i,
            len(sampled),
            img_path.name,
            canonical_rating,
            pred_rating,
            jaccard,
            precision,
            recall_in_vocab,
            len(canon_set),
            len(canon_in_vocab),
            len(pred_set),
        )

    if rows:
        with rows_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    n = len(jaccards)
    metrics = {
        "n_images": n,
        "mean_jaccard": _mean(jaccards),
        "mean_precision": _mean(precisions),
        "mean_recall_in_vocab": _mean(recalls_in_vocab),
        "mean_raw_recall": _mean(raw_recalls),
        "rating_accuracy": (
            rating_correct / rating_seen if rating_seen else None
        ),
        "rating_seen": rating_seen,
        "top_false_positives": fp_counter.most_common(20),
        "top_missed_in_vocab": fn_counter.most_common(20),
        "top_oov_skipped": fn_oov_counter.most_common(20),
    }

    write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        artifacts=["per_image.csv"],
        label=args.label,
    )

    print()
    print(f"=== wd-tagger vs canonical  ({n} images, seed={args.seed}) ===")
    print(f"  mean Jaccard           : {metrics['mean_jaccard']:.3f}")
    print(f"  mean precision         : {metrics['mean_precision']:.3f}")
    print(f"  mean recall (in-vocab) : {metrics['mean_recall_in_vocab']:.3f}")
    print(f"  mean recall (raw)      : {metrics['mean_raw_recall']:.3f}")
    if metrics["rating_accuracy"] is not None:
        print(
            f"  rating accuracy        : {metrics['rating_accuracy']:.3f}"
            f"  ({rating_correct}/{rating_seen})"
        )
    print()
    print("Top 10 hallucinations (predicted, canon didn't have):")
    for tag, c in fp_counter.most_common(10):
        print(f"  {c:3d}  {tag}")
    print()
    print("Top 10 missed (in-vocab, canon had, tagger didn't):")
    for tag, c in fn_counter.most_common(10):
        print(f"  {c:3d}  {tag}")
    print()
    print("Top 10 OOV (canon had, not in tagger vocab — artists/copyrights):")
    for tag, c in fn_oov_counter.most_common(10):
        print(f"  {c:3d}  {tag}")
    print()
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
