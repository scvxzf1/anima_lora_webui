"""AnimaTagger — multi-label tagger trained on the Anima caption distribution.

Drop-in replacement for :class:`WDTagger` when feeding ψ_src to DirectEdit.
Mirrors the same public surface (``predict``, ``predict_caption``) so existing
callers swap by changing only the import.

Checkpoint layout (produced by ``scripts/train_anima_tagger.py``):

::

    ckpt_dir/
      config.json              # model config + training metadata
      model.safetensors        # AnimaTaggerHead state dict
      thresholds.safetensors   # per-tag F1-optimal thresholds
      vocab.json               # tag list with category + median_pos info
      rules.yaml               # caption-normalization rules snapshot

The vision encoder (PE-Core-L14-336 by default) is loaded lazily on first
``predict`` call. Captions are emitted in Anima's canonical slot order:
``rating, count_tags, characters, copyrights, @artists, generals``, with
underscores replaced by spaces (matching how Anima's training-time T5 saw
the data).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as st_load

from library.captioning import tag_rules as tr
from library.captioning.anima_tagger_data import pil_resize_to_bucket
from library.captioning.anima_tagger_model import AnimaTaggerConfig, AnimaTaggerHead
from library.datasets.image_utils import IMAGE_TRANSFORMS
from library.vision.encoder import (
    VisionEncoderBundle,
    encode_pe_from_imageminus1to1,
    load_pe_encoder,
)

logger = logging.getLogger(__name__)

# Canonical caption-format slot order (matches Anima training captions).
SLOT_ORDER: Tuple[str, ...] = (
    "rating",
    "count",
    "character",
    "copyright",
    "artist",
    "general",
)


@dataclass
class _TagEntry:
    name: str
    index: int
    category: str
    median_pos: float


def _underscore_to_space(s: str) -> str:
    """Anima caption format: tags with spaces, not underscores.

    The cache key uses underscores; the canonical caption uses spaces.
    Apply at emit time (not vocab-build) so tag indexing stays stable.
    """
    return s.replace("_", " ")


def _load_thresholds(path: Path, n_tags: int, default: float = 0.5) -> torch.Tensor:
    """Load per-tag thresholds; missing → uniform default."""
    if not path.exists():
        logger.warning("no thresholds.safetensors at %s — using default=%.2f", path, default)
        return torch.full((n_tags,), default)
    d = st_load(str(path))
    t = d["thresholds"]
    if t.shape != (n_tags,):
        raise ValueError(f"thresholds shape {tuple(t.shape)} != ({n_tags},)")
    return t


class AnimaTagger:
    """Drop-in replacement for ``WDTagger`` with Anima-distribution vocabulary."""

    def __init__(
        self,
        ckpt_dir: str | Path = "models/captioners/anima-tagger-v1",
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.ckpt_dir = Path(ckpt_dir)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype

        with open(self.ckpt_dir / "config.json") as f:
            cfg_d = json.load(f)
        self.encoder_name: str = cfg_d.get("encoder", "pe")
        self.cfg = AnimaTaggerConfig.from_dict(cfg_d["model"])

        self.model = AnimaTaggerHead(self.cfg)
        self.model.load_state_dict(st_load(str(self.ckpt_dir / "model.safetensors")))
        self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        with open(self.ckpt_dir / "vocab.json") as f:
            vocab = json.load(f)
        self.tag_entries: List[_TagEntry] = [
            _TagEntry(
                name=t["name"],
                index=int(t["index"]),
                category=str(t["category"]),
                median_pos=float(t.get("median_pos", 0.0)),
            )
            for t in vocab["tags"]
        ]
        self.ratings: List[str] = list(vocab["ratings"])
        # Map category → list of (index, median_pos, name) sorted by median_pos.
        self._by_cat: Dict[str, List[Tuple[int, float, str]]] = {}
        for e in self.tag_entries:
            cat = e.category if e.category in SLOT_ORDER else "general"
            self._by_cat.setdefault(cat, []).append((e.index, e.median_pos, e.name))
        for cat in self._by_cat:
            self._by_cat[cat].sort(key=lambda triple: (triple[1], triple[2]))

        self.thresholds = _load_thresholds(
            self.ckpt_dir / "thresholds.safetensors", n_tags=self.cfg.n_tags
        )
        self.thresholds_dev = self.thresholds.to(self.device)

        self.rules = tr.load_rules(self.ckpt_dir / "rules.yaml")
        self._encoder: Optional[VisionEncoderBundle] = None

    # ── Encoder lazy-load ──────────────────────────────────────────────

    def _bundle(self) -> VisionEncoderBundle:
        if self._encoder is None:
            self._encoder = load_pe_encoder(
                self.device, name=self.encoder_name, dtype=self.dtype
            )
        return self._encoder

    @torch.no_grad()
    def _encode_image(self, pil_img: Image.Image) -> torch.Tensor:
        """Image → pooled feature ``[d_in]`` on ``self.device``."""
        bundle = self._bundle()
        pil_img = pil_resize_to_bucket(pil_img.convert("RGB"), bundle.bucket_spec)
        tensor = IMAGE_TRANSFORMS(np.array(pil_img)).unsqueeze(0)
        feats_list = encode_pe_from_imageminus1to1(bundle, tensor, same_bucket=True)
        feats = feats_list[0]                # [T, d_enc]
        return feats.mean(dim=0).to(torch.float32)

    # ── Public API (mirrors WDTagger) ──────────────────────────────────

    @torch.no_grad()
    def predict(self, pil_img: Image.Image) -> Dict[str, object]:
        """Run one image through the head; return raw + thresholded outputs.

        Returns a dict with:

        * ``rating``: predicted rating string (one of ``self.ratings``)
        * ``rating_scores``: dict ``{rating: prob}``
        * ``scores``: dict ``{tag: prob}`` for *all* in-vocab tags
        * ``kept``: dict ``{tag: prob}`` for tags above their per-tag threshold
        """
        feat = self._encode_image(pil_img).unsqueeze(0).to(self.device)
        tag_logits, rating_logits = self.model(feat)
        tag_probs = tag_logits.sigmoid()[0]                  # [n_tags]
        rating_probs = rating_logits.softmax(dim=-1)[0]      # [n_ratings]
        kept_mask = (tag_probs >= self.thresholds_dev).cpu()
        tag_probs_cpu = tag_probs.cpu()
        scores = {
            self.tag_entries[i].name: float(tag_probs_cpu[i])
            for i in range(self.cfg.n_tags)
        }
        kept = {
            self.tag_entries[i].name: float(tag_probs_cpu[i])
            for i in range(self.cfg.n_tags)
            if kept_mask[i]
        }
        rating_idx = int(rating_probs.argmax().item())
        return {
            "rating": self.ratings[rating_idx],
            "rating_scores": {
                r: float(rating_probs[i].cpu()) for i, r in enumerate(self.ratings)
            },
            "scores": scores,
            "kept": kept,
        }

    def predict_caption(self, pil_img: Image.Image) -> str:
        """Image → canonical Anima caption string (rating + slotted tags)."""
        out = self.predict(pil_img)
        kept_idxs = {
            self.tag_entries[i].index
            for i, name in enumerate([e.name for e in self.tag_entries])
            if name in out["kept"]
        }
        # Slot tags by canonical category order, within-slot by median_pos.
        slotted: Dict[str, List[str]] = {cat: [] for cat in SLOT_ORDER}
        slotted["rating"].append(out["rating"])
        for cat, entries in self._by_cat.items():
            for idx, _, name in entries:
                if idx in kept_idxs:
                    slotted.setdefault(cat, []).append(name)
        # Re-apply tag rules at emit time as a safety net (the dedup map
        # already fired during training-data normalization, but the model
        # could in principle predict both ``bra`` and ``black bra``;
        # apply_rules drops ``bra`` in that case).
        flat: List[str] = []
        for cat in SLOT_ORDER:
            flat.extend(slotted.get(cat, []))
        rating_held = flat[:1]
        rest = tr.apply_rules(flat[1:], self.rules)
        out_tags = rating_held + rest
        return ", ".join(_underscore_to_space(t) for t in out_tags)
