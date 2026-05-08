"""Dataset plumbing for the Anima tagger.

Three pieces:

* :class:`TaggerManifest` — loads ``dataset.json`` (the per-stem
  image-path + multi-hot tag indices + rating-class index emitted by
  ``scripts/train_anima_tagger.py --mode build_vocab``).
* :class:`FeatureCacheBuilder` — encodes each manifest image through a
  frozen PE-Core trunk, mean-pools over patch tokens, and writes a
  per-stem ``.safetensors`` to the cache dir. Idempotent: skips entries
  that already exist.
* :class:`CachedFeatureDataset` — reads the per-stem cache into one
  in-memory tensor and exposes ``(feature, multi_hot, rating_idx)`` tuples
  for the trainer.

The pooling decision (mean over patch tokens) is locked into the cache
file format. Swap pooling → invalidate the cache dir → rebuild.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as st_load
from safetensors.torch import save_file as st_save
from torch.utils.data import Dataset
from tqdm import tqdm

from library.datasets.image_utils import IMAGE_TRANSFORMS
from library.vision.buckets import (
    BucketSpec,
    bucket_pixel_size,
    pick_bucket,
)
from library.vision.encoder import (
    VisionEncoderBundle,
    encode_pe_from_imageminus1to1,
    load_pe_encoder,
)

logger = logging.getLogger(__name__)


def pil_resize_to_bucket(img: Image.Image, spec: BucketSpec) -> Image.Image:
    """LANCZOS-resize a PIL image to its closest bucket size for ``spec``.

    Pre-resizing on the PIL side (high quality LANCZOS) avoids decoding
    multi-megapixel source images into a tensor only to bilinear-resize
    them down inside the encoder. Speeds up cache builds 5–10× on
    high-resolution corpora and removes a quality penalty (LANCZOS >
    bilinear for severe downscales).
    """
    w, h = img.size
    h_p, w_p = pick_bucket(h, w, spec)
    target_h, target_w = bucket_pixel_size((h_p, w_p), spec)
    if (h, w) != (target_h, target_w):
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return img


@dataclass
class TaggerManifest:
    """Trainable-sample manifest emitted by ``--mode build_vocab``."""

    stems: List[str]
    image_paths: List[Path]
    tag_indices: List[List[int]]
    rating_indices: List[int]
    train_stems: List[str]
    val_stems: List[str]
    n_tags: int
    n_ratings: int

    @classmethod
    def from_path(cls, path: Path) -> "TaggerManifest":
        with open(path) as f:
            d = json.load(f)
        return cls(
            stems=list(d["stems"]),
            image_paths=[Path(p) for p in d["image_paths"]],
            tag_indices=[list(idxs) for idxs in d["tag_indices"]],
            rating_indices=list(d["rating_indices"]),
            train_stems=list(d["split"]["train"]),
            val_stems=list(d["split"]["val"]),
            n_tags=int(d["n_tags"]),
            n_ratings=int(d["n_ratings"]),
        )

    def stem_index(self) -> Dict[str, int]:
        return {s: i for i, s in enumerate(self.stems)}


# ── Feature cache ─────────────────────────────────────────────────────────


def _cache_path(cache_dir: Path, stem: str) -> Path:
    return cache_dir / f"{stem}.safetensors"


class FeatureCacheBuilder:
    """Build per-stem mean-pooled PE-Core features into ``cache_dir``.

    Uses a single-image-per-forward path for simplicity (PE-Core supports
    dynamic resolution; we don't need to bucket-batch). One forward per
    image is fast enough that 12K stems finish in ~10–20 minutes on a
    single GPU; a bucketed-batch path can be added later if it shows up
    in profiling.
    """

    def __init__(
        self,
        manifest: TaggerManifest,
        cache_dir: Path,
        device: torch.device,
        encoder_name: str = "pe",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.manifest = manifest
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.encoder_name = encoder_name
        self.dtype = dtype
        self._bundle: Optional[VisionEncoderBundle] = None

    def _bundle_lazy(self) -> VisionEncoderBundle:
        if self._bundle is None:
            self._bundle = load_pe_encoder(
                self.device, name=self.encoder_name, dtype=self.dtype
            )
        return self._bundle

    def missing_stems(self) -> List[int]:
        return [
            i
            for i, stem in enumerate(self.manifest.stems)
            if not _cache_path(self.cache_dir, stem).exists()
        ]

    @torch.no_grad()
    def build(self) -> int:
        """Encode + cache every stem missing from ``cache_dir``.

        Returns the count of newly cached entries (0 if everything was
        already cached). Errors on individual images are logged and the
        loop continues — a single corrupt image shouldn't tank the run.
        """
        missing = self.missing_stems()
        if not missing:
            logger.info("feature cache: all %d entries present", len(self.manifest.stems))
            return 0

        logger.info(
            "feature cache: encoding %d missing entries (out of %d total)",
            len(missing),
            len(self.manifest.stems),
        )
        bundle = self._bundle_lazy()
        spec = bundle.bucket_spec
        d_enc = bundle.d_enc
        n_done = 0
        for i in tqdm(missing, desc="pooled-pe", unit="img"):
            stem = self.manifest.stems[i]
            img_path = self.manifest.image_paths[i]
            try:
                with Image.open(img_path) as im:
                    im = pil_resize_to_bucket(im.convert("RGB"), spec)
                    arr = np.array(im)
                tensor = IMAGE_TRANSFORMS(arr).unsqueeze(0)
                feats_list = encode_pe_from_imageminus1to1(
                    bundle, tensor, same_bucket=True
                )
                feats = feats_list[0]                      # [T, d_enc]
                pooled = feats.mean(dim=0).to(torch.float32).cpu()  # [d_enc]
                assert pooled.shape == (d_enc,), pooled.shape
                st_save({"feature": pooled}, str(_cache_path(self.cache_dir, stem)))
                n_done += 1
            except Exception as e:
                logger.warning("failed to encode %s (%s): %s", stem, img_path, e)
        logger.info("feature cache: wrote %d new entries", n_done)
        return n_done


# ── Dataset for the trainer ───────────────────────────────────────────────


class CachedFeatureDataset(Dataset):
    """In-memory ``(feature, multi_hot, rating_idx)`` tuples.

    Loads every cached feature for the requested stems into one tensor at
    init. The full training feature tensor at 12K × 1024 × float32 is ~48
    MB — small enough that we don't bother with lazy disk reads.
    """

    def __init__(
        self,
        manifest: TaggerManifest,
        cache_dir: Path,
        stems_subset: Optional[Sequence[str]] = None,
    ):
        idx_of = manifest.stem_index()
        if stems_subset is None:
            stems_subset = manifest.stems
        kept_stems: List[str] = []
        kept_features: List[torch.Tensor] = []
        kept_tag_idx: List[List[int]] = []
        kept_rating_idx: List[int] = []
        n_missing = 0
        for stem in stems_subset:
            i = idx_of.get(stem)
            if i is None:
                n_missing += 1
                continue
            cache_file = _cache_path(cache_dir, stem)
            if not cache_file.exists():
                n_missing += 1
                continue
            t = st_load(str(cache_file))["feature"]
            kept_stems.append(stem)
            kept_features.append(t)
            kept_tag_idx.append(manifest.tag_indices[i])
            kept_rating_idx.append(manifest.rating_indices[i])
        if not kept_stems:
            raise RuntimeError(
                f"no cached features found in {cache_dir} for the requested "
                f"stems (n_requested={len(stems_subset)}, n_missing={n_missing})"
            )
        if n_missing:
            logger.warning(
                "CachedFeatureDataset: %d stems missing from cache (out of %d "
                "requested) — they will not contribute to training",
                n_missing,
                len(stems_subset),
            )
        self.stems = kept_stems
        self.features = torch.stack(kept_features, dim=0)               # [N, d]
        self.multi_hot = torch.zeros(len(kept_stems), manifest.n_tags)  # [N, T]
        for row, idxs in enumerate(kept_tag_idx):
            self.multi_hot[row, idxs] = 1.0
        self.rating_idx = torch.tensor(kept_rating_idx, dtype=torch.long)
        self.n_tags = manifest.n_tags
        self.n_ratings = manifest.n_ratings
        self.d_in = self.features.shape[-1]

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], self.multi_hot[idx], self.rating_idx[idx]
