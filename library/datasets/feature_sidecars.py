"""Feature/sidecar helpers for BaseDataset variants.

Houses IP-Adapter PE cache loading, identity-pair setup, soft-token contrastive
negatives, inversion runs, BYG tuples, and related image reloads. Extracted from
``BaseDataset`` so the core dataset class stays focused on buckets/caching/getitem.
"""

from __future__ import annotations

import logging
import os
import random
from typing import Dict, Optional

import torch

from library.datasets.subsets import ImageInfo  # noqa: F401 - type annotations

logger = logging.getLogger(__name__)


class FeatureSidecarMixin:
    """Mixin providing optional training-feature loaders for BaseDataset."""

    def _try_load_ip_features(self, image_abs_path: str) -> Optional[torch.Tensor]:
        """Load ``{stem}_anima_{encoder}.safetensors`` produced by
        ``scripts/preprocess/cache_pe_encoder.py``.

        Looks first in the subset's ``cache_dir`` (when set) and falls back to
        the legacy sidecar location next to the source image, so existing
        datasets keep working unchanged.

        Returns a ``[T_pe, d_enc]`` float tensor, or ``None`` if disabled. When
        the flag is on but the file is missing, raises so the user gets a clear
        pointer to re-run ``make preprocess-pe`` instead of silently training
        with a partially-cached dataset.
        """
        if not self.ip_features_cache_to_disk:
            return None
        from safetensors.torch import load_file

        stem = os.path.splitext(os.path.basename(image_abs_path))[0]
        suffix = f"_anima_{self.ip_features_encoder}.safetensors"
        subset = self.image_to_subset.get(image_abs_path)
        cache_dir = getattr(subset, "cache_dir", None) if subset is not None else None
        image_dir = getattr(subset, "image_dir", None) if subset is not None else None
        candidates: list[str] = []
        if cache_dir:
            # Nested-mirror lookup first (image_dataset/charA/img1.png →
            # cache_dir/charA/img1_anima_pe.safetensors); fall back to the
            # legacy flat layout so caches written before nested support
            # still resolve when the source image sits at the tree root.
            from library.io.cache import resolve_cache_path

            nested = resolve_cache_path(
                image_abs_path, suffix, cache_dir=str(cache_dir), image_dir=image_dir
            )
            candidates.append(nested)
            flat = os.path.join(str(cache_dir), stem + suffix)
            if flat != nested:
                candidates.append(flat)
        candidates.append(os.path.join(os.path.dirname(image_abs_path), stem + suffix))
        cache_path = next((c for c in candidates if os.path.exists(c)), None)
        if cache_path is None:
            raise FileNotFoundError(
                f"PE feature cache missing for {image_abs_path}. "
                f"Looked in: {candidates}. Run `make preprocess-pe`, or set "
                f"ip_features_cache_to_disk=false to fall back to live PE encoding."
            )
        sd = load_file(cache_path)
        feats = sd.get("image_features")
        if feats is None:
            raise KeyError(
                f"Cache {cache_path} has no 'image_features' key; "
                f"keys={list(sd.keys())}. Re-run `make preprocess-pe`."
            )
        # Hand back the on-disk dtype unchanged (bf16 by default; see
        # scripts/preprocess/cache_pe_encoder.py --dtype). The IP-Adapter resampler
        # runs in bf16, so upcasting to fp32 here only doubles CPU memory and
        # H2D bandwidth before being cast right back down.
        return feats

    def setup_identity_pairs(
        self,
        index_path: str,
        *,
        mode: str,
        prob: float,
        min_level: str,
        caption_strip_p: float,
        is_validation: bool,
    ) -> None:
        """Attach an IdentityPairSampler so ``__getitem__`` draws a distinct
        same-identity reference for the IP path. ``mode`` is one of
        ``identity`` / ``identity_cross_artist`` (``self`` should not call
        this). For training the candidate pool is restricted to this dataset's
        registered stems (no validation-image leakage); for validation it spans
        the whole index so each held-out target can reach its identity siblings
        in the training pool (the deployment condition)."""
        from library.datasets.identity_pairs import IdentityPairSampler

        registered = {
            os.path.splitext(os.path.basename(info.absolute_path))[0]
            for info in self.image_data.values()
        }
        restrict = None if is_validation else registered
        self.identity_pair_sampler = IdentityPairSampler(
            index_path,
            min_level=min_level,
            cross_artist=(mode == "identity_cross_artist"),
            restrict_stems=restrict,
        )
        self.ip_pair_prob = float(prob)
        self.ip_pair_caption_strip_p = float(caption_strip_p)
        self.ip_pair_is_validation = bool(is_validation)
        n_missing = sum(1 for s in registered if not self.identity_pair_sampler.has(s))
        if n_missing:
            logger.warning(
                f"[ip-pair] {n_missing}/{len(registered)} registered stems are "
                f"absent from {index_path} (will self-pair). Re-run "
                f"`make caption-index` if the dataset changed."
            )

    def _load_ip_features_for_stem(
        self, stem: str, subset, rel_dir: str
    ) -> Optional[torch.Tensor]:
        """Load a *reference* stem's cached PE features by reconstructing its
        nested cache path (``cache_dir/<rel_dir>/<stem>_anima_<enc>.safetensors``,
        with a flat fallback). Unlike ``_try_load_ip_features`` this resolves a
        stem that may not be a registered image of this dataset (the pair
        partner often lives in a different subset/split)."""
        if not self.ip_features_cache_to_disk:
            return None
        from safetensors.torch import load_file

        suffix = f"_anima_{self.ip_features_encoder}.safetensors"
        cache_dir = getattr(subset, "cache_dir", None) if subset is not None else None
        candidates: list[str] = []
        if cache_dir:
            if rel_dir:
                candidates.append(os.path.join(str(cache_dir), rel_dir, stem + suffix))
            candidates.append(os.path.join(str(cache_dir), stem + suffix))
        cache_path = next((c for c in candidates if os.path.exists(c)), None)
        if cache_path is None:
            raise FileNotFoundError(
                f"PE feature cache missing for reference stem {stem!r}. "
                f"Looked in: {candidates}. Run `make preprocess-pe`."
            )
        feats = load_file(cache_path).get("image_features")
        if feats is None:
            raise KeyError(
                f"Cache {cache_path} has no 'image_features' key. "
                f"Re-run `make preprocess-pe`."
            )
        return feats

    def setup_contrastive_negatives(
        self,
        index_path: str,
        *,
        k: int,
        mode: str,
        is_validation: bool,
    ) -> None:
        """Attach an IdentityPairSampler so ``__getitem__`` surfaces ``k``
        cached negative text embeddings (``neg_crossattn_emb``) per example for
        the soft-tokens contrastive objective.

        ``mode`` (docs/proposal/soft_tokens_contrastive.md):
          - ``shuffled``    — an unrelated image (no character/copyright overlap).
          - ``jaccard``     — shuffled sourcing + a per-negative tag-overlap
            weight (``neg_jaccard``) the loss uses to down-weight near-misses.
          - ``hard``        — a same-artist / different-character sibling (falls
            back to shuffled for orphan artists).
          - ``hard_backoff`` — tiered hard negative: same-artist/different-
            character → same-copyright/different-character → shuffled. The
            copyright tier rescues most of ``hard``'s ~71% orphan fallback.

        The candidate pool is restricted to this dataset's registered stems so
        negatives never leak in from another split."""
        if mode not in ("shuffled", "jaccard", "hard", "hard_backoff"):
            raise ValueError(
                "contrastive_negative_mode must be shuffled/jaccard/hard/"
                f"hard_backoff, got {mode!r}"
            )
        from library.datasets.identity_pairs import IdentityPairSampler

        registered = {
            os.path.splitext(os.path.basename(info.absolute_path))[0]
            for info in self.image_data.values()
        }
        self.contrastive_neg_sampler = IdentityPairSampler(
            index_path,
            min_level="artist",
            cross_artist=False,
            restrict_stems=registered,
        )
        self.contrastive_neg_k = int(k)
        self.contrastive_neg_mode = str(mode)
        n_missing = sum(
            1 for s in registered if not self.contrastive_neg_sampler.has(s)
        )
        if n_missing:
            logger.warning(
                f"[contrastive] {n_missing}/{len(registered)} registered stems "
                f"are absent from {index_path} (will skip negatives for those). "
                f"Re-run `make caption-index` if the dataset changed."
            )

        # One-shot hardness diagnostic: tally the negative *level* each registered
        # stem would draw under this mode (one deterministic draw per stem). Lets
        # you read the strict-vs-shuffled mix before committing to a run — e.g.
        # how much of `hard`'s shuffled fallback the `hard_backoff` copyright tier
        # actually rescues. Skipped for shuffled/jaccard (every draw is shuffled).
        if mode in ("hard", "hard_backoff"):
            from collections import Counter

            diag_rng = random.Random(0)
            hist: Counter[str] = Counter()
            for s in sorted(registered):
                if self.contrastive_neg_sampler.has(s):
                    _, lvl = self.contrastive_neg_sampler.draw(s, mode, diag_rng)
                    hist[lvl] += 1
            total = sum(hist.values())
            if total:
                breakdown = ", ".join(
                    f"{lvl}={n} ({100 * n / total:.0f}%)"
                    for lvl, n in sorted(hist.items(), key=lambda kv: -kv[1])
                )
                logger.info(
                    f"[contrastive] negative-level mix ({mode}, n={total}): {breakdown}"
                )

    def _load_te_for_stem(
        self, stem: str, subset, rel_dir: str
    ) -> Optional[torch.Tensor]:
        """Load a *negative* stem's cached text embedding (post-LLM-adapter
        ``crossattn_emb``) by reconstructing its nested cache path. Mirrors
        ``_load_ip_features_for_stem`` but swaps the PE feature for the TE
        feature (``{stem}_anima_te.safetensors``). Returns ``(S, D)`` or None."""
        from safetensors import safe_open

        suffix = "_anima_te.safetensors"
        cache_dir = getattr(subset, "cache_dir", None) if subset is not None else None
        candidates: list[str] = []
        if cache_dir:
            if rel_dir:
                candidates.append(os.path.join(str(cache_dir), rel_dir, stem + suffix))
            candidates.append(os.path.join(str(cache_dir), stem + suffix))
        cache_path = next((c for c in candidates if os.path.exists(c)), None)
        if cache_path is None:
            raise FileNotFoundError(
                f"TE cache missing for contrastive negative stem {stem!r}. "
                f"Looked in: {candidates}. Run `make preprocess-te` with "
                f"cache_llm_adapter_outputs=true."
            )
        with safe_open(cache_path, framework="pt") as f:
            keys = set(f.keys())
            # Prefer the pristine v0 variant; fall back to single-variant cache.
            for key in ("crossattn_emb_v0", "crossattn_emb"):
                if key in keys:
                    return f.get_tensor(key)
        raise KeyError(
            f"TE cache {cache_path} has no 'crossattn_emb' key — the negative "
            f"requires cache_llm_adapter_outputs=true. Re-run `make preprocess-te`."
        )

    @staticmethod
    def _strip_identity_tags(caption: str, meta: dict) -> str:
        """Drop the target's character/copyright tags from a comma-separated
        caption (case-insensitive), so identity must flow through the IP image
        path rather than the text. Leaves all other tags (incl. artist) intact.
        No-op when ``caption`` carries no comma structure or no identity tag
        matches."""
        drop = {
            t.strip().lower()
            for t in (meta.get("character", []) + meta.get("copyright", []))
            if t.strip()
        }
        if not drop or "," not in caption:
            return caption
        kept = [tok for tok in caption.split(",") if tok.strip().lower() not in drop]
        return ",".join(kept)

    def _try_load_inversion_runs(self, image_abs_path: str) -> Optional[torch.Tensor]:
        """Load <stem>_inverted_run{0..N-1}.safetensors from self.inversion_dir.

        Returns a [N_runs, S, D] tensor, or None if any of the expected runs is missing
        (caller masks samples without inversions out of the functional loss).
        """
        if not self.inversion_dir:
            return None
        stem = os.path.splitext(os.path.basename(image_abs_path))[0]
        from safetensors.torch import load_file

        runs = []
        for i in range(self.inversion_num_runs):
            p = os.path.join(self.inversion_dir, f"{stem}_inverted_run{i}.safetensors")
            if not os.path.exists(p):
                return None
            sd = load_file(p)
            t = sd.get("crossattn_emb")
            if t is None:
                return None
            runs.append(t.float())
        return torch.stack(runs, dim=0)  # [N_runs, S, D]

    def _load_cond_latent(
        self, subset, image_info, flipped: bool
    ) -> Optional[torch.Tensor]:
        """Load a stem-matched condition latent for cond!=target tasks."""
        cond_dir = getattr(subset, "cond_cache_dir", None)
        if not cond_dir:
            return None
        npz_path = self.latents_caching_strategy.get_latents_npz_path(
            image_info.absolute_path,
            image_info.bucket_reso,
            cache_dir=str(cond_dir),
            image_dir=subset.image_dir,
        )
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"Condition latent cache missing for {image_info.absolute_path!r}: "
                f"{npz_path}. Run the condition prep step first."
            )
        cond, _, _, cond_flipped, _ = self.latents_caching_strategy.load_latents_from_disk(
            npz_path, image_info.bucket_reso
        )
        if flipped:
            if cond_flipped is None:
                raise ValueError(
                    f"flip_aug is on but condition cache {npz_path} has no "
                    "flipped latent. Set flip_aug=false or regenerate the cache."
                )
            cond = cond_flipped
        return torch.FloatTensor(cond)

    def restrict_to_byg_tuples(self) -> tuple[int, int]:
        """Drop images without a BYG edit-tuple sidecar and rebuild buckets."""
        if not self.byg_text_dir:
            return (0, 0)
        kept: Dict[str, ImageInfo] = {}
        dropped = 0
        for key, info in self.image_data.items():
            stem = os.path.splitext(os.path.basename(info.absolute_path))[0]
            path = os.path.join(self.byg_text_dir, f"{stem}_byg.safetensors")
            if os.path.exists(path):
                kept[key] = info
            else:
                dropped += 1
        if dropped == 0:
            return (len(kept), 0)
        self.image_data = kept
        self.num_train_images = sum(info.num_repeats for info in kept.values())
        self.bucket_manager = None
        self.make_buckets(
            constant_token_buckets=getattr(self, "_constant_token_buckets", True)
        )
        return (len(kept), dropped)

    def _try_load_byg_tuple(self, image_abs_path: str) -> Optional[dict]:
        """Load <stem>_byg.safetensors from ``self.byg_text_dir``."""
        if not self.byg_text_dir:
            return None
        stem = os.path.splitext(os.path.basename(image_abs_path))[0]
        path = os.path.join(self.byg_text_dir, f"{stem}_byg.safetensors")
        if not os.path.exists(path):
            return None
        from safetensors.torch import load_file

        sd = load_file(path)
        out = {}
        for role in self._byg_roles:
            emb = sd.get(f"{role}_emb")
            if emb is None:
                return None
            out[f"{role}_emb"] = emb.float()
            mask = sd.get(f"{role}_mask")
            if mask is not None:
                out[f"{role}_mask"] = mask
        return out

    def _load_image_at_bucket(self, subset, image_info, flipped: bool) -> torch.Tensor:
        """Reload the source image at bucket resolution for IP-Adapter live
        PE encoding alongside cached latents.

        Skips augmentation, alpha-mask, and face-crop logic — those are
        already baked into the cached latent. PE will resize to its own
        bucket on the GPU side, so we only need a tensor that matches the
        latent's spatial alignment (resize to bucket + flip if the latent
        is its flipped variant).
        """
        from library.datasets.image_utils import trim_and_resize_if_required

        img, _, _, _, _ = self.load_image_with_face_info(
            subset, image_info.absolute_path, subset.alpha_mask
        )
        img, _, _ = trim_and_resize_if_required(
            False,  # force deterministic crop — must match the cached latent
            img,
            image_info.bucket_reso,
            image_info.resized_size,
            resize_interpolation=image_info.resize_interpolation,
        )
        if flipped:
            img = img[:, ::-1, :].copy()
        img = img[:, :, :3]
        return self.image_transforms(img)


