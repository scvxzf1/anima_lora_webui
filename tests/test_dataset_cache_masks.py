from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from library.anima.strategy import AnimaLatentsCachingStrategy
from library.datasets.dataset_cache import alpha_mask_required_for_cache


def test_external_preloaded_mask_does_not_require_alpha_npz_key():
    info = SimpleNamespace(mask_path="/tmp/subject_mask.png", preloaded_alpha_mask=None)
    subset = SimpleNamespace(alpha_mask=True, mask_dir="/tmp/masks")

    assert alpha_mask_required_for_cache(info, subset) is False


def test_embedded_alpha_mask_still_requires_alpha_npz_key():
    info = SimpleNamespace(mask_path=None, preloaded_alpha_mask=None)
    subset = SimpleNamespace(alpha_mask=True, mask_dir=None)

    assert alpha_mask_required_for_cache(info, subset) is True


def test_unmasked_full_cache_never_requires_alpha_npz_key():
    info = SimpleNamespace(mask_path=None, preloaded_alpha_mask=None)
    subset = SimpleNamespace(alpha_mask=False, mask_dir="")

    assert alpha_mask_required_for_cache(info, subset) is False


def test_skip_cache_check_still_requires_current_bucket_latent_key(tmp_path):
    npz_path = tmp_path / "image_1024x1024_anima.npz"
    np.savez(
        npz_path,
        latents_128x128=np.zeros((16, 128, 128), dtype=np.float32),
    )
    strategy = AnimaLatentsCachingStrategy(
        cache_to_disk=True,
        batch_size=1,
        skip_disk_cache_validity_check=True,
    )

    assert not strategy.is_disk_cached_latents_expected(
        (768, 768),
        str(npz_path),
        flip_aug=False,
        alpha_mask=False,
    )


def test_skip_cache_check_accepts_matching_bucket_latent_key(tmp_path):
    npz_path = tmp_path / "image_0768x0768_anima.npz"
    np.savez(
        npz_path,
        latents_96x96=np.zeros((16, 96, 96), dtype=np.float32),
    )
    strategy = AnimaLatentsCachingStrategy(
        cache_to_disk=True,
        batch_size=1,
        skip_disk_cache_validity_check=True,
    )

    assert strategy.is_disk_cached_latents_expected(
        (768, 768),
        str(npz_path),
        flip_aug=False,
        alpha_mask=False,
    )
