from __future__ import annotations

from types import SimpleNamespace

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
