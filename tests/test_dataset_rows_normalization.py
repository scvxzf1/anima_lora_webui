from __future__ import annotations

import pytest

from web.services.config.dataset_rows import _normalize_preprocess_dataset_settings


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({}, {}),
        ({"resolution": 1536}, {"resolution": 1536}),
        ({"max_bucket_reso": 768}, {"max_bucket_reso": 768}),
    ],
)
def test_partial_preprocess_settings_do_not_require_sibling_keys(raw, expected):
    assert _normalize_preprocess_dataset_settings(raw) == expected


def test_preprocess_settings_raise_max_bucket_to_explicit_resolution():
    assert _normalize_preprocess_dataset_settings(
        {"resolution": 1536, "max_bucket_reso": 1024}
    ) == {"resolution": 1536, "max_bucket_reso": 1536}
