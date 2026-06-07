from __future__ import annotations

from library.datasets.buckets import (
    CONSTANT_TOKEN_BUCKETS,
    BucketManager,
    make_constant_token_bucket_resolutions,
)


def _token_counts(resos: list[tuple[int, int]]) -> set[int]:
    return {(width // 16) * (height // 16) for width, height in resos}


def test_constant_token_bucket_table_stays_on_canonical_token_families():
    assert _token_counts(list(CONSTANT_TOKEN_BUCKETS)) == {4032, 4200}


def test_constant_token_buckets_keep_canonical_1024_table():
    manager = BucketManager(
        max_reso=(1024, 1024), min_size=256, max_size=2048, reso_steps=64
    )
    manager.make_buckets(constant_token_buckets=True)

    assert manager.predefined_resos == list(CONSTANT_TOKEN_BUCKETS)
    assert _token_counts(manager.predefined_resos) == {4032, 4200}


def test_constant_token_buckets_scale_with_non_1024_resolution():
    manager = BucketManager(
        max_reso=(768, 768), min_size=256, max_size=2048, reso_steps=64
    )
    manager.make_buckets(constant_token_buckets=True)

    assert manager.predefined_resos == make_constant_token_bucket_resolutions(
        (768, 768), 256, 2048
    )
    assert manager.predefined_resos != list(CONSTANT_TOKEN_BUCKETS)
    assert max(max(width, height) for width, height in manager.predefined_resos) <= 1512
    assert manager.select_bucket(768, 768)[0] != (1008, 1024)
