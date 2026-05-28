from __future__ import annotations

from types import SimpleNamespace

import pytest

from library.datasets.group import DatasetGroup


def _group_with_resos(resos):
    group = DatasetGroup.__new__(DatasetGroup)
    group.datasets = [
        SimpleNamespace(
            bucket_manager=SimpleNamespace(resos=resos),
            image_data={},
        )
    ]
    return group


def test_verify_bucket_reso_steps_accepts_divisible_buckets():
    group = _group_with_resos([(896, 1200), (1024, 1024)])

    group.verify_bucket_reso_steps(16)


def test_verify_bucket_reso_steps_rejects_misaligned_buckets():
    group = _group_with_resos([(896, 1200), (895, 1200)])

    with pytest.raises(ValueError, match="895x1200"):
        group.verify_bucket_reso_steps(16)
