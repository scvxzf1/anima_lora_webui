from __future__ import annotations

from types import SimpleNamespace

import pytest

from library.datasets.group import DatasetGroup
from library.datasets.subsets import split_train_val


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


def test_validation_split_fraction_controls_train_and_val_slices():
    paths = [f"image-{i}.png" for i in range(10)]
    sizes = [None] * len(paths)

    train_paths, _train_sizes = split_train_val(
        paths,
        sizes,
        True,
        validation_split=0.2,
        validation_seed=42,
    )
    val_paths, _val_sizes = split_train_val(
        paths,
        sizes,
        False,
        validation_split=0.2,
        validation_seed=42,
    )

    assert len(train_paths) == 8
    assert len(val_paths) == 2
    assert set(train_paths).isdisjoint(val_paths)
    assert set(train_paths) | set(val_paths) == set(paths)


def test_validation_split_num_overrides_fraction_in_training_split():
    paths = [f"image-{i}.png" for i in range(10)]
    sizes = [None] * len(paths)

    train_paths, _train_sizes = split_train_val(
        paths,
        sizes,
        True,
        validation_split=0.8,
        validation_seed=42,
        validation_split_num=3,
    )
    val_paths, _val_sizes = split_train_val(
        paths,
        sizes,
        False,
        validation_split=0.8,
        validation_seed=42,
        validation_split_num=3,
    )

    assert len(train_paths) == 7
    assert len(val_paths) == 3
    assert set(train_paths).isdisjoint(val_paths)
    assert set(train_paths) | set(val_paths) == set(paths)
