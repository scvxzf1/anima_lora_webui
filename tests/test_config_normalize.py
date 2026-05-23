from __future__ import annotations

import pickle
from pathlib import Path

import toml

from library.config.loader import load_user_config
from library.datasets.subsets import DreamBoothSubset


def test_dataset_inline_table_config_is_pickle_safe(tmp_path: Path) -> None:
    config_path = tmp_path / "dataset.toml"
    config_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "",
                "[[datasets.subsets]]",
                'image_dir = "images"',
                'custom_attributes = {source_dir = "raw/images"}',
            ]
        ),
        encoding="utf-8",
    )

    raw = toml.load(config_path)
    raw_attrs = raw["datasets"][0]["subsets"][0]["custom_attributes"]
    try:
        pickle.dumps(raw_attrs)
    except AttributeError:
        pass
    else:
        raise AssertionError("test fixture no longer reproduces toml inline table")

    loaded = load_user_config(str(config_path))
    attrs = loaded["datasets"][0]["subsets"][0]["custom_attributes"]

    assert type(attrs) is dict
    assert attrs == {"source_dir": "raw/images"}
    pickle.dumps(attrs)


def test_subset_custom_attributes_are_pickle_safe(tmp_path: Path) -> None:
    raw = toml.loads('custom_attributes = {source_dir = "raw/images"}')
    subset = DreamBoothSubset(
        image_dir=str(tmp_path),
        is_reg=False,
        class_tokens=None,
        caption_extension=".txt",
        cache_info=False,
        alpha_mask=False,
        num_repeats=1,
        sample_ratio=1.0,
        caption_separator=",",
        keep_tokens=0,
        keep_tokens_separator=None,
        secondary_separator=None,
        enable_wildcard=False,
        color_aug=False,
        flip_aug=False,
        face_crop_aug_range=None,
        random_crop=False,
        caption_dropout_rate=0.0,
        caption_dropout_every_n_epochs=0,
        caption_tag_dropout_rate=0.0,
        caption_prefix=None,
        caption_suffix=None,
        token_warmup_min=1,
        token_warmup_step=0,
        custom_attributes=raw["custom_attributes"],
    )

    assert type(subset.custom_attributes) is dict
    assert subset.custom_attributes == {"source_dir": "raw/images"}
    pickle.dumps(subset.custom_attributes)
