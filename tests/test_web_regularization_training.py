from __future__ import annotations

import argparse

from PIL import Image
import toml

from library.config.loader import (
    BlueprintGenerator,
    ConfigSanitizer,
    generate_dataset_group_by_blueprint,
)
from web.services.config.dataset_rows import _build_dataset_config_doc


def test_web_rows_register_cross_dataset_regularization_images(tmp_path):
    train_dir = tmp_path / "train"
    reg_dir = tmp_path / "reg"
    train_dir.mkdir()
    reg_dir.mkdir()
    Image.new("RGB", (64, 64), color=(20, 40, 60)).save(train_dir / "train.png")
    Image.new("RGB", (64, 64), color=(60, 40, 20)).save(reg_dir / "reg.png")
    (train_dir / "train.txt").write_text("subject", encoding="utf-8")
    (reg_dir / "reg.txt").write_text("class", encoding="utf-8")

    settings = {
        "resolution": 64,
        "batch_size": 1,
        "enable_bucket": False,
        "prior_loss_weight": 0.25,
    }
    rows = [
        {
            "source_dir": str(train_dir),
            "image_dir": str(train_dir),
            "cache_dir": str(tmp_path / "train-cache"),
            "num_repeats": 3,
            "is_reg": False,
            "settings": settings,
        },
        {
            "source_dir": str(reg_dir),
            "image_dir": str(reg_dir),
            "cache_dir": str(tmp_path / "reg-cache"),
            "num_repeats": 1,
            "is_reg": True,
            "settings": settings,
        },
    ]
    config = toml.loads(
        _build_dataset_config_doc(
            rows,
            {**settings, "caption_extension": ".txt", "keep_tokens": 0},
        )
    )
    args = argparse.Namespace(
        train_batch_size=None,
        debug_dataset=False,
        max_token_length=None,
        prior_loss_weight=1.0,
    )

    blueprint = BlueprintGenerator(ConfigSanitizer(support_dropout=True)).generate(
        config,
        args,
    )
    group, _ = generate_dataset_group_by_blueprint(blueprint.dataset_group)

    assert len(config["datasets"]) == 2
    assert len(group.datasets[0].image_data) == 1
    assert len(group.datasets[1].image_data) == 1
    assert len(group.datasets[1]) == 3
    reg_info = next(iter(group.datasets[1].image_data.values()))
    assert reg_info.is_reg is True
    assert reg_info.num_repeats == 3
    assert group.datasets[1].prior_loss_weight == 0.25
    assert group.num_train_images == 3
    assert group.num_reg_images == 1


def test_same_dataset_regularization_balance_remains_unchanged(tmp_path):
    train_dir = tmp_path / "train"
    reg_dir = tmp_path / "reg"
    train_dir.mkdir()
    reg_dir.mkdir()
    Image.new("RGB", (64, 64), color=(20, 40, 60)).save(train_dir / "train.png")
    Image.new("RGB", (64, 64), color=(60, 40, 20)).save(reg_dir / "reg.png")
    (train_dir / "train.txt").write_text("subject", encoding="utf-8")
    (reg_dir / "reg.txt").write_text("class", encoding="utf-8")
    config = {
        "general": {"caption_extension": ".txt"},
        "datasets": [
            {
                "batch_size": 1,
                "prior_loss_weight": 0.5,
                "subsets": [
                    {"image_dir": str(train_dir), "num_repeats": 2},
                    {
                        "image_dir": str(reg_dir),
                        "num_repeats": 1,
                        "is_reg": True,
                    },
                ],
            }
        ],
    }
    args = argparse.Namespace(
        train_batch_size=None,
        debug_dataset=False,
        max_token_length=None,
        prior_loss_weight=1.0,
    )

    blueprint = BlueprintGenerator(ConfigSanitizer(support_dropout=True)).generate(
        config,
        args,
    )
    group, _ = generate_dataset_group_by_blueprint(blueprint.dataset_group)

    assert len(group.datasets) == 1
    reg_infos = [info for info in group.image_data.values() if info.is_reg]
    assert len(reg_infos) == 1
    assert reg_infos[0].num_repeats == 2
    assert len(group.datasets[0]) == 4
