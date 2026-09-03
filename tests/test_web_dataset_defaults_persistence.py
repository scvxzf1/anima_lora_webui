from __future__ import annotations

from pathlib import Path

import toml

from library.config.loader import ConfigSanitizer
from tests.web_config_test_support import (
    _patch_config_service_paths,
    _write_minimal_config_tree,
)
from web.services import config_service
from web.services.config.metadata import WEBUI_DATASET_DEFAULTS_ATTR_KEY


def test_dataset_preset_roundtrips_defaults_independently_from_first_row(
    tmp_path: Path,
    monkeypatch,
):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    defaults = {
        "resolution": 768,
        "batch_size": 2,
        "prior_loss_weight": 1.25,
        "enable_bucket": False,
        "min_bucket_reso": 192,
        "max_bucket_reso": 1280,
        "bucket_reso_steps": 32,
        "bucket_no_upscale": True,
        "validation_split": 0.2,
        "validation_split_num": 7,
        "validation_seed": 17,
        "caption_extension": ".caption",
        "keep_tokens": 9,
        "prefer_json_caption": True,
        "caption_source_mode": "auto",
    }

    saved = config_service.save_dataset_preset(
        "configs/datasets/default-overrides.toml",
        [
            {
                "source_dir": "image_dataset/a",
                "image_dir": "post_image_dataset/a_resized",
                "cache_dir": "post_image_dataset/a_cache",
                "num_repeats": 1,
                "settings": {
                    "resolution": 1024,
                    "batch_size": 1,
                    "enable_bucket": True,
                    "caption_extension": ".txt",
                    "prefer_json_caption": False,
                    "caption_source_mode": "txt",
                },
            }
        ],
        defaults,
    )

    data = toml.loads(saved["content"])
    stored_defaults = data["general"]["custom_attributes"][WEBUI_DATASET_DEFAULTS_ATTR_KEY]
    assert stored_defaults == defaults

    loaded = config_service.load_dataset_preset(saved["file"])
    assert loaded["defaults"] == defaults

    row_settings = loaded["datasets"][0]["settings"]
    assert row_settings["resolution"] == 1024
    assert row_settings["batch_size"] == 1
    assert row_settings["enable_bucket"] is True
    assert row_settings["caption_extension"] == ".txt"
    assert row_settings["prefer_json_caption"] is False
    assert row_settings["caption_source_mode"] == "txt"

    ConfigSanitizer(support_dropout=True).sanitize_user_config(data)


def test_dataset_editor_roundtrips_defaults_independently_from_first_row(
    tmp_path: Path,
    monkeypatch,
):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    row = {
        "source_dir": "image_dataset/a",
        "image_dir": "post_image_dataset/a_resized",
        "cache_dir": "post_image_dataset/a_cache",
        "num_repeats": 1,
        "settings": {
            "resolution": 1024,
            "prefer_json_caption": False,
            "caption_source_mode": "txt",
        },
    }
    defaults = {
        "resolution": 768,
        "prefer_json_caption": True,
        "caption_source_mode": "auto",
    }

    saved = config_service.save_dataset_editor(
        "lora",
        "default",
        "imported",
        [row],
        defaults=defaults,
        train_file="configs/imported/lora.toml",
    )
    loaded = config_service.load_dataset_editor(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/lora.toml",
    )

    assert saved["defaults"]["prefer_json_caption"] is True
    assert loaded["defaults"]["resolution"] == 768
    assert loaded["defaults"]["prefer_json_caption"] is True
    assert loaded["defaults"]["caption_source_mode"] == "auto"
    assert loaded["datasets"][0]["settings"]["resolution"] == 1024
    assert loaded["datasets"][0]["settings"]["prefer_json_caption"] is False
    assert loaded["datasets"][0]["settings"]["caption_source_mode"] == "txt"
