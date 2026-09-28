from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
from voluptuous import MultipleInvalid

from library.config.loader import ConfigSanitizer, DreamBoothSubsetParams
from library.datasets.subsets import DreamBoothSubset, normalize_sample_ratio
from library.models.family_registry import model_family_capability_catalog
from web.services.config.preflight_compat import _check_core_training_semantics
from web.services.config.preflight_runtime import _sync_from_facade


def _checks(cfg: dict, rows: list[dict] | None = None) -> list[dict[str, str]]:
    _sync_from_facade()
    checks: list[dict[str, str]] = []

    def add(level: str, key: str, message: str, *_args) -> None:
        checks.append({"level": level, "key": key, "message": message})

    _check_core_training_semantics(cfg, add, dataset_rows=rows)
    return checks


@pytest.mark.parametrize(("value", "expected"), [(1, 1.0), (0.5, 0.5), ("0.25", 0.25)])
def test_sample_ratio_accepts_only_finite_unit_interval_values(value, expected):
    assert normalize_sample_ratio(value) == expected


@pytest.mark.parametrize("value", [0, -0.1, 1.01, float("nan"), float("inf"), True, "bad"])
def test_sample_ratio_rejects_values_the_training_runtime_cannot_use(value):
    with pytest.raises(ValueError, match="sample_ratio"):
        normalize_sample_ratio(value)

    config = {"datasets": [{"subsets": [{"image_dir": "images", "sample_ratio": value}]}]}
    with pytest.raises(MultipleInvalid):
        ConfigSanitizer(support_dropout=True).sanitize_user_config(config)

    params = DreamBoothSubsetParams(image_dir="images", sample_ratio=value)
    with pytest.raises(ValueError, match="sample_ratio"):
        DreamBoothSubset(**asdict(params))


def test_preflight_matches_epoch_override_and_fixed_step_semantics():
    invalid_epochs = _checks({"max_train_epochs": 0, "max_train_steps": 100})
    assert [(item["level"], item["key"]) for item in invalid_epochs] == [
        ("error", "max_train_epochs")
    ]

    epoch_override = _checks({"max_train_epochs": 2, "max_train_steps": 100})
    assert [(item["level"], item["key"]) for item in epoch_override] == [
        ("warning", "max_train_steps")
    ]

    assert _checks({"max_train_steps": 100}) == []
    assert _checks({"max_train_steps": 0})[0]["key"] == "max_train_steps"


def test_preflight_keeps_weight_and_resume_retention_semantics_separate():
    zero = _checks(
        {
            "max_train_steps": 1,
            "save_last_n_epochs": 0,
            "checkpointing_last_n_epochs": 0,
        }
    )
    assert {item["key"] for item in zero} == {
        "save_last_n_epochs",
        "checkpointing_last_n_epochs",
    }

    keep_all = _checks(
        {
            "max_train_steps": 1,
            "save_last_n_epochs": -2,
            "checkpointing_last_n_epochs": -1,
        }
    )
    assert keep_all == []

    invalid_resume = _checks(
        {"max_train_steps": 1, "checkpointing_last_n_epochs": -2}
    )
    assert invalid_resume[0]["key"] == "checkpointing_last_n_epochs"


def test_preflight_requires_weights_before_dimensions_can_be_inferred():
    missing = _checks({"max_train_steps": 1, "dim_from_weights": True})
    assert missing[0]["key"] == "dim_from_weights"
    assert "rank/alpha" in missing[0]["message"]

    configured = _checks(
        {
            "max_train_steps": 1,
            "dim_from_weights": True,
            "network_weights": "adapter.safetensors",
        }
    )
    assert configured == []


def _image_rows(image_dir: Path, count: int) -> list[dict]:
    image_dir.mkdir()
    for index in range(count):
        (image_dir / f"{index:03d}.png").write_bytes(b"image")
    return [
        {
            "source_dir": str(image_dir),
            "recursive": True,
            "path_pattern": "*",
            "settings": {"validation_split": 0.2},
        }
    ]


def test_preflight_matches_small_dataset_validation_fallback(tmp_path: Path):
    small = _checks(
        {"max_train_steps": 1, "use_cmmd": True},
        _image_rows(tmp_path / "small", 3),
    )
    assert {item["key"] for item in small} == {"validation_split", "use_cmmd"}

    boundary = _checks(
        {"max_train_steps": 1, "use_cmmd": True},
        _image_rows(tmp_path / "boundary", 100),
    )
    assert boundary == []


def test_public_model_family_catalog_exposes_runtime_relevant_capabilities():
    catalog = {item["name"]: item for item in model_family_capability_catalog()}

    assert catalog["anima"]["latent_cache_suffix"] == "_anima.npz"
    assert catalog["krea2_raw"]["aliases"] == ["krea2", "krea2_raw"]
    assert catalog["krea2_raw"]["plain_lora_only"] is False
    assert catalog["krea2_raw"]["supported_network_specs"] is None
    assert set(catalog["krea2_raw"]["supported_attention_modes"]) == {
        "flash",
        "sdpa",
        "torch",
    }
    assert catalog["qwen_image_2_1"]["supported_tasks"] == ["edit", "t2i"]
    assert catalog["z_image"]["text_cache_suffix"] == "_z_image_te.safetensors"
    assert catalog["z_image"]["pipeline_parallel"]["runtime_available"] is False
