"""Tests for the M3 config schema: validation, provenance, print-config.

Covers:

* schema population (known keys present, aliases resolved)
* typo detection (unknown key → warning with file:line; strict → raises)
* off-list ``choices`` rejection
* soft type coercion (TOML ``1`` → ``float`` when schema says float)
* every ``methods × presets`` combination round-trips without warnings
* ``_render_merged_toml`` output re-parses as valid TOML whose keys are
  a subset of the populated schema
"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import pytest
import toml

from library.config import schema as config_schema
from library.config.io import (
    _flatten_toml,
    _render_merged_toml,
    load_method_preset,
    load_preset_section,
    list_presets,
)
from library.config.loader import load_user_config
from library.config.provenance import explain_key, trace_method_config
from library.datasets.subsets import DreamBoothSubset
from library.env import project_root
from tests.conftest import iter_method_names


def _repo_configs_root() -> Path:
    return project_root() / "configs"


def test_cuda_track_stays_bitsandbytes_compatible():
    root = project_root()
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    lock = (root / "uv.lock").read_text(encoding="utf-8")

    assert "https://download.pytorch.org/whl/cu130" in pyproject
    assert "https://download.pytorch.org/whl/cu132" not in pyproject
    assert "bitsandbytes>=0.49.2" in pyproject
    assert 'name = "bitsandbytes"' in lock
    assert 'name = "cuda-toolkit"\nversion = "13.0.2"' in lock
    assert "flash_attn-2.8.3+cu130torch2.12" in lock
    assert "flash_attn-2.8.3+cu132torch2.12" not in lock


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
    raw_attributes = raw["datasets"][0]["subsets"][0]["custom_attributes"]
    try:
        pickle.dumps(raw_attributes)
    except AttributeError:
        pass
    else:
        raise AssertionError("test fixture no longer reproduces toml inline table")

    loaded = load_user_config(str(config_path))
    attributes = loaded["datasets"][0]["subsets"][0]["custom_attributes"]

    assert type(attributes) is dict
    assert attributes == {"source_dir": "raw/images"}
    pickle.dumps(attributes)


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


def _write_provenance_config_tree(root: Path) -> None:
    (root / "methods").mkdir(parents=True)
    (root / "base.toml").write_text(
        "\n".join(
            [
                'network_module = "networks.lora_anima"',
                "network_dim = 8",
                "blocks_to_swap = 0",
                "learning_rate = 0.0001",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "presets.toml").write_text(
        "\n".join(
            [
                "[default]",
                "blocks_to_swap = 0",
                "[low]",
                "blocks_to_swap = 12",
                "learning_rate = 0.0002",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "methods" / "demo.toml").write_text(
        "network_dim = 32\nlearning_rate = 0.0003\n",
        encoding="utf-8",
    )


def test_trace_method_config_records_layer_history(tmp_path: Path) -> None:
    _write_provenance_config_tree(tmp_path)

    trace = trace_method_config("demo", "low", configs_dir=tmp_path)

    assert trace["values"]["network_dim"] == 32
    assert trace["values"]["blocks_to_swap"] == 12
    assert trace["values"]["learning_rate"] == 0.0003
    learning_rate = explain_key(trace, "learning_rate")
    assert [item["kind"] for item in learning_rate["history"]] == [
        "base",
        "preset",
        "method",
    ]
    assert learning_rate["source"].endswith("configs/methods/demo.toml") or learning_rate[
        "source"
    ].endswith("/methods/demo.toml")


def test_trace_method_config_layers_runtime_and_overrides(tmp_path: Path) -> None:
    _write_provenance_config_tree(tmp_path)
    runtime = tmp_path / "runtime.toml"
    runtime.write_text("network_dim = 48\nblocks_to_swap = 20\n", encoding="utf-8")

    trace = trace_method_config(
        "demo",
        "low",
        configs_dir=tmp_path,
        runtime_config=runtime,
        overrides={"network_dim": 64},
    )

    assert trace["values"]["network_dim"] == 64
    assert trace["values"]["blocks_to_swap"] == 20
    network_dim = explain_key(trace, "network_dim")
    assert [item["kind"] for item in network_dim["history"]] == [
        "base",
        "method",
        "runtime",
        "override",
    ]
    assert network_dim["source"] == "CLI/override"


# ---------------------------------------------------------------------------
# Schema population
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def populated_parser():
    import train

    parser = train.setup_parser()
    config_schema.populate_schema(parser, extras=train.build_network_extras())
    return parser


def test_schema_has_known_keys(populated_parser):
    schema = config_schema.get_schema()
    # a handful of must-have keys that come from different argparse layers
    for k in (
        "network_dim",
        "network_alpha",
        "optimizer_type",
        "learning_rate",
        "max_train_epochs",
        "mixed_precision",
        "attn_mode",
        "v100_flash_stability",
        "debug_finite_checks",
        "compile_block_scope",
        "block_swap_transfer_dtype",
        "block_swap_restore_mode",
        "selective_checkpoint_blocks",
        "memory_probe_jsonl",
        "memory_probe_max_steps",
        "peak_probe_jsonl",
        "peak_probe_max_steps",
        "peak_probe_level",
        "preprocess_memory_profile",
        "preprocess_vae_cache_batch_size",
        "preprocess_text_cache_batch_size",
        "preprocess_precision_preference",
        "base_config",  # manual extra
        "stage_schedule_target_groups",  # internal Web runtime mapping
        "use_moe_style",  # network-module allowlist (three-axis routing)
        "lora_fp32_compute",  # LoRA family rank-path fp32 fallback
        "down_init",  # plain LoRA down-projection init policy
        "use_glora",  # GLoRA plugin selector
        "lokr_factor_group_size",  # LoKr grouped projection tuning
        "lokr_project_chunk_bytes",  # LoKr projection row-chunk threshold
        "lokr_grouped_delta_backend",  # LoKr grouped-delta backend selector
        "lokr_use_einsum",  # LoKr decomposed einsum runtime path toggle
        "lokr_decompose_w2",  # LoKr w2 decomposition compatibility override
        "lokr_full_factor",  # LoKr full Kronecker factors with normal scale
        "lokr_allow_legacy_dim",  # resume-only legacy dim=114514 sentinel
    ):
        assert k in schema, f"expected {k!r} in populated schema"


def test_schema_accepts_stage_schedule_target_groups(populated_parser):
    flattened = _flatten_toml(
        {"stage_schedule_target_groups": [[0, 1], [2]]},
        source="config.runtime.toml",
        strict=True,
    )

    assert flattened["stage_schedule_target_groups"] == [[0, 1], [2]]


def test_choices_preserved(populated_parser):
    assert "mem_efficient" in config_schema.get_schema()["attn_mode"].choices
    assert list(config_schema.get_schema()["v100_flash_stability"].choices) == [
        "off",
        "hybrid",
        "safe",
    ]
    mp = config_schema.get_schema()["mixed_precision"]
    assert "bf16" in mp.choices
    assert "no" in mp.choices
    transfer_dtype = config_schema.get_schema()["block_swap_transfer_dtype"]
    assert "bf16" in transfer_dtype.choices
    assert "fp8_e4m3" in transfer_dtype.choices
    assert "int8" in transfer_dtype.choices
    restore_mode = config_schema.get_schema()["block_swap_restore_mode"]
    assert "foreach" in restore_mode.choices
    assert "slab" in restore_mode.choices
    compile_block_scope = config_schema.get_schema()["compile_block_scope"]
    assert "resident" in compile_block_scope.choices
    assert "all" in compile_block_scope.choices
    preprocess_precision = config_schema.get_schema()["preprocess_precision_preference"]
    assert "bf16" in preprocess_precision.choices
    assert "fp16" in preprocess_precision.choices
    assert "fp32" in preprocess_precision.choices
    selective_checkpoint = config_schema.get_schema()["selective_checkpoint"]
    assert "adapter_aware" in selective_checkpoint.choices
    assert "peak_blocks_adapter_aware" in selective_checkpoint.choices
    sample_sampler = config_schema.get_schema()["sample_sampler"]
    assert sample_sampler.default == "euler"
    for option in ("euler", "er_sde", "lcm", "ddim", "dpmsolver++"):
        assert option in sample_sampler.choices



def test_list_presets_includes_new_and_legacy_custom_layouts(tmp_path: Path):
    configs = tmp_path / "configs"
    (configs / "custom" / "presets").mkdir(parents=True)
    (configs / "presets.toml").write_text("[default]\n", encoding="utf-8")
    (configs / "custom" / "presets" / "V100.toml").write_text(
        'attn_mode = "torch"\n', encoding="utf-8"
    )
    (configs / "custom" / "V100.toml").write_text(
        'attn_mode = "flash"\n', encoding="utf-8"
    )
    (configs / "custom" / "legacy.toml").write_text(
        'attn_mode = "torch"\n', encoding="utf-8"
    )

    assert list_presets(str(configs)) == ["V100", "default", "legacy"]
    assert load_preset_section("V100", str(configs)) == {"attn_mode": "torch"}
    assert load_preset_section("legacy", str(configs)) == {"attn_mode": "torch"}


# ---------------------------------------------------------------------------
# Typo / choice detection
# ---------------------------------------------------------------------------


def test_unknown_key_warns(populated_parser, tmp_path: Path, caplog):
    bogus = tmp_path / "bogus.toml"
    bogus.write_text("network_ditm = 16\n")
    with caplog.at_level(logging.WARNING):
        out = _flatten_toml({"a": {"network_ditm": 16}}, source=str(bogus))
    assert out == {"network_ditm": 16}
    assert any("unknown key 'network_ditm'" in rec.getMessage() for rec in caplog.records)
    # line locator should include the line number
    assert any(":1:" in rec.getMessage() for rec in caplog.records)


def test_unknown_key_strict_raises(populated_parser, tmp_path: Path):
    bogus = tmp_path / "bogus.toml"
    bogus.write_text("network_ditm = 16\n")
    with pytest.raises(config_schema.ConfigSchemaError):
        _flatten_toml(
            {"a": {"network_ditm": 16}}, source=str(bogus), strict=True
        )


def test_off_list_choice_warns(populated_parser, caplog):
    with caplog.at_level(logging.WARNING):
        _flatten_toml({"a": {"mixed_precision": "fp4"}}, source="x.toml")
    assert any(
        "mixed_precision" in rec.getMessage() and "not in choices" in rec.getMessage()
        for rec in caplog.records
    )


def test_int_to_float_coerced(populated_parser):
    # schema says network_alpha is float; TOML ``1`` comes in as int.
    out = _flatten_toml({"a": {"network_alpha": 64}}, source="x.toml")
    assert isinstance(out["network_alpha"], float)
    assert out["network_alpha"] == 64.0


def test_train_batch_size_overrides_dataset_config_batch_size():
    import train

    user_config = {
        "datasets": [
            {"batch_size": 1, "subsets": []},
            {"subsets": []},
        ]
    }

    train.AnimaTrainer._apply_train_batch_size_to_user_config(
        user_config,
        argparse.Namespace(train_batch_size=2),
    )

    assert [dataset["batch_size"] for dataset in user_config["datasets"]] == [2, 2]


def test_default_train_batch_size_overrides_dataset_config_batch_size():
    import train

    user_config = {"datasets": [{"batch_size": 4, "subsets": []}]}

    train.AnimaTrainer._apply_train_batch_size_to_user_config(
        user_config,
        argparse.Namespace(train_batch_size=1),
    )

    assert user_config["datasets"][0]["batch_size"] == 1


def test_dataset_config_ignores_legacy_preprocess_only_keys():
    from library.config.loader import ConfigSanitizer

    user_config = {
        "general": {"caption_extension": ".txt"},
        "datasets": [
            {
                "resolution": 1024,
                "batch_size": 1,
                "enable_bucket": True,
                "min_bucket_reso": 256,
                "max_bucket_reso": 1024,
                "bucket_reso_steps": 64,
                "bucket_no_upscale": False,
                "validation_split": 0.025,
                "validation_seed": 42,
                "subsets": [
                    {
                        "image_dir": "post_image_dataset/resized",
                        "cache_dir": "post_image_dataset/lora",
                        "custom_attributes": {"source_dir": "image_dataset"},
                    }
                ],
            }
        ],
    }

    sanitized = ConfigSanitizer(support_dropout=True).sanitize_user_config(user_config)
    dataset = sanitized["datasets"][0]

    for key in ConfigSanitizer.PREPROCESS_ONLY_DATASET_KEYS:
        assert key not in dataset
    assert dataset["batch_size"] == 1
    assert dataset["validation_seed"] == 42
    assert dataset["subsets"][0]["image_dir"] == "post_image_dataset/resized"
    assert dataset["subsets"][0]["custom_attributes"]["preprocess"] == {
        "resolution": 1024,
        "enable_bucket": True,
        "min_bucket_reso": 256,
        "max_bucket_reso": 1024,
        "bucket_reso_steps": 64,
        "bucket_no_upscale": False,
    }


def test_dataset_config_ignores_top_level_stage_schedule_fields():
    from library.config.loader import ConfigSanitizer

    user_config = {
        "stage_schedule_enabled": True,
        "stage_schedule": [
            {
                "name": "low",
                "subset_index": 0,
                "start_pct": 0.0,
                "end_pct": 1.0,
            }
        ],
        "stage_schedule_target_groups": [[0, 1]],
        "general": {"caption_extension": ".txt"},
        "datasets": [
            {
                "batch_size": 1,
                "subsets": [
                    {
                        "image_dir": "post_image_dataset/resized",
                        "cache_dir": "post_image_dataset/lora",
                    }
                ],
            }
        ],
    }

    sanitized = ConfigSanitizer(support_dropout=True).sanitize_user_config(user_config)

    assert "stage_schedule_enabled" not in sanitized
    assert "stage_schedule" not in sanitized
    assert "stage_schedule_target_groups" not in sanitized
    assert sanitized["general"]["caption_extension"] == ".txt"
    assert sanitized["datasets"][0]["batch_size"] == 1
    assert sanitized["datasets"][0]["subsets"][0]["cache_dir"] == "post_image_dataset/lora"


def test_dataset_config_ignores_general_stage_schedule_fields_without_mutation():
    from library.config.loader import ConfigSanitizer

    user_config = {
        "general": {
            "caption_extension": ".txt",
            "stage_schedule_enabled": True,
            "stage_schedule": [
                {
                    "name": "general-stage",
                    "subset_index": 0,
                    "start_pct": 0.0,
                    "end_pct": 1.0,
                }
            ],
            "stage_schedule_target_groups": [[0, 1]],
        },
        "datasets": [
            {
                "subsets": [
                    {
                        "image_dir": "post_image_dataset/resized",
                        "cache_dir": "post_image_dataset/lora",
                    }
                ]
            }
        ],
    }

    sanitized = ConfigSanitizer(support_dropout=True).sanitize_user_config(user_config)

    assert sanitized["general"] == {"caption_extension": ".txt"}
    assert user_config["general"]["stage_schedule_enabled"] is True
    assert user_config["general"]["stage_schedule"][0]["name"] == "general-stage"
    assert user_config["general"]["stage_schedule_target_groups"] == [[0, 1]]


def test_dataset_preprocess_resolution_drives_training_bucket_params():
    from library.config.loader import BlueprintGenerator, ConfigSanitizer

    user_config = {
        "general": {"caption_extension": ".txt"},
        "datasets": [
            {
                "batch_size": 1,
                "subsets": [
                    {
                        "image_dir": "post_image_dataset/resized",
                        "cache_dir": "post_image_dataset/lora",
                        "custom_attributes": {
                            "preprocess": {
                                "resolution": 768,
                                "enable_bucket": "false",
                                "min_bucket_reso": 256,
                                "max_bucket_reso": 1344,
                                "bucket_reso_steps": 32,
                                "bucket_no_upscale": "true",
                            }
                        },
                    }
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
        user_config, args
    )
    params = blueprint.dataset_group.datasets[0].params

    assert params.resolution == 768
    assert params.enable_bucket is False
    assert params.min_bucket_reso == 256
    assert params.max_bucket_reso == 1344
    assert params.bucket_reso_steps == 32
    assert params.bucket_no_upscale is True


def test_regularization_dataset_flags_reach_training_blueprint():
    from library.config.loader import BlueprintGenerator, ConfigSanitizer

    user_config = {
        "general": {"caption_extension": ".txt"},
        "datasets": [
            {
                "batch_size": 1,
                "prior_loss_weight": 2.5,
                "subsets": [
                    {
                        "image_dir": "post_image_dataset/reg",
                        "cache_dir": "post_image_dataset/reg_cache",
                        "num_repeats": 1,
                        "is_reg": True,
                    }
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
        user_config,
        args,
    )
    dataset = blueprint.dataset_group.datasets[0]

    assert dataset.params.prior_loss_weight == 2.5
    assert dataset.subsets[0].params.is_reg is True


def test_training_dataset_uses_square_bucket_when_preprocess_bucket_disabled(tmp_path):
    from library.config.loader import (
        BlueprintGenerator,
        ConfigSanitizer,
        generate_dataset_group_by_blueprint,
    )

    image_dir = tmp_path / "resized"
    image_dir.mkdir()
    from PIL import Image

    Image.new("RGB", (768, 768), color=(20, 40, 60)).save(image_dir / "square.png")

    user_config = {
        "general": {"caption_extension": ".txt"},
        "datasets": [
            {
                "batch_size": 1,
                "subsets": [
                    {
                        "image_dir": str(image_dir),
                        "custom_attributes": {
                            "preprocess": {
                                "resolution": 768,
                                "enable_bucket": False,
                            }
                        },
                    }
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
        user_config, args
    )
    group, _ = generate_dataset_group_by_blueprint(
        blueprint.dataset_group,
        constant_token_buckets=True,
    )
    dataset = group.datasets[0]
    info = next(iter(dataset.image_data.values()))

    assert dataset.enable_bucket is False
    assert dataset.bucket_manager.resos == [(768, 768)]
    assert info.bucket_reso == (768, 768)
    assert info.resized_size == (768, 768)


def test_training_dataset_no_upscale_uses_preprocessed_image_size(tmp_path):
    from library.config.loader import (
        BlueprintGenerator,
        ConfigSanitizer,
        generate_dataset_group_by_blueprint,
    )

    image_dir = tmp_path / "resized"
    image_dir.mkdir()
    from PIL import Image

    Image.new("RGB", (512, 768), color=(20, 40, 60)).save(image_dir / "portrait.png")

    user_config = {
        "general": {"caption_extension": ".txt"},
        "datasets": [
            {
                "batch_size": 1,
                "subsets": [
                    {
                        "image_dir": str(image_dir),
                        "custom_attributes": {
                            "preprocess": {
                                "resolution": 768,
                                "bucket_no_upscale": True,
                            }
                        },
                    }
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
        user_config, args
    )
    group, _ = generate_dataset_group_by_blueprint(
        blueprint.dataset_group,
        constant_token_buckets=True,
    )
    dataset = group.datasets[0]
    info = next(iter(dataset.image_data.values()))

    assert dataset.bucket_no_upscale is True
    assert dataset.bucket_manager.resos == [(512, 768)]
    assert info.bucket_reso == (512, 768)
    assert info.resized_size == (512, 768)


def test_collect_bucket_resolutions_prefers_active_bucket_manager_resos():
    import train

    class BucketManager:
        resos = [(752, 768), (768, 752)]

    class Dataset:
        bucket_manager = BucketManager()
        image_data = {}

    class Group:
        datasets = [Dataset()]

    assert train._collect_bucket_resolutions(Group()) == [(752, 768), (768, 752)]


# ---------------------------------------------------------------------------
# Round-trip: all methods × presets produce no warnings
# ---------------------------------------------------------------------------


METHODS = list(iter_method_names())


def _load_preset_names() -> list[str]:
    configs_root = _repo_configs_root()
    return list(toml.load(configs_root / "presets.toml").keys())


@pytest.mark.parametrize("method", METHODS)
def test_method_configs_clean(populated_parser, method: str, caplog):
    presets = _load_preset_names()
    configs_root = _repo_configs_root()
    for preset in presets:
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            load_method_preset(method, preset, configs_dir=str(configs_root))
        offenders = [
            rec.getMessage()
            for rec in caplog.records
            if rec.levelno >= logging.WARNING and rec.name.startswith("library.train_util")
        ]
        assert not offenders, f"{method} × {preset} warnings: {offenders}"


def test_low_vram_blockswap_preset_is_available(populated_parser):
    configs_root = _repo_configs_root()
    preset = toml.load(configs_root / "presets.toml")["low_vram_blockswap"]
    assert preset["blocks_to_swap"] == 8
    assert preset["gradient_checkpointing"] is True
    assert preset["unsloth_offload_checkpointing"] is False
    assert preset["torch_compile"] is False

    merged = load_method_preset("lora", "low_vram_blockswap", configs_dir=str(configs_root))
    assert merged["blocks_to_swap"] == 8
    assert merged["disable_block_swap_for_eval"] is False


def test_balanced_16g_preset_is_block_swap_first(populated_parser):
    configs_root = _repo_configs_root()
    preset = toml.load(configs_root / "presets.toml")["balanced_16g"]
    assert preset["blocks_to_swap"] == 12
    assert preset["gradient_checkpointing"] is False
    assert preset["unsloth_offload_checkpointing"] is False
    assert preset["torch_compile"] is True
    assert "compile_inductor_mode" not in preset
    assert preset["selective_checkpoint"] == "off"
    assert preset["block_swap_profile_jsonl"] == "off"

    merged = load_method_preset("lora", "balanced_16g", configs_dir=str(configs_root))
    assert merged["blocks_to_swap"] == 12
    assert merged["gradient_checkpointing"] is False
    assert merged["unsloth_offload_checkpointing"] is False
    assert merged["selective_checkpoint"] == "off"
    assert merged["block_swap_profile_jsonl"] == "off"
    assert merged["block_swap_transfer_dtype"] == "bf16"
    assert merged.get("block_swap_restore_mode", "slab") == "slab"
    assert merged["compile_block_scope"] == "resident"


def test_gui_lora_respects_balanced_16g_blockswap(populated_parser):
    configs_root = _repo_configs_root()
    merged = load_method_preset(
        "lora",
        "balanced_16g",
        configs_dir=str(configs_root),
        methods_subdir="gui-methods",
    )
    assert merged["blocks_to_swap"] == 12
    assert merged["unsloth_offload_checkpointing"] is False


# ---------------------------------------------------------------------------
# Provenance + render
# ---------------------------------------------------------------------------


def test_provenance_returned():
    configs_root = _repo_configs_root()
    merged, provenance = load_method_preset(
        "lora", "default", configs_dir=str(configs_root), return_provenance=True
    )
    # base key - provenance 使用标准化的 configs/... 格式
    assert provenance["network_module"] == "configs/base.toml"
    # method key
    assert provenance["network_dim"] == "configs/methods/lora.toml"
    assert set(provenance) == set(merged)


def _reparse_without_comments(text: str) -> dict:
    # toml.loads ignores comments natively, but our output has `# --- from ... ---`
    # headers that are valid TOML comments, so it round-trips directly.
    return toml.loads(text)


def test_render_roundtrips_to_valid_toml(populated_parser):
    import train

    configs_root = _repo_configs_root()
    parser = train.setup_parser()
    config_schema.populate_schema(parser, extras=train.build_network_extras())

    merged, provenance = load_method_preset(
        "lora", "default", configs_dir=str(configs_root), return_provenance=True
    )
    ns = argparse.Namespace(**merged)
    args = parser.parse_args(["--method", "lora", "--preset", "default"], namespace=ns)

    rendered = _render_merged_toml(args, parser, provenance)
    parsed = _reparse_without_comments(rendered)

    schema = config_schema.get_schema()
    for key in parsed:
        assert key in schema, f"rendered key {key!r} not in schema"


def test_render_header_includes_method_and_preset(populated_parser):
    import train

    configs_root = _repo_configs_root()
    parser = train.setup_parser()
    config_schema.populate_schema(parser, extras=train.build_network_extras())

    merged, provenance = load_method_preset(
        "lora", "low_vram", configs_dir=str(configs_root), return_provenance=True
    )
    ns = argparse.Namespace(**merged)
    args = parser.parse_args(
        ["--method", "lora", "--preset", "low_vram"], namespace=ns
    )
    rendered = _render_merged_toml(args, parser, provenance)
    assert "Method: lora" in rendered
    assert "Preset: low_vram" in rendered
    # section ordering: base → preset → method
    # provenance 使用标准化的 configs/... 格式
    base_idx = rendered.index("configs/base.toml")
    preset_idx = rendered.index("configs/presets.toml[low_vram]")
    method_idx = rendered.index("configs/methods/lora.toml")
    assert base_idx < preset_idx < method_idx


def test_base_config_cycle_is_rejected(tmp_path, monkeypatch):
    from library.config import io as config_io
    from library import env as env_mod

    monkeypatch.setattr(env_mod, "get_configs_root", lambda: tmp_path)
    a = tmp_path / "a.toml"
    b = tmp_path / "b.toml"
    a.write_text('base_config = "b.toml"\nnetwork_dim = 8\n', encoding="utf-8")
    b.write_text('base_config = "a.toml"\nnetwork_alpha = 1.0\n', encoding="utf-8")

    with pytest.raises(ValueError, match="cycle"):
        config_io._load_toml_with_base(str(a))


def test_base_config_rejects_path_escape(tmp_path, monkeypatch):
    from library.config import io as config_io
    from library import env as env_mod

    monkeypatch.setattr(env_mod, "get_configs_root", lambda: tmp_path)
    child = tmp_path / "child.toml"
    outside = tmp_path.parent / "outside.toml"
    outside.write_text("network_dim = 4\n", encoding="utf-8")
    child.write_text('base_config = "../outside.toml"\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"\.\.|escapes"):
        config_io._load_toml_with_base(str(child))


def test_base_config_merges_within_root(tmp_path, monkeypatch):
    from library.config import io as config_io
    from library import env as env_mod

    monkeypatch.setattr(env_mod, "get_configs_root", lambda: tmp_path)
    base = tmp_path / "base.toml"
    child = tmp_path / "child.toml"
    base.write_text("network_dim = 4\nnetwork_alpha = 1.0\n", encoding="utf-8")
    child.write_text('base_config = "base.toml"\nnetwork_dim = 16\n', encoding="utf-8")

    merged = config_io._load_toml_with_base(str(child))
    assert merged["network_dim"] == 16
    assert merged["network_alpha"] == 1.0
