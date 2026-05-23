from __future__ import annotations

from pathlib import Path

import pytest
import toml
from PIL import Image

from web.services import config_service


def test_save_dataset_editor_does_not_overwrite_dataset_when_train_patch_fails(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    original_dataset = "# keep me\n[[datasets]]\nresolution = 512\n"
    dataset_path.write_text(original_dataset, encoding="utf-8")
    _patch_config_service_paths(monkeypatch, tmp_path)

    with pytest.raises(ValueError, match="TOML 更新失败"):
        config_service.save_dataset_editor(
            "lora",
            "default",
            "imported",
            [
                {
                    "source_dir": "new_source",
                    "image_dir": "new_resized",
                    "cache_dir": "new_cache",
                    "num_repeats": 2,
                }
            ],
            train_file="configs/imported/lora.toml",
            train_content='dataset_config = "configs/datasets/lora.toml"\ninvalid = [\n',
        )

    assert dataset_path.read_text(encoding="utf-8") == original_dataset


def test_save_dataset_editor_restores_dataset_when_train_write_fails(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    original_dataset = "# original\n[[datasets]]\nresolution = 512\n"
    dataset_path.write_text(original_dataset, encoding="utf-8")
    _patch_config_service_paths(monkeypatch, tmp_path)

    def fail_train_save(rel_path: str, content: str, **kwargs):
        if rel_path == "configs/imported/lora.toml":
            return False, "训练 TOML 写入失败"
        return original_save_raw_file(rel_path, content, **kwargs)

    original_save_raw_file = config_service.save_raw_file
    monkeypatch.setattr(config_service, "save_raw_file", fail_train_save)

    with pytest.raises(ValueError, match="训练 TOML 写入失败"):
        config_service.save_dataset_editor(
            "lora",
            "default",
            "imported",
            [
                {
                    "source_dir": "new_source",
                    "image_dir": "new_resized",
                    "cache_dir": "new_cache",
                    "num_repeats": 2,
                }
            ],
            train_file="configs/imported/lora.toml",
        )

    assert dataset_path.read_text(encoding="utf-8") == original_dataset


def test_sample_prompts_roundtrip_preserves_comments_blank_lines_and_spacing(tmp_path: Path, monkeypatch):
    root = tmp_path
    configs = root / "configs"
    configs.mkdir()
    monkeypatch.setattr(config_service, "ROOT", root)

    original = "# 角色 A\n\n  masterpiece, best quality  \n# 角色 B\nsolo, 1girl\n"
    saved = config_service.save_sample_prompts_file(original, "configs/sample_prompts.txt")
    loaded = config_service.load_sample_prompts_file("configs/sample_prompts.txt")

    assert (configs / "sample_prompts.txt").read_text(encoding="utf-8") == original
    assert saved["content"] == original
    assert loaded["content"] == original
    assert loaded["prompts"] == ["masterpiece, best quality", "solo, 1girl"]


def test_raw_patch_ignores_dataset_picker_ui_field(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    train_rel = "configs/imported/lora.toml"

    ok, msg, content, changed = config_service.patch_raw_file_values(
        train_rel,
        {
            "dataset_config_picker": "configs/datasets/character_a.toml",
            "output_name": "clean",
        },
    )

    assert ok is True, msg
    assert changed == ["output_name"]
    assert 'output_name = "clean"' in content
    assert "dataset_config_picker" not in content
    assert "dataset_config_picker" not in (configs / "imported" / "lora.toml").read_text(encoding="utf-8")


def test_locked_user_group_cannot_be_deleted(tmp_path: Path, monkeypatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "web-file-groups.toml").write_text(
        "\n".join(
            [
                "[[groups]]",
                'id = "custom_group"',
                'label = "自定义分组"',
                "open = true",
                "locked = false",
                "trainable = true",
                "user_managed = true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (configs / "web-user-locks.toml").write_text(
        'locked_groups = ["custom_group"]\n',
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    group = config_service.list_config_file_groups()[0]
    assert group["user_group_locked"] is True
    assert group["renamable"] is True
    assert group["deletable"] is False

    ok, message, renamed = config_service.rename_config_file_group("custom_group", "锁定但可重命名")
    assert ok is True
    assert message == "分组已重命名"
    assert renamed["label"] == "锁定但可重命名"

    ok, message = config_service.delete_config_file_group("custom_group")
    assert ok is False
    assert "已锁定" in message


def test_unlocked_default_group_can_be_deleted_without_hiding_files(tmp_path: Path, monkeypatch):
    configs = tmp_path / "configs"
    imported = configs / "imported"
    imported.mkdir(parents=True)
    (imported / "demo.toml").write_text('output_name = "demo"\n', encoding="utf-8")
    (configs / "web-file-groups.toml").write_text(
        "\n".join(
            [
                "[[groups]]",
                'id = "imported"',
                'label = "导入配置"',
                "open = true",
                "locked = false",
                "trainable = true",
                'methods_subdir = "imported"',
                'patterns = ["configs/imported/*.toml"]',
                "",
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    group = config_service.list_config_file_groups()[0]
    assert group["id"] == "imported"
    assert group["deletable"] is True

    ok, message = config_service.delete_config_file_group("imported")
    assert ok is True
    assert "已保留" in message

    groups = config_service.list_config_file_groups()
    assert [group["id"] for group in groups] == ["unfiled_imported"]
    assert groups[0]["deletable"] is True
    assert [item["path"] for item in groups[0]["files"]] == ["configs/imported/demo.toml"]


def test_unlocked_default_group_can_be_renamed(tmp_path: Path, monkeypatch):
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "web-file-groups.toml").write_text(
        "\n".join(
            [
                "[[groups]]",
                'id = "imported"',
                'label = "导入配置"',
                "open = true",
                "locked = false",
                "trainable = true",
                'methods_subdir = "imported"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    _patch_config_service_paths(monkeypatch, tmp_path)

    group = config_service.list_config_file_groups()[0]
    assert group["renamable"] is True

    ok, message, renamed = config_service.rename_config_file_group("imported", "常用导入配置")
    assert ok is True
    assert message == "分组已重命名"
    assert renamed["label"] == "常用导入配置"


def test_dataset_preset_save_read_list_and_apply(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    saved = config_service.save_dataset_preset(
        "configs/datasets/character_a.toml",
        [
            {
                "source_dir": "image_dataset/a",
                "image_dir": "post_image_dataset/a_resized",
                "cache_dir": "post_image_dataset/a_cache",
                "num_repeats": 3,
            },
            {
                "source_dir": "image_dataset/b",
                "image_dir": "post_image_dataset/b_resized",
                "cache_dir": "post_image_dataset/b_cache",
                "num_repeats": 2,
            },
        ],
        {"resolution": 768, "batch_size": 1, "enable_bucket": True},
    )

    assert saved["ok"] is True
    assert saved["file"] == "configs/datasets/character_a.toml"

    loaded = config_service.load_dataset_preset("configs/datasets/character_a.toml")
    assert loaded["summary"]["dataset_count"] == 2
    assert loaded["summary"]["repeat_total"] == 5
    assert loaded["defaults"]["resolution"] == 768

    listed = config_service.list_dataset_presets()
    assert "configs/datasets/character_a.toml" in [item["path"] for item in listed["presets"]]

    applied = config_service.apply_dataset_preset_to_training_config(
        "configs/datasets/character_a.toml",
        "configs/imported/lora.toml",
    )
    assert applied["ok"] is True
    train_text = (configs / "imported" / "lora.toml").read_text(encoding="utf-8")
    assert 'dataset_config = "configs/datasets/character_a.toml"' in train_text
    assert 'source_image_dir = "image_dataset/a"' in train_text
    assert 'resized_image_dir = "post_image_dataset/a_resized"' in train_text
    assert 'lora_cache_dir = "post_image_dataset/a_cache"' in train_text


def test_system_dataset_preset_is_readonly_but_can_be_saved_as(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    system_preset = tmp_path / "configs" / "datasets" / "ip_adapter.toml"
    system_preset.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "resolution = 1024",
                "batch_size = 1",
                "",
                "[[datasets.subsets]]",
                'image_dir = "ip-adapter-dataset"',
                'cache_dir = "post_image_dataset/ip_adapter"',
                "num_repeats = 1",
                'custom_attributes = {source_dir = "ip-adapter-dataset"}',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="只读"):
        config_service.save_dataset_preset(
            "configs/datasets/ip_adapter.toml",
            [{"source_dir": "x", "image_dir": "y", "cache_dir": "z", "num_repeats": 1}],
            {},
        )

    with pytest.raises(ValueError, match="不能删除"):
        config_service.delete_dataset_preset("configs/datasets/ip_adapter.toml")

    copied = config_service.save_dataset_preset_as(
        "ip_adapter_copy",
        [{"source_dir": "x", "image_dir": "y", "cache_dir": "z", "num_repeats": 1}],
        {},
    )
    assert copied["file"] == "configs/datasets/ip_adapter_copy.toml"


def test_step_estimate_defaults_max_train_steps_to_disabled_when_epoch_missing(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    _write_step_estimate_dataset(tmp_path, dataset_path)

    estimate = config_service.estimate_training_steps("lora", "default", "imported")

    assert estimate["steps_per_epoch"] == 15
    assert estimate["max_train_epochs"] is None
    assert estimate["max_train_steps"] == 0
    assert estimate["uses_max_train_epochs"] is False
    assert estimate["duration_configured"] is False
    assert estimate["duration_mode"] == "unset"
    assert estimate["total_steps"] == 0


def test_step_estimate_uses_explicit_max_train_steps_when_epoch_missing(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    _write_step_estimate_dataset(tmp_path, dataset_path)
    (configs / "imported" / "lora.toml").write_text(
        "\n".join(
            [
                'dataset_config = "configs/datasets/lora.toml"',
                "max_train_steps = 1600",
            ]
        ),
        encoding="utf-8",
    )

    estimate = config_service.estimate_training_steps("lora", "default", "imported")

    assert estimate["steps_per_epoch"] == 15
    assert estimate["max_train_epochs"] is None
    assert estimate["max_train_steps"] == 1600
    assert estimate["uses_max_train_epochs"] is False
    assert estimate["duration_configured"] is True
    assert estimate["duration_mode"] == "steps"
    assert estimate["total_steps"] == 1600


def test_step_estimate_prefers_epochs_over_max_train_steps(tmp_path: Path, monkeypatch):
    configs, dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    _write_step_estimate_dataset(tmp_path, dataset_path)
    (configs / "imported" / "lora.toml").write_text(
        "\n".join(
            [
                'dataset_config = "configs/datasets/lora.toml"',
                "max_train_epochs = 2",
                "max_train_steps = 1600",
            ]
        ),
        encoding="utf-8",
    )

    estimate = config_service.estimate_training_steps("lora", "default", "imported")

    assert estimate["steps_per_epoch"] == 15
    assert estimate["max_train_epochs"] == 2
    assert estimate["max_train_steps"] == 1600
    assert estimate["uses_max_train_epochs"] is True
    assert estimate["duration_configured"] is True
    assert estimate["duration_mode"] == "epochs"
    assert estimate["total_steps"] == 30


def test_imported_config_can_move_to_rokkotsu_group(tmp_path: Path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    train_file = configs / "imported" / "copy.toml"
    train_file.write_text('output_name = "copy"\n', encoding="utf-8")
    (configs / "web-file-groups.toml").write_text(
        "\n".join(
            [
                "[[groups]]",
                'id = "rokkotsu_goddess"',
                'label = "肋骨女神配置"',
                "open = true",
                "locked = false",
                "trainable = true",
                'methods_subdir = "imported"',
                "",
                "[[groups]]",
                'id = "imported"',
                'label = "导入配置"',
                "open = true",
                "locked = false",
                "trainable = true",
                'methods_subdir = "imported"',
                'patterns = ["configs/imported/*.toml"]',
            ]
        ),
        encoding="utf-8",
    )

    ok, msg, group = config_service.move_config_file_to_group(
        "configs/imported/copy.toml",
        "rokkotsu_goddess",
    )

    assert ok is True, msg
    assert group is not None
    assert group["id"] == "rokkotsu_goddess"
    assert [item["path"] for item in group["files"]] == ["configs/imported/copy.toml"]


def test_dataset_preset_writes_independent_dataset_settings_per_path(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    saved = config_service.save_dataset_preset(
        "configs/datasets/multi_bucket.toml",
        [
            {
                "source_dir": "image_dataset/a",
                "image_dir": "post_image_dataset/a_resized",
                "cache_dir": "post_image_dataset/a_cache",
                "num_repeats": 2,
                "settings": {
                    "resolution": 768,
                    "min_bucket_reso": 256,
                    "max_bucket_reso": 768,
                    "bucket_reso_steps": 32,
                    "bucket_no_upscale": True,
                    "validation_split_num": 4,
                    "validation_seed": 7,
                },
            },
            {
                "source_dir": "image_dataset/b",
                "image_dir": "post_image_dataset/b_resized",
                "cache_dir": "post_image_dataset/b_cache",
                "num_repeats": 5,
                "settings": {
                    "resolution": 1024,
                    "min_bucket_reso": 384,
                    "max_bucket_reso": 1344,
                    "bucket_reso_steps": 64,
                    "bucket_no_upscale": False,
                    "validation_split": 0.1,
                    "validation_seed": 99,
                },
            },
        ],
        {"caption_extension": ".txt", "keep_tokens": 2},
    )

    data = toml.loads(saved["content"])
    assert len(data["datasets"]) == 2
    assert data["datasets"][0]["resolution"] == 768
    assert data["datasets"][0]["max_bucket_reso"] == 768
    assert data["datasets"][0]["bucket_reso_steps"] == 32
    assert data["datasets"][0]["bucket_no_upscale"] is True
    assert data["datasets"][0]["validation_split_num"] == 4
    assert data["datasets"][1]["resolution"] == 1024
    assert data["datasets"][1]["min_bucket_reso"] == 384
    assert data["datasets"][1]["max_bucket_reso"] == 1344
    assert data["datasets"][1]["validation_split"] == 0.1

    loaded = config_service.load_dataset_preset("configs/datasets/multi_bucket.toml")
    assert loaded["datasets"][0]["settings"]["resolution"] == 768
    assert loaded["datasets"][1]["settings"]["max_bucket_reso"] == 1344


def test_dataset_preset_image_preview_reads_training_images_and_captions(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    source_dir = tmp_path / "image_dataset" / "a"
    image_dir = tmp_path / "post_image_dataset" / "a_resized"
    source_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    Image.new("RGB", (8, 6), color=(120, 20, 40)).save(image_dir / "hero.png")
    (source_dir / "hero.txt").write_text("1girl, blue eyes", encoding="utf-8")

    config_service.save_dataset_preset(
        "configs/datasets/preview.toml",
        [{
            "source_dir": "image_dataset/a",
            "image_dir": "post_image_dataset/a_resized",
            "cache_dir": "post_image_dataset/a_cache",
            "num_repeats": 2,
        }],
        {"caption_extension": ".txt", "keep_tokens": 1},
    )

    listing = config_service.list_dataset_preset_images("configs/datasets/preview.toml", 0)

    assert listing["ok"] is True
    assert listing["total"] == 1
    assert listing["images"][0]["name"] == "hero.png"
    assert listing["images"][0]["caption"]["ok"] is True
    assert listing["images"][0]["caption"]["text"] == "1girl, blue eyes"
    assert "dataset_index=0" in listing["images"][0]["url"]


def test_dataset_preview_image_resolver_rejects_files_outside_selected_row(tmp_path: Path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    image_dir = tmp_path / "post_image_dataset" / "a_resized"
    other_dir = tmp_path / "post_image_dataset" / "other"
    image_dir.mkdir(parents=True)
    other_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(image_dir / "ok.png")
    Image.new("RGB", (4, 4), color=(10, 20, 30)).save(other_dir / "bad.png")
    config_service.save_dataset_preset(
        "configs/datasets/preview_guard.toml",
        [{
            "source_dir": "image_dataset/a",
            "image_dir": "post_image_dataset/a_resized",
            "cache_dir": "post_image_dataset/a_cache",
            "num_repeats": 1,
        }],
        {},
    )

    resolved = config_service.resolve_dataset_preview_image(
        "configs/datasets/preview_guard.toml",
        0,
        "post_image_dataset/a_resized/ok.png",
    )
    assert resolved.name == "ok.png"

    with pytest.raises(ValueError, match="不属于当前数据集路径"):
        config_service.resolve_dataset_preview_image(
            "configs/datasets/preview_guard.toml",
            0,
            "post_image_dataset/other/bad.png",
        )


def _write_minimal_config_tree(root: Path) -> tuple[Path, Path]:
    configs = root / "configs"
    (configs / "imported").mkdir(parents=True)
    (configs / "datasets").mkdir(parents=True)
    (configs / "base.toml").write_text(
        "\n".join(
            [
                'source_image_dir = "image_dataset"',
                'resized_image_dir = "post_image_dataset/resized"',
                'lora_cache_dir = "post_image_dataset/lora"',
            ]
        ),
        encoding="utf-8",
    )
    (configs / "presets.toml").write_text("[default]\n", encoding="utf-8")
    (configs / "imported" / "lora.toml").write_text(
        'dataset_config = "configs/datasets/lora.toml"\n',
        encoding="utf-8",
    )
    dataset_path = configs / "datasets" / "lora.toml"
    return configs, dataset_path


def _write_step_estimate_dataset(root: Path, dataset_path: Path) -> None:
    image_dir = root / "post_image_dataset" / "a_resized"
    image_dir.mkdir(parents=True)
    for idx in range(3):
        Image.new("RGB", (8, 8), color=(idx, 20, 40)).save(image_dir / f"{idx}.png")
    dataset_path.write_text(
        "\n".join(
            [
                "[[datasets]]",
                "",
                "[[datasets.subsets]]",
                'image_dir = "post_image_dataset/a_resized"',
                "num_repeats = 5",
            ]
        ),
        encoding="utf-8",
    )


def _patch_config_service_paths(monkeypatch, root: Path) -> None:
    configs = root / "configs"
    monkeypatch.setattr(config_service, "ROOT", root)
    monkeypatch.setattr(config_service, "CONFIGS_DIR", configs)
    monkeypatch.setattr(config_service, "DATASET_PRESETS_DIR", configs / "datasets")
    monkeypatch.setattr(config_service, "GUI_METHODS_DIR", configs / "gui-methods")
    monkeypatch.setattr(config_service, "IMPORTED_CONFIGS_DIR", configs / "imported")
    monkeypatch.setattr(config_service, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(config_service, "WEB_FILE_GROUPS_FILE", configs / "web-file-groups.toml")
    monkeypatch.setattr(config_service, "WEB_USER_LOCKS_FILE", configs / "web-user-locks.toml")
