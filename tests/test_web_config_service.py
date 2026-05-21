from __future__ import annotations

from pathlib import Path

import pytest

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


def _patch_config_service_paths(monkeypatch, root: Path) -> None:
    configs = root / "configs"
    monkeypatch.setattr(config_service, "ROOT", root)
    monkeypatch.setattr(config_service, "CONFIGS_DIR", configs)
    monkeypatch.setattr(config_service, "GUI_METHODS_DIR", configs / "gui-methods")
    monkeypatch.setattr(config_service, "IMPORTED_CONFIGS_DIR", configs / "imported")
    monkeypatch.setattr(config_service, "PRESETS_FILE", configs / "presets.toml")
    monkeypatch.setattr(config_service, "WEB_FILE_GROUPS_FILE", configs / "web-file-groups.toml")
    monkeypatch.setattr(config_service, "WEB_USER_LOCKS_FILE", configs / "web-user-locks.toml")
