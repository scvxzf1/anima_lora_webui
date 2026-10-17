from __future__ import annotations

from pathlib import Path

import tomlkit

from web.services.config import file_groups
from web.services import config_service


def _configure_file_group_paths(monkeypatch, tmp_path: Path) -> tuple[Path, Path]:
    configs = tmp_path / "configs"
    (configs / "methods").mkdir(parents=True)
    (configs / "methods" / "moss_image_lora.toml").write_text(
        'model_family = "moss_image"\n', encoding="utf-8"
    )
    registry = configs / "web-file-groups.toml"
    monkeypatch.setattr(file_groups, "ROOT", tmp_path)
    monkeypatch.setattr(file_groups, "CONFIGS_DIR", configs)
    monkeypatch.setattr(file_groups, "WEB_FILE_GROUPS_FILE", registry)
    monkeypatch.setattr(config_service, "ROOT", tmp_path)
    monkeypatch.setattr(config_service, "CONFIGS_DIR", configs)
    monkeypatch.setattr(config_service, "WEB_FILE_GROUPS_FILE", registry)
    return configs, registry


def test_default_training_groups_include_moss_methods_group(monkeypatch, tmp_path):
    _configure_file_group_paths(monkeypatch, tmp_path)

    groups = file_groups.list_config_file_groups(kind="training")
    methods = next(group for group in groups if group["id"] == "methods")

    assert methods["locked"] is True
    assert methods["group_locked"] is True
    assert methods["trainable"] is True
    assert methods["methods_subdir"] == "methods"
    assert [item["path"] for item in methods["files"]] == [
        "configs/methods/moss_image_lora.toml"
    ]
    assert methods["files"][0]["label"] == "MOSS Image LoRA 内置方法配置"


def test_existing_file_group_registry_overrides_default_methods_group(monkeypatch, tmp_path):
    _configs, registry = _configure_file_group_paths(monkeypatch, tmp_path)
    registry.write_text(
        tomlkit.dumps(
            {
                "groups": [
                    {
                        "id": "custom_training",
                        "label": "用户训练配置",
                        "trainable": True,
                        "methods_subdir": "imported",
                        "files": [],
                        "patterns": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    groups = file_groups.list_config_file_groups(kind="training")

    assert [group["id"] for group in groups] == ["custom_training"]
