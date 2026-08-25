from __future__ import annotations

from pathlib import Path

import pytest
import toml

from web.services import model_config_service, settings_service


def _patch_settings_file(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", path)


def _item(
    item_id: str,
    name: str,
    family: str = "anima",
) -> dict[str, str]:
    return {
        "id": item_id,
        "name": name,
        "model_family": family,
        "pretrained_model_name_or_path": f"models/{item_id}/dit.safetensors",
        "qwen3": f"models/{item_id}/qwen.safetensors",
        "vae": f"models/{item_id}/vae.safetensors",
    }


def test_model_config_get_migrates_legacy_values_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = tmp_path / "external-configs"
    configs.mkdir()
    settings_file = configs / "web-ui-settings.toml"
    settings_file.write_text(
        toml.dumps(
            {
                "global": {
                    "pretrained_model_name_or_path": "custom/dit.safetensors",
                    "model_family": "krea2_raw",
                }
            }
        ),
        encoding="utf-8",
    )
    (configs / "base.toml").write_text(
        toml.dumps(
            {
                "pretrained_model_name_or_path": "base/dit.safetensors",
                "qwen3": "base/qwen.safetensors",
                "vae": "base/vae.safetensors",
            }
        ),
        encoding="utf-8",
    )
    original = settings_file.read_text(encoding="utf-8")
    _patch_settings_file(monkeypatch, settings_file)

    payload = model_config_service.get_model_configs()

    assert payload["ok"] is True
    assert payload["migrated"] is True
    assert payload["groups_migrated"] is False
    assert payload["default_id"] == "legacy-default"
    assert payload["items"] == [
        {
            "id": "legacy-default",
            "name": "Krea-2 默认配置",
            "model_family": "krea2_raw",
            "pretrained_model_name_or_path": "custom/dit.safetensors",
            "qwen3": "base/qwen.safetensors",
            "vae": "base/vae.safetensors",
            "complete": True,
        }
    ]
    assert payload["groups"] == [
        {"id": "ungrouped", "label": "未分组", "item_ids": ["legacy-default"]}
    ]
    assert settings_file.read_text(encoding="utf-8") == original


def test_model_config_legacy_migration_uses_environment_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs = tmp_path / "configs"
    configs.mkdir()
    settings_file = configs / "web-ui-settings.toml"
    (configs / "base.toml").write_text(
        toml.dumps(
            {
                "pretrained_model_name_or_path": "base/dit.safetensors",
                "qwen3": "base/qwen.safetensors",
                "vae": "base/vae.safetensors",
            }
        ),
        encoding="utf-8",
    )
    _patch_settings_file(monkeypatch, settings_file)
    monkeypatch.setenv("ANIMA_MODEL_FAMILY", "krea2_raw")

    payload = model_config_service.get_model_configs()

    assert payload["items"][0]["model_family"] == "krea2_raw"
    assert payload["items"][0]["name"] == "Krea-2 默认配置"


def test_model_config_legacy_migration_rejects_unknown_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    settings_file.parent.mkdir()
    settings_file.write_text(
        toml.dumps({"global": {"model_family": "unknown"}}),
        encoding="utf-8",
    )
    _patch_settings_file(monkeypatch, settings_file)

    with pytest.raises(ValueError, match="模型格式仅支持"):
        model_config_service.get_model_configs()


def test_model_config_save_roundtrip_preserves_sections_and_mirrors_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    settings_file.parent.mkdir()
    settings_file.write_text(
        toml.dumps(
            {
                "global": {"output_root": "output/custom", "unknown": "keep"},
                "training_policy": {"max_attempts": 3},
                "custom_section": {"enabled": True},
            }
        ),
        encoding="utf-8",
    )
    _patch_settings_file(monkeypatch, settings_file)
    revision = model_config_service.get_model_configs()["revision"]

    saved = model_config_service.save_model_configs(
        {
            "revision": revision,
            "default_id": "krea",
            "items": [
                _item("krea", "Krea 主模型", "krea2"),
                _item("anima", "Anima 备用模型"),
            ],
            "groups": [
                {"id": "primary", "label": "主力模型", "item_ids": ["krea"]},
                {"id": "fallback", "label": "备用模型", "item_ids": ["anima"]},
            ],
        }
    )

    assert [item["id"] for item in saved["items"]] == ["krea", "anima"]
    assert saved["items"][0]["model_family"] == "krea2_raw"
    raw = toml.loads(settings_file.read_text(encoding="utf-8"))
    assert [item["id"] for item in raw["model_config_library"]["items"]] == [
        "krea",
        "anima",
    ]
    assert raw["model_config_library"]["groups"] == [
        {"id": "primary", "label": "主力模型", "item_ids": ["krea"]},
        {"id": "fallback", "label": "备用模型", "item_ids": ["anima"]},
    ]
    assert raw["global"]["model_family"] == "krea2_raw"
    assert (
        raw["global"]["pretrained_model_name_or_path"] == "models/krea/dit.safetensors"
    )
    assert raw["global"]["output_root"] == "output/custom"
    assert raw["global"]["unknown"] == "keep"
    assert raw["training_policy"] == {"max_attempts": 3}
    assert raw["custom_section"] == {"enabled": True}

    settings_service.save_global_settings({"output_root": "output/next"})
    reloaded = model_config_service.get_model_configs()
    assert [item["id"] for item in reloaded["items"]] == ["krea", "anima"]
    assert reloaded["default_id"] == "krea"


def test_model_config_save_without_groups_preserves_existing_group_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_file = tmp_path / "web-ui-settings.toml"
    _patch_settings_file(monkeypatch, settings_file)
    initial_revision = model_config_service.get_model_configs()["revision"]
    first = model_config_service.save_model_configs(
        {
            "revision": initial_revision,
            "default_id": "one",
            "items": [_item("one", "One"), _item("two", "Two")],
            "groups": [
                {"id": "a", "label": "A", "item_ids": ["one"]},
                {"id": "b", "label": "B", "item_ids": ["two"]},
            ],
        }
    )

    saved = model_config_service.save_model_configs(
        {
            "revision": first["revision"],
            "default_id": "one",
            "items": [
                _item("two", "Two"),
                _item("one", "One"),
                _item("three", "Three"),
            ],
        }
    )

    assert saved["groups"] == [
        {"id": "a", "label": "A", "item_ids": ["one", "three"]},
        {"id": "b", "label": "B", "item_ids": ["two"]},
    ]
    assert [item["id"] for item in saved["items"]] == ["one", "three", "two"]


def test_model_config_existing_library_without_groups_is_migrated_in_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_file = tmp_path / "web-ui-settings.toml"
    settings_file.write_text(
        toml.dumps(
            {
                "model_config_library": {
                    "default_id": "one",
                    "items": [_item("one", "One"), _item("two", "Two")],
                }
            }
        ),
        encoding="utf-8",
    )
    _patch_settings_file(monkeypatch, settings_file)

    payload = model_config_service.get_model_configs()

    assert payload["migrated"] is False
    assert payload["groups_migrated"] is True
    assert payload["groups"] == [
        {"id": "ungrouped", "label": "未分组", "item_ids": ["one", "two"]}
    ]


def test_model_config_anima_default_removes_legacy_family_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_file = tmp_path / "web-ui-settings.toml"
    settings_file.write_text(
        toml.dumps({"global": {"model_family": "krea2_raw"}}), encoding="utf-8"
    )
    _patch_settings_file(monkeypatch, settings_file)
    revision = model_config_service.get_model_configs()["revision"]

    model_config_service.save_model_configs(
        {
            "revision": revision,
            "default_id": "anima",
            "items": [_item("anima", "Anima 主模型")],
        }
    )

    raw = toml.loads(settings_file.read_text(encoding="utf-8"))
    assert "model_family" not in raw["global"]


def test_model_config_saves_z_image_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_file = tmp_path / "web-ui-settings.toml"
    _patch_settings_file(monkeypatch, settings_file)
    revision = model_config_service.get_model_configs()["revision"]

    result = model_config_service.save_model_configs(
        {
            "revision": revision,
            "default_id": "z-image",
            "items": [_item("z-image", "Z-Image BF16", "z_image")],
        }
    )

    assert result["items"][0]["model_family"] == "z_image"
    raw = toml.loads(settings_file.read_text(encoding="utf-8"))
    assert raw["global"]["model_family"] == "z_image"


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"items": []}, "至少需要保留一个"),
        ({"default_id": "missing"}, "默认模型配置"),
        ({"items": [_item("one", "Same"), _item("two", "same")]}, "名称不能重复"),
        (
            {"items": [{**_item("one", "One"), "vae": ""}]},
            "缺少 vae",
        ),
        (
            {"items": [{**_item("one", "One"), "model_family": "unknown"}]},
            "仅支持 anima、krea2_raw 或 z_image",
        ),
        (
            {"groups": [{"id": "a", "label": "A", "item_ids": []}]},
            "都必须属于一个分组",
        ),
        (
            {
                "groups": [
                    {"id": "a", "label": "Same", "item_ids": ["one"]},
                    {"id": "b", "label": "same", "item_ids": []},
                ]
            },
            "分组名称不能重复",
        ),
    ],
)
def test_model_config_validation_rejects_invalid_library(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    patch: dict[str, object],
    message: str,
) -> None:
    settings_file = tmp_path / "web-ui-settings.toml"
    _patch_settings_file(monkeypatch, settings_file)
    payload: dict[str, object] = {
        "revision": model_config_service.get_model_configs()["revision"],
        "default_id": "one",
        "items": [_item("one", "One")],
        **patch,
    }

    with pytest.raises(ValueError, match=message):
        model_config_service.save_model_configs(payload)

    assert not settings_file.exists()


def test_model_config_rejects_stale_revision_and_broken_toml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings_file = tmp_path / "web-ui-settings.toml"
    _patch_settings_file(monkeypatch, settings_file)
    revision = model_config_service.get_model_configs()["revision"]
    settings_file.write_text("[global]\noutput_root = 'changed'\n", encoding="utf-8")

    with pytest.raises(model_config_service.ModelConfigConflictError):
        model_config_service.save_model_configs(
            {
                "revision": revision,
                "default_id": "one",
                "items": [_item("one", "One")],
            }
        )

    settings_file.write_text("[global\nbroken = true", encoding="utf-8")
    with pytest.raises(model_config_service.ModelConfigFileError):
        model_config_service.get_model_configs()
