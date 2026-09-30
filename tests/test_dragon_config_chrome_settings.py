from __future__ import annotations

from pathlib import Path

import toml

from web.services import settings_service


def _patch_settings(tmp_path: Path, monkeypatch) -> Path:
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    monkeypatch.setattr(settings_service, "ROOT", tmp_path)
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", settings_file)
    return settings_file

def test_config_chrome_defaults_contextual_and_roundtrips(tmp_path, monkeypatch):
    settings_file = _patch_settings(tmp_path, monkeypatch)
    defaults = settings_service.get_global_settings()

    assert defaults["dragon_config_help_always_visible"] is False
    assert defaults["dragon_config_tags_always_visible"] is False

    saved = settings_service.save_global_settings(
        {
            "dragon_config_help_always_visible": True,
            "dragon_config_tags_always_visible": True,
        }
    )

    assert saved["dragon_config_help_always_visible"] is True
    assert saved["dragon_config_tags_always_visible"] is True
    raw = toml.loads(settings_file.read_text(encoding="utf-8"))["global"]
    assert raw["dragon_config_help_always_visible"] is True
    assert raw["dragon_config_tags_always_visible"] is True


def test_partial_global_save_preserves_config_chrome_settings(tmp_path, monkeypatch):
    _patch_settings(tmp_path, monkeypatch)
    settings_service.save_global_settings({"dragon_config_help_always_visible": True})

    saved = settings_service.save_global_settings({"output_root": "output/next"})

    assert saved["dragon_config_help_always_visible"] is True
    assert saved["dragon_config_tags_always_visible"] is False
