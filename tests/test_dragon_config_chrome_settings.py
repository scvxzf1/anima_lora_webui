from __future__ import annotations

from pathlib import Path

import toml

from web.services import settings_service


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "web" / "static"


def _patch_settings(tmp_path: Path, monkeypatch) -> Path:
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    monkeypatch.setattr(settings_service, "ROOT", tmp_path)
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", settings_file)
    return settings_file


def _read(relative: str) -> str:
    return (STATIC / relative).read_text(encoding="utf-8")


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


def test_global_settings_exposes_and_applies_config_chrome_toggles():
    settings_page = _read("js/dragon-ui/pages/global-settings.js")
    runtime = _read("js/dragon-ui/config-chrome.js")

    assert "['dragon_config_help_always_visible', '常态显示参数“？”', 'boolean'" in settings_page
    assert "['dragon_config_tags_always_visible', '常态显示参数标签', 'boolean'" in settings_page
    assert "applyDragonConfigChromeSettings(payload);" in settings_page
    assert "dataset.dragonConfigHelp" in runtime
    assert "dataset.dragonConfigTags" in runtime
