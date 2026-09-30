from __future__ import annotations

from pathlib import Path

import toml

from web.services import settings_service


def _patch_settings(tmp_path: Path, monkeypatch) -> Path:
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    monkeypatch.setattr(settings_service, "ROOT", tmp_path)
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", settings_file)
    return settings_file

def test_dragon_motion_defaults_enabled_and_roundtrips(tmp_path, monkeypatch):
    settings_file = _patch_settings(tmp_path, monkeypatch)

    assert settings_service.get_global_settings()["dragon_motion_enabled"] is True

    saved = settings_service.save_global_settings({"dragon_motion_enabled": False})

    assert saved["dragon_motion_enabled"] is False
    assert settings_service.get_global_settings()["dragon_motion_enabled"] is False
    assert toml.loads(settings_file.read_text(encoding="utf-8"))["global"]["dragon_motion_enabled"] is False


def test_partial_global_save_preserves_disabled_dragon_motion(tmp_path, monkeypatch):
    _patch_settings(tmp_path, monkeypatch)
    settings_service.save_global_settings({"dragon_motion_enabled": False})

    saved = settings_service.save_global_settings({"output_root": "output/next"})

    assert saved["dragon_motion_enabled"] is False
