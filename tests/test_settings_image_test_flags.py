"""image_test_allow_home_search first-class global settings."""

from __future__ import annotations

from pathlib import Path

import toml

from web.services import image_test_service, settings_service


def _patch_settings(tmp_path: Path, monkeypatch):
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    settings_file.parent.mkdir(parents=True)
    settings_file.write_text('[global]\noutput_root = "output/runs"\n', encoding="utf-8")
    monkeypatch.setattr(settings_service, "ROOT", tmp_path)
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", settings_file)
    monkeypatch.setattr(image_test_service, "ROOT", tmp_path)
    return settings_file


def test_image_test_allow_home_search_default_false(tmp_path, monkeypatch):
    _patch_settings(tmp_path, monkeypatch)
    got = settings_service.get_global_settings()
    assert got["image_test_allow_home_search"] is False
    assert image_test_service._image_test_allow_home_search() is False


def test_image_test_allow_home_search_roundtrip(tmp_path, monkeypatch):
    settings_file = _patch_settings(tmp_path, monkeypatch)
    saved = settings_service.save_global_settings({"image_test_allow_home_search": True})
    assert saved["ok"] is True
    assert saved["image_test_allow_home_search"] is True
    raw = toml.loads(settings_file.read_text(encoding="utf-8"))
    assert raw["global"]["image_test_allow_home_search"] is True
    loaded = settings_service.get_global_settings()
    assert loaded["image_test_allow_home_search"] is True
    assert image_test_service._image_test_allow_home_search() is True
