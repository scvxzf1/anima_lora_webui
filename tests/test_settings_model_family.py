"""model_family global setting (Krea-2-Raw migration, stage 6).

Pins the round-trip behavior of the ``model_family`` selector in
``settings_service``: the value is an enum string (anima / krea2_raw), stored
under ``[global]`` in ``web-ui-settings.toml``. Empty value means anima
default (the resolve_model_family fallback chain still governs at training
time), so we drop the key when empty rather than writing an empty string.
Unknown explicit values are rejected so execution cannot silently select Anima.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import toml

from web.services import settings_service


def _patch_settings(tmp_path: Path, monkeypatch):
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    settings_file.parent.mkdir(parents=True)
    settings_file.write_text('[global]\noutput_root = "output/runs"\n', encoding="utf-8")
    monkeypatch.setattr(settings_service, "ROOT", tmp_path)
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", settings_file)
    return settings_file


def test_model_family_default_empty(tmp_path, monkeypatch):
    _patch_settings(tmp_path, monkeypatch)
    got = settings_service.get_global_settings()
    assert got["model_family"] == ""


def test_model_family_krea2_roundtrip(tmp_path, monkeypatch):
    settings_file = _patch_settings(tmp_path, monkeypatch)
    saved = settings_service.save_global_settings({"model_family": "krea2_raw"})
    assert saved["ok"] is True
    assert saved["model_family"] == "krea2_raw"
    raw = toml.loads(settings_file.read_text(encoding="utf-8"))
    assert raw["global"]["model_family"] == "krea2_raw"
    loaded = settings_service.get_global_settings()
    assert loaded["model_family"] == "krea2_raw"


def test_model_family_anima_stored_as_empty(tmp_path, monkeypatch):
    """Saving the anima value drops the key so anima-default holds.

    Empty == "use resolve_model_family() fallback", which defaults to anima.
    Writing an explicit ``model_family = "anima"`` would mask the env-var
    override path, so we normalize anima back to empty on save.
    """
    settings_file = _patch_settings(tmp_path, monkeypatch)
    settings_service.save_global_settings({"model_family": "krea2_raw"})
    saved = settings_service.save_global_settings({"model_family": "anima"})
    assert saved["model_family"] == ""
    raw = toml.loads(settings_file.read_text(encoding="utf-8"))
    assert "model_family" not in raw["global"]


def test_model_family_unknown_value_is_rejected_on_save(tmp_path, monkeypatch):
    _patch_settings(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="WebUI global model_family"):
        settings_service.save_global_settings({"model_family": "bogus_family"})


def test_model_family_unknown_on_disk_is_rejected(tmp_path, monkeypatch):
    settings_file = _patch_settings(tmp_path, monkeypatch)
    settings_file.write_text(
        '[global]\noutput_root = "output/runs"\nmodel_family = "bogus"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="WebUI global model_family"):
        settings_service.get_global_settings()
