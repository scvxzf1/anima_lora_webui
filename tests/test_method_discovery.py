from __future__ import annotations

from pathlib import Path

import web.services.config.merge as merge


def test_list_methods_omits_missing_known_files(tmp_path: Path, monkeypatch) -> None:
    methods = tmp_path / "methods"
    methods.mkdir()
    (methods / "lora.toml").write_text("[network]\n", encoding="utf-8")
    monkeypatch.setattr(merge, "CONFIGS_DIR", tmp_path)

    names = merge.list_methods.__wrapped__()

    assert "lora" in names
    assert "lokr" not in names
    assert "tlora" not in names
    assert "hydralora" not in names
    assert "spd" not in names


def test_list_methods_includes_disk_extras(tmp_path: Path, monkeypatch) -> None:
    methods = tmp_path / "methods"
    methods.mkdir()
    (methods / "turbo.toml").write_text("[network]\n", encoding="utf-8")
    (methods / "lora.toml").write_text("[network]\n", encoding="utf-8")
    monkeypatch.setattr(merge, "CONFIGS_DIR", tmp_path)

    names = merge.list_methods.__wrapped__()

    assert "turbo" in names
    assert "lora" in names
    # Known order first: lora before turbo extras.
    assert names.index("lora") < names.index("turbo")


def test_list_methods_empty_dir_returns_empty(tmp_path: Path, monkeypatch) -> None:
    methods = tmp_path / "methods"
    methods.mkdir()
    monkeypatch.setattr(merge, "CONFIGS_DIR", tmp_path)

    assert merge.list_methods.__wrapped__() == []
