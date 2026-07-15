from __future__ import annotations

from pathlib import Path

import pytest

from web.services import atomic_io


def test_atomic_write_text_creates_parent_and_replaces_content(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "settings.toml"

    atomic_io.atomic_write_text(target, "value = 1\n")
    atomic_io.atomic_write_text(target, "value = 2\n")

    assert target.read_text(encoding="utf-8") == "value = 2\n"
    assert list(target.parent.glob(f".{target.name}.*.tmp")) == []


def test_atomic_write_text_replace_failure_preserves_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "settings.toml"
    target.write_text("value = 'old'\n", encoding="utf-8")
    observed: dict[str, Path] = {}

    def fail_replace(source, destination) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        observed.update(source=source_path, destination=destination_path)
        assert source_path.parent == target.parent
        assert source_path.read_text(encoding="utf-8") == "value = 'new'\n"
        raise OSError("simulated replace failure")

    monkeypatch.setattr(atomic_io.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        atomic_io.atomic_write_text(target, "value = 'new'\n")

    assert observed["destination"] == target
    assert target.read_text(encoding="utf-8") == "value = 'old'\n"
    assert list(tmp_path.glob(f".{target.name}.*.tmp")) == []
