"""P0: asyncio.to_thread offload contract for preview/dataset listing handlers.

Locks the invariant that wrapping sync listing calls in ``asyncio.to_thread``
does not change return values or exception types — so the event loop stays
non-blocking while handler behavior is preserved.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from PIL import Image

from web.services import preview_service, settings_service


def _patch_preview_settings(monkeypatch, settings_file: Path, *, root: Path) -> None:
    monkeypatch.setattr(preview_service, "SETTINGS_FILE", settings_file)
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", settings_file)
    monkeypatch.setattr(preview_service, "ROOT", root)
    monkeypatch.setattr(settings_service, "ROOT", root)


def _make_sample_dir(root: Path, rel: str, name: str, mtime: float = 100.0) -> Path:
    sample = root / rel
    sample.mkdir(parents=True, exist_ok=True)
    img = sample / name
    Image.new("RGB", (16, 24)).save(img)
    os.utime(img, (mtime, mtime))
    return sample


def test_to_thread_offload_returns_same_payload_as_sync(tmp_path, monkeypatch):
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    settings_file.parent.mkdir(parents=True)
    settings_file.write_text('[global]\noutput_root = "output/runs"\n', encoding="utf-8")
    _patch_preview_settings(monkeypatch, settings_file, root=tmp_path)

    _make_sample_dir(tmp_path, "output/runs/522/training_output/sample", "img.png")

    sync_payload = preview_service.list_preview_images("training")
    async_payload = asyncio.run(
        asyncio.to_thread(preview_service.list_preview_images, "training")
    )

    assert async_payload["images"][0]["name"] == "img.png"
    assert async_payload["images"][0]["width"] == 16
    assert async_payload["images"][0]["height"] == 24
    assert async_payload["directory"] == sync_payload["directory"]


def test_to_thread_offload_propagates_value_error(tmp_path, monkeypatch):
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    settings_file.parent.mkdir(parents=True)
    settings_file.write_text('[global]\noutput_root = "output/runs"\n', encoding="utf-8")
    _patch_preview_settings(monkeypatch, settings_file, root=tmp_path)

    with pytest.raises(ValueError):
        asyncio.run(
            asyncio.to_thread(preview_service.list_preview_images, "bogus_source")
        )
