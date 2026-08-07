from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from web.services.image_size import probe_image_size


def _make_image(path: Path, size: tuple[int, int], fmt: str) -> None:
    Image.new("RGB", size).save(path, format=fmt)


@pytest.mark.parametrize(
    "ext, fmt",
    [(".png", "PNG"), (".jpg", "JPEG"), (".webp", "WEBP"), (".bmp", "BMP")],
)
def test_probe_image_size_matches_pil(tmp_path: Path, ext: str, fmt: str) -> None:
    path = tmp_path / f"img{ext}"
    _make_image(path, (123, 45), fmt)

    assert probe_image_size(path) == (123, 45)


def test_probe_image_size_returns_none_for_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")

    assert probe_image_size(path) == (None, None)


def test_probe_image_size_falls_back_to_pil_when_header_parse_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "img.png"
    _make_image(path, (64, 32), "PNG")

    def _bad_header(_path):
        return (-1, -1)

    monkeypatch.setattr("web.services.image_size.imagesize.get", _bad_header)

    assert probe_image_size(path) == (64, 32)
