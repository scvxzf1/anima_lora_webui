"""Fast image dimension probing for WebUI preview/dataset listings.

Reads width/height lazily with Pillow for common raster formats and falls back
to ``imagesize`` for less common formats. Results are cached by file metadata,
so WebUI listings do not reopen unchanged files on every page request.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import imagesize
from PIL import Image


def probe_image_size(path: Path) -> tuple[int | None, int | None]:
    """Return ``(width, height)`` for ``path``, or ``(None, None)`` on failure.

    Both parsers are defensive because one broken file must not abort a listing.
    """
    try:
        stat = path.stat()
    except OSError:
        return None, None
    return _probe_image_size_cached(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=2048)
def _probe_image_size_cached(path: str, _mtime_ns: int, _size: int) -> tuple[int | None, int | None]:
    image_path = Path(path)
    if image_path.suffix.lower() in {'.bmp', '.gif', '.jpeg', '.jpg', '.png', '.tif', '.tiff', '.webp'}:
        dimensions = _pil_image_size(image_path)
        if dimensions[0] is not None:
            return dimensions

    try:
        width, height = imagesize.get(image_path)
    except Exception:
        width, height = -1, -1

    if width and width > 0 and height and height > 0:
        return int(width), int(height)

    try:
        width, height = _pil_image_size(image_path)
    except Exception:
        return None, None
    if width and width > 0 and height and height > 0:
        return int(width), int(height)
    return None, None


def _pil_image_size(path: Path) -> tuple[int | None, int | None]:
    try:
        with Image.open(path) as img:
            width, height = img.size
    except Exception:
        return None, None
    if width and width > 0 and height and height > 0:
        return int(width), int(height)
    return None, None
