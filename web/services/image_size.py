"""Fast image dimension probing for WebUI preview/dataset listings.

Reads width/height from file headers via ``imagesize`` (sub-millisecond, no full
decode), falling back to PIL only when the header parse returns an invalid
size. Mirrors the pattern in ``library/datasets/dataset_image_io.py`` so WebUI
handlers stop blocking the event loop on 200-image PIL decodes per request.
"""

from __future__ import annotations

from pathlib import Path

import imagesize
from PIL import Image


def probe_image_size(path: Path) -> tuple[int | None, int | None]:
    """Return ``(width, height)`` for ``path``, or ``(None, None)`` on failure.

    Header parse first; PIL decode only as a fallback. Both arms are
    defensive — callers use this in listing loops where one broken file
    must not abort the whole response.
    """
    try:
        width, height = imagesize.get(path)
    except Exception:
        width, height = -1, -1

    if width and width > 0 and height and height > 0:
        return int(width), int(height)

    try:
        with Image.open(path) as img:
            width, height = img.size
    except Exception:
        return None, None
    if width and width > 0 and height and height > 0:
        return int(width), int(height)
    return None, None
