"""Bounded in-memory thumbnails for dataset preview cards."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

DATASET_PREVIEW_THUMBNAIL_SIZE = (640, 480)
DATASET_PREVIEW_THUMBNAIL_CACHE_ITEMS = 256


@dataclass(frozen=True)
class DatasetPreviewThumbnail:
    content: bytes
    content_type: str
    etag: str


def render_dataset_preview_thumbnail(path: Path) -> DatasetPreviewThumbnail:
    resolved = path.resolve()
    stat = resolved.stat()
    width, height = DATASET_PREVIEW_THUMBNAIL_SIZE
    content, content_type = _render_thumbnail_cached(
        str(resolved),
        stat.st_mtime_ns,
        stat.st_size,
        width,
        height,
    )
    fingerprint = f"{resolved}\0{stat.st_mtime_ns}\0{stat.st_size}\0{width}x{height}".encode()
    return DatasetPreviewThumbnail(
        content=content,
        content_type=content_type,
        etag=f'"{sha256(fingerprint).hexdigest()[:24]}"',
    )


@lru_cache(maxsize=DATASET_PREVIEW_THUMBNAIL_CACHE_ITEMS)
def _render_thumbnail_cached(
    path: str,
    _mtime_ns: int,
    _size: int,
    width: int,
    height: int,
) -> tuple[bytes, str]:
    try:
        with Image.open(path) as source:
            source.draft("RGB", (width, height))
            image = ImageOps.exif_transpose(source)
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
            output = BytesIO()
            try:
                image.save(output, format="WEBP", quality=80, method=4)
                return output.getvalue(), "image/webp"
            except OSError:
                output = BytesIO()
                image.save(output, format="PNG", optimize=True)
                return output.getvalue(), "image/png"
    except (Image.DecompressionBombError, OSError) as exc:
        raise ValueError("无法生成数据集预览缩略图") from exc
