"""Read-only, opt-in image-directory dimension census for step estimates.

This is a pre-sampling census, not a reconstruction of a running dataloader.
Only image headers are read. Source projections reuse preprocessing geometry.
"""

from collections import Counter
from functools import lru_cache
from pathlib import Path
from time import monotonic
from typing import Any

from PIL import Image

from web.services.config.dataset_media import _dataset_image_files
from web.services.config.common import _bool_value, _positive_int

MAX_CENSUS_IMAGES = 20_000
MAX_CENSUS_SECONDS = 10.0


@lru_cache(maxsize=20_000)
def _image_size(path: Path, mtime_ns: int, file_size: int) -> tuple[int, int]:
    # The stat signature invalidates cached headers when an image is replaced.
    with Image.open(path) as image:
        return image.size


def _source_bucket_selector(settings: dict[str, Any]):
    from library.datasets.buckets import BucketManager
    from library.preprocess.bucket_geometry import select_resize_bucket

    resolution = _positive_int(settings.get("resolution"), 1024)
    minimum = _positive_int(settings.get("min_bucket_reso"), 256)
    manager = BucketManager(
        max_reso=(resolution, resolution), min_size=minimum,
        max_size=max(resolution, minimum, _positive_int(settings.get("max_bucket_reso"), 2048)),
        reso_steps=_positive_int(settings.get("bucket_reso_steps"), 64),
    )
    manager.make_buckets(constant_token_buckets=True)

    def select(width: int, height: int):
        return select_resize_bucket(
            manager, width, height,
            enable_bucket=_bool_value(settings.get("enable_bucket"), True),
            bucket_no_upscale=_bool_value(settings.get("bucket_no_upscale"), False),
        )

    return select


def inspect_estimate_buckets(
    directory: Path,
    *,
    recursive: bool,
    path_pattern: str,
    available: bool,
    source_settings: dict[str, Any] | None = None,
    min_pixels: int = 0,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "basis": "source_projection" if source_settings is not None else "image_dimensions",
        "status": "ready" if available else "pending",
        "image_count": 0,
        "unreadable_count": 0,
        "unscanned_count": 0,
        "filtered_count": 0,
        "buckets": [],
    }
    if not available:
        return result
    select_bucket = _source_bucket_selector(source_settings) if source_settings is not None else None
    counts: Counter[tuple[int, int]] = Counter()
    files = _dataset_image_files(
        directory,
        {".png", ".jpg", ".jpeg", ".webp", ".bmp"},
        recursive=recursive,
        path_pattern=path_pattern,
    )
    deadline = monotonic() + MAX_CENSUS_SECONDS
    for index, path in enumerate(files):
        if index >= MAX_CENSUS_IMAGES or monotonic() >= deadline:
            result["unscanned_count"] = len(files) - index
            break
        try:
            stat = path.stat()
            width, height = _image_size(path, stat.st_mtime_ns, stat.st_size)
            if width <= 0 or height <= 0:
                raise ValueError("Invalid image dimensions")
            if select_bucket and width * height < min_pixels:
                result["filtered_count"] += 1
                continue
            counts[select_bucket(width, height) if select_bucket else (width, height)] += 1
        except (OSError, ValueError, Image.DecompressionBombError):
            result["unreadable_count"] += 1
    result["image_count"] = sum(counts.values())
    result["status"] = "partial" if result["unreadable_count"] or result["unscanned_count"] else "ready"
    result["buckets"] = [
        {"width": width, "height": height, "count": count}
        for (width, height), count in sorted(counts.items())
    ]
    return result
