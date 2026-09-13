"""Pixel geometry shared by mask-editor reads and save validation."""

from pathlib import Path

from PIL import Image

from web.services.config.estimate_buckets import _source_bucket_selector

MAX_MASK_PIXELS = 16_777_216


def check_size(size: tuple[int, int]) -> None:
    if min(size) < 1 or size[0] * size[1] > MAX_MASK_PIXELS:
        raise ValueError("蒙版编辑支持最多 1600 万像素的图片")


def prepare_image(path: Path, *, project: bool, settings: dict) -> Image.Image:
    # Match training's raw pixel orientation, not the browser's EXIF rotation.
    with Image.open(path) as source:
        check_size(source.size)
        image = source.convert("RGBA")
    if project:
        width, height = image.size
        bw, bh = _source_bucket_selector(settings)(width, height)
        check_size((bw, bh))
        if width / height > bw / bh:
            resized = (round(bh * width / height), bh)
        else:
            resized = (bw, round(bw * height / width))
        check_size(resized)
        image = image.resize(resized, Image.Resampling.LANCZOS)
        left, top = (resized[0] - bw) // 2, (resized[1] - bh) // 2
        image = image.crop((left, top, left + bw, top + bh))
    image.info.clear()
    return image
