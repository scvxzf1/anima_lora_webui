"""Durable, content-addressed sample reference images, separate from datasets."""

import hashlib
import io
from pathlib import Path

from library.env import get_configs_root
from library.training.preview_spec import MAX_REFERENCE_BYTES, publish_reference, read_reference, reference_path


def import_reference(*, data: bytes | None = None, path: str = "") -> dict:
    if data is not None:
        if not data or len(data) > MAX_REFERENCE_BYTES:
            raise ValueError("参考图不能为空或超过 20 MiB")
        image = read_reference(io.BytesIO(data))
    else:
        image = read_reference(reference_path(path))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = buffer.getvalue()
    if len(encoded) > MAX_REFERENCE_BYTES:
        raise ValueError("参考图转换为 PNG 后超过 20 MiB，请先缩小图片")
    key = hashlib.sha256(encoded).hexdigest()
    root = Path(get_configs_root()) / "sample-references"
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{key}.png"
    publish_reference(target, encoded)
    return {"ok": True, "reference_image": str(target.resolve()), "width": image.width,
            "height": image.height, "url": f"/api/config/sample-references/{key}"}


def resolve_reference_asset(key: str) -> Path:
    if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
        raise ValueError("无效参考图标识")
    root = (Path(get_configs_root()) / "sample-references").resolve()
    path = (root / f"{key}.png").resolve()
    if path.parent != root or not path.is_file():
        raise ValueError("参考图不存在")
    return path
