from io import BytesIO
import base64
from pathlib import Path

from PIL import Image

from web.services.config import dataset_cover as cover
from tests.web_config_test_support import _patch_config_service_paths, _write_minimal_config_tree
from web.services import config_service


def test_cover_source_fallback_cache_and_small_thumbnail(tmp_path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (500, 300), "red").save(images / "one.png")
    file = "configs/datasets/cover.toml"
    config_service.save_dataset_preset(file, [{"source_dir": "missing", "image_dir": "images", "num_repeats": 1}], {})
    result = cover.dataset_cover(file)
    with Image.open(BytesIO(base64.b64decode(result["image"].split(",", 1)[1]))) as image:
        assert max(image.size) <= 96
    monkeypatch.setattr(cover, "_detect", lambda *_: (_ for _ in ()).throw(AssertionError("cache miss")))
    assert cover.dataset_cover(file) == result


def test_cover_missing_empty_corrupt_and_symlink(tmp_path):
    resolve = lambda value: Path(value).resolve()
    assert "不存在" in cover._detect([{"source_dir": str(tmp_path / "missing")}], resolve)["reason"]
    assert cover._detect([{"source_dir": str(tmp_path)}], resolve)["image"] is None
    (tmp_path / "bad.png").write_bytes(b"broken")
    assert "读取失败" in cover._detect([{"source_dir": str(tmp_path)}], resolve)["reason"]
    (tmp_path / "bad.png").unlink()
    (tmp_path / "loop").symlink_to(tmp_path, target_is_directory=True)
    assert cover._detect([{"source_dir": str(tmp_path)}], resolve)["image"] is None


def test_cover_scan_limit_and_filter(tmp_path, monkeypatch):
    for index in range(5):
        Image.new("RGB", (3, 3)).save(tmp_path / f"{index}.png")
    resolve = lambda value: Path(value).resolve()
    row = {"source_dir": str(tmp_path), "path_pattern": "*.jpg"}
    monkeypatch.setattr(cover, "MAX_ENTRIES", 2)
    assert "扫描上限" in cover._detect([row], resolve)["reason"]
    monkeypatch.setattr(cover, "MAX_ENTRIES", 100)
    assert cover._detect([row], resolve)["image"] is None


def test_cover_invalid_path_rejected(tmp_path, monkeypatch):
    import pytest
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    with pytest.raises(ValueError):
        cover.dataset_cover("../outside.toml")
