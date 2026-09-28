import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from PIL import Image
import pytest

from library.training import preview_spec
from web.routes import sample_references as routes
from web.services import sample_references as service


def image_bytes(format="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 32), "red").save(buffer, format)
    return buffer.getvalue()


@pytest.fixture
def isolated_root(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "get_configs_root", lambda: tmp_path)
    return tmp_path


def test_concurrent_import_is_complete_and_content_addressed(isolated_root):
    data = image_bytes()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: service.import_reference(data=data), range(24)))
    assert all(result == results[0] for result in results)
    path = Path(results[0]["reference_image"])
    assert path.read_bytes() == data
    with Image.open(path) as image:
        image.load()
        assert image.size == (64, 32)
    assert list(path.parent.iterdir()) == [path]
    assert service.resolve_reference_asset(results[0]["url"].rsplit("/", 1)[1]) == path


@pytest.mark.parametrize("data", [b"", b"not an image"])
def test_import_rejects_invalid_bytes(isolated_root, data):
    with pytest.raises(ValueError):
        service.import_reference(data=data)


def test_import_accepts_jpeg_and_normalizes_to_png(isolated_root):
    result = service.import_reference(data=image_bytes("JPEG"))
    path = Path(result["reference_image"])
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.size == (64, 32)


def test_import_rejects_animated_gif(isolated_root):
    buffer = io.BytesIO()
    Image.new("RGB", (64, 32), "red").save(
        buffer,
        "GIF",
        save_all=True,
        append_images=[Image.new("RGB", (64, 32), "blue")],
        duration=10,
        loop=0,
    )
    with pytest.raises(ValueError, match="单帧"):
        service.import_reference(data=buffer.getvalue())


def test_import_rejects_byte_and_pixel_limits(isolated_root, monkeypatch):
    data = image_bytes()
    monkeypatch.setattr(service, "MAX_REFERENCE_BYTES", len(data) - 1)
    with pytest.raises(ValueError):
        service.import_reference(data=data)
    monkeypatch.setattr(service, "MAX_REFERENCE_BYTES", 20 * 1024 * 1024)
    monkeypatch.setattr(preview_spec, "MAX_REFERENCE_PIXELS", 100)
    with pytest.raises(ValueError):
        service.import_reference(data=data)


def test_asset_boundary_rejects_invalid_key_and_symlink(isolated_root, tmp_path):
    for key in ("../outside", "A" * 64, "f" * 63, "f" * 64):
        with pytest.raises(ValueError):
            service.resolve_reference_asset(key)
    root = isolated_root / "sample-references"
    root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_bytes(image_bytes())
    (root / ("e" * 64 + ".png")).symlink_to(outside)
    with pytest.raises(ValueError):
        service.resolve_reference_asset("e" * 64)


@pytest.mark.parametrize("body", [None, [], {}, {"path": 42}])
def test_import_route_rejects_bad_body(body):
    request = SimpleNamespace(content_type="application/json", json=AsyncMock(return_value=body))
    response = asyncio.run(routes.handle_import(request))
    assert response.status == 400
    assert json.loads(response.body)["ok"] is False


def test_import_route_bad_json_and_missing_asset():
    request = SimpleNamespace(content_type="application/json", json=AsyncMock(side_effect=ValueError("bad json")))
    assert asyncio.run(routes.handle_import(request)).status == 400
    assert asyncio.run(routes.handle_image(SimpleNamespace(match_info={"key": "../bad"}))).status == 404


def test_import_route_accepts_path(isolated_root):
    source = isolated_root / "参考 图 --.png"
    source.write_bytes(image_bytes())
    request = SimpleNamespace(content_type="application/json", json=AsyncMock(return_value={"path": str(source)}))
    response = asyncio.run(routes.handle_import(request))
    result = json.loads(response.body)
    assert response.status == 200 and result["ok"]
    assert result["width"] == 64 and result["height"] == 32


def test_upload_route_accepts_file_and_enforces_limit(isolated_root, monkeypatch):
    data = image_bytes()

    def request():
        part = SimpleNamespace(name="file", read_chunk=AsyncMock(side_effect=[data, b""]))
        reader = SimpleNamespace(next=AsyncMock(return_value=part))
        return SimpleNamespace(content_type="multipart/form-data", multipart=AsyncMock(return_value=reader))

    assert asyncio.run(routes.handle_import(request())).status == 200
    monkeypatch.setattr(routes, "MAX_REFERENCE_BYTES", len(data) - 1)
    assert asyncio.run(routes.handle_import(request())).status == 400


def test_upload_route_accepts_jpeg(isolated_root):
    data = image_bytes("JPEG")
    part = SimpleNamespace(name="file", read_chunk=AsyncMock(side_effect=[data, b""]))
    reader = SimpleNamespace(next=AsyncMock(return_value=part))
    request = SimpleNamespace(content_type="multipart/form-data", multipart=AsyncMock(return_value=reader))
    response = asyncio.run(routes.handle_import(request))
    result = json.loads(response.body)
    assert response.status == 200 and result["ok"]


def test_upload_route_returns_json_for_unexpected_import_error(monkeypatch):
    monkeypatch.setattr(routes, "import_reference", lambda **_: (_ for _ in ()).throw(RuntimeError("boom")))
    data = image_bytes()
    part = SimpleNamespace(name="file", read_chunk=AsyncMock(side_effect=[data, b""]))
    reader = SimpleNamespace(next=AsyncMock(return_value=part))
    request = SimpleNamespace(content_type="multipart/form-data", multipart=AsyncMock(return_value=reader))
    response = asyncio.run(routes.handle_import(request))
    assert response.status == 500
    assert json.loads(response.body) == {"ok": False, "error": "参考图载入失败，请检查图片格式后重试"}


def test_atomic_publish_failure_removes_temporary_file(tmp_path, monkeypatch):
    target = tmp_path / "references" / "image.png"

    def fail_replace(*args):
        raise OSError("disk write failed")

    monkeypatch.setattr(preview_spec.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk write failed"):
        preview_spec.publish_reference(target, image_bytes())
    assert list(target.parent.iterdir()) == []


def test_atomic_publish_rejects_symlink_target(tmp_path):
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"unchanged")
    target = tmp_path / "target.png"
    target.symlink_to(outside)
    with pytest.raises(ValueError):
        preview_spec.publish_reference(target, image_bytes())
    assert outside.read_bytes() == b"unchanged"
