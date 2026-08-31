from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path

from PIL import Image

from tests.web_config_test_support import (
    _QueryRequest,
    _patch_config_service_paths,
    _write_minimal_config_tree,
)
from web.routes import config as config_routes
from web.services import config_service
from web.services.config import dataset_media
from web.services.config.dataset_media import _dataset_image_files, _list_dataset_image_files
from web.services.config.dataset_preview_thumbnail import (
    _render_thumbnail_cached,
    render_dataset_preview_thumbnail,
)


def test_dataset_image_listing_pages_beyond_legacy_preview_limit(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(130):
        (image_dir / f"image_{index:03d}.png").touch()

    page = _list_dataset_image_files(image_dir, 48, offset=96)

    assert [path.name for path in page["items"]] == [
        f"image_{index:03d}.png" for index in range(96, 130)
    ]
    assert page == {
        "items": page["items"],
        "total": 130,
        "offset": 96,
        "limit": 48,
        "returned": 34,
        "next_offset": 130,
        "has_more_before": True,
        "has_more_after": False,
    }


def test_dataset_preview_service_only_builds_metadata_for_requested_page(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    source_dir = tmp_path / "image_dataset" / "paged"
    train_dir = tmp_path / "post_image_dataset" / "paged"
    source_dir.mkdir(parents=True)
    train_dir.mkdir(parents=True)
    for index in range(5):
        stem = f"image_{index:02d}"
        Image.new("RGB", (4, 3), color=(index, 20, 40)).save(source_dir / f"{stem}.png")
        (source_dir / f"{stem}.txt").write_text(f"caption {index}", encoding="utf-8")

    config_service.save_dataset_preset(
        "configs/datasets/paged-preview.toml",
        [{
            "source_dir": "image_dataset/paged",
            "image_dir": "post_image_dataset/paged",
            "cache_dir": "post_image_dataset/paged_cache",
            "num_repeats": 1,
        }],
        {"caption_extension": ".txt"},
    )

    page = config_service.list_dataset_preset_images(
        "configs/datasets/paged-preview.toml",
        0,
        source="source",
        limit=2,
        offset=2,
    )

    assert [image["name"] for image in page["images"]] == ["image_02.png", "image_03.png"]
    assert [image["caption"]["text"] for image in page["images"]] == ["caption 2", "caption 3"]
    assert page["total"] == 5
    assert page["offset"] == 2
    assert page["returned"] == 2
    assert page["next_offset"] == 4
    assert page["has_more_before"] is True
    assert page["has_more_after"] is True
    assert page["images"][0]["thumbnail_url"].startswith(
        "/api/config/dataset-presets/thumbnail?"
    )
    assert "&v=" in page["images"][0]["thumbnail_url"]


def test_dataset_preview_route_forwards_offset(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_listing(file: str, dataset_index: int, **kwargs):
        captured.update(file=file, dataset_index=dataset_index, **kwargs)
        return {"ok": True, "images": []}

    monkeypatch.setattr(config_routes, "list_dataset_preset_images", fake_listing)
    response = asyncio.run(config_routes.handle_dataset_preset_images(_QueryRequest({
        "file": "configs/datasets/paged.toml",
        "dataset_index": "2",
        "source": "source",
        "limit": "48",
        "offset": "144",
    })))

    assert response.status == 200
    assert captured == {
        "file": "configs/datasets/paged.toml",
        "dataset_index": 2,
        "source": "source",
        "limit": 48,
        "offset": 144,
    }


def test_dataset_image_listing_cache_is_bounded_and_invalidates_on_directory_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "one.png").touch()
    dataset_media._DATASET_IMAGE_LIST_CACHE.clear()
    calls = 0

    def fake_walk_images(path: Path, **_kwargs):
        nonlocal calls
        calls += 1
        return list(path.iterdir())

    monkeypatch.setattr(dataset_media, "walk_images", fake_walk_images)
    first = _dataset_image_files(image_dir, {".png"})
    second = _dataset_image_files(image_dir, {".png"})
    (image_dir / "two.png").touch()
    third = _dataset_image_files(image_dir, {".png"})

    assert [path.name for path in first] == ["one.png"]
    assert second == first
    assert [path.name for path in third] == ["one.png", "two.png"]
    assert calls == 2
    assert len(dataset_media._DATASET_IMAGE_LIST_CACHE) <= dataset_media._DATASET_IMAGE_LIST_CACHE_ITEMS


def test_dataset_preview_thumbnail_is_resized_and_reused(tmp_path: Path) -> None:
    image_path = tmp_path / "large.png"
    Image.new("RGB", (1600, 1200), color=(80, 120, 160)).save(image_path)
    _render_thumbnail_cached.cache_clear()

    first = render_dataset_preview_thumbnail(image_path)
    second = render_dataset_preview_thumbnail(image_path)
    with Image.open(BytesIO(first.content)) as thumbnail:
        assert thumbnail.width <= 640
        assert thumbnail.height <= 480

    assert first == second
    assert first.content_type in {"image/webp", "image/png"}
    assert _render_thumbnail_cached.cache_info().hits == 1


def test_dataset_preview_thumbnail_route_keeps_raw_resolver_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (1000, 750), color=(20, 40, 60)).save(image_path)
    monkeypatch.setattr(config_routes, "resolve_dataset_preview_image", lambda *_args, **_kwargs: image_path)

    class Request:
        query = {
            "file": "configs/datasets/paged.toml",
            "dataset_index": "0",
            "source": "source",
            "image": str(image_path),
        }
        headers: dict[str, str] = {}

    response = asyncio.run(config_routes.handle_dataset_preset_thumbnail(Request()))
    assert response.status == 200
    assert response.content_type in {"image/webp", "image/png"}
    assert response.headers["Cache-Control"] == "private, max-age=86400"
    assert response.headers["ETag"]

    Request.headers = {"If-None-Match": response.headers["ETag"]}
    cached = asyncio.run(config_routes.handle_dataset_preset_thumbnail(Request()))
    assert cached.status == 304
