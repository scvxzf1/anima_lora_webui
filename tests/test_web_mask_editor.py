from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import pytest
import tomlkit
from PIL import Image

from tests.web_config_test_support import _patch_config_service_paths, _write_minimal_config_tree
from web.services import config_service
from web.services.config import mask_editor as masks

FILE = "configs/datasets/manual-mask.toml"


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    for directory in ["source/nested", "training/nested"]:
        (tmp_path / directory).mkdir(parents=True)
    Image.new("RGB", (80, 60), (80, 120, 200)).save(tmp_path / "source/nested/photo.jpg")
    Image.new("RGB", (64, 48), (80, 120, 200)).save(tmp_path / "training/nested/photo.png")
    config_service.save_dataset_preset(FILE, [{
        "source_dir": "source", "image_dir": "training", "cache_dir": "cache",
        "mask_mode": "none", "num_repeats": 1,
    }], {"resolution": 64, "enable_bucket": False})
    return tmp_path


def png(size=(64, 48), value=0):
    output = BytesIO()
    image = Image.new("L", size, value)
    image.paste(255, (0, 0, size[0] // 2, size[1]))
    image.save(output, format="PNG")
    return output.getvalue()


def test_save_reload_apply_and_training_loader(dataset):
    from library.datasets.image_utils import load_mask_from_dir

    source = "source/nested/photo.jpg"
    before = masks.read_mask(FILE, 0, source)
    assert before["basis"] == "training"
    assert (before["width"], before["height"]) == (64, 48)
    result = masks.save_mask(FILE, 0, source, before["revision"], png())
    after = masks.read_mask(FILE, 0, source)
    assert result["revision"] == after["revision"] != before["revision"]
    assert after["has_mask"]
    decoded = Image.open(BytesIO(base64.b64decode(after["mask_url"].split(",")[1])))
    assert decoded.getpixel((0, 0)) == 255 and decoded.getpixel((63, 0)) == 0
    assert masks.list_masks(FILE, 0)["images"][0]["has_mask"]
    # Save alone must not silently change training semantics.
    assert config_service.load_dataset_preset(FILE)["datasets"][0]["mask_mode"] == "none"
    listing = masks.list_masks(FILE, 0)
    masks.apply_masks(FILE, 0, listing["config_revision"])
    row = config_service.load_dataset_preset(FILE)["datasets"][0]
    assert row["mask_mode"] == "external" and row["alpha_mask"] is True
    tensor = load_mask_from_dir(str(dataset / row["mask_dir"]),
                               str(dataset / "training/nested/photo.png"), (64, 48),
                               str(dataset / "training"))
    assert tensor is not None and tensor[0, 0] == 1 and tensor[0, -1] == 0
    assert not list((dataset / "source").rglob("*_mask.png"))


def test_conflicts_and_invalid_images_are_non_destructive(dataset):
    source = "source/nested/photo.jpg"
    before = masks.read_mask(FILE, 0, source)
    for content in [b"not an image", png((1, 1))]:
        with pytest.raises((ValueError, OSError)):
            masks.save_mask(FILE, 0, source, before["revision"], content)
    with pytest.raises(masks.MaskConflict):
        masks.save_mask(FILE, 0, source, "stale", png())
    assert not masks.read_mask(FILE, 0, source)["has_mask"]
    masks.save_mask(FILE, 0, source, before["revision"], png())
    with pytest.raises(masks.MaskConflict):
        masks.save_mask(FILE, 0, source, before["revision"], png(value=255))


def test_paths_and_symlinks_fail_closed(dataset):
    with pytest.raises(ValueError):
        masks.read_mask(FILE, 0, "training/nested/photo.png")
    with pytest.raises(ValueError):
        masks.read_mask(FILE, -1, "source/nested/photo.jpg")
    page = masks.list_masks(FILE, 0)
    root = dataset / page["mask_dir"]
    root.mkdir(parents=True)
    (root / "nested").symlink_to(dataset / "source/nested", target_is_directory=True)
    with pytest.raises(ValueError, match="越界"):
        masks.read_mask(FILE, 0, "source/nested/photo.jpg")


def test_projected_geometry_matches_preprocessing(dataset):
    from library.preprocess.images import process_image

    (dataset / "training/nested/photo.png").unlink()
    path = dataset / "source/nested/photo.jpg"
    pattern = Image.new("RGB", (80, 60))
    pattern.putdata([(x * 3 % 256, y * 4 % 256, (x + y) * 2 % 256)
                     for y in range(60) for x in range(80)])
    exif = Image.Exif()
    exif[274] = 6
    pattern.save(path, exif=exif)
    process_image(path, dataset / "expected", ((64, 64), 256, 2048, 64, True, False, False))
    result = masks.read_mask(FILE, 0, "source/nested/photo.jpg")
    assert result["basis"] == "projected"
    actual = Image.open(BytesIO(base64.b64decode(result["image_url"].split(",")[1])))
    expected = Image.open(dataset / "expected/photo.png")
    assert actual.size == expected.size == (64, 64)
    assert actual.tobytes() == expected.tobytes()
    assert actual.getexif().get(274) is None


def test_apply_preserves_comments_and_unrelated_fields(dataset):
    path = dataset / FILE
    content = path.read_text()
    path.write_text("# keep this comment\n" + content)
    page = masks.list_masks(FILE, 0)
    masks.apply_masks(FILE, 0, page["config_revision"])
    assert path.read_text().startswith("# keep this comment\n")
    assert tomlkit.parse(path.read_text())["datasets"][0]["subsets"][0]["num_repeats"] == 1
    with pytest.raises(masks.MaskConflict):
        masks.apply_masks(FILE, 0, page["config_revision"])


def test_locked_preset_cannot_write_masks(dataset, monkeypatch):
    original = masks.presets.load_dataset_preset
    def locked(file):
        result = original(file)
        result["readonly"] = True
        return result
    monkeypatch.setattr(masks.presets, "load_dataset_preset", locked)
    before = masks.read_mask(FILE, 0, "source/nested/photo.jpg")
    with pytest.raises(PermissionError):
        masks.save_mask(FILE, 0, "source/nested/photo.jpg", before["revision"], png())


def test_duplicate_stems_cannot_overwrite_each_other(dataset):
    Image.new("RGB", (80, 60)).save(dataset / "source/nested/photo.png")
    with pytest.raises(ValueError, match="重名"):
        masks.read_mask(FILE, 0, "source/nested/photo.jpg")


def test_legacy_auto_masks_are_loaded_and_applied_in_place(dataset):
    path = dataset / FILE
    path.write_text(path.read_text().replace('mask_mode = "none"', 'mask_mode = "auto"'))
    legacy = dataset / "masks/sam/nested"
    legacy.mkdir(parents=True)
    (legacy / "photo_mask.png").write_bytes(png())
    page = masks.list_masks(FILE, 0)
    assert page["mask_dir"] == "masks/sam"
    assert masks.read_mask(FILE, 0, "source/nested/photo.jpg")["has_mask"]
    masks.apply_masks(FILE, 0, page["config_revision"])
    assert config_service.load_dataset_preset(FILE)["datasets"][0]["mask_dir"] == "masks/sam"


def test_mask_directory_cannot_overwrite_source_assets(dataset):
    path = dataset / FILE
    path.write_text(path.read_text().replace('mask_mode = "none"',
                    'mask_mode = "external"\nmask_dir = "source"'))
    with pytest.raises(ValueError, match="独立目录"):
        masks.list_masks(FILE, 0)


def test_http_round_trip_and_conflict_status(dataset):
    import asyncio
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from web.routes.mask_editor import setup_mask_editor_routes

    async def exercise():
        app = web.Application()
        setup_mask_editor_routes(app)
        async with TestClient(TestServer(app)) as client:
            params = {"file": FILE, "dataset_index": "0", "image": "source/nested/photo.jpg"}
            response = await client.get("/api/config/dataset-masks/image", params=params)
            assert response.status == 200 and response.headers["Cache-Control"] == "no-store"
            revision = (await response.json())["revision"]
            headers = {"Content-Type": "image/png", "If-Match": revision}
            response = await client.put("/api/config/dataset-masks/image", params=params, headers=headers, data=png())
            assert response.status == 200
            response = await client.put("/api/config/dataset-masks/image", params=params, headers=headers, data=png())
            assert response.status == 409
            response = await client.put("/api/config/dataset-masks/image", params=params, data=b"invalid")
            assert response.status == 415
            response = await client.get("/api/config/dataset-masks", params={"file": FILE, "dataset_index": "bad"})
            assert response.status == 400
    asyncio.run(exercise())
