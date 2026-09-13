from pathlib import Path

from PIL import Image
import pytest

from web.services.config.estimate_buckets import inspect_estimate_buckets


def test_dimension_census_filters_paths_and_reports_unreadable(tmp_path: Path):
    folder = tmp_path / "selected"
    folder.mkdir()
    for name, size in [("a.png", (64, 64)), ("b.png", (64, 64)), ("c.png", (96, 64))]:
        Image.new("RGB", size).save(folder / name)
    (folder / "broken.png").write_bytes(b"invalid")
    Image.new("RGB", (32, 32)).save(tmp_path / "excluded.png")
    result = inspect_estimate_buckets(
        tmp_path, recursive=True, path_pattern="selected/*", available=True,
    )
    assert result == {
        "basis": "image_dimensions", "status": "partial", "image_count": 3,
        "unreadable_count": 1, "unscanned_count": 0, "filtered_count": 0,
        "buckets": [
            {"width": 64, "height": 64, "count": 2},
            {"width": 96, "height": 64, "count": 1},
        ],
    }


def test_pending_census_does_not_fall_back_to_source_images(tmp_path: Path):
    Image.new("RGB", (64, 64)).save(tmp_path / "source.png")
    result = inspect_estimate_buckets(tmp_path, recursive=True, path_pattern="*", available=False)
    assert result["status"] == "pending"
    assert result["buckets"] == []


def test_non_recursive_census_and_empty_directory(tmp_path: Path):
    (tmp_path / "nested").mkdir()
    Image.new("RGB", (64, 64)).save(tmp_path / "nested" / "source.png")
    result = inspect_estimate_buckets(tmp_path, recursive=False, path_pattern="*", available=True)
    assert result["image_count"] == 0
    assert result["unreadable_count"] == 0


def test_bounded_scan_and_header_cache_invalidation(tmp_path: Path, monkeypatch):
    from web.services.config import estimate_buckets

    image = tmp_path / "a.png"
    Image.new("RGB", (64, 64)).save(image)
    Image.new("RGB", (64, 64)).save(tmp_path / "b.png")
    monkeypatch.setattr(estimate_buckets, "MAX_CENSUS_IMAGES", 1)
    first = inspect_estimate_buckets(tmp_path, recursive=True, path_pattern="*", available=True)
    assert first["status"] == "partial"
    assert first["image_count"] == first["unscanned_count"] == 1
    Image.new("RGB", (96, 64)).save(image)
    second = inspect_estimate_buckets(tmp_path, recursive=True, path_pattern="*", available=True)
    assert second["buckets"] == [{"width": 96, "height": 64, "count": 1}]


def test_estimation_opt_in_and_regularization(monkeypatch, tmp_path: Path):
    from web.services import config_service
    from tests.web_config_test_support import (
        _patch_config_service_paths, _write_minimal_config_tree, _write_step_estimate_dataset,
    )

    _, dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    _write_step_estimate_dataset(tmp_path, dataset_path)
    baseline = config_service.estimate_training_steps("lora", "default", "imported")
    assert all("bucket_distribution" not in row for row in baseline["datasets"])
    detailed = config_service.estimate_training_steps("lora", "default", "imported", include_buckets=True)
    assert detailed["total_steps"] == baseline["total_steps"]
    assert detailed["steps_per_epoch"] == baseline["steps_per_epoch"]
    assert all("bucket_distribution" in row and "is_reg" in row for row in detailed["datasets"])


@pytest.mark.parametrize("settings", [
    {"resolution": 512},
    {"resolution": 1024, "enable_bucket": False},
    {"resolution": 1024, "bucket_no_upscale": True},
])
def test_source_projection_matches_preprocessing_without_writing(tmp_path: Path, settings):
    from library.preprocess.images import process_image

    source = tmp_path / "source"
    source.mkdir()
    image = source / "image.png"
    Image.new("RGB", (700, 900)).save(image)
    before = image.read_bytes()
    result = inspect_estimate_buckets(
        source, recursive=True, path_pattern="*", available=True, source_settings=settings,
    )
    assert result["basis"] == "source_projection"
    assert list(source.iterdir()) == [image]
    assert image.read_bytes() == before
    resolution = settings["resolution"]
    _, size = process_image(image, tmp_path / "output", (
        (resolution, resolution), 256, 2048, 64, True,
        settings.get("bucket_no_upscale", False), settings.get("enable_bucket", True),
    ))
    assert result["buckets"] == [{"width": size[0], "height": size[1], "count": 1}]


def test_source_projection_filters_low_resolution(tmp_path: Path):
    Image.new("RGB", (100, 100)).save(tmp_path / "small.png")
    Image.new("RGB", (1000, 800)).save(tmp_path / "large.png")
    result = inspect_estimate_buckets(
        tmp_path, recursive=True, path_pattern="*", available=True,
        source_settings={"resolution": 1024}, min_pixels=500_000,
    )
    assert result["filtered_count"] == result["image_count"] == 1
    assert result["status"] == "ready"


def test_estimation_falls_back_to_source_projection(monkeypatch, tmp_path: Path):
    from web.services import config_service
    from tests.web_config_test_support import _patch_config_service_paths

    _patch_config_service_paths(monkeypatch, tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    Image.new("RGB", (900, 700)).save(source / "image.png")
    monkeypatch.setattr(config_service, "_load_training_config_for_web_run", lambda *a, **kw: {"drop_lowres_images": False})
    monkeypatch.setattr(config_service, "_dataset_rows_for_estimate", lambda cfg: [{
        "source_dir": str(source), "image_dir": str(tmp_path / "not-preprocessed"),
        "settings": {"resolution": 512, "enable_bucket": False},
    }])
    result = config_service.estimate_training_steps("lora", "default", include_buckets=True)
    distribution = result["datasets"][0]["bucket_distribution"]
    assert distribution["basis"] == "source_projection"
    assert distribution["buckets"] == [{"width": 512, "height": 512, "count": 1}]
