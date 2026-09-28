import asyncio
import json
import os
from types import SimpleNamespace
import pytest

from PIL import Image

from tests.test_preview_service import _patch_preview_settings_file
from web.routes import preview as preview_routes
from web.services import preview_service


def setup_preview(tmp_path, monkeypatch):
    settings = tmp_path / "configs" / "web-ui-settings.toml"
    settings.parent.mkdir()
    settings.write_text('[global]\noutput_root="output/runs"\n', encoding="utf-8")
    _patch_preview_settings_file(monkeypatch, settings, root=tmp_path)


def test_images_offset_only_reads_page_metadata(tmp_path, monkeypatch):
    setup_preview(tmp_path, monkeypatch)
    directory = tmp_path / "output" / "tests"
    directory.mkdir(parents=True)
    for index in range(125):
        path = directory / f"image-{index}.png"
        Image.new("RGB", (2, 2)).save(path)
        os.utime(path, (index + 100, index + 100))
    seen = []
    original = preview_service._image_meta

    def read(path, **kwargs):
        seen.append(path.name)
        return original(path, **kwargs)

    monkeypatch.setattr(preview_service, "_image_meta", read)
    page = preview_service.list_preview_images("inference", limit=60, offset=120)
    assert page["total"] == 125
    assert page["count"] == 5
    assert page["next_offset"] is None
    assert seen == [f"image-{index}.png" for index in range(4, -1, -1)]


def test_config_group_image_route_forwards_nonnegative_offset(monkeypatch):
    captured = {}

    def list_page(tasks, **kwargs):
        captured["tasks"] = tasks
        captured.update(kwargs)
        return {"ok": True, "mode": "config_group", "images": [], "offset": kwargs["offset"]}

    monkeypatch.setattr(preview_routes, "_selected_config_group_tasks", lambda _request: [{"id": "task-a"}])
    monkeypatch.setattr(preview_routes, "list_config_group_preview_images", list_page)
    request = SimpleNamespace(
        query={"source": "training", "mode": "config_group", "limit": "25", "offset": "50"},
        app={},
    )

    response = asyncio.run(preview_routes.handle_preview_images(request))

    assert response.status == 200
    assert json.loads(response.text)["offset"] == 50
    assert captured["offset"] == 50
    assert captured["limit"] == 25


def test_config_group_image_route_rejects_negative_offset():
    request = SimpleNamespace(
        query={"source": "training", "mode": "config_group", "offset": "-1"},
        app={},
    )

    response = asyncio.run(preview_routes.handle_preview_images(request))

    assert response.status == 400
    assert "offset" in json.loads(response.text)["error"]


def test_weights_beyond_500_are_paged_with_bounded_header_reads(tmp_path, monkeypatch):
    setup_preview(tmp_path, monkeypatch)
    directory = tmp_path / "output" / "runs" / "task"
    directory.mkdir(parents=True)
    for index in range(505):
        path = directory / f"weight-{index:04}.safetensors"
        path.write_bytes(b"stub")
        os.utime(path, (index + 100, index + 100))
    seen = []
    monkeypatch.setattr(preview_service, "_read_safetensors_metadata", lambda path: seen.append(path.name) or {})
    task = {"output_dir": "output/runs/task", "id": "task"}
    page = preview_service.list_training_weights(task, limit=100, offset=500, sort="recent", allow_latest_fallback=False)
    assert page["total"] == 505
    assert page["count"] == 5
    assert page["next_offset"] is None
    assert seen == [f"weight-{index:04}.safetensors" for index in range(4, -1, -1)]
    by_name = preview_service.list_training_weights(task, limit=2, offset=1, sort="name", allow_latest_fallback=False)
    assert [item["name"] for item in by_name["weights"]] == ["weight-0001.safetensors", "weight-0002.safetensors"]
    assert by_name["next_offset"] == 3
    with pytest.raises(ValueError, match="分页需选择"):
        preview_service.list_training_weights(task, offset=1)
