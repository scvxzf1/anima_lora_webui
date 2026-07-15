from __future__ import annotations

import toml

from tests.web_config_test_support import (
    _patch_config_service_paths,
    _write_minimal_config_tree,
)
from web.services import config_service


def _row(source: str) -> dict:
    stem = source.rsplit("/", 1)[-1]
    return {
        "source_dir": source,
        "image_dir": f"post_image_dataset/{stem}_resized",
        "cache_dir": f"post_image_dataset/{stem}_cache",
        "num_repeats": 1,
    }


def _import_content(source: str, *, stage_name: str | None = None) -> str:
    lines: list[str] = []
    if stage_name is not None:
        lines.extend(
            [
                "stage_schedule_enabled = true",
                (
                    "stage_schedule = [{name = "
                    f'"{stage_name}", subset_index = 0, '
                    "start_pct = 0.0, end_pct = 1.0}]"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "[[datasets]]",
            "resolution = 1024",
            "",
            "[[datasets.subsets]]",
            f'image_dir = "post_image_dataset/{source.rsplit("/", 1)[-1]}_resized"',
            f'cache_dir = "post_image_dataset/{source.rsplit("/", 1)[-1]}_cache"',
            "num_repeats = 1",
            "",
            "[datasets.subsets.custom_attributes]",
            f'source_dir = "{source}"',
            "",
        ]
    )
    return "\n".join(lines)


def _save_old_stage(rel_path: str) -> None:
    config_service.save_dataset_preset(
        rel_path,
        [_row("image_dataset/old")],
        overwrite=True,
        stage_schedule_enabled=True,
        stage_schedule=[
            {
                "name": "old-stage",
                "subset_index": 0,
                "start_pct": 0.0,
                "end_pct": 1.0,
            }
        ],
    )


def test_overwrite_import_without_stage_removes_existing_schedule(tmp_path, monkeypatch):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    rel_path = "configs/datasets/character.toml"
    _save_old_stage(rel_path)

    imported = config_service.import_dataset_preset(
        "character.toml",
        _import_content("image_dataset/new"),
        overwrite=True,
    )

    loaded = config_service.load_dataset_preset(rel_path)
    persisted = toml.loads(
        (configs / "datasets" / "character.toml").read_text(encoding="utf-8")
    )
    assert imported["datasets"][0]["source_dir"] == "image_dataset/new"
    assert "stage_schedule_enabled" not in loaded
    assert "stage_schedule" not in loaded
    assert "stage_schedule_enabled" not in persisted
    assert "stage_schedule" not in persisted


def test_regular_save_without_stage_arguments_preserves_existing_schedule(
    tmp_path, monkeypatch
):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    rel_path = "configs/datasets/character.toml"
    _save_old_stage(rel_path)

    config_service.save_dataset_preset(
        rel_path,
        [_row("image_dataset/edited")],
        overwrite=True,
    )

    loaded = config_service.load_dataset_preset(rel_path)
    assert loaded["datasets"][0]["source_dir"] == "image_dataset/edited"
    assert loaded["stage_schedule_enabled"] is True
    assert loaded["stage_schedule"][0]["name"] == "old-stage"


def test_overwrite_import_with_stage_replaces_existing_schedule(tmp_path, monkeypatch):
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    rel_path = "configs/datasets/character.toml"
    _save_old_stage(rel_path)

    config_service.import_dataset_preset(
        "character.toml",
        _import_content("image_dataset/new", stage_name="new-stage"),
        overwrite=True,
    )

    loaded = config_service.load_dataset_preset(rel_path)
    assert loaded["stage_schedule_enabled"] is True
    assert loaded["stage_schedule"][0]["name"] == "new-stage"
