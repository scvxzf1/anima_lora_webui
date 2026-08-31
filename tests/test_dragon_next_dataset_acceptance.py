from __future__ import annotations

import asyncio
import json
from pathlib import Path

import toml

from web.routes import config as config_routes

from tests.web_config_test_support import (
    _JsonRequest,
    _patch_config_service_paths,
    _write_minimal_config_tree,
)


def _payload(response) -> dict:
    return json.loads(response.text or "{}")


def _save_request(
    file: str,
    source: str,
    *,
    repeats: int,
    schedule: bool = False,
    secondary_is_reg: bool = True,
):
    return _JsonRequest({
        "file": file,
        "overwrite": False,
        "defaults": {
            "resolution": 1024,
            "batch_size": 1,
            "enable_bucket": True,
            "prior_loss_weight": 1.25,
        },
        "datasets": [{
            "source_dir": source,
            "image_dir": f"post_{source}",
            "cache_dir": f"cache_{source}",
            "num_repeats": repeats,
            "is_reg": False,
            "settings": {"caption_extension": ".txt"},
        }, {
            "source_dir": f"{source}_secondary",
            "image_dir": f"post_{source}_secondary",
            "cache_dir": f"cache_{source}_secondary",
            "num_repeats": 1,
            "is_reg": secondary_is_reg,
            "settings": {"prior_loss_weight": 1.5},
        }],
        "stage_schedule_enabled": schedule,
        "stage_schedule": ([{
            "name": "全程",
            "subset_index": 0,
            "start_pct": 0,
            "end_pct": 1,
        }] if schedule else []),
    })


def test_dragon_dataset_write_order_and_apply_persist_in_isolated_config_root(
    tmp_path: Path,
    monkeypatch,
):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    train_path = configs / "imported" / "dragon_acceptance.toml"
    train_path.write_text('output_name = "dragon-acceptance"\n', encoding="utf-8")

    alpha_response = asyncio.run(config_routes.handle_dataset_preset_put(
        _save_request(
            "configs/datasets/dragon_alpha.toml",
            "image_dataset/dragon_alpha",
            repeats=5,
            schedule=True,
            secondary_is_reg=False,
        )
    ))
    beta_response = asyncio.run(config_routes.handle_dataset_preset_put(
        _save_request(
            "configs/datasets/dragon_beta.toml",
            "image_dataset/dragon_beta",
            repeats=3,
        )
    ))
    assert alpha_response.status == beta_response.status == 200

    group_response = asyncio.run(config_routes.handle_file_group_create(_JsonRequest({
        "label": "Dragon 隔离验收",
        "kind": "dataset",
    })))
    assert group_response.status == 200
    group_id = _payload(group_response)["group"]["id"]

    first_place = asyncio.run(config_routes.handle_file_group_place(_JsonRequest({
        "target": "file",
        "file": "configs/datasets/dragon_beta.toml",
        "group": group_id,
        "order": ["configs/datasets/dragon_beta.toml"],
    })))
    assert first_place.status == 200
    ordered_place = asyncio.run(config_routes.handle_file_group_place(_JsonRequest({
        "target": "file",
        "file": "configs/datasets/dragon_alpha.toml",
        "group": group_id,
        "order": [
            "configs/datasets/dragon_beta.toml",
            "configs/datasets/dragon_alpha.toml",
        ],
    })))
    assert ordered_place.status == 200
    assert [item["path"] for item in _payload(ordered_place)["group"]["files"]] == [
        "configs/datasets/dragon_beta.toml",
        "configs/datasets/dragon_alpha.toml",
    ]

    apply_response = asyncio.run(config_routes.handle_dataset_preset_apply(_JsonRequest({
        "dataset_file": "configs/datasets/dragon_alpha.toml",
        "train_file": "configs/imported/dragon_acceptance.toml",
    })))
    assert apply_response.status == 200
    applied = _payload(apply_response)
    assert applied["dataset_config"] == "configs/datasets/dragon_alpha.toml"
    assert applied["values"]["stage_schedule_enabled"] is True

    alpha_text = (configs / "datasets" / "dragon_alpha.toml").read_text(encoding="utf-8")
    train_text = train_path.read_text(encoding="utf-8")
    groups_text = (configs / "web-file-groups.toml").read_text(encoding="utf-8")
    groups_doc = toml.loads(groups_text)
    assert 'source_dir = "image_dataset/dragon_alpha"' in alpha_text
    assert "stage_schedule_enabled = true" in alpha_text
    assert alpha_text.index('source_dir = "image_dataset/dragon_alpha"') < alpha_text.index(
        'source_dir = "image_dataset/dragon_alpha_secondary"'
    )
    assert 'dataset_config = "configs/datasets/dragon_alpha.toml"' in train_text
    assert 'source_image_dir = "image_dataset/dragon_alpha"' in train_text
    assert "stage_schedule_enabled = true" in train_text
    assert "dragon_beta.toml" in groups_text
    stored_group = next(group for group in groups_doc["groups"] if group["id"] == group_id)
    assert stored_group["order"] == [
        "configs/datasets/dragon_beta.toml",
        "configs/datasets/dragon_alpha.toml",
    ]


def test_dragon_dataset_new_preset_refuses_overwrite_in_isolated_config_root(
    tmp_path: Path,
    monkeypatch,
):
    configs, _dataset_path = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    request = _save_request(
        "configs/datasets/dragon_collision.toml",
        "image_dataset/original",
        repeats=2,
    )
    first = asyncio.run(config_routes.handle_dataset_preset_put(request))
    second = asyncio.run(config_routes.handle_dataset_preset_put(request))

    assert first.status == 200
    assert second.status == 400
    assert "已存在" in _payload(second)["error"]
    content = (configs / "datasets" / "dragon_collision.toml").read_text(encoding="utf-8")
    assert 'source_dir = "image_dataset/original"' in content
