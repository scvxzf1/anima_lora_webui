from dataclasses import replace
from types import SimpleNamespace

import pytest
import toml

from library.models.family_registry import (
    MODEL_FAMILY_REGISTRY,
    model_family_capability_catalog,
)
from library.training.compat_matrix import check_training_compat
from library.training.task_contracts import task_config_values
from tests.web_config_test_support import _write_selected_checkpoint_preflight_config
from web.services import config_service


def _edit_config(family="qwen_image_2_1"):
    return {
        "model_family": family,
        "qwen_image_2_1_task": "edit",
        "cache_latents": True,
        "cache_text_encoder_outputs": True,
        "datasets": [
            {
                "batch_size": 1,
                "subsets": [
                    {
                        "image_dir": "target",
                        "reference_image_dir": "reference",
                    }
                ],
            }
        ],
    }


def _codes(cfg):
    return {issue.code for issue in check_training_compat(cfg).errors}


def test_catalog_declares_ordinary_tasks_for_all_existing_families():
    items = {item["name"]: item for item in model_family_capability_catalog()}
    assert all("t2i" in item["supported_tasks"] for item in items.values())
    assert items["qwen_image_2_1"]["supported_tasks"] == ["edit", "t2i"]


@pytest.mark.parametrize("family", ["anima", "krea2", "z_image"])
def test_unsupported_edit_is_rejected_even_with_legacy_default_task(family):
    cfg = _edit_config(family)
    cfg["qwen_image_2_1_task"] = "t2i"
    assert "unsupported_training_task" in _codes(cfg)
    from library.training.bootstrap import TrainingBootstrap

    args = SimpleNamespace(model_family=family)
    with pytest.raises(ValueError, match="不支持编辑数据集"):
        TrainingBootstrap.validate_qwen_dataset_config(args, cfg)


@pytest.mark.parametrize("family", ["qwen_image_2_1", "qwen21", "qwen_image_21"])
def test_task_projection_and_validation_accept_aliases(family):
    assert task_config_values(family, "edit") == {"qwen_image_2_1_task": "edit"}
    assert not _codes(_edit_config(family))


def test_registry_declaration_cannot_enable_missing_runtime_contract(monkeypatch):
    spec = MODEL_FAMILY_REGISTRY["anima"]
    monkeypatch.setitem(
        MODEL_FAMILY_REGISTRY,
        "anima",
        replace(
            spec,
            supported_tasks=frozenset({"t2i", "edit"}),
            training_task_key="example_task",
        ),
    )
    with pytest.raises(ValueError, match="尚未接入"):
        task_config_values("anima", "edit")
    assert "training_task_unavailable" in _codes(
        {**_edit_config("anima"), "example_task": "edit"}
    )


def test_missing_task_keeps_t2i_default_and_does_not_silently_enable_edit():
    cfg = _edit_config()
    cfg.pop("qwen_image_2_1_task")
    assert "qwen_image_2_1_edit_task_mismatch" in _codes(cfg)
    assert not _codes({"model_family": "qwen_image_2_1"})
    assert "invalid_model_family" in _codes({"model_family": "made-up"})


@pytest.mark.parametrize(
    "override,code",
    [
        ({"cache_latents": False}, "edit_latent_cache"),
        ({"cache_text_encoder_outputs": False}, "edit_text_cache"),
        ({"train_batch_size": 2}, "edit_batch_size"),
        ({"general": {"color_aug": True}}, "edit_augmentation"),
    ],
)
def test_shared_edit_contract_keeps_runtime_constraints(override, code):
    assert f"qwen_image_2_1_{code}" in _codes({**_edit_config(), **override})


def test_preflight_reads_linked_dataset_and_returns_model_task_summary(
    tmp_path, monkeypatch
):
    _write_selected_checkpoint_preflight_config(
        tmp_path,
        monkeypatch,
        [
            'model_family = "anima"',
            'dataset_config = "configs/datasets/edit.toml"',
        ],
    )
    path = tmp_path / "configs/datasets/edit.toml"
    path.write_text(
        toml.dumps({"datasets": _edit_config()["datasets"]}), encoding="utf-8"
    )
    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/selected.toml",
    )
    assert not result["ok"]
    assert any(
        item["key"] == "model_family" and "不支持编辑数据集" in item["message"]
        for item in result["errors"]
    )
    assert result["training_task"] == {
        "model_family": "anima",
        "model_name": "Anima",
        "supported_tasks": ["t2i"],
        "configured_task": "t2i",
        "dataset_task": "edit",
    }


@pytest.mark.parametrize(
    "content", [None, "invalid [toml", "[general]\nbatch_size = 1"]
)
def test_preflight_missing_or_malformed_linked_dataset_does_not_pass(
    tmp_path, monkeypatch, content
):
    _write_selected_checkpoint_preflight_config(
        tmp_path,
        monkeypatch,
        [
            'dataset_config = "configs/datasets/edit.toml"',
            'model_family = "qwen_image_2_1"',
            'qwen_image_2_1_task = "edit"',
        ],
    )
    if content is not None:
        (tmp_path / "configs/datasets/edit.toml").write_text(content, encoding="utf-8")
    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/selected.toml",
    )
    assert not result["ok"]
    assert any(item["key"] == "dataset_config" for item in result["errors"])


def test_ordinary_preprocess_can_generate_dataset_without_claiming_task_verified(
    tmp_path, monkeypatch
):
    _write_selected_checkpoint_preflight_config(tmp_path, monkeypatch, [])
    result = config_service.preflight_training_config(
        "lora",
        "default",
        "imported",
        config_file="configs/imported/selected.toml",
    )
    assert result["ok"]
    assert result["training_task"]["dataset_task"] == "unknown"
