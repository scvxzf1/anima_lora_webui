from __future__ import annotations

from pathlib import Path

import pytest
import toml
from PIL import Image

from tests.web_config_test_support import _patch_config_service_paths, _write_minimal_config_tree
from web.services import config_service


def _write_pair(target_dir: Path, reference_dir: Path, *, reference: bool = True) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 32), color=(10, 20, 30)).save(target_dir / "target.png")
    if reference:
        reference_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 32), color=(40, 50, 60)).save(reference_dir / "target.jpg")


def _edit_rows() -> list[dict]:
    return [{
        "source_dir": "image_dataset/reference",
        "image_dir": "post_image_dataset/targets",
        "reference_image_dir": "image_dataset/reference",
        "cache_dir": "post_image_dataset/cache",
        "num_repeats": 1,
        "settings": {"batch_size": 1, "resolution": 64},
    }]


def _paired_rows(count: int = 2) -> list[dict]:
    rows = []
    for number in range(1, count + 1):
        rows.extend([
            {"edit_role": "before", "edit_pair_id": str(number),
             "source_dir": f"image_dataset/reference_{number}"},
            {"edit_role": "after", "edit_pair_id": str(number),
             "source_dir": f"image_dataset/target_{number}",
             "image_dir": f"post_image_dataset/target_{number}",
             "cache_dir": f"post_image_dataset/cache_{number}",
             "settings": {"batch_size": 1, "resolution": 64}},
        ])
    return rows


def test_edit_roles_roundtrip_as_target_only_runtime_rows(tmp_path: Path, monkeypatch) -> None:
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    rows = _paired_rows()
    rows[0], rows[1] = rows[1], rows[0]
    config_service.save_dataset_preset("configs/datasets/pairs.toml", rows, {"batch_size": 1})
    loaded = config_service.load_dataset_preset("configs/datasets/pairs.toml")
    data = toml.loads((tmp_path / "configs/datasets/pairs.toml").read_text(encoding="utf-8"))

    assert [(row["edit_role"], row["edit_pair_id"]) for row in loaded["datasets"]] == [
        (row["edit_role"], row["edit_pair_id"]) for row in rows
    ]
    assert loaded["defaults"]["qwen_edit_enabled"] is True
    assert loaded["summary"]["train_dataset_count"] == 2
    assert len(data["datasets"]) == 2
    assert [item["subsets"][0]["reference_image_dir"] for item in data["datasets"]] == [
        "image_dataset/reference_1", "image_dataset/reference_2"
    ]

    data["datasets"][0]["subsets"][0]["reference_image_dir"] = "image_dataset/wrong"
    (tmp_path / "configs/datasets/pairs.toml").write_text(toml.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="不一致"):
        config_service.load_dataset_preset("configs/datasets/pairs.toml")


@pytest.mark.parametrize("mutate,match", [
    (lambda rows: rows.pop(1), "必须各有一个"),
    (lambda rows: rows[1].update(edit_role="before"), "重复"),
    (lambda rows: rows[1].update(edit_pair_id="2"), "必须各有一个"),
    (lambda rows: rows.append({"source_dir": "image_dataset/normal"}), "不能混用"),
])
def test_edit_roles_reject_invalid_groups(tmp_path: Path, monkeypatch, mutate, match: str) -> None:
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    rows = _paired_rows(1)
    mutate(rows)
    with pytest.raises(ValueError, match=match):
        config_service.save_dataset_preset("configs/datasets/pairs.toml", rows, {"batch_size": 1})


@pytest.mark.parametrize("family", ["qwen_image_2_1", "qwen21", "qwen_image_21"])
def test_apply_multiple_edit_pairs_uses_target_rows_only(tmp_path: Path, monkeypatch, family) -> None:
    configs, _ = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    for number in (1, 2):
        _write_pair(tmp_path / f"post_image_dataset/target_{number}",
                    tmp_path / f"image_dataset/reference_{number}")
    config_service.save_dataset_preset(
        "configs/datasets/pairs.toml", _paired_rows(), {"batch_size": 1},
    )
    train_path = configs / "imported/qwen.toml"
    train_path.write_text(
        f'model_family = "{family}"\n'
        'network_module = "networks.lora_anima"\n'
        'cache_latents = true\ncache_text_encoder_outputs = true\n',
        encoding="utf-8",
    )

    applied = config_service.apply_dataset_preset_to_training_config(
        "configs/datasets/pairs.toml", "configs/imported/qwen.toml",
    )

    assert applied["values"]["qwen_image_2_1_task"] == "edit"
    assert applied["values"]["source_image_dir"] == "image_dataset/target_1"
    assert applied["summary"]["train_dataset_count"] == 2


def test_qwen_edit_dataset_preset_roundtrips_ab_fields(tmp_path: Path, monkeypatch) -> None:
    _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)

    saved = config_service.save_dataset_preset(
        "configs/datasets/qwen_edit.toml",
        _edit_rows(),
        {"qwen_edit_enabled": True, "batch_size": 1},
    )
    loaded = config_service.load_dataset_preset("configs/datasets/qwen_edit.toml")
    data = toml.loads((tmp_path / "configs/datasets/qwen_edit.toml").read_text(encoding="utf-8"))

    assert saved["ok"]
    assert loaded["defaults"]["qwen_edit_enabled"] is True
    assert loaded["datasets"][0]["reference_image_dir"] == "image_dataset/reference"
    assert data["general"]["custom_attributes"]["webui_dataset_defaults"]["qwen_edit_enabled"] is True
    assert data["datasets"][0]["subsets"][0]["reference_image_dir"] == "image_dataset/reference"


def test_apply_qwen_edit_sets_task_only_after_plain_lora_and_pair_validation(
    tmp_path: Path, monkeypatch
) -> None:
    configs, _ = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    target_dir = tmp_path / "post_image_dataset" / "targets"
    reference_dir = tmp_path / "image_dataset" / "reference"
    _write_pair(target_dir, reference_dir)
    config_service.save_dataset_preset(
        "configs/datasets/qwen_edit.toml",
        _edit_rows(),
        {"qwen_edit_enabled": True, "batch_size": 1},
    )
    train_path = configs / "imported" / "qwen.toml"
    train_path.write_text(
        '\n'.join([
            'model_family = "qwen_image_2_1"',
            'network_module = "networks.lora_anima"',
            "cache_latents = true",
            "cache_text_encoder_outputs = true",
            "use_moe_style = false",
            "route_per_layer = false",
            'router_source = "none"',
        ]),
        encoding="utf-8",
    )

    applied = config_service.apply_dataset_preset_to_training_config(
        "configs/datasets/qwen_edit.toml",
        "configs/imported/qwen.toml",
    )

    assert applied["values"]["qwen_image_2_1_task"] == "edit"
    assert toml.loads(train_path.read_text(encoding="utf-8"))["qwen_image_2_1_task"] == "edit"

    before = train_path.read_text(encoding="utf-8")
    train_path.write_text(before.replace("cache_text_encoder_outputs = true\n", ""), encoding="utf-8")
    missing_cache = train_path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="Qwen3-VL 条件缓存"):
        config_service.apply_dataset_preset_to_training_config(
            "configs/datasets/qwen_edit.toml",
            "configs/imported/qwen.toml",
        )
    assert train_path.read_text(encoding="utf-8") == missing_cache

    train_path.write_text(before.replace("networks.lora_anima", "networks.methods.easycontrol"), encoding="utf-8")
    incompatible = train_path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="仅支持 plain LoRA"):
        config_service.apply_dataset_preset_to_training_config(
            "configs/datasets/qwen_edit.toml",
            "configs/imported/qwen.toml",
        )
    assert train_path.read_text(encoding="utf-8") == incompatible


def test_apply_qwen_edit_rejects_incomplete_pair_before_writing_task(
    tmp_path: Path, monkeypatch
) -> None:
    configs, _ = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    reference_dir = tmp_path / "image_dataset/reference"
    reference_dir.mkdir(parents=True)
    _write_pair(tmp_path / "post_image_dataset/targets", reference_dir, reference=False)
    config_service.save_dataset_preset(
        "configs/datasets/qwen_edit.toml",
        _edit_rows(),
        {"qwen_edit_enabled": True, "batch_size": 1},
    )
    train_path = configs / "imported" / "qwen.toml"
    original = (
        'model_family = "qwen_image_2_1"\n'
        'network_module = "networks.lora_anima"\n'
        'cache_latents = true\n'
        'cache_text_encoder_outputs = true\n'
    )
    train_path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="no paired reference"):
        config_service.apply_dataset_preset_to_training_config(
            "configs/datasets/qwen_edit.toml",
            "configs/imported/qwen.toml",
        )
    assert train_path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("invalid_config,match", [
    ('mixed_precision = "fp16"\n', "mixed_precision=bf16"),
    ('base_compute = "nf4"\n', "base_compute=bf16"),
])
def test_apply_qwen_edit_rejects_other_compat_errors(
    tmp_path: Path, monkeypatch, invalid_config: str, match: str,
) -> None:
    configs, _ = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    _write_pair(tmp_path / "post_image_dataset/targets", tmp_path / "image_dataset/reference")
    config_service.save_dataset_preset(
        "configs/datasets/qwen_edit.toml", _edit_rows(),
        {"qwen_edit_enabled": True, "batch_size": 1},
    )
    train_path = configs / "imported/qwen.toml"
    original = (
        'model_family = "qwen_image_2_1"\n'
        'cache_latents = true\n'
        'cache_text_encoder_outputs = true\n'
        + invalid_config
    )
    train_path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        config_service.apply_dataset_preset_to_training_config(
            "configs/datasets/qwen_edit.toml", "configs/imported/qwen.toml",
        )
    assert train_path.read_text(encoding="utf-8") == original


def test_apply_qwen_edit_rejects_augmented_dataset(tmp_path: Path, monkeypatch) -> None:
    configs, _ = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    _write_pair(tmp_path / "post_image_dataset/targets", tmp_path / "image_dataset/reference")
    rows = _edit_rows()
    rows[0]["flip_aug"] = True
    config_service.save_dataset_preset(
        "configs/datasets/qwen_edit.toml", rows,
        {"qwen_edit_enabled": True, "batch_size": 1},
    )
    train_path = configs / "imported/qwen.toml"
    original = (
        'model_family = "qwen_image_2_1"\n'
        'cache_latents = true\n'
        'cache_text_encoder_outputs = true\n'
    )
    train_path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="deterministic reference/target"):
        config_service.apply_dataset_preset_to_training_config(
            "configs/datasets/qwen_edit.toml", "configs/imported/qwen.toml",
        )
    assert train_path.read_text(encoding="utf-8") == original


def test_apply_qwen_t2i_rejects_leftover_reference_dir(tmp_path: Path, monkeypatch) -> None:
    configs, _ = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    config_service.save_dataset_preset(
        "configs/datasets/qwen_t2i.toml", _edit_rows(),
        {"qwen_edit_enabled": False, "batch_size": 1},
    )
    train_path = configs / "imported/qwen.toml"
    original = 'model_family = "qwen_image_2_1"\n'
    train_path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="仍包含参考图目录"):
        config_service.apply_dataset_preset_to_training_config(
            "configs/datasets/qwen_t2i.toml", "configs/imported/qwen.toml",
        )
    assert train_path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize("family", ["anima", "krea2", "z_image"])
def test_apply_edit_rejects_unsupported_capability_without_writing(tmp_path, monkeypatch, family):
    configs, _ = _write_minimal_config_tree(tmp_path)
    _patch_config_service_paths(monkeypatch, tmp_path)
    config_service.save_dataset_preset(
        "configs/datasets/pairs.toml", _paired_rows(1), {"batch_size": 1},
    )
    train_path = configs / "imported/other.toml"
    original = f'model_family = "{family}"\n'
    train_path.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="不支持编辑数据集训练") as error:
        config_service.apply_dataset_preset_to_training_config(
            "configs/datasets/pairs.toml", "configs/imported/other.toml",
        )
    assert "Qwen" not in str(error.value)
    assert train_path.read_text(encoding="utf-8") == original
