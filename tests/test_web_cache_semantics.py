from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from library.datasets.mask_mode import normalize_mask_mode
from library.config.loader import DreamBoothSubsetParams
from library.datasets.subsets import DreamBoothSubset
from library.io.cache_names import classify_cache_file, count_preprocess_caches
from web.services.config.cache_audit import audit_dataset_row_caches
from library.models.qwen_image_2_1.strategy import (
    QwenImage21EditTextCache, QwenImage21LatentCache,
)
from library.datasets.qwen_image_edit import edit_condition_fingerprint


def _image(root: Path, name: str = "sample.png") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    Image.new("RGB", (8, 8), color=(20, 40, 60)).save(path)
    return path


def _latent(path: Path, *, suffix: str, alpha: bool = False) -> Path:
    target = path.parent / f"{path.stem}_0008x0008{suffix}"
    payload = {"latents_1x1": np.zeros((16, 1, 1), dtype=np.float32)}
    if alpha:
        payload["alpha_mask_1x1"] = np.ones((1, 1), dtype=np.float32)
    np.savez(target, **payload)
    return target


def _audit(tmp_path: Path, family: str, *, config=None, row=None):
    image_dir = tmp_path / "images"
    image = _image(image_dir)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    cfg = {
        "model_family": family,
        "use_vae_cache": True,
        "use_text_cache": True,
        "skip_cache_check": True,
        **(config or {}),
    }
    dataset_row = {
        "image_dir": str(image_dir),
        "cache_dir": str(cache_dir),
        "mask_mode": "none",
        **(row or {}),
    }
    results = audit_dataset_row_caches(
        cfg,
        dataset_row,
        [image],
        resolve_path=lambda value: Path(value),
    )
    return image, cache_dir, {result["kind"]: result for result in results}


@pytest.mark.parametrize(
    ("family", "latent_suffix", "text_suffix"),
    [
        ("anima", "_anima.npz", "_anima_te.safetensors"),
        ("krea2_raw", "_anima.npz", "_krea2_te.safetensors"),
        ("z_image", "_z_image.npz", "_z_image_te.safetensors"),
    ],
)
def test_cache_audit_uses_family_registry_suffixes(
    tmp_path: Path,
    family: str,
    latent_suffix: str,
    text_suffix: str,
):
    image, cache_dir, _ = _audit(tmp_path, family)
    _latent(cache_dir / image.name, suffix=latent_suffix)
    (cache_dir / f"{image.stem}{text_suffix}").write_bytes(b"exists")

    _, _, results = _audit(tmp_path, family)

    assert results["latent_cache"]["valid"] == 1
    assert results["text_cache"]["valid"] == 1


def test_cache_audit_rejects_partial_runtime_cache(tmp_path: Path):
    image_dir = tmp_path / "images"
    first = _image(image_dir, "first.png")
    second = _image(image_dir, "second.png")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _latent(cache_dir / first.name, suffix="_anima.npz")

    results = audit_dataset_row_caches(
        {
            "model_family": "anima",
            "use_vae_cache": True,
            "use_text_cache": False,
        },
        {"image_dir": str(image_dir), "cache_dir": str(cache_dir), "mask_mode": "none"},
        [first, second],
        resolve_path=lambda value: Path(value),
    )

    latent = results[0]
    assert latent["total"] == 2
    assert latent["valid"] == 1
    assert [Path(path).name for path in latent["missing"]] == [
        "second_0008x0008_anima.npz"
    ]


def _edit_audit_fixture(tmp_path: Path):
    image_dir = tmp_path / "images"
    image = _image(image_dir / "nested", "sample.png")
    reference_dir = tmp_path / "references"
    reference = _image(reference_dir / "nested", "sample.png")
    caption = image.with_suffix(".txt")
    caption.write_text("make it blue", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    row = {
        "image_dir": str(image_dir),
        "source_dir": str(image_dir),
        "reference_image_dir": str(reference_dir),
        "cache_dir": str(cache_dir),
        "mask_mode": "none",
        "settings": {"enable_bucket": False, "resolution": 64},
    }
    info = SimpleNamespace(absolute_path=str(image), reference_image_path=str(reference),
                           caption="make it blue", bucket_reso=(64, 64))
    subset = SimpleNamespace(cache_dir=str(cache_dir), image_dir=str(image_dir))
    strategy = QwenImage21EditTextCache(True, 1, False)
    path = Path(strategy.get_outputs_npz_path_for_info(info, subset))
    path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = edit_condition_fingerprint(str(reference), info.caption, info.bucket_reso)
    slots = torch.zeros(8, dtype=torch.bool)
    slots[:4] = True  # 64x64 reference -> 4x4 latent -> four packed image slots.
    tensors = {
        "hiddens": torch.zeros(8, 4096, dtype=torch.bfloat16),
        "mask": torch.ones(8, dtype=torch.bool),
        "image_slots": slots,
        "caption_dropout_rate": torch.tensor(0.0),
    }
    metadata = {"edit_cache_schema": strategy.EDIT_CACHE_SCHEMA,
                "edit_condition_fingerprint": fingerprint}
    save_file(tensors, str(path), metadata=metadata)
    latent_strategy = QwenImage21LatentCache(True, 1, False)
    reference_path = Path(latent_strategy.get_edit_reference_latent_path(info, subset))
    save_file(
        {"latent": torch.zeros(64, 4, 4)}, str(reference_path),
        metadata={"edit_cache_schema": latent_strategy.EDIT_LATENT_CACHE_SCHEMA,
                  "edit_condition_fingerprint": fingerprint},
    )
    cfg = {"model_family": "qwen_image_2_1", "qwen_image_2_1_task": "edit",
           "use_vae_cache": False, "use_text_cache": True}
    return cfg, row, image, reference, path, reference_path, tensors, metadata


def _edit_results(cfg, row, image):
    return {result["kind"]: result for result in audit_dataset_row_caches(
        cfg, row, [image], resolve_path=Path,
    )}


def test_qwen_edit_cache_audit_accepts_valid_nested_runtime_sidecar(tmp_path: Path):
    cfg, row, image, *_ = _edit_audit_fixture(tmp_path)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "sample.txt").write_text("wrong source caption")
    row["source_dir"] = str(source_dir)
    results = _edit_results(cfg, row, image)
    for result in results.values():
        assert result["valid"] == result["total"] == 1
        assert result["missing"] == result["invalid"] == []


def test_qwen_edit_cache_audit_missing_directory_is_read_only(tmp_path: Path):
    cfg, row, image, *_ = _edit_audit_fixture(tmp_path)
    missing_cache_dir = tmp_path / "absent-cache"
    row["cache_dir"] = str(missing_cache_dir)
    for result in _edit_results(cfg, row, image).values():
        assert result["valid"] == 0
        assert len(result["missing"]) == 1
    assert not missing_cache_dir.exists()


@pytest.mark.parametrize("changed", ["caption", "reference", "bucket"])
def test_qwen_edit_cache_audit_rejects_changed_condition(tmp_path: Path, changed: str):
    cfg, row, image, reference, *_ = _edit_audit_fixture(tmp_path)
    if changed == "caption":
        image.with_suffix(".txt").write_text("make it red")
    elif changed == "reference":
        Image.new("RGB", (8, 8), color=(200, 40, 60)).save(reference)
    else:
        row["settings"]["resolution"] = 128
    for result in _edit_results(cfg, row, image).values():
        assert result["valid"] == 0
        assert len(result["missing"]) == 1


@pytest.mark.parametrize("changed", ["stale", "slots", "slot_shape", "missing"])
def test_qwen_edit_cache_audit_rejects_invalid_text_cache(tmp_path: Path, changed: str):
    cfg, row, image, _, path, _, tensors, metadata = _edit_audit_fixture(tmp_path)
    if changed == "stale":
        metadata["edit_condition_fingerprint"] = "stale"
    elif changed == "slots":
        tensors["image_slots"] = torch.ones(8, dtype=torch.bool)
    elif changed == "slot_shape":
        tensors["image_slots"] = torch.ones(7, dtype=torch.bool)
    if changed == "missing":
        path.unlink()
    else:
        save_file(tensors, str(path), metadata=metadata)
    results = _edit_results(cfg, row, image)
    text = results["text_cache"]
    assert text["valid"] == 0
    assert len(text["missing"] if changed == "missing" else text["invalid"]) == 1
    assert results["edit_reference_latent_cache"]["valid"] == 1


@pytest.mark.parametrize("changed", ["stale", "shape", "missing"])
def test_qwen_edit_cache_audit_rejects_invalid_reference_cache(tmp_path: Path, changed: str):
    cfg, row, image, _, _, path, _, metadata = _edit_audit_fixture(tmp_path)
    metadata["edit_cache_schema"] = QwenImage21LatentCache.EDIT_LATENT_CACHE_SCHEMA
    if changed == "stale":
        metadata["edit_condition_fingerprint"] = "stale"
    if changed == "missing":
        path.unlink()
    else:
        save_file(
            {"latent": torch.zeros(64, 8 if changed == "shape" else 4, 4)},
            str(path), metadata=metadata,
        )
    results = _edit_results(cfg, row, image)
    reference_result = results["edit_reference_latent_cache"]
    assert reference_result["valid"] == 0
    assert len(reference_result["missing"] if changed == "missing" else reference_result["invalid"]) == 1
    assert results["text_cache"]["valid"] == 1


def test_qwen_t2i_audit_keeps_plain_text_suffix(tmp_path: Path):
    image_dir = tmp_path / "images"
    image = _image(image_dir)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    plain = cache_dir / "sample_qwen_image_2_1_te.safetensors"
    plain.write_bytes(b"not a structured cache")
    result = audit_dataset_row_caches(
        {"model_family": "qwen_image_2_1", "use_vae_cache": False, "use_text_cache": True},
        {"image_dir": str(image_dir), "cache_dir": str(cache_dir), "mask_mode": "none"},
        [image], resolve_path=lambda value: Path(value),
    )
    assert result[0]["kind"] == "text_cache"
    assert result[0]["valid"] == 0


def test_ip_cache_is_required_only_for_disk_mode_and_uses_encoder(tmp_path: Path):
    image, cache_dir, results = _audit(
        tmp_path,
        "anima",
        config={
            "use_vae_cache": False,
            "use_text_cache": False,
            "use_ip_adapter": True,
            "ip_features_cache_to_disk": False,
        },
    )
    assert "pe_cache" not in results

    save_file(
        {"image_features": torch.zeros((1, 4))},
        str(cache_dir / f"{image.stem}_anima_siglip.safetensors"),
    )
    _, _, results = _audit(
        tmp_path,
        "anima",
        config={
            "use_vae_cache": False,
            "use_text_cache": False,
            "use_ip_adapter": True,
            "ip_features_cache_to_disk": True,
            "ip_encoder": "siglip",
        },
    )
    assert results["pe_cache"]["valid"] == 1


def test_embedded_mask_requires_alpha_but_external_mask_is_optional(tmp_path: Path):
    image, cache_dir, _ = _audit(
        tmp_path,
        "anima",
        config={"use_text_cache": False},
        row={"mask_mode": "embedded"},
    )
    _latent(cache_dir / image.name, suffix="_anima.npz", alpha=False)
    _, _, embedded = _audit(
        tmp_path,
        "anima",
        config={"use_text_cache": False},
        row={"mask_mode": "embedded"},
    )
    assert embedded["latent_cache"]["invalid"]

    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    _, _, external = _audit(
        tmp_path,
        "anima",
        config={"use_text_cache": False},
        row={"mask_mode": "external", "mask_dir": str(mask_dir)},
    )
    assert external["latent_cache"]["valid"] == 1
    assert external["external_masks"]["missing_level"] == "warning"
    assert external["external_masks"]["missing"]


def test_mask_modes_resolve_legacy_fields_without_overloading_none():
    assert normalize_mask_mode(None).mode == "auto"
    assert normalize_mask_mode(None, alpha_mask=True).mode == "embedded"
    assert normalize_mask_mode(None, mask_dir="masks").mode == "external"
    disabled = normalize_mask_mode("none", alpha_mask=True, mask_dir="masks")
    assert disabled.alpha_mask is False
    assert disabled.mask_dir is None


def test_cache_classifier_is_family_aware(tmp_path: Path):
    assert classify_cache_file("a_krea2_te.safetensors", model_family="krea2") == "te"
    assert classify_cache_file("a_0008x0008_z_image.npz", model_family="zimage") == "latents"
    (tmp_path / "a_krea2_te.safetensors").write_bytes(b"cache")
    counts = count_preprocess_caches(tmp_path, model_family="krea2")
    assert counts == {"latents": 0, "te": 1, "pe": 0}


@pytest.mark.parametrize(
    ("family", "suffix"),
    [
        ("anima", "_anima_te.safetensors"),
        ("krea2", "_krea2_te.safetensors"),
        ("zimage", "_z_image_te.safetensors"),
    ],
)
def test_training_subset_resolves_text_cache_suffix_from_family_registry(
    tmp_path: Path,
    family: str,
    suffix: str,
):
    subset = DreamBoothSubset(
        **asdict(
            DreamBoothSubsetParams(
                image_dir=str(tmp_path),
                model_family=family,
                mask_mode="none",
            )
        )
    )

    assert subset.text_cache_suffix == suffix
