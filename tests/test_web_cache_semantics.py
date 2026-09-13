from __future__ import annotations

from pathlib import Path
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
