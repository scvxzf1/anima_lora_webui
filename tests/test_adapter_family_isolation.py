"""Fail-closed cross-family isolation contracts (CPU only).

Covers the three artefacts that must never cross MOSS-Image / Anima (or any
other registered family): LoRA adapters, text-encoder caches and latent
caches. No checkpoints, downloads, CUDA or preprocessing are required.
"""
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from library.io.cache_names import classify_cache_file, family_cache_suffixes
from library.models.family_registry import validate_text_cache_metadata
from library.models.moss_image.cache_policy import text_cache_metadata
from networks.lora_anima.factory import (
    create_network_from_weights,
    require_adapter_family,
)
from networks.lora_anima.persistence import load_lora_network_weights


def _write_adapter(path, metadata, key="lora_unet_x.lora_down.weight"):
    save_file({key: torch.zeros(4, 4)}, str(path), metadata=metadata)
    return str(path)


class _StubNetwork:
    """Minimal LoRANetwork surface used by the resume/warm-start loader."""

    def __init__(self, family):
        self.cfg = SimpleNamespace(model_family=family)
        self.unet_loras = []
        self.text_encoder_loras = []
        self.loaded = None

    def load_state_dict(self, weights_sd, strict):
        self.loaded = weights_sd
        return SimpleNamespace(unexpected_keys=[], missing_keys=[])


def test_unstamped_checkpoint_reads_as_anima():
    require_adapter_family({}, "anima")
    require_adapter_family(None, "anima")
    with pytest.raises(ValueError, match="cross-family adapter loading is refused"):
        require_adapter_family({}, "moss_image")


def test_stamped_family_must_match_the_target():
    require_adapter_family({"ss_model_family": "moss_image"}, "moss_image")
    require_adapter_family({"ss_model_family": "moss"}, "moss_image")
    with pytest.raises(ValueError, match="moss_image"):
        require_adapter_family({"ss_model_family": "moss_image"}, "anima")
    with pytest.raises(ValueError, match="krea2_raw"):
        require_adapter_family({"ss_model_family": "krea2_raw"}, "qwen_image_2_1")


def test_unknown_target_family_keeps_stamp_driven_loading():
    """Callers that pass no runtime family keep the previous behaviour."""
    require_adapter_family({"ss_model_family": "moss_image"}, None)
    require_adapter_family({"ss_model_family": "anima"}, None)


def test_create_network_from_weights_refuses_foreign_family(tmp_path):
    path = _write_adapter(tmp_path / "moss.safetensors", {"ss_model_family": "moss_image"})
    with pytest.raises(ValueError, match="cross-family adapter loading is refused"):
        create_network_from_weights(1.0, path, None, [], None, model_family="anima")
    with pytest.raises(ValueError, match="cross-family adapter loading is refused"):
        create_network_from_weights(
            1.0,
            None,
            None,
            [],
            None,
            weights_sd={"lora_unet_x.lora_down.weight": torch.zeros(4, 4)},
            metadata={"ss_model_family": "krea2_raw"},
            model_family="moss_image",
        )


def test_cross_family_resume_is_refused(tmp_path):
    unstamped = _write_adapter(tmp_path / "anima.safetensors", {})
    with pytest.raises(ValueError, match="cross-family adapter loading is refused"):
        load_lora_network_weights(_StubNetwork("moss_image"), unstamped)

    matching = _write_adapter(tmp_path / "moss.safetensors", {"ss_model_family": "moss_image"})
    network = _StubNetwork("moss_image")
    load_lora_network_weights(network, matching)
    assert network.loaded is not None


def test_text_cache_identity_carries_the_token_cap():
    """MOSS stores real-length states, so the cap must be part of the identity."""
    from library.models.moss_image.strategy import MossImageTextCache

    capped = MossImageTextCache(True, 1, False, quantization="nf4", max_text_tokens=1024)
    full = MossImageTextCache(True, 1, False, quantization="nf4", max_text_tokens=4096)
    assert capped.cache_suffix == full.cache_suffix, "suffix unchanged; identity lives in metadata"
    assert text_cache_metadata("nf4", 1024)["max_text_tokens"] == "1024"
    assert text_cache_metadata("nf4")["max_text_tokens"] == "4096"
    import torch
    from safetensors.torch import save_file

    tensors = {"hiddens": torch.ones(2, 4096, dtype=torch.bfloat16),
               "mask": torch.ones(2, dtype=torch.bool),
               "caption_dropout_rate": torch.tensor(0.0)}
    path = "text.safetensors"
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "text.safetensors")
        save_file(tensors, path, metadata=text_cache_metadata("nf4", 1024))
        assert capped.is_disk_cached_outputs_expected(path)
        assert not full.is_disk_cached_outputs_expected(path)
        with pytest.raises(ValueError, match="max_text_tokens"):
            full.load_outputs_npz(path)
        save_file(tensors, path, metadata=text_cache_metadata("nf4", 4096))
        with pytest.raises(ValueError, match="max_text_tokens"):
            capped.load_outputs_npz(path)


def test_legacy_v1_text_cache_stays_readable():
    """v1 caches carry no cap; they stay readable rather than forced-rebuilt."""
    from library.models.moss_image.cache_policy import validate_cache_metadata

    legacy = {"model_family": "moss_image", "cache_schema": "moss_image_te_v1",
              "quantization": "nf4", "compute_dtype": "bfloat16",
              "tokenizer_version": "qwen3_5_qwen2_v1",
              "template_version": "user_generation_thinking_v1"}
    validate_cache_metadata(legacy, "nf4", "c", 1024)
    with pytest.raises(ValueError, match="quantization"):
        validate_cache_metadata(dict(legacy, quantization="bf16"), "nf4", "c", 1024)


def test_text_cache_metadata_is_family_scoped():
    validate_text_cache_metadata(text_cache_metadata("nf4"), family="moss_image", path="c")
    with pytest.raises(ValueError, match="family mismatch"):
        validate_text_cache_metadata(
            {"model_family": "anima", "cache_schema": "anima_te_v1"},
            family="moss_image",
            path="c",
        )
    with pytest.raises(ValueError, match="family mismatch"):
        validate_text_cache_metadata(
            {"model_family": "moss_image", "cache_schema": "moss_image_te_v2"},
            family="anima",
            path="c",
        )


def test_registry_schema_tracks_the_moss_cache_policy():
    """Tripwire: the registry literal and the MOSS cache policy must agree."""
    from library.models.family_registry import get_model_family_spec
    from library.models.moss_image.cache_policy import MOSS_TEXT_CACHE_SCHEMA

    spec = get_model_family_spec("moss_image")
    assert spec.text_cache.schema == MOSS_TEXT_CACHE_SCHEMA


def test_moss_latent_cache_does_not_accept_an_anima_stride(tmp_path):
    """A stride-8 (Anima) latent npz must not satisfy the MOSS stride-16 read."""
    import numpy as np

    from library.models.moss_image.strategy import MossImageLatentCache

    cache = MossImageLatentCache(True, 1, False)
    npz_path = str(tmp_path / "img_64x64_moss_image.npz")
    # Anima writes stride-8 keys; MOSS reads stride-16 keys for the same bucket.
    np.savez(
        npz_path,
        latents_128x128=np.zeros((16, 128, 128), dtype=np.float32),
        original_size_128x128=np.array([1024, 1024]),
        crop_ltrb_128x128=np.array([0, 0, 1024, 1024]),
    )
    assert not cache.is_disk_cached_latents_expected((1024, 1024), npz_path, False, False)
    np.savez(
        npz_path,
        latents_64x64=np.zeros((128, 64, 64), dtype=np.float32),
        original_size_64x64=np.array([1024, 1024]),
        crop_ltrb_64x64=np.array([0, 0, 1024, 1024]),
    )
    assert cache.is_disk_cached_latents_expected((1024, 1024), npz_path, False, False)


def test_cache_suffixes_are_disjoint_and_classified_per_family():
    other_families = ("anima", "krea2_raw", "z_image", "qwen_image_2_1")
    moss_latent, moss_te = family_cache_suffixes("moss_image")
    assert moss_latent == "_moss_image.npz"
    assert moss_te == "_moss_image_te_nf4.safetensors"
    for family in other_families:
        latent, text = family_cache_suffixes(family)
        assert latent != moss_latent, family
        assert not text.startswith("_moss_image_te_"), family
    name = "a_64x96_moss_image.npz"
    assert classify_cache_file(name, model_family="moss_image") == "latents"
    assert classify_cache_file(name, model_family="anima") is None
    te_name = "a_moss_image_te_nf4.safetensors"
    assert classify_cache_file(te_name, model_family="moss_image") == "te"
    assert classify_cache_file(te_name, model_family="anima") is None
