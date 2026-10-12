"""All-family preview policy, legacy configuration merging, and persistence."""

import argparse
from types import SimpleNamespace

import pytest
import torch

from library.config import io, schema
from library.training import preview_decode as shared
from library.training.preview_decode_config import add_arguments, decode_policy, configuration_errors
from library.training.preview_decode_policy import PreviewDecodeRestoreError, decode_with_policy


@pytest.fixture
def decode_schema(monkeypatch):
    monkeypatch.setattr(schema, "CONFIG_SCHEMA", {})
    monkeypatch.setattr(schema, "ALIAS_MAP", {})
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    schema.populate_schema(parser)
    return parser


def test_legacy_alias_survives_schema_cli_and_config_layers(decode_schema):
    assert decode_schema.parse_args([]).sample_vae_decode == "tiled"
    assert decode_schema.parse_args(["--qwen_sample_vae_decode", "cpu"]).sample_vae_decode == "cpu"
    assert schema.validate_entry("qwen_sample_vae_decode", "auto", strict=True) == ("sample_vae_decode", "auto")
    merged = io._flatten_toml({"sample_vae_decode": "tiled"})
    merged.update(io._flatten_toml({"qwen_sample_vae_decode": "auto"}))
    assert merged["sample_vae_decode"] == "auto"


@pytest.mark.parametrize("items", [
    [("sample_vae_decode", "cpu"), ("qwen_sample_vae_decode", "auto")],
    [("qwen_sample_vae_decode", "auto"), ("sample_vae_decode", "cpu")],
])
def test_canonical_key_wins_with_or_without_schema(decode_schema, monkeypatch, items):
    assert io._flatten_toml(dict(items))["sample_vae_decode"] == "cpu"
    monkeypatch.setattr(schema, "CONFIG_SCHEMA", {})
    assert io._flatten_toml(dict(items))["sample_vae_decode"] == "cpu"
    assert decode_policy(dict(items)) == "cpu"
    assert decode_policy(SimpleNamespace(**dict(items))) == "cpu"


@pytest.mark.parametrize("family", ["anima", "krea2_raw", "z_image", "qwen_image_2_1"])
@pytest.mark.parametrize("policy", ["tiled", "auto", "cpu"])
def test_policy_preflight_for_all_families(family, policy):
    from library.training.compat_matrix import check_training_compat

    config = dict(model_family=family, sample_vae_decode=policy)
    assert not configuration_errors(config)
    result = check_training_compat(config)
    assert not any(error.code.endswith("decode_contract") for error in result.errors)
    invalid = check_training_compat(dict(config, sample_vae_decode="invalid"))
    assert any(error.code == "sample_vae_decode_contract" for error in invalid.errors)


class VAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.use_tiling = False
        self.config = SimpleNamespace(scaling_factor=2.0, shift_factor=0.5)
        self.seen = None
        self.states = []

    @property
    def dtype(self):
        return self.anchor.dtype

    def enable_tiling(self):
        self.use_tiling = True

    def disable_tiling(self):
        self.use_tiling = False

    def decode_to_pixels(self, latents):
        self.seen = latents.clone()
        self.states.append(self.use_tiling)
        return torch.zeros(1, 3, 1, 8, 8)

    def decode(self, latents, return_dict=False):
        self.seen = latents.clone()
        self.states.append(self.use_tiling)
        return (torch.zeros(1, 3, 8, 8),)


@pytest.mark.parametrize("family", ["anima", "krea2_raw", "z_image"])
@pytest.mark.parametrize("policy", ["tiled", "auto", "cpu"])
def test_pending_decode_routes_policy_and_restores_device_dtype_and_tiling(tmp_path, family, policy):
    from library.anima.training import decode_pending_samples

    staged = tmp_path / "sample" / "latents"
    staged.mkdir(parents=True)
    latent = torch.full((1, 16, 1, 1), 2.0)
    torch.save({"latents": latent}, staged / "one.pt")
    args = SimpleNamespace(output_dir=str(tmp_path), model_family=family, sample_vae_decode=policy)
    accelerator = SimpleNamespace(device=torch.device("cpu"), trackers=[])
    vae = VAE()
    decode_pending_samples(accelerator, args, vae)
    assert (tmp_path / "sample" / "one.png").is_file() and not (staged / "one.pt").exists()
    assert vae.anchor.device.type == "cpu" and vae.dtype == torch.float32 and not vae.use_tiling
    assert vae.states == [policy != "auto"]
    assert torch.equal(vae.seen, torch.full_like(latent, 1.5) if family == "z_image" else latent)


@pytest.mark.parametrize("fatal", [False, True])
def test_pending_decode_retains_failure_and_propagates_restore_error(tmp_path, monkeypatch, fatal):
    staged = tmp_path / "sample" / "latents"
    staged.mkdir(parents=True)
    path = staged / "one.pt"
    torch.save({"latents": torch.zeros(1, 16, 1, 1)}, path)
    error = PreviewDecodeRestoreError("restore failed") if fatal else ValueError("invalid latent")
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(shared, "decode_with_policy", fail)
    accelerator = SimpleNamespace(device=torch.device("cpu"))
    args = SimpleNamespace(output_dir=str(tmp_path))
    if fatal:
        with pytest.raises(PreviewDecodeRestoreError):
            shared.decode_pending_samples(accelerator, args, VAE(), decode_latent=shared.decode_anima_latent)
    else:
        shared.decode_pending_samples(accelerator, args, VAE(), decode_latent=shared.decode_anima_latent)
    assert path.exists() and not (tmp_path / "sample" / "one.png").exists()


def test_remote_decode_failure_prevents_tail_barrier(monkeypatch):
    from library.training import sample_preview_common
    monkeypatch.setattr(sample_preview_common, "failed_on_any_process", lambda *args: True)
    with pytest.raises(RuntimeError, match="another process"):
        shared.raise_decode_failure(SimpleNamespace(), None, 2)


def test_flux_1024_threshold_is_reduced_and_restored():
    vae = VAE()
    vae.tile_sample_min_size, vae.tile_latent_min_size = 1024, 128
    def decode(model, latent):
        assert model.use_tiling
        assert (model.tile_sample_min_size, model.tile_latent_min_size) == (512, 64)
        return "image"
    assert decode_with_policy(vae, object(), torch.device("cpu"), "tiled", decode) == "image"
    assert not vae.use_tiling
    assert (vae.tile_sample_min_size, vae.tile_latent_min_size) == (1024, 128)
