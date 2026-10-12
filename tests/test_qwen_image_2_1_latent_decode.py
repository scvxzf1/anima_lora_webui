"""CLI latent decoding uses Qwen's VAE contract without generation loaders."""

import sys
from types import SimpleNamespace
import weakref

from PIL import Image
import pytest
from safetensors.torch import save_file
import torch

import inference
from library.models.qwen_image_2_1 import inference_runner as runner


def _cli(monkeypatch, tmp_path, paths, *flags):
    monkeypatch.setattr(sys, "argv", [
        "inference.py", "--model_family", "qwen_image_2_1",
        "--text_encoder", "unused", "--vae", "qwen-vae",
        "--save_path", str(tmp_path / "out"), "--device", "cpu",
        "--latent_path", *(str(path) for path in paths), *flags,
    ])
    def forbidden(*args, **kwargs):
        pytest.fail("latent decode must not load shared VAE, TE or DiT")
    monkeypatch.setattr(inference.qwen_image_autoencoder_kl, "load_vae", forbidden)
    monkeypatch.setattr(runner, "load_qwen_image_2_1_text_encoder", forbidden)
    monkeypatch.setattr(runner, "load_qwen_image_2_1_transformer", forbidden)


def _save(path, latent=None, metadata=None):
    save_file({"latent": torch.zeros(1, 64, 2, 3) if latent is None else latent},
              str(path), metadata=metadata)
    return path


class _VAE:
    device = torch.device("cpu")
    dtype = torch.bfloat16
    config = SimpleNamespace(latents_mean=[2.] * 64, latents_std=[3.] * 64)

    def decode(self, latent, *, return_dict):
        assert latent.shape == (1, 64, 1, 2, 3)
        assert latent.dtype == torch.bfloat16
        assert torch.all(latent == 5)
        assert not return_dict and torch.is_inference_mode_enabled()
        return (torch.zeros(1, 4, 1, 32, 48),)


def test_cli_decodes_with_qwen_normalization_and_releases_vae(monkeypatch, tmp_path):
    path = _save(tmp_path / "source.safetensors", torch.ones(1, 64, 2, 3),
                 {"height": "32", "width": "48", "seeds": "17", "prompt": "saved"})
    _cli(monkeypatch, tmp_path, [path])
    residents, cleanup = [], []
    def load(path, *, dtype, device):
        assert path == "qwen-vae" and dtype == torch.bfloat16 and device == torch.device("cpu")
        vae = _VAE()
        residents.append(weakref.ref(vae))
        return vae
    monkeypatch.setattr(runner, "load_qwen_image_2_1_vae", load)
    monkeypatch.setattr(runner, "clean_memory_on_device", cleanup.append)
    monkeypatch.setattr(runner, "get_time_flag", lambda: "decode")
    inference.main()
    with Image.open(tmp_path / "out/decode_17_source.png") as image:
        assert image.size == (48, 32)
        assert image.info["prompt"] == "saved"
    assert residents[0]() is None
    assert cleanup == [torch.device("cpu")]


@pytest.mark.parametrize("latent,metadata", [
    (torch.zeros(1, 64, 1, 2, 3), None),
    (torch.zeros(2, 64, 2, 3), None),
    (torch.zeros(1, 16, 2, 3), None),
    (torch.zeros(1, 64, 0, 3), None),
    (torch.zeros(1, 64, 2, 3, dtype=torch.int32), None),
    (torch.full((1, 64, 2, 3), float("nan")), None),
    (torch.zeros(1, 64, 2, 3), {"height": "16"}),
    (torch.zeros(1, 64, 2, 3), {"width": "24"}),
])
def test_cli_invalid_latents_fail_before_any_vae(monkeypatch, tmp_path, latent, metadata):
    valid = _save(tmp_path / "first.safetensors")
    invalid = _save(tmp_path / "bad.safetensors", latent, metadata)
    _cli(monkeypatch, tmp_path, [valid, invalid])
    monkeypatch.setattr(runner, "save_qwen_image_2_1_output", lambda *a, **k: pytest.fail("preflight first"))
    with pytest.raises(ValueError, match="Qwen Image 2.1 latent"):
        inference.main()


def test_cli_missing_key_fails_before_vae(monkeypatch, tmp_path):
    path = tmp_path / "bad.safetensors"
    save_file({"wrong": torch.zeros(1)}, str(path))
    _cli(monkeypatch, tmp_path, [path])
    monkeypatch.setattr(runner, "save_qwen_image_2_1_output", lambda *a, **k: pytest.fail("bad key"))
    with pytest.raises(ValueError, match="lacks 'latent'"):
        inference.main()


@pytest.mark.parametrize("flags,message", [
    (["--qwen_image_2_1_task", "edit", "--reference_images", "a", "b", "c", "d", "e"], "latent decode"),
    (["--reference_image", "a"], "latent decode"),
    (["--lora_weight", "adapter"], "latent decode"),
    (["--from_file", "prompts"], "latent decode"),
    (["--interactive"], "latent decode"),
    (["--runtime_dtype", "fp16"], "BF16"),
    (["--sampler", "lcm"], "Euler"),
    (["--compile"], "--compile"),
])
def test_cli_generation_flags_fail_before_vae(monkeypatch, tmp_path, flags, message):
    path = _save(tmp_path / "source.safetensors")
    _cli(monkeypatch, tmp_path, [path], *flags)
    monkeypatch.setattr(runner, "save_qwen_image_2_1_output", lambda *a, **k: pytest.fail("flags preflight"))
    with pytest.raises(SystemExit, match=message):
        inference.main()


def test_multiple_files_restore_metadata_without_mutating_cli_args(monkeypatch, tmp_path):
    first = _save(tmp_path / "first.name.safetensors", metadata={
        "seeds": "9", "prompt": "first", "negative_prompt": "bad", "height": "32", "width": "48",
    })
    second = tmp_path / "second.pt"
    torch.save(torch.zeros(1, 64, 3, 2), second)
    _cli(monkeypatch, tmp_path, [first, second], "--seed", "7", "--prompt", "cli",
         "--negative_prompt", "cli negative", "--qwen_image_2_1_task", "t2i")
    original_parse = inference.parse_args
    parsed, saved, loads = [], [], []
    def parse():
        args = original_parse()
        parsed.append(args)
        return args
    def save(args, latent, device, *, original_base_name):
        saved.append((args.seed, args.prompt, args.negative_prompt, args.image_size, original_base_name))
        args.prompt = "mutation"
    original_load = torch.load
    def load(*args, **kwargs):
        loads.append(kwargs)
        return original_load(*args, **kwargs)
    monkeypatch.setattr(inference, "parse_args", parse)
    monkeypatch.setattr(runner, "save_qwen_image_2_1_output", save)
    monkeypatch.setattr(torch, "load", load)
    inference.main()
    assert saved == [(9, "first", "bad", [32, 48], "first.name"),
                     (7, "cli", "cli negative", [48, 32], "second")]
    assert parsed[0].prompt == "cli" and parsed[0].seed == 7
    assert parsed[0].image_size == [1024, 1024]
    assert loads == [{"map_location": "cpu", "weights_only": True}]


@pytest.mark.parametrize("output_type", ["latent", "latent_images"])
def test_cli_preserves_latent_output_types(monkeypatch, tmp_path, output_type):
    path = _save(tmp_path / "source.safetensors", metadata={"seeds": "3"})
    _cli(monkeypatch, tmp_path, [path], "--output_type", output_type)
    monkeypatch.setattr(runner, "get_time_flag", lambda: "decode")
    loaded = []
    monkeypatch.setattr(runner, "load_qwen_image_2_1_vae", lambda *a, **k: loaded.append(True) or object())
    monkeypatch.setattr(runner, "decode_latent", lambda *a: Image.new("RGB", (48, 32)))
    inference.main()
    from safetensors import safe_open
    with safe_open(str(next((tmp_path / "out").glob("*.safetensors"))), framework="pt") as handle:
        assert handle.get_tensor("latent").shape == (1, 64, 2, 3)
        assert handle.metadata()["height"] == "32" and handle.metadata()["width"] == "48"
    assert bool(loaded) == (output_type == "latent_images")
    assert (tmp_path / "out/decode_3_source.png").exists() == bool(loaded)
