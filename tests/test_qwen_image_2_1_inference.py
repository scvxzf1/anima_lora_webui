from types import SimpleNamespace

import pytest
import torch

from library.inference.family_runtime import prepare_inference_family
from library.models.family_registry import get_model_family_spec
from library.models.qwen_image_2_1 import inference_runner


def _args(**overrides):
    values = dict(
        qwen_image_2_1_task="t2i",
        sampler="euler",
        image_size=[512, 768],
        infer_steps=28,
        lora_weight=None,
        pgraft=False,
        spectrum=False,
        spd=False,
        tiled_diffusion=False,
        dcw=False,
        pooled_text_proj=None,
        ip_adapter_weight=None,
        easycontrol_weight=None,
        soft_tokens_weight=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_qwen_registry_exposes_initial_standalone_t2i_contract():
    spec = get_model_family_spec("qwen_image_2_1")
    assert spec.supported_inference_modes == frozenset({"single"})
    assert spec.supported_inference_samplers == frozenset({"euler"})


def test_qwen_runner_validation_is_single_euler_only():
    inference_runner.validate_qwen_image_2_1_inference_args(_args())
    with pytest.raises(SystemExit, match="edit requires"):
        inference_runner.validate_qwen_image_2_1_inference_args(_args(qwen_image_2_1_task="edit"))
    with pytest.raises(SystemExit, match="Euler"):
        inference_runner.validate_qwen_image_2_1_inference_args(_args(sampler="lcm"))
    with pytest.raises(SystemExit, match="automatic token-count"):
        inference_runner.validate_qwen_image_2_1_inference_args(_args(flow_shift=2.0))
    with pytest.raises(ValueError, match="divisible by 32"):
        inference_runner.validate_qwen_image_2_1_inference_args(_args(image_size=[510, 512]))
    with pytest.raises(ValueError, match="divisible by 32"):
        inference_runner.validate_qwen_image_2_1_inference_args(_args(image_size=[528, 512]))
    with pytest.raises(ValueError, match="infer_steps must be positive"):
        inference_runner.validate_qwen_image_2_1_inference_args(_args(infer_steps=0))
    with pytest.raises(SystemExit, match="--compile"):
        inference_runner.validate_qwen_image_2_1_inference_args(_args(compile=True))


def test_prepare_inference_family_calls_qwen_validator(monkeypatch):
    calls = []
    monkeypatch.setattr(
        inference_runner,
        "validate_qwen_image_2_1_inference_args",
        lambda args, mode: calls.append(mode),
    )
    prepare_inference_family(_args(), "qwen_image_2_1", "single")
    assert calls == ["single"]


def test_generate_qwen_image_2_1_uses_t2i_euler_preview_sampler(monkeypatch):
    observed = {}

    def fake_generate(dit, prompt, positive, negative, reference, **kwargs):
        observed.update(prompt)
        assert positive[1].dtype is torch.bool
        assert negative[0].dtype == torch.float32
        assert reference is None
        assert kwargs["seed"] == 17
        return torch.zeros(1, 64, 32, 48)

    monkeypatch.setattr(inference_runner, "generate_preview", fake_generate)
    args = _args(prompt="a cat", negative_prompt="blurry", infer_steps=28, guidance_scale=3.5)
    result = inference_runner.generate_qwen_image_2_1(
        args,
        object(),
        None,
        (torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.bool)),
        (torch.zeros(1, 2, 4), torch.ones(1, 2, dtype=torch.bool)),
        torch.device("cpu"),
        17,
        torch.float32,
    )
    assert result.shape == (1, 64, 32, 48)
    assert observed["width"] == 768
    assert observed["height"] == 512
    assert observed["sample_steps"] == 28


@pytest.mark.parametrize("device_name", ["cpu", "cuda:2"])
def test_qwen_output_decodes_on_selected_device(monkeypatch, tmp_path, device_name):
    from PIL import Image, PngImagePlugin

    device = torch.device(device_name)
    latent = torch.zeros(1, 64, 32, 48)
    cleanup = []

    def fake_load(path, *, dtype, device):
        assert path == "qwen-vae.safetensors"
        assert dtype == torch.bfloat16
        assert device == torch.device(device_name)
        return SimpleNamespace(device=device)

    def fake_decode(vae, value):
        assert vae.device == device
        assert value is latent
        assert torch.is_inference_mode_enabled()
        return Image.new("RGB", (768, 512))

    monkeypatch.setattr(inference_runner, "load_qwen_image_2_1_vae", fake_load)
    monkeypatch.setattr(inference_runner, "decode_latent", fake_decode)
    monkeypatch.setattr(inference_runner, "get_time_flag", lambda: "smoke")
    monkeypatch.setattr(inference_runner, "_build_png_info", lambda *args: PngImagePlugin.PngInfo())
    monkeypatch.setattr(inference_runner, "clean_memory_on_device", cleanup.append)
    args = _args(
        vae="qwen-vae.safetensors", output_type="images", save_path=str(tmp_path), seed=17,
    )

    inference_runner.save_qwen_image_2_1_output(args, latent, device)

    with Image.open(tmp_path / "smoke_17.png") as result:
        assert result.size == (768, 512)
    assert cleanup == [device]


@pytest.mark.parametrize(
    "damage", [None, "alpha", "missing_up", "missing_down", "extra", "non_dit", "shape",
               "wrong_family", "unstamped"]
)
def test_qwen_plain_lora_strict_checkpoint_loading(tmp_path, damage):
    from copy import deepcopy

    from diffusers import QwenImage21Transformer2DModel
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    from library.models.qwen_image_2_1.lora_targets import qwen_image_2_1_target_kwargs
    from networks.lora_anima import create_network

    model = QwenImage21Transformer2DModel(
        num_layers=1, num_attention_heads=1, attention_head_dim=128,
        context_in_dim=4096, mlp_ratio=1,
    )
    restored_model = deepcopy(model)
    original_forward = restored_model.transformer_blocks[0].attn.to_q.forward
    network = create_network(
        multiplier=1.0, network_dim=4, network_alpha=4,
        vae=None, text_encoders=[], unet=model,
        model_family="qwen_image_2_1", **qwen_image_2_1_target_kwargs(),
    )
    network.apply_to([], model, False, True)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.fill_(0.125)
    checkpoint = tmp_path / "plain.safetensors"
    network.save_weights(str(checkpoint), torch.float32, metadata={})
    with safe_open(str(checkpoint), framework="pt") as saved:
        metadata = dict(saved.metadata())
    weights = load_file(str(checkpoint))
    up_key = next(key for key in weights if key.endswith(".lora_up.weight"))
    if damage == "alpha":
        weights = {key: value for key, value in weights.items() if not key.endswith(".alpha")}
    elif damage == "missing_up":
        del weights[up_key]
    elif damage == "missing_down":
        del weights[up_key.replace("lora_up", "lora_down")]
    elif damage == "extra":
        weights["lora_unet_unmatched.lora_up.weight"] = torch.zeros(4, 4)
    elif damage == "non_dit":
        weights["lora_te_unmatched.lora_up.weight"] = torch.zeros(4, 4)
    elif damage == "shape":
        weights[up_key] = torch.zeros(1, 1)
    elif damage == "wrong_family":
        metadata["ss_model_family"] = "anima"
    elif damage == "unstamped":
        del metadata["ss_model_family"]
    save_file(weights, str(checkpoint), metadata=metadata)

    if damage is None:
        loaded = inference_runner._load_plain_lora(
            _args(lora_weight=str(checkpoint)), restored_model, torch.device("cpu"), torch.float32
        )
        assert loaded is not None
        assert not loaded.training
        for key, value in weights.items():
            torch.testing.assert_close(loaded.state_dict()[key], value)
        assert restored_model.transformer_blocks[0].attn.to_q.forward != original_forward
    else:
        with pytest.raises(ValueError) as error:
            inference_runner._load_plain_lora(
                _args(lora_weight=str(checkpoint)), restored_model, torch.device("cpu"), torch.float32
            )
        assert str(checkpoint) in str(error.value)
        assert any(text in str(error.value) for text in ("keys", "shape mismatches", "stamped"))
        assert restored_model.transformer_blocks[0].attn.to_q.forward == original_forward


@pytest.mark.parametrize("spec", ["lokr", "loha", "hydra"])
def test_qwen_plain_lora_rejects_other_stamped_specs(monkeypatch, spec):
    from library.inference import models

    monkeypatch.setattr(models, "_read_lora_metadata", lambda path: {"ss_network_spec": spec})
    with pytest.raises(ValueError, match="plain LoRA only"):
        inference_runner._load_plain_lora(
            _args(lora_weight="unsupported.safetensors"), torch.nn.Linear(1, 1),
            torch.device("cpu"), torch.float32,
        )
