"""Local tiny-model FP16 loading and opt-in CUDA AMP training contracts."""
from copy import deepcopy
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from library.models.moss_image.dit import MOSSImageTransformer
from library.models.moss_image.family import compute_noise_pred_and_target
from library.models.moss_image.lora_targets import moss_image_target_kwargs
from library.models.moss_image.math import _RMSNorm
from library.models.moss_image.weights import load_moss_transformer


def _tiny_model():
    torch.manual_seed(43)
    return MOSSImageTransformer(dim=32, ffn_dim=48, num_heads=1,
                                num_layers=3, text_dim=8, freq_dim=8)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_strict_local_component_loading(tmp_path, dtype, monkeypatch):
    model = _tiny_model()
    component = tmp_path / "transformer"
    model.save_pretrained(component)
    original = MOSSImageTransformer.from_pretrained
    calls = []

    def local_only(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["local_files_only"] is True
        assert kwargs["output_loading_info"] is True
        return original(*args, **kwargs)

    monkeypatch.setattr(MOSSImageTransformer, "from_pretrained", local_only)
    loaded = load_moss_transformer(tmp_path, dtype=dtype)
    assert calls and not loaded.training
    assert all(p.dtype == dtype for p in loaded.parameters())
    assert loaded.rope.bands.dtype == torch.float32
    for name, parameter in loaded.named_parameters():
        torch.testing.assert_close(parameter, model.state_dict()[name].to(dtype), rtol=0, atol=0)

    # A real local checkpoint with a missing tensor must still fail strictly.
    weights_path = component / "diffusion_pytorch_model.safetensors"
    weights = load_file(weights_path)
    weights.pop("final_linear_video.weight")
    save_file(weights, weights_path, metadata={"format": "pt"})
    with pytest.raises(ValueError, match="strict load failed.*missing_keys"):
        load_moss_transformer(component, dtype=dtype)


def test_fp16_rmsnorm_large_activations_and_backward():
    norm = _RMSNorm(8).half().requires_grad_(False)
    x = torch.linspace(-60000, 60000, 8).half().requires_grad_()
    output = norm(x)
    expected = x.float() * torch.rsqrt(x.float().square().mean() + 1e-6)
    torch.testing.assert_close(output.float(), expected, rtol=1e-3, atol=1e-3)
    output.float().square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.hardware
@pytest.mark.parametrize("checkpoint,swap", [(False, False), (True, False), (True, True)])
def test_fp16_accelerate_plain_lora_optimizer_step(tmp_path, checkpoint, swap):
    if os.environ.get("MOSS_FP16_AMP_CUDA") != "1":
        pytest.skip("Set MOSS_FP16_AMP_CUDA=1 on an explicitly selected idle CUDA GPU")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from accelerate import Accelerator
    from networks.lora_anima import create_network, create_network_from_weights

    accelerator = Accelerator(mixed_precision="fp16")
    assert accelerator.scaler is not None and accelerator.scaler.is_enabled()
    device = accelerator.device
    model = _tiny_model().half().requires_grad_(False)
    base = deepcopy(model)
    network = create_network(1.0, 2, 2, None, [], model,
                             model_family="moss_image", **moss_image_target_kwargs())
    network.apply_to([], model, False, True)
    network.to(device=device, dtype=torch.float32)
    network.prepare_grad_etc([], model)
    if checkpoint:
        model.enable_gradient_checkpointing()
    if swap:
        model.enable_block_swap(1, device)
        model.move_to_device_except_swap_blocks(device)
        model.prepare_block_swap_before_forward()
    else:
        model.to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=1e-3)
    network, optimizer = accelerator.prepare(network, optimizer)
    raw_network = accelerator.unwrap_model(network)
    before = [p.detach().clone() for p in raw_network.parameters()]
    projection_dtypes = []
    handle = model.video_embedder.register_forward_hook(
        lambda module, inputs, output: projection_dtypes.append(output.dtype))
    ctx = SimpleNamespace(network=network, unet=model, weight_dtype=torch.float16,
                          accelerator=accelerator,
                          args=SimpleNamespace(gradient_checkpointing=checkpoint))
    latents = torch.randn(1, 128, 2, 3, device=device, dtype=torch.float16)
    text = torch.randn(1, 4, 8, device=device, dtype=torch.bfloat16)
    mask = torch.tensor([[1, 1, 0, 0]], device=device, dtype=torch.bool)
    text[:, 2:] = float("nan")  # Padding must remain excluded under AMP.
    pred, target, times, weights = compute_noise_pred_and_target(
        SimpleNamespace(), ctx, latents, {}, [text, mask])
    handle.remove()
    assert projection_dtypes == [torch.float16]
    assert pred.shape == target.shape == latents.shape and torch.isfinite(pred).all()
    assert times.dtype == weights.dtype == torch.float32
    loss = (pred.float() - target.float()).square().mean()
    accelerator.backward(loss)
    accelerator.unscale_gradients(optimizer)
    gradients = [p.grad for p in raw_network.parameters()]
    assert gradients and all(g is not None and g.dtype == torch.float32 for g in gradients)
    assert all(torch.isfinite(g).all() for g in gradients)
    assert any(g.count_nonzero() for g in gradients)
    assert all(p.grad is None for p in model.parameters())
    optimizer.step()
    assert not accelerator.optimizer_step_was_skipped
    assert any(not torch.equal(old, p) for old, p in zip(before, raw_network.parameters()))
    assert all(p.dtype == torch.float32 and torch.isfinite(p).all() for p in raw_network.parameters())

    if swap:
        model.switch_block_swap_for_inference()
    model.disable_gradient_checkpointing()
    path = str(tmp_path / "fp16-amp-lora.safetensors")
    raw_network.save_weights(path, torch.float32, metadata={})
    restored, _ = create_network_from_weights(1.0, path, None, [], base)
    restored.apply_to([], base, False, True)
    restored.load_weights(path)
    base.to(device)
    restored.to(device=device, dtype=torch.float32)
    with torch.no_grad(), accelerator.autocast():
        expected = model(latents, 500., text, mask).sample
        actual = base(latents, 500., text, mask).sample
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert model.rope.bands.dtype == torch.float32
    if swap:
        model.switch_block_swap_for_training()
        assert model.offloader.forward_only is False
    accelerator.free_memory()
