"""Transactional decode fallback without allocating CUDA tensors."""
from types import SimpleNamespace

import pytest
import torch

from library.models.qwen_image_2_1 import preview_decode
from library.models.qwen_image_2_1.preview_decode_policy import (
    PreviewDecodeRestoreError, decode_with_policy,
)


class VAE:
    def __init__(self, *, tiling=False, move_ooms=0, restore_failure=False):
        self.device, self.dtype, self.use_tiling = torch.device("cpu"), torch.bfloat16, tiling
        self.moves, self.clears = [], 0
        self.move_ooms, self.restore_failure = move_ooms, restore_failure

    def parameters(self):
        return iter(())

    def buffers(self):
        return iter(())

    def to(self, device=None, *, dtype=None):
        if dtype is not None:
            self.dtype = dtype
            return self
        self.moves.append(str(device))
        if torch.device(device).type == "cuda" and self.move_ooms:
            self.move_ooms -= 1
            raise torch.cuda.OutOfMemoryError("move CUDA out of memory")
        if torch.device(device).type == "cpu" and self.restore_failure:
            raise RuntimeError("restore broken")
        self.device = torch.device(device)
        return self

    def enable_tiling(self):
        self.use_tiling = True

    def disable_tiling(self):
        self.use_tiling = False

    def clear_cache(self):
        self.clears += 1


@pytest.fixture(autouse=True)
def no_cuda_allocator(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


def test_auto_normal_preserves_original_device_dtype_and_tiling():
    vae, latent = VAE(tiling=True), torch.ones(1)
    observed = []
    def decode(current, value):
        observed.append((current.device.type, current.dtype, current.use_tiling, value is latent))
        current.dtype = torch.float32
        return "image"
    assert decode_with_policy(vae, latent, torch.device("cuda"), "auto", decode) == "image"
    assert observed == [("cuda", torch.bfloat16, True, True)]
    assert (vae.device.type, vae.dtype, vae.use_tiling) == ("cpu", torch.bfloat16, True)
    assert vae.clears == 0


def test_auto_retries_cuda_oom_as_tiled_then_cpu():
    vae, stages = VAE(), []
    def decode(current, latent):
        stages.append((current.device.type, current.use_tiling))
        if current.device.type == "cuda":
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return "image"
    assert decode_with_policy(vae, torch.zeros(1), torch.device("cuda"), "auto", decode) == "image"
    assert stages == [("cuda", False), ("cuda", True), ("cpu", True)]
    assert vae.clears == 2 and vae.moves == ["cuda", "cuda", "cpu", "cpu"]
    assert not vae.use_tiling and vae.device.type == "cpu"


@pytest.mark.parametrize("error", [ValueError("bad latent"), RuntimeError("CPU out of memory")])
def test_non_cuda_oom_is_not_retried(error):
    vae, calls = VAE(), []
    def decode(current, latent):
        calls.append(current.use_tiling)
        raise error
    with pytest.raises(type(error), match=str(error)):
        decode_with_policy(vae, torch.zeros(1), torch.device("cuda"), "auto", decode)
    assert calls == [False] and vae.clears == 1
    assert vae.device.type == "cpu" and not vae.use_tiling


def test_gpu_movement_oom_also_falls_back_without_decoding_on_failed_device():
    vae, calls = VAE(move_ooms=2), []
    def decode(current, latent):
        calls.append((current.device.type, current.use_tiling))
        return "image"
    assert decode_with_policy(vae, torch.zeros(1), torch.device("cuda"), "auto", decode) == "image"
    assert calls == [("cpu", True)] and vae.clears == 2
    assert vae.moves == ["cuda", "cuda", "cpu", "cpu"]


@pytest.mark.parametrize("policy,device", [("tiled", "cuda"), ("cpu", "cpu")])
def test_explicit_policy_has_no_retry_and_restores_state(policy, device):
    vae, calls = VAE(), []
    def decode(current, latent):
        calls.append((current.device.type, current.use_tiling))
        current.dtype = torch.float32
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")
    with pytest.raises(torch.cuda.OutOfMemoryError):
        decode_with_policy(vae, torch.zeros(1), torch.device("cuda"), policy, decode)
    assert calls == [(device, True)] and vae.clears == 1
    assert (vae.device.type, vae.dtype, vae.use_tiling) == ("cpu", torch.bfloat16, False)


@pytest.mark.parametrize("body_error", [False, True])
def test_restore_failure_is_explicit_and_never_retried(body_error):
    vae, calls = VAE(restore_failure=True), []
    def decode(current, latent):
        calls.append(1)
        if body_error:
            raise ValueError("decode broken")
        return "image"
    with pytest.raises(PreviewDecodeRestoreError, match="restore VAE") as caught:
        decode_with_policy(vae, torch.zeros(1), torch.device("cuda"), "auto", decode)
    assert calls == [1] and str(caught.value.__cause__) == "restore broken"
    assert not vae.use_tiling
    if body_error:
        assert any("decode broken" in note for note in caught.value.__notes__)


@pytest.mark.parametrize("fatal", [False, True])
def test_failed_pending_decode_retains_latent_and_propagates_fatal(tmp_path, monkeypatch, fatal):
    root = tmp_path / "sample" / "latents"
    root.mkdir(parents=True)
    path = root / "pending.pt"
    torch.save({"latents": torch.zeros(1, 64, 2, 2), "metadata": {}}, path)
    error = PreviewDecodeRestoreError("restore failed") if fatal else ValueError("invalid decode")
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(preview_decode, "decode_with_policy", fail)
    monkeypatch.setattr(preview_decode, "clean_memory_on_device", lambda device: None)
    args, accelerator = SimpleNamespace(output_dir=str(tmp_path)), SimpleNamespace(device=torch.device("cpu"))
    if fatal:
        with pytest.raises(PreviewDecodeRestoreError):
            preview_decode.decode_pending_samples(accelerator, args, VAE())
    else:
        preview_decode.decode_pending_samples(accelerator, args, VAE())
    assert path.exists() and not (tmp_path / "sample" / "pending.png").exists()


def test_tiling_restore_failure_is_fatal_even_when_device_restores():
    vae = VAE()
    def fail_restore():
        raise RuntimeError("tiling restore broken")
    vae.disable_tiling = fail_restore
    with pytest.raises(PreviewDecodeRestoreError) as caught:
        decode_with_policy(vae, torch.zeros(1), torch.device("cuda"), "tiled", lambda *args: "image")
    assert vae.device.type == "cpu"
    assert str(caught.value.__cause__) == "tiling restore broken"
