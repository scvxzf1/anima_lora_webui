import pytest
import torch
from bench.convrot_90hx.inputs import validate_tensors
from bench.convrot_90hx.manifest import validate_device


def test_1024_native_latent_and_maxpadding_contract():
    validate_tensors(torch.zeros(16, 128, 128), torch.zeros(512, 4))
    with pytest.raises(ValueError, match="max-padded"):
        validate_tensors(torch.zeros(16, 128, 128), torch.zeros(12, 4))
    with pytest.raises(ValueError, match="1024"):
        validate_tensors(torch.zeros(16, 64, 64), torch.zeros(512, 4))


def test_nonfinite_cache_rejected():
    context = torch.zeros(512, 4)
    context[0, 0] = float("inf")
    with pytest.raises(ValueError, match="NaN/Inf"):
        validate_tensors(torch.zeros(16, 128, 128), context)


def test_wrong_device_visibility_rejected_before_cuda(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(RuntimeError, match="UUID"):
        validate_device()
