from __future__ import annotations

import types

import torch

from library.training.precision_policy import resolve_vae_dtype


def _fake_args(mp="fp16", no_half_vae=False, half_vae=False):
    return types.SimpleNamespace(
        mixed_precision=mp,
        no_half_vae=no_half_vae,
        half_vae=half_vae,
    )


def test_no_half_vae_forces_fp32(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 0))
    args = _fake_args(mp="bf16", no_half_vae=True)
    assert resolve_vae_dtype(args, torch.bfloat16) == torch.float32


def test_no_half_vae_beats_half_vae(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 0))
    args = _fake_args(mp="fp16", no_half_vae=True, half_vae=True)
    assert resolve_vae_dtype(args, torch.float16) == torch.float32


def test_half_vae_overrides_auto_fp32_on_v100(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 0))
    args = _fake_args(mp="fp16", half_vae=True)
    assert resolve_vae_dtype(args, torch.float16) == torch.float16


def test_auto_fp32_on_v100_fp16(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 0))
    args = _fake_args(mp="fp16")
    assert resolve_vae_dtype(args, torch.float16) == torch.float32


def test_no_force_on_ampere_fp16(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 0))
    args = _fake_args(mp="fp16")
    assert resolve_vae_dtype(args, torch.float16) == torch.float16


def test_no_force_on_bf16(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 0))
    args = _fake_args(mp="bf16")
    assert resolve_vae_dtype(args, torch.bfloat16) == torch.bfloat16


def test_capability_probe_failure_keeps_weight_dtype(monkeypatch, caplog):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    def _raise(*a, **k):
        raise RuntimeError("cuda init failed")

    monkeypatch.setattr("torch.cuda.get_device_capability", _raise)
    args = _fake_args(mp="fp16")
    with caplog.at_level("WARNING"):
        dtype = resolve_vae_dtype(args, torch.float16)
    assert dtype == torch.float16
