from __future__ import annotations

from pathlib import Path
import types

import pytest

from library.training.precision_policy import resolve_mixed_precision

ROOT = Path(__file__).resolve().parents[1]


def _fake_args(mp="bf16"):
    return types.SimpleNamespace(mixed_precision=mp)


def test_no_switch_on_ampere(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (8, 0))
    args = _fake_args()
    resolve_mixed_precision(args)
    assert args.mixed_precision == "bf16"


def test_switch_on_v100_back_writes_args(monkeypatch, caplog):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 0))
    args = _fake_args("bf16")
    with caplog.at_level("WARNING"):
        resolve_mixed_precision(args)
    assert args.mixed_precision == "fp16"
    assert any("fp16" in r.getMessage() for r in caplog.records)


def test_switch_on_t4(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 5))
    args = _fake_args("bf16")
    resolve_mixed_precision(args)
    assert args.mixed_precision == "fp16"


def test_explicit_fp16_left_alone(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 0))
    args = _fake_args("fp16")
    resolve_mixed_precision(args)
    assert args.mixed_precision == "fp16"


def test_capability_probe_failure_is_safe(monkeypatch, caplog):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    def _raise(*a, **k):
        raise RuntimeError("cuda init failed")

    monkeypatch.setattr("torch.cuda.get_device_capability", _raise)
    args = _fake_args("bf16")
    with caplog.at_level("WARNING"):
        resolve_mixed_precision(args)
    assert args.mixed_precision == "bf16"


def test_no_cuda_is_noop(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    args = _fake_args("bf16")
    resolve_mixed_precision(args)
    assert args.mixed_precision == "bf16"


def test_missing_mixed_precision_attr_is_safe(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda *a, **k: (7, 0))
    args = types.SimpleNamespace()
    resolve_mixed_precision(args)
    assert not hasattr(args, "mixed_precision")


def test_train_session_resolves_mixed_precision_before_first_prepare_dtype():
    source = (ROOT / "library" / "training" / "train_session.py").read_text(
        encoding="utf-8"
    )
    resolve_idx = source.find("resolve_mixed_precision(args)")
    first_prepare_idx = source.find("prepare_dtype(args)")
    accelerator_idx = source.find("prepare_accelerator(args)")
    assert resolve_idx != -1
    assert first_prepare_idx != -1
    assert accelerator_idx != -1
    assert resolve_idx < first_prepare_idx
    assert resolve_idx < accelerator_idx

