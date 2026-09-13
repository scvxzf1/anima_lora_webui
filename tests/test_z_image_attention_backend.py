from __future__ import annotations

from contextlib import contextmanager
from enum import Enum

import pytest
import torch

from library.models.z_image import attention_backend


class _BackendName(str, Enum):
    FLASH_VARLEN = "flash_varlen"


class _Processor:
    def __init__(self) -> None:
        self._attention_backend = None


class _AttentionModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.processor = _Processor()
        self.set_calls: list[str] = []

    def set_attention_backend(self, backend: str) -> None:
        self.set_calls.append(backend)
        self.processor._attention_backend = _BackendName(backend)


def _install_fake_api(monkeypatch, *, context_error: Exception | None = None):
    context_calls = []

    @contextmanager
    def backend_context(backend):
        context_calls.append(backend)
        if context_error is not None:
            raise context_error
        yield

    monkeypatch.setattr(
        attention_backend,
        "_load_diffusers_attention_api",
        lambda: (_BackendName, backend_context, _Processor),
    )
    return context_calls


@pytest.mark.parametrize("value", [None, "", "flash", "FLASH"])
def test_z_image_attention_defaults_to_flash(value) -> None:
    assert attention_backend.normalize_z_image_attention_mode(value) == "flash"


@pytest.mark.parametrize("value", ["torch", "sdpa", "native"])
def test_z_image_attention_normalizes_native_aliases(value) -> None:
    assert attention_backend.normalize_z_image_attention_mode(value) == "torch"


def test_z_image_flash_maps_to_diffusers_flash_varlen(monkeypatch) -> None:
    context_calls = _install_fake_api(monkeypatch)
    model = _AttentionModule()

    selected = attention_backend.prepare_z_image_attention(
        model,
        "flash",
        dtype=torch.bfloat16,
    )

    assert selected == "flash"
    assert context_calls == [_BackendName.FLASH_VARLEN]
    assert model.set_calls == ["flash_varlen"]
    assert model.processor._attention_backend is _BackendName.FLASH_VARLEN


def test_z_image_torch_restores_native_processor_backend(monkeypatch) -> None:
    context_calls = _install_fake_api(monkeypatch)
    model = _AttentionModule()
    model.processor._attention_backend = _BackendName.FLASH_VARLEN

    selected = attention_backend.prepare_z_image_attention(
        model,
        "sdpa",
        dtype=torch.bfloat16,
    )

    assert selected == "torch"
    assert context_calls == []
    assert model.set_calls == []
    assert model.processor._attention_backend is None


def test_z_image_flash_rejects_non_bf16_before_provider_check(monkeypatch) -> None:
    monkeypatch.setattr(
        attention_backend,
        "_load_diffusers_attention_api",
        lambda: pytest.fail("provider should not be loaded"),
    )
    with pytest.raises(RuntimeError, match="only for bf16"):
        attention_backend.prepare_z_image_attention(
            _AttentionModule(),
            "flash",
            dtype=torch.float16,
        )


def test_z_image_flash_rejects_unavailable_provider(monkeypatch) -> None:
    _install_fake_api(monkeypatch, context_error=RuntimeError("missing provider"))
    with pytest.raises(RuntimeError, match="compatible FlashAttention 2 provider"):
        attention_backend.prepare_z_image_attention(
            _AttentionModule(),
            "flash",
            dtype=torch.bfloat16,
        )


def test_z_image_attention_rejects_model_without_expected_processor(
    monkeypatch,
) -> None:
    _install_fake_api(monkeypatch)
    with pytest.raises(RuntimeError, match="no ZSingleStreamAttnProcessor"):
        attention_backend.prepare_z_image_attention(
            torch.nn.Linear(2, 2),
            "flash",
            dtype=torch.bfloat16,
        )
