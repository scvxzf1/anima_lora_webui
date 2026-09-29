from __future__ import annotations

import pytest

from web.services import image_test_service


def _image_payload(*, attn_mode: str, runtime_dtype: str = "bf16") -> dict:
    return {
        "prompt": "test",
        "attn_mode": attn_mode,
        "runtime_dtype": runtime_dtype,
        "config": {
            "pretrained_model_name_or_path": "dit.safetensors",
            "qwen3": "qwen3.safetensors",
            "vae": "vae.safetensors",
        },
    }


def _stub_image_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        image_test_service,
        "_apply_global_model_path_defaults",
        lambda cfg: cfg,
    )
    monkeypatch.setattr(
        image_test_service,
        "_resolve_image_test_model_paths",
        lambda cfg: cfg,
    )
    monkeypatch.setattr(
        image_test_service.settings_service,
        "get_global_settings",
        lambda: {"model_family": "krea2_raw"},
    )


def test_image_service_canonicalizes_krea2_sdpa_alias(monkeypatch) -> None:
    _stub_image_paths(monkeypatch)

    normalized = image_test_service._normalize_image_test_request(
        _image_payload(attn_mode="sdpa")
    )

    assert normalized["attn_mode"] == "torch"
    assert normalized["model_family"] == "krea2_raw"


def test_image_service_rejects_krea2_anima_only_attention(monkeypatch) -> None:
    _stub_image_paths(monkeypatch)

    with pytest.raises(ValueError, match="Krea-2"):
        image_test_service._normalize_image_test_request(
            _image_payload(attn_mode="xformers")
        )


def test_image_service_rejects_krea2_flash_fp32(monkeypatch) -> None:
    _stub_image_paths(monkeypatch)

    with pytest.raises(ValueError, match="fp16.*bf16"):
        image_test_service._normalize_image_test_request(
            _image_payload(attn_mode="flash", runtime_dtype="fp32")
        )
