from __future__ import annotations

from pathlib import Path

import pytest

from web.services import image_test_service


ROOT = Path(__file__).resolve().parents[1]


def _text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


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


def test_training_form_filters_and_locks_krea2_attention_options() -> None:
    form = _text("web/static/js/features/config-form/form-fields-ui.js")
    live = _text("web/static/js/features/config-form/live-compat.js")

    assert "attn_mode: new Set(['torch', 'flash'])" in form
    assert "selective_checkpoint: new Set(['off', 'every_other'])" in form
    assert "key === 'compile_dynamic_seq'" in form
    assert "key === 'compile_seq_bands'" in form
    assert "key === 'compile_inductor_mode'" in form
    assert "key === 'v100_flash_stability'" in form
    assert "option.disabled = true" in form
    assert "input.disabled = true" in form
    assert "krea2_invalid_attn_mode" in live
    assert "krea2_compile_dynamic_seq" in live
    assert "krea2_compile_seq_bands" in live


def test_image_test_options_are_family_aware() -> None:
    state = _text("web/static/js/features/image-test/state.js")
    feature = _text("web/static/js/features/image-test/index.js")
    bridge = _text(
        "web/static/js/features/anima-app/chunks/01a-image-test-feature.js"
    )

    assert "imageTestAttnModeOptionsForFamily" in state
    assert "['torch', 'flash'].includes(item.value)" in state
    assert "imageTestSamplerOptionsForFamily" in state
    assert "item.value === 'euler'" in state
    assert "renderer.setAttentionModeOptions(modelFamily" in feature
    assert "renderer.setSamplerOptions(modelFamily" in feature
    assert "flowShiftInput.disabled = isKrea2" in feature
    assert "getModelFamily" in bridge
    assert "configState.currentConfig?.model_family" in bridge


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
