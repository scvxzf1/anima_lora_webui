"""S-R6: image_test_save_root global setting with output/tests fallback."""

from __future__ import annotations

from pathlib import Path

import pytest
import toml

from web.services import image_test_service, settings_service


def _patch_settings(tmp_path: Path, monkeypatch):
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    settings_file.parent.mkdir(parents=True)
    settings_file.write_text('[global]\noutput_root = "output/runs"\n', encoding="utf-8")
    monkeypatch.setattr(settings_service, "ROOT", tmp_path)
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", settings_file)
    monkeypatch.setattr(image_test_service, "ROOT", tmp_path)
    monkeypatch.setattr(image_test_service, "DEFAULT_INFERENCE_DIR", "output/tests")
    return settings_file


def test_empty_save_root_falls_back_to_output_tests(tmp_path, monkeypatch):
    _patch_settings(tmp_path, monkeypatch)
    got = settings_service.get_global_settings()
    assert got.get("image_test_save_root", "") in {"", None} or got["image_test_save_root"] == ""
    assert settings_service.resolve_image_test_save_root() == "output/tests"


def test_custom_relative_save_root_persists(tmp_path, monkeypatch):
    settings_file = _patch_settings(tmp_path, monkeypatch)
    saved = settings_service.save_global_settings({"image_test_save_root": "output/my-tests"})
    assert saved["ok"] is True
    assert saved["image_test_save_root"] == "output/my-tests"
    raw = toml.loads(settings_file.read_text(encoding="utf-8"))
    assert raw["global"]["image_test_save_root"] == "output/my-tests"
    assert settings_service.resolve_image_test_save_root() == "output/my-tests"
    assert settings_service.get_global_settings()["image_test_save_root"] == "output/my-tests"


def test_save_root_rejects_dotdot(tmp_path, monkeypatch):
    _patch_settings(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        settings_service.save_global_settings({"image_test_save_root": "foo/../bar"})


def test_normalize_request_uses_save_root_default(tmp_path, monkeypatch):
    _patch_settings(tmp_path, monkeypatch)
    settings_service.save_global_settings({"image_test_save_root": "output/my-tests"})
    # minimal request without save_path; model paths stubbed via monkeypatch
    monkeypatch.setattr(
        image_test_service,
        "_resolve_image_test_model_paths",
        lambda cfg: {
            **cfg,
            "pretrained_model_name_or_path": "models/dit.safetensors",
            "qwen3": "models/qwen",
            "vae": "models/vae.safetensors",
        },
    )
    monkeypatch.setattr(
        image_test_service,
        "_apply_global_model_path_defaults",
        lambda cfg: cfg,
    )
    out = image_test_service._normalize_image_test_request(
        {
            "prompt": "a cat",
            "config": {
                "pretrained_model_name_or_path": "models/dit.safetensors",
                "qwen3": "models/qwen",
                "vae": "models/vae.safetensors",
            },
        }
    )
    assert out["save_path"] == "output/my-tests"


def test_explicit_request_save_path_wins(tmp_path, monkeypatch):
    _patch_settings(tmp_path, monkeypatch)
    settings_service.save_global_settings({"image_test_save_root": "output/my-tests"})
    monkeypatch.setattr(
        image_test_service,
        "_resolve_image_test_model_paths",
        lambda cfg: {
            **cfg,
            "pretrained_model_name_or_path": "models/dit.safetensors",
            "qwen3": "models/qwen",
            "vae": "models/vae.safetensors",
        },
    )
    monkeypatch.setattr(
        image_test_service,
        "_apply_global_model_path_defaults",
        lambda cfg: cfg,
    )
    out = image_test_service._normalize_image_test_request(
        {
            "prompt": "a cat",
            "save_path": "output/other",
            "config": {
                "pretrained_model_name_or_path": "models/dit.safetensors",
                "qwen3": "models/qwen",
                "vae": "models/vae.safetensors",
            },
        }
    )
    assert out["save_path"] == "output/other"
