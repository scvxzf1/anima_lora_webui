from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

from aiohttp import web
from PIL import Image
import pytest

from web.services import image_test_service


def test_normalize_image_test_request_reads_config_and_overrides(monkeypatch) -> None:
    monkeypatch.setattr(image_test_service, "_apply_global_model_path_defaults", lambda cfg: cfg)
    monkeypatch.setattr(image_test_service, "_resolve_image_test_model_paths", lambda cfg: cfg)
    monkeypatch.setattr(
        image_test_service.settings_service,
        "get_global_settings",
        lambda: {"model_family": "anima"},
    )
    monkeypatch.setattr(
        image_test_service,
        "_resolve_image_test_weight_path",
        lambda value, app=None: Path("/tmp/weights/test.safetensors"),
    )

    payload = {
        "prompt": "1girl, masterpiece",
        "negative_prompt": "low quality",
        "width": "768",
        "height": "512",
        "infer_steps": "16",
        "guidance_scale": "4.5",
        "flow_shift": "1.25",
        "sampler": "LCM",
        "attn_mode": "FLASH",
        "runtime_dtype": "fp32",
        "text_encoder_dtype": "fp16",
        "seed": "42",
        "weight_path": "output/runs/demo/anima.safetensors",
        "lora_multiplier": "0.8",
        "anima_selective_lora": True,
        "anima_selective_preset": "late_main",
        "anima_selective_block_strengths": {
            "block_20": 0.65,
            "block_21": 0.25,
            "final_layer": 1.0,
            "block_4": 0.0,
        },
        "config": {
            "pretrained_model_name_or_path": "models/base.safetensors",
            "qwen3": "models/qwen3.safetensors",
            "vae": "models/vae.safetensors",
            "precision_preference": "fp16",
            "preprocess_precision_preference": "bf16",
        },
    }

    normalized = image_test_service._normalize_image_test_request(payload)

    assert normalized["prompt"] == "1girl, masterpiece"
    assert normalized["negative_prompt"] == "low quality"
    assert normalized["width"] == 768
    assert normalized["height"] == 512
    assert normalized["infer_steps"] == 16
    assert normalized["guidance_scale"] == 4.5
    assert normalized["flow_shift"] == 1.25
    assert normalized["sampler"] == "lcm"
    assert normalized["attn_mode"] == "flash"
    assert normalized["runtime_dtype"] == "fp32"
    assert normalized["text_encoder_dtype"] == "fp16"
    assert normalized["device"] is None
    assert normalized["gpu_index"] is None
    assert normalized["gpu_label"] == "自动"
    assert normalized["seed"] == 42
    assert normalized["weight_path"] == "/tmp/weights/test.safetensors"
    assert normalized["lora_multiplier"] == 0.8
    assert normalized["anima_selective_lora"] is True
    assert normalized["anima_selective_preset"] == "late_main"
    assert normalized["anima_selective_blocks"] == ["block_20", "block_21", "final_layer"]
    assert normalized["anima_selective_block_strengths"]["block_20"] == 0.65
    assert normalized["anima_selective_block_strengths"]["block_21"] == 0.25
    assert normalized["anima_selective_block_strengths"]["final_layer"] == 1.0
    assert normalized["anima_selective_block_strengths"]["block_4"] == 0.0


def _stub_image_test_model_resolution(monkeypatch, *, global_family: str) -> None:
    monkeypatch.setattr(image_test_service, "_apply_global_model_path_defaults", lambda cfg: cfg)
    monkeypatch.setattr(image_test_service, "_resolve_image_test_model_paths", lambda cfg: cfg)
    monkeypatch.setattr(
        image_test_service.settings_service,
        "get_global_settings",
        lambda: {"model_family": global_family},
    )


def _minimal_image_test_payload(*, model_family=None) -> dict:
    config = {
        "pretrained_model_name_or_path": "models/base.safetensors",
        "qwen3": "models/qwen3.safetensors",
        "vae": "models/vae.safetensors",
    }
    if model_family is not None:
        config["model_family"] = model_family
    return {
        "prompt": "test",
        "sampler": "euler",
        "attn_mode": "torch",
        "flow_shift": "",
        "config": config,
    }


def test_image_test_config_family_overrides_global_family(monkeypatch) -> None:
    _stub_image_test_model_resolution(monkeypatch, global_family="anima")
    normalized = image_test_service._normalize_image_test_request(
        _minimal_image_test_payload(model_family="krea2_raw")
    )
    assert normalized["model_family"] == "krea2_raw"
    assert normalized["flow_shift"] == 3.0


def test_image_test_global_family_is_legacy_fallback(monkeypatch) -> None:
    _stub_image_test_model_resolution(monkeypatch, global_family="krea2_raw")
    normalized = image_test_service._normalize_image_test_request(
        _minimal_image_test_payload()
    )
    assert normalized["model_family"] == "krea2_raw"


def test_image_test_env_family_is_final_fallback(monkeypatch) -> None:
    _stub_image_test_model_resolution(monkeypatch, global_family="")
    monkeypatch.setenv("ANIMA_MODEL_FAMILY", "krea2_raw")
    normalized = image_test_service._normalize_image_test_request(
        _minimal_image_test_payload()
    )
    assert normalized["model_family"] == "krea2_raw"


def test_image_test_rejects_unknown_explicit_config_family(monkeypatch) -> None:
    _stub_image_test_model_resolution(monkeypatch, global_family="anima")
    with pytest.raises(ValueError, match="image-test config.model_family"):
        image_test_service._normalize_image_test_request(
            _minimal_image_test_payload(model_family="unknown")
        )


def test_image_test_rejects_krea2_non_euler_and_flow_shift(monkeypatch) -> None:
    _stub_image_test_model_resolution(monkeypatch, global_family="anima")
    payload = _minimal_image_test_payload(model_family="krea2_raw")
    payload["sampler"] = "lcm"
    with pytest.raises(ValueError, match="Euler"):
        image_test_service._normalize_image_test_request(payload)
    payload["sampler"] = "euler"
    payload["flow_shift"] = "1.0"
    with pytest.raises(ValueError, match="mu shift"):
        image_test_service._normalize_image_test_request(payload)


def test_build_generation_command_includes_expected_cli_flags(monkeypatch) -> None:
    monkeypatch.setattr(image_test_service, "resolve_web_python_executable", lambda: "/venv/bin/python")

    request = {
        "prompt": "cat",
        "negative_prompt": "bad anatomy",
        "width": 1024,
        "height": 1024,
        "infer_steps": 20,
        "guidance_scale": 4.0,
        "flow_shift": 1.0,
        "sampler": "euler",
        "attn_mode": "flash",
        "runtime_dtype": "fp16",
        "text_encoder_dtype": "same",
        "model_family": "anima",
        "device": "cuda",
        "gpu_index": 1,
        "gpu_label": "GPU 1 · Demo",
        "seed": 123,
        "weight_path": "/tmp/demo.safetensors",
        "lora_multiplier": 0.75,
        "anima_selective_lora": True,
        "anima_selective_preset": "half_strength",
        "anima_selective_blocks": ["block_0", "block_1"],
        "anima_selective_strength": 1.0,
        "anima_selective_block_strengths": {
            "block_0": 0.5,
            "block_1": 0.25,
            "block_2": 0.0,
        },
        "save_path": "output/tests",
        "config": {
            "pretrained_model_name_or_path": "models/base.safetensors",
            "vae": "models/vae.safetensors",
            "qwen3": "models/qwen3.safetensors",
        },
    }

    cmd = image_test_service._build_generation_command(request)

    assert cmd[0] == "/venv/bin/python"
    assert cmd[1] == "inference.py"
    assert "--prompt" in cmd
    assert "--negative_prompt" in cmd
    assert "--image_size" in cmd
    assert "--infer_steps" in cmd
    assert "--sampler" in cmd
    assert "--attn_mode" in cmd
    assert "--device" in cmd
    assert "--runtime_dtype" in cmd
    assert "--text_encoder_dtype" in cmd
    family_index = cmd.index("--model_family")
    assert cmd[family_index + 1] == "anima"
    assert "--lora_weight" in cmd
    assert "--lora_multiplier" in cmd
    assert "--anima_selective_lora" in cmd
    assert "--anima_selective_preset" in cmd
    assert "--anima_selective_block_strengths" in cmd
    assert "--save_path" in cmd
    assert cmd[cmd.index("--device") + 1] == "cuda"
    assert cmd[cmd.index("--runtime_dtype") + 1] == "fp16"
    assert cmd[cmd.index("--text_encoder_dtype") + 1] == "same"
    assert cmd[cmd.index("--anima_selective_preset") + 1] == "half_strength"
    strengths_index = cmd.index("--anima_selective_block_strengths") + 1
    assert "block_0=0.50" in cmd[strengths_index:]
    assert "block_1=0.25" in cmd[strengths_index:]
    assert "block_2=0.00" in cmd[strengths_index:]


def test_normalize_image_test_request_requires_weight_for_selective_lora(monkeypatch) -> None:
    monkeypatch.setattr(image_test_service, "_apply_global_model_path_defaults", lambda cfg: cfg)
    monkeypatch.setattr(image_test_service, "_resolve_image_test_model_paths", lambda cfg: cfg)

    payload = {
        "prompt": "1girl",
        "anima_selective_lora": True,
        "config": {
            "pretrained_model_name_or_path": "models/base.safetensors",
            "qwen3": "models/qwen3.safetensors",
            "vae": "models/vae.safetensors",
        },
    }

    try:
        image_test_service._normalize_image_test_request(payload)
    except ValueError as exc:
        assert "启用 LoRA 分层加载时，需要先选择一个 LoRA 权重" in str(exc)
    else:
        raise AssertionError("expected ValueError when selective LoRA has no weight")


def test_normalize_image_test_request_accepts_detected_gpu(monkeypatch) -> None:
    monkeypatch.setattr(image_test_service, "_apply_global_model_path_defaults", lambda cfg: cfg)
    monkeypatch.setattr(image_test_service, "_resolve_image_test_model_paths", lambda cfg: cfg)

    payload = {
        "prompt": "1girl, masterpiece",
        "gpu_index": "1",
        "config": {
            "pretrained_model_name_or_path": "models/base.safetensors",
            "qwen3": "models/qwen3.safetensors",
            "vae": "models/vae.safetensors",
        },
    }

    normalized = image_test_service._normalize_image_test_request(
        payload,
        available_gpus=[
            {"index": 0, "label": "GPU 0 · A"},
            {"index": 1, "label": "GPU 1 · B"},
        ],
    )

    assert normalized["device"] == "cuda"
    assert normalized["gpu_index"] == 1
    assert normalized["gpu_label"] == "GPU 1 · B"


def test_normalize_image_test_request_rejects_unknown_gpu(monkeypatch) -> None:
    monkeypatch.setattr(image_test_service, "_apply_global_model_path_defaults", lambda cfg: cfg)
    monkeypatch.setattr(image_test_service, "_resolve_image_test_model_paths", lambda cfg: cfg)

    payload = {
        "prompt": "1girl, masterpiece",
        "gpu_index": "3",
        "config": {
            "pretrained_model_name_or_path": "models/base.safetensors",
            "qwen3": "models/qwen3.safetensors",
            "vae": "models/vae.safetensors",
        },
    }

    try:
        image_test_service._normalize_image_test_request(
            payload,
            available_gpus=[{"index": 0, "label": "GPU 0 · A"}],
        )
    except ValueError as exc:
        assert "当前未检测到 GPU 3" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown gpu")


def test_build_generation_env_applies_single_gpu_whitelist() -> None:
    env = image_test_service._build_generation_env({"gpu_index": 2})

    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["CUDA_VISIBLE_DEVICES"] == "2"


def test_resolve_image_test_model_paths_prefers_existing_global_model_files(monkeypatch, tmp_path) -> None:
    dit = tmp_path / "models" / "anima-base.safetensors"
    qwen3 = tmp_path / "models" / "qwen3.safetensors"
    vae = tmp_path / "models" / "vae.safetensors"
    dit.parent.mkdir(parents=True, exist_ok=True)
    dit.write_bytes(b"dit")
    qwen3.write_bytes(b"qwen3")
    vae.write_bytes(b"vae")
    monkeypatch.setattr(
        image_test_service.settings_service,
        "get_global_settings",
        lambda: {
            "pretrained_model_name_or_path": str(dit),
            "qwen3": str(qwen3),
            "vae": str(vae),
        },
    )

    resolved = image_test_service._resolve_image_test_model_paths({
        "pretrained_model_name_or_path": "models/diffusion_models/anima-base-v1.0.safetensors",
        "qwen3": "models/text_encoders/missing-qwen3.safetensors",
        "vae": "models/vae/missing-vae.safetensors",
    })

    assert resolved["pretrained_model_name_or_path"] == str(dit.resolve())
    assert resolved["qwen3"] == str(qwen3.resolve())
    assert resolved["vae"] == str(vae.resolve())


def test_resolve_image_test_model_paths_prefers_global_over_placeholder_defaults(monkeypatch, tmp_path) -> None:
    placeholder_dit = tmp_path / "project" / "models" / "diffusion_models" / "anima-base-v1.0.safetensors"
    placeholder_qwen3 = tmp_path / "project" / "models" / "text_encoders" / "qwen_3_06b_base.safetensors"
    placeholder_vae = tmp_path / "project" / "models" / "vae" / "qwen_image_vae.safetensors"
    global_dit = tmp_path / "global" / "anima-preview3-base.safetensors"
    global_qwen3 = tmp_path / "global" / "anima_qwen_3_06b_base.safetensors"
    global_vae = tmp_path / "global" / "anima_vae.safetensors"
    for path, content in (
        (placeholder_dit, b"placeholder-dit"),
        (placeholder_qwen3, b"placeholder-qwen3"),
        (placeholder_vae, b"placeholder-vae"),
        (global_dit, b"global-dit"),
        (global_qwen3, b"global-qwen3"),
        (global_vae, b"global-vae"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    mapping = {
        "models/diffusion_models/anima-base-v1.0.safetensors": placeholder_dit,
        "models/text_encoders/qwen_3_06b_base.safetensors": placeholder_qwen3,
        "models/vae/qwen_image_vae.safetensors": placeholder_vae,
        str(global_dit): global_dit,
        str(global_qwen3): global_qwen3,
        str(global_vae): global_vae,
    }
    monkeypatch.setattr(
        image_test_service,
        "_resolve_image_test_model_config_path",
        lambda value: mapping.get(str(value or "").strip()),
    )
    monkeypatch.setattr(
        image_test_service.settings_service,
        "get_global_settings",
        lambda: {
            "pretrained_model_name_or_path": str(global_dit),
            "qwen3": str(global_qwen3),
            "vae": str(global_vae),
            "defaults": {
                "pretrained_model_name_or_path": "models/diffusion_models/anima-base-v1.0.safetensors",
                "qwen3": "models/text_encoders/qwen_3_06b_base.safetensors",
                "vae": "models/vae/qwen_image_vae.safetensors",
            },
        },
    )

    resolved = image_test_service._resolve_image_test_model_paths({
        "pretrained_model_name_or_path": "models/diffusion_models/anima-base-v1.0.safetensors",
        "qwen3": "models/text_encoders/qwen_3_06b_base.safetensors",
        "vae": "models/vae/qwen_image_vae.safetensors",
    })

    assert resolved["pretrained_model_name_or_path"] == str(global_dit.resolve())
    assert resolved["qwen3"] == str(global_qwen3.resolve())
    assert resolved["vae"] == str(global_vae.resolve())


def test_resolve_image_test_model_paths_rejects_missing_dit_before_spawn(monkeypatch, tmp_path) -> None:
    qwen3 = tmp_path / "models" / "qwen3.safetensors"
    vae = tmp_path / "models" / "vae.safetensors"
    qwen3.parent.mkdir(parents=True, exist_ok=True)
    qwen3.write_bytes(b"qwen3")
    vae.write_bytes(b"vae")
    monkeypatch.setattr(image_test_service.settings_service, "get_global_settings", lambda: {})

    try:
        image_test_service._resolve_image_test_model_paths({
            "pretrained_model_name_or_path": "models/diffusion_models/anima-base-v1.0.safetensors",
            "qwen3": str(qwen3),
            "vae": str(vae),
        })
    except ValueError as exc:
        assert "基础 DiT 模型 不存在" in str(exc)
        assert "全局设置" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing dit model path")


def test_resolve_image_test_weight_path_accepts_global_output_root(monkeypatch, tmp_path) -> None:
    output_root = tmp_path / "output"
    weight_path = output_root / "runs" / "demo.safetensors"
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path.write_bytes(b"stub")
    monkeypatch.setattr(image_test_service.settings_service, "resolve_output_root", lambda value=None: output_root)

    resolved = image_test_service._resolve_image_test_weight_path(str(weight_path))

    assert resolved == weight_path.resolve()


def test_resolve_image_test_weight_path_accepts_unique_bare_filename(monkeypatch, tmp_path) -> None:
    output_root = tmp_path / "output"
    weight_path = output_root / "runs" / "nested" / "demo.safetensors"
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path.write_bytes(b"stub")
    monkeypatch.setattr(image_test_service.settings_service, "resolve_output_root", lambda value=None: output_root)

    resolved = image_test_service._resolve_image_test_weight_path("demo.safetensors")

    assert resolved == weight_path.resolve()


def test_image_test_service_resolve_weight_path_returns_full_path_payload(monkeypatch, tmp_path) -> None:
    output_root = tmp_path / "output"
    weight_path = output_root / "runs" / "nested" / "demo.safetensors"
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path.write_bytes(b"stub")
    monkeypatch.setattr(image_test_service.settings_service, "resolve_output_root", lambda value=None: output_root)
    app = web.Application()
    svc = image_test_service.ImageTestService(app)

    payload = svc.resolve_weight_path("demo.safetensors")

    assert payload["ok"] is True
    assert payload["name"] == "demo.safetensors"
    assert payload["weight_path"] == str(weight_path.resolve())


def test_resolve_image_test_weight_path_accepts_current_training_output_dir(monkeypatch, tmp_path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    training_output_dir = tmp_path / "current-run"
    weight_path = training_output_dir / "current.safetensors"
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path.write_bytes(b"stub")
    monkeypatch.setattr(image_test_service.settings_service, "resolve_output_root", lambda value=None: output_root)
    app = {
        "training_service": SimpleNamespace(
            current_output_dir=str(training_output_dir),
            current_sample_dir="",
        )
    }

    resolved = image_test_service._resolve_image_test_weight_path(str(weight_path), app=app)

    assert resolved == weight_path.resolve()


def test_resolve_image_test_weight_path_prefers_current_training_output_dir_for_duplicate_name(monkeypatch, tmp_path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    first = output_root / "runs" / "a" / "same.safetensors"
    second = tmp_path / "current-run" / "same.safetensors"
    first.parent.mkdir(parents=True, exist_ok=True)
    second.parent.mkdir(parents=True, exist_ok=True)
    first.write_bytes(b"stub-a")
    second.write_bytes(b"stub-b")
    monkeypatch.setattr(image_test_service.settings_service, "resolve_output_root", lambda value=None: output_root)
    app = {
        "training_service": SimpleNamespace(
            current_output_dir=str(second.parent),
            current_sample_dir="",
        )
    }

    resolved = image_test_service._resolve_image_test_weight_path("same.safetensors", app=app)

    assert resolved == second.resolve()


def test_resolve_image_test_weight_path_prefers_newest_duplicate_name(monkeypatch, tmp_path) -> None:
    output_root = tmp_path / "output"
    older = output_root / "runs" / "a" / "same.safetensors"
    newer = output_root / "runs" / "b" / "same.safetensors"
    older.parent.mkdir(parents=True, exist_ok=True)
    newer.parent.mkdir(parents=True, exist_ok=True)
    older.write_bytes(b"older")
    newer.write_bytes(b"newer")
    os.utime(older, (1_700_000_000, 1_700_000_000))
    os.utime(newer, (1_800_000_000, 1_800_000_000))
    monkeypatch.setattr(image_test_service.settings_service, "resolve_output_root", lambda value=None: output_root)

    resolved = image_test_service._resolve_image_test_weight_path("same.safetensors")

    assert resolved == newer.resolve()


def test_resolve_image_test_weight_path_accepts_bare_filename_from_global_model_root(monkeypatch, tmp_path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    model_root = tmp_path / "pac" / "models"
    dit = model_root / "diffusion_models" / "anima" / "anima-preview3-base.safetensors"
    qwen3 = model_root / "text_encoders" / "anima" / "anima_qwen_3_06b_base.safetensors"
    vae = model_root / "vae" / "anima" / "anima_vae.safetensors"
    external_weight = model_root / "loras" / "es1" / "ichika87_style-es2-2.safetensors"
    for path in (dit, qwen3, vae, external_weight):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub")
    monkeypatch.setattr(image_test_service.settings_service, "resolve_output_root", lambda value=None: output_root)
    monkeypatch.setattr(
        image_test_service.settings_service,
        "get_global_settings",
        lambda: {
            "pretrained_model_name_or_path": str(dit),
            "qwen3": str(qwen3),
            "vae": str(vae),
        },
    )

    resolved = image_test_service._resolve_image_test_weight_path("ichika87_style-es2-2.safetensors")

    assert resolved == external_weight.resolve()


def test_resolve_image_test_weight_path_does_not_search_home_by_default(monkeypatch, tmp_path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    home_dir = tmp_path / "user-home"
    external_weight = home_dir / "random" / "loras" / "elsewhere.safetensors"
    external_weight.parent.mkdir(parents=True, exist_ok=True)
    external_weight.write_bytes(b"stub")
    monkeypatch.setattr(image_test_service.settings_service, "resolve_output_root", lambda value=None: output_root)
    monkeypatch.setattr(
        image_test_service.settings_service,
        "get_global_settings",
        lambda: {},
    )
    monkeypatch.setenv("HOME", str(home_dir))
    app = {"training_service": SimpleNamespace(current_output_dir="", current_sample_dir="")}

    try:
        image_test_service._resolve_image_test_weight_path("elsewhere.safetensors", app=app)
    except FileNotFoundError as exc:
        assert "未找到对应的 LoRA / LokR 权重文件" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError when home search is disabled")


def test_resolve_image_test_weight_path_can_opt_in_home_search(monkeypatch, tmp_path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    home_dir = tmp_path / "user-home"
    external_weight = home_dir / "random" / "loras" / "elsewhere.safetensors"
    external_weight.parent.mkdir(parents=True, exist_ok=True)
    external_weight.write_bytes(b"stub")
    monkeypatch.setattr(image_test_service.settings_service, "resolve_output_root", lambda value=None: output_root)
    monkeypatch.setattr(
        image_test_service.settings_service,
        "get_global_settings",
        lambda: {"image_test_allow_home_search": True},
    )
    monkeypatch.setenv("HOME", str(home_dir))
    app = {"training_service": SimpleNamespace(current_output_dir="", current_sample_dir="")}

    resolved = image_test_service._resolve_image_test_weight_path("elsewhere.safetensors", app=app)

    assert resolved == external_weight.resolve()


def test_resolve_image_test_weight_path_does_not_search_workspace_parent_tree(monkeypatch, tmp_path) -> None:
    output_root = tmp_path / "project" / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    workspace_hit = tmp_path / "sibling-project" / "mystery.safetensors"
    workspace_hit.parent.mkdir(parents=True, exist_ok=True)
    workspace_hit.write_bytes(b"stub")
    monkeypatch.setattr(image_test_service, "ROOT", tmp_path / "project")
    monkeypatch.setattr(image_test_service.settings_service, "resolve_output_root", lambda value=None: output_root)
    monkeypatch.setattr(
        image_test_service.settings_service,
        "get_global_settings",
        lambda: {},
    )
    app = {"training_service": SimpleNamespace(current_output_dir="", current_sample_dir="")}

    try:
        image_test_service._resolve_image_test_weight_path("mystery.safetensors", app=app)
    except FileNotFoundError as exc:
        assert "未找到对应的 LoRA / LokR 权重文件" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError when workspace parent search is disabled")


def test_resolve_image_test_weight_path_rejects_outside_allowlist(monkeypatch, tmp_path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    outside_path = tmp_path / "outside" / "bad.safetensors"
    outside_path.parent.mkdir(parents=True, exist_ok=True)
    outside_path.write_bytes(b"stub")
    monkeypatch.setattr(image_test_service.settings_service, "resolve_output_root", lambda value=None: output_root)
    monkeypatch.setattr(image_test_service, "ROOT", tmp_path / "repo")
    (tmp_path / "repo").mkdir(parents=True, exist_ok=True)

    try:
        image_test_service._resolve_image_test_weight_path(str(outside_path))
    except ValueError as exc:
        assert "允许范围" in str(exc)
    else:
        raise AssertionError("expected ValueError for outside allowlist path")


def test_resolve_image_test_weight_path_rejects_parent_escape(monkeypatch, tmp_path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(image_test_service.settings_service, "resolve_output_root", lambda value=None: output_root)
    monkeypatch.setattr(image_test_service, "ROOT", tmp_path / "repo")
    (tmp_path / "repo").mkdir(parents=True, exist_ok=True)

    try:
        image_test_service._resolve_image_test_weight_path("../secret.safetensors")
    except ValueError as exc:
        assert ".." in str(exc) or "允许范围" in str(exc)
    else:
        raise AssertionError("expected ValueError for parent escape path")


def test_resolve_image_test_weight_path_accepts_under_output_root(monkeypatch, tmp_path) -> None:
    output_root = tmp_path / "output"
    weight = output_root / "runs" / "ok.safetensors"
    weight.parent.mkdir(parents=True, exist_ok=True)
    weight.write_bytes(b"stub")
    monkeypatch.setattr(image_test_service.settings_service, "resolve_output_root", lambda value=None: output_root)
    monkeypatch.setattr(image_test_service, "ROOT", tmp_path / "repo")
    (tmp_path / "repo").mkdir(parents=True, exist_ok=True)

    resolved = image_test_service._resolve_image_test_weight_path(str(weight))
    assert resolved == weight.resolve()


def test_resolve_image_test_weight_path_rejects_missing_bare_filename(monkeypatch, tmp_path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(image_test_service.settings_service, "resolve_output_root", lambda value=None: output_root)

    try:
        image_test_service._resolve_image_test_weight_path("missing.safetensors")
    except FileNotFoundError as exc:
        assert "未找到对应的 LoRA / LokR 权重文件" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError for missing bare filename")


def test_resolve_image_test_weight_path_rejects_non_safetensors(monkeypatch, tmp_path) -> None:
    output_root = tmp_path / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    wrong_ext = output_root / "runs" / "demo.ckpt"
    wrong_ext.parent.mkdir(parents=True, exist_ok=True)
    wrong_ext.write_bytes(b"stub")
    monkeypatch.setattr(image_test_service.settings_service, "resolve_output_root", lambda value=None: output_root)

    try:
        image_test_service._resolve_image_test_weight_path(str(wrong_ext))
    except ValueError as exc:
        assert "只支持 .safetensors 权重文件" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-safetensors weight")


def test_image_test_service_initial_snapshot_is_idle(tmp_path) -> None:
    service = image_test_service.ImageTestService(web.Application())
    service.output_dir = tmp_path

    snapshot = service.get_status_snapshot()

    assert snapshot["ok"] is True
    assert snapshot["status"] == "idle"
    assert snapshot["running"] is False
    assert snapshot["output_count"] == 0
    assert snapshot["output_files"] == []


def test_image_test_status_keeps_large_recent_gallery(tmp_path) -> None:
    service = image_test_service.ImageTestService(web.Application())
    service.output_dir = tmp_path
    for index in range(14):
        Image.new("RGB", (8, 8), color=(index, index, index)).save(tmp_path / f"image-{index:02d}.png")

    snapshot = service.get_status_snapshot()

    assert snapshot["output_count"] == 14
    assert len(snapshot["output_files"]) == 14


def test_list_output_images_probes_only_recent_top_k(tmp_path, monkeypatch) -> None:
    for index in range(8):
        path = tmp_path / f"image-{index}.png"
        path.write_bytes(b"stub")
        os.utime(path, (100 + index, 100 + index))

    probe_calls: list[str] = []

    def fake_probe(path: Path):
        probe_calls.append(path.name)
        return 8, 8

    monkeypatch.setattr(image_test_service, "probe_image_size", fake_probe)

    images = image_test_service._list_output_images(tmp_path, limit=3)

    assert [item["name"] for item in images] == ["image-7.png", "image-6.png", "image-5.png"]
    assert probe_calls == ["image-7.png", "image-6.png", "image-5.png"]


def test_list_output_images_skips_file_that_disappears_after_scan(tmp_path, monkeypatch) -> None:
    stable = tmp_path / "stable.png"
    disappearing = tmp_path / "disappearing.png"
    Image.new("RGB", (8, 8)).save(stable)
    Image.new("RGB", (8, 8)).save(disappearing)
    os.utime(stable, (100, 100))
    os.utime(disappearing, (200, 200))

    original_lstat = Path.lstat
    disappearing_calls = 0

    def flaky_lstat(path: Path):
        nonlocal disappearing_calls
        if path == disappearing:
            disappearing_calls += 1
            if disappearing_calls == 2:
                disappearing.unlink()
                raise FileNotFoundError(path)
        return original_lstat(path)

    monkeypatch.setattr(Path, "lstat", flaky_lstat)

    images = image_test_service._list_output_images(tmp_path, limit=2)

    assert [item["name"] for item in images] == ["stable.png"]


def test_list_output_images_keeps_corrupt_file_with_unknown_dimensions(tmp_path) -> None:
    (tmp_path / "broken.png").write_bytes(b"not an image")

    images = image_test_service._list_output_images(tmp_path)

    assert len(images) == 1
    assert images[0]["width"] is None
    assert images[0]["height"] is None


def test_list_output_images_clamps_gallery_to_500(tmp_path, monkeypatch) -> None:
    for index in range(505):
        path = tmp_path / f"image-{index:03d}.png"
        path.write_bytes(b"stub")
        os.utime(path, (100 + index, 100 + index))
    probe_count = 0

    def fake_probe(_path: Path):
        nonlocal probe_count
        probe_count += 1
        return 8, 8

    monkeypatch.setattr(image_test_service, "probe_image_size", fake_probe)

    images = image_test_service._list_output_images(tmp_path, limit=999)

    assert len(images) == 500
    assert probe_count == 500
    assert images[0]["name"] == "image-504.png"
    assert images[-1]["name"] == "image-005.png"
