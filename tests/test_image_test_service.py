from __future__ import annotations

from pathlib import Path

from aiohttp import web

from web.services import image_test_service


def test_normalize_image_test_request_reads_config_and_overrides(monkeypatch) -> None:
    monkeypatch.setattr(image_test_service, "_apply_global_model_path_defaults", lambda cfg: cfg)
    monkeypatch.setattr(
        image_test_service,
        "resolve_analysis_weight",
        lambda value: Path("/tmp/weights/test.safetensors"),
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
    assert normalized["seed"] == 42
    assert normalized["weight_path"] == "/tmp/weights/test.safetensors"
    assert normalized["lora_multiplier"] == 0.8


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
        "seed": 123,
        "weight_path": "/tmp/demo.safetensors",
        "lora_multiplier": 0.75,
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
    assert "--runtime_dtype" in cmd
    assert "--text_encoder_dtype" in cmd
    assert "--lora_weight" in cmd
    assert "--lora_multiplier" in cmd
    assert "--save_path" in cmd
    assert cmd[cmd.index("--runtime_dtype") + 1] == "fp16"
    assert cmd[cmd.index("--text_encoder_dtype") + 1] == "same"


def test_image_test_service_initial_snapshot_is_idle(tmp_path) -> None:
    service = image_test_service.ImageTestService(web.Application())
    service.output_dir = tmp_path

    snapshot = service.get_status_snapshot()

    assert snapshot["ok"] is True
    assert snapshot["status"] == "idle"
    assert snapshot["running"] is False
    assert snapshot["output_count"] == 0
    assert snapshot["output_files"] == []
