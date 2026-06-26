from __future__ import annotations

import sys
from pathlib import Path

from web.services import environment_check_service as env_service
from web.services import project_python
from web.services.config._legacy import PREPROCESS_ENV_REQUIRED_FILES
from web.services.environment_check_service import (
    check_preprocess_environment_for_preflight,
    run_environment_check,
)


def test_project_file_checks_are_unique():
    assert len(env_service.PROJECT_FILE_CHECKS) == len(set(env_service.PROJECT_FILE_CHECKS))
    assert env_service.PROJECT_FILE_CHECKS.count("tasks.py") == 1


def test_cuda_track_from_probe_runtime_version():
    assert env_service._cuda_track_from_probe({"runtime_version": "13.0"}) == "cu130"
    assert env_service._cuda_track_from_probe({"runtime_version": "12.8"}) == "cu128"
    assert env_service._cuda_track_from_probe({"runtime_version": ""}) == "unknown"


def test_venv_python_path_linux_layout(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(project_python, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "platform", "linux")
    bindir = tmp_path / ".venv" / "bin"
    bindir.mkdir(parents=True)
    py = bindir / "python"
    py.write_text("#!/bin/sh\n", encoding="utf-8")
    py.chmod(0o755)
    assert project_python.venv_python_path(tmp_path) == py


def test_venv_python_path_windows_layout(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(project_python, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "platform", "win32")
    scripts = tmp_path / ".venv" / "Scripts"
    scripts.mkdir(parents=True)
    exe = scripts / "python.exe"
    exe.write_bytes(b"")
    assert project_python.venv_python_path(tmp_path) == exe


def test_run_environment_check_reports_missing_venv(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("web.services.environment_check_service.ROOT", tmp_path)
    monkeypatch.setattr("web.services.project_python.ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    result = run_environment_check()
    assert result["summary"]["checks"] >= 1
    assert any(c["key"] == "project_venv" and c["level"] == "error" for c in result["checks"])


def test_run_environment_check_reports_configured_model_paths(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(env_service, "ROOT", tmp_path)
    monkeypatch.setattr(project_python, "ROOT", tmp_path)
    monkeypatch.setattr("web.services.settings_service.ROOT", tmp_path)
    monkeypatch.setattr("web.services.settings_service.SETTINGS_FILE", tmp_path / "configs" / "web-ui-settings.toml")
    monkeypatch.setattr(env_service, "_run_cmd", lambda *args, **kwargs: (True, "tool ok"))
    monkeypatch.setattr(
        env_service,
        "_probe_imports",
        lambda _python: {
            "modules": {name: {"ok": True, "version": "1.0"} for name in env_service.CORE_IMPORT_MODULES},
            "torch_cuda": {"available": False, "devices": []},
        },
    )
    monkeypatch.setattr(sys, "platform", "linux")

    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    py = tmp_path / ".venv" / "bin" / "python"
    py.write_text("#!/bin/sh\n", encoding="utf-8")
    py.chmod(0o755)
    for rel in ("pyproject.toml", "uv.lock", "tasks.py", *PREPROCESS_ENV_REQUIRED_FILES):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#\n", encoding="utf-8")
    for rel in (
        "models/diffusion_models/anima-base-v1.0.safetensors",
        "models/text_encoders/qwen_3_06b_base.safetensors",
        "models/vae/qwen_image_vae.safetensors",
    ):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub")
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "base.toml").write_text(
        "\n".join([
            'pretrained_model_name_or_path = "models/diffusion_models/anima-base-v1.0.safetensors"',
            'qwen3 = "models/text_encoders/qwen_3_06b_base.safetensors"',
            'vae = "models/vae/qwen_image_vae.safetensors"',
        ]),
        encoding="utf-8",
    )

    result = run_environment_check()
    model_checks = [item for item in result["checks"] if item["key"].startswith("model_")]
    assert len(model_checks) == 3
    assert all(item["level"] == "ok" for item in model_checks)
    assert any(group["key"] == "model_paths" for group in result["groups"])


def test_run_environment_check_uses_external_global_model_paths(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(env_service, "ROOT", tmp_path)
    monkeypatch.setattr(project_python, "ROOT", tmp_path)
    monkeypatch.setattr("web.services.settings_service.ROOT", tmp_path)
    external_configs = tmp_path / "external-configs"
    monkeypatch.setattr(
        "web.services.settings_service.SETTINGS_FILE",
        external_configs / "web-ui-settings.toml",
    )
    monkeypatch.setattr(env_service, "_run_cmd", lambda *args, **kwargs: (True, "tool ok"))
    monkeypatch.setattr(
        env_service,
        "_probe_imports",
        lambda _python: {
            "modules": {name: {"ok": True, "version": "1.0"} for name in env_service.CORE_IMPORT_MODULES},
            "torch_cuda": {"available": False, "devices": []},
        },
    )
    monkeypatch.setattr(sys, "platform", "linux")

    py = tmp_path / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\n", encoding="utf-8")
    py.chmod(0o755)
    for rel in ("pyproject.toml", "uv.lock", "tasks.py", *PREPROCESS_ENV_REQUIRED_FILES):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#\n", encoding="utf-8")
    for rel in (
        "models/global-anima.safetensors",
        "models/global-qwen.safetensors",
        "models/global-vae.safetensors",
    ):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub")
    external_configs.mkdir(parents=True)
    (external_configs / "web-ui-settings.toml").write_text(
        "\n".join(
            [
                "[global]",
                'pretrained_model_name_or_path = "models/global-anima.safetensors"',
                'qwen3 = "models/global-qwen.safetensors"',
                'vae = "models/global-vae.safetensors"',
            ]
        ),
        encoding="utf-8",
    )

    result = run_environment_check()
    model_checks = {item["key"]: item for item in result["checks"] if item["key"].startswith("model_")}
    assert model_checks["model_pretrained_model_name_or_path"]["level"] == "ok"
    assert model_checks["model_qwen3"]["level"] == "ok"
    assert model_checks["model_vae"]["level"] == "ok"


def test_preflight_preprocess_ok_when_files_present(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("web.services.environment_check_service.ROOT", tmp_path)
    monkeypatch.setattr("web.services.project_python.ROOT", tmp_path)
    monkeypatch.setattr(sys, "platform", "linux")
    py = tmp_path / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.write_text("#!/bin/sh\n", encoding="utf-8")
    py.chmod(0o755)
    for rel in PREPROCESS_ENV_REQUIRED_FILES:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("#\n", encoding="utf-8")
    captured = []

    def add(level, key, message, path=None, **kwargs):
        captured.append({"level": level, "key": key, "message": message})

    check_preprocess_environment_for_preflight(add)
    assert any(item["key"] == "preprocess_environment" and item["level"] == "ok" for item in captured)
