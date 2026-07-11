"""S-R8: environment check reports effective storage roots."""

from __future__ import annotations

from pathlib import Path

import toml

from web.services import environment_check_service as env_service
from web.services import settings_service
from web.services.environment_check_service import run_environment_check


def test_run_environment_check_includes_effective_roots(tmp_path, monkeypatch):
    # Isolate project root + settings so paths are deterministic.
    monkeypatch.setattr(env_service, "ROOT", tmp_path)
    monkeypatch.setattr(settings_service, "ROOT", tmp_path)
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    settings_file.parent.mkdir(parents=True)
    settings_file.write_text(
        "\n".join(
            [
                "[global]",
                'output_root = "output/runs"',
                'configs_root = "configs"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", settings_file)

    # Avoid heavy import probes / model scans dominating the test.
    monkeypatch.setattr(env_service, "_probe_imports", lambda _py: {"modules": {}, "torch_cuda": {}})
    monkeypatch.setattr(env_service, "_check_model_paths", lambda add: None)
    monkeypatch.setattr(env_service, "_run_cmd", lambda *_a, **_k: (True, "ok"))
    monkeypatch.setattr(env_service, "venv_python_path", lambda: tmp_path / ".venv" / "bin" / "python")
    monkeypatch.setattr(env_service, "resolve_web_python_executable", lambda: str(tmp_path / ".venv" / "bin" / "python"))
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").write_text("", encoding="utf-8")
    for rel in ("pyproject.toml", "uv.lock"):
        (tmp_path / rel).write_text("x", encoding="utf-8")
    # preprocess required files may be many; stub PROJECT_FILE_CHECKS
    monkeypatch.setattr(env_service, "PROJECT_FILE_CHECKS", ("pyproject.toml", "uv.lock"))

    result = run_environment_check()
    platform = result["platform"]
    assert "effective_roots" in platform
    roots = platform["effective_roots"]
    for key in ("project_root", "configs_root", "output_root", "history_root", "queue_root"):
        assert key in roots, key
        assert roots[key], key

    # Also surface as explicit checks under web_runtime.
    keys = {c["key"] for c in result["checks"]}
    assert "effective_configs_root" in keys
    assert "effective_output_root" in keys
    assert "effective_history_root" in keys
    assert "effective_queue_root" in keys
