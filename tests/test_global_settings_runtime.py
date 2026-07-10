from __future__ import annotations

import asyncio
import json
from pathlib import Path

import toml

import library.env as library_env
from web.routes import settings as settings_routes
from web.services import config_service, preview_service, settings_service


ROOT = Path(__file__).resolve().parents[1]
GLOBAL_SETTINGS_JS = ROOT / "web" / "static" / "js" / "features" / "global-settings" / "settings.js"
EVENT_LISTENERS_JS = ROOT / "web" / "static" / "js" / "features" / "app-shell" / "event-listeners.js"
INDEX_HTML = ROOT / "web" / "static" / "index.html"


class _JsonRequest:
    def __init__(self, payload: dict[str, object], app: dict[str, object] | None = None):
        self._payload = payload
        self.app = app or {}

    async def json(self) -> dict[str, object]:
        return self._payload


def test_save_global_settings_moves_to_new_configs_root_and_preserves_preview_section(tmp_path, monkeypatch):
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    settings_file.write_text(
        "\n".join(
            [
                "[preview]",
                'training_dir = "output/ckpt/sample"',
                'inference_dir = "output/tests"',
                'custom_dir = ""',
                "",
                "[global]",
                'output_root = "output/runs"',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_service, "ROOT", tmp_path)
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", settings_file)

    saved = settings_service.save_global_settings(
        {
            "output_root": "custom/runs",
            "configs_root": "external-configs",
        }
    )

    target_settings = tmp_path / "external-configs" / "web-ui-settings.toml"
    assert saved["ok"] is True
    assert saved["requires_reload"] is True
    assert saved["configs_root"] == "external-configs"
    assert target_settings.exists()

    raw = toml.loads(target_settings.read_text(encoding="utf-8"))
    assert raw["preview"]["training_dir"] == "output/ckpt/sample"
    assert raw["global"]["output_root"] == "custom/runs"
    assert raw["global"]["configs_root"] == "external-configs"

    override = toml.loads((tmp_path / ".anima-webui-settings.toml").read_text(encoding="utf-8"))
    assert override["paths"]["configs_root"] == "external-configs"


def test_preview_settings_follow_runtime_external_configs_root(tmp_path, monkeypatch):
    external_configs = tmp_path / "external-configs"
    (tmp_path / ".anima-webui-settings.toml").write_text(
        '[paths]\nconfigs_root = "external-configs"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(library_env, "project_root", lambda: tmp_path)
    monkeypatch.setattr(settings_service, "ROOT", tmp_path)
    monkeypatch.setattr(preview_service, "ROOT", tmp_path)
    monkeypatch.delenv("ANIMA_CONFIGS_ROOT", raising=False)

    payload = preview_service.save_preview_settings(
        {
            "training_dir": "output/ckpt/sample",
            "inference_dir": "output/tests",
            "custom_dir": "",
        }
    )

    target_settings = external_configs / "web-ui-settings.toml"
    assert payload["ok"] is True
    assert target_settings.exists()

    raw = toml.loads(target_settings.read_text(encoding="utf-8"))
    assert raw["preview"]["training_dir"] == "output/ckpt/sample"
    assert raw["preview"]["inference_dir"] == "output/tests"
    assert preview_service.get_preview_settings()["training_dir"] == "output/ckpt/sample"


def test_config_service_set_configs_root_updates_synced_paths(tmp_path, monkeypatch):
    for name in (
        "CONFIGS_DIR",
        "GUI_METHODS_DIR",
        "IMPORTED_CONFIGS_DIR",
        "PRESETS_FILE",
        "WEB_FILE_GROUPS_FILE",
        "WEB_USER_LOCKS_FILE",
        "DEFAULT_SAMPLE_PROMPTS_FILE",
        "DATASET_PRESETS_DIR",
    ):
        monkeypatch.setattr(config_service, name, getattr(config_service, name))

    resolved = config_service.set_configs_root(tmp_path / "alt-configs")

    assert resolved == (tmp_path / "alt-configs").resolve()
    assert config_service.CONFIGS_DIR == resolved
    assert config_service.GUI_METHODS_DIR == resolved / "gui-methods"
    assert config_service.IMPORTED_CONFIGS_DIR == resolved / "imported"
    assert config_service.PRESETS_FILE == resolved / "presets.toml"
    assert config_service.WEB_FILE_GROUPS_FILE == resolved / "web-file-groups.toml"
    assert config_service.WEB_USER_LOCKS_FILE == resolved / "web-user-locks.toml"
    assert config_service.DEFAULT_SAMPLE_PROMPTS_FILE == str(resolved / "sample_prompts.txt")
    assert config_service.DATASET_PRESETS_DIR == resolved / "datasets"


def test_global_settings_frontend_reload_after_configs_root_switch():
    source = GLOBAL_SETTINGS_JS.read_text(encoding="utf-8")

    assert "if (res.requires_reload)" in source
    assert "location.reload()" in source
    assert "正在切换新的配置根目录" in source


def test_global_settings_tooltips_and_help_describe_configs_root_reload():
    listeners = EVENT_LISTENERS_JS.read_text(encoding="utf-8")
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "修改配置根目录后页面会自动刷新" in listeners
    assert "'global-configs-root': '配置根目录" in listeners
    assert "'global-ui-scale': '默认界面缩放比例" in listeners
    assert "保存后页面会自动刷新并切换到新的配置根目录" in html


def test_settings_route_hot_swaps_configs_root_and_refreshes_training_state(monkeypatch, tmp_path):
    payload = {
        "ok": True,
        "requires_reload": True,
        "configs_root": "external-configs",
    }
    calls: dict[str, object] = {}
    service = object()

    monkeypatch.setattr(settings_routes, "save_global_settings", lambda data: payload)
    monkeypatch.setattr(settings_routes.settings_service, "SETTINGS_FILE", tmp_path / "external-configs" / "web-ui-settings.toml")
    monkeypatch.setattr(
        settings_routes.config_service,
        "set_configs_root",
        lambda root: calls.setdefault("configs_root", Path(root)),
    )
    monkeypatch.setattr(
        settings_routes.training_service,
        "reload_runtime_storage_state",
        lambda training_service: calls.setdefault("training_service", training_service),
    )

    response = asyncio.run(
        settings_routes.handle_global_settings_put(
            _JsonRequest({"configs_root": "external-configs"}, app={"training_service": service})
        )
    )

    body = json.loads(response.text)
    assert response.status == 200
    assert body["requires_reload"] is True
    assert calls["configs_root"] == (tmp_path / "external-configs")
    assert calls["training_service"] is service


def test_settings_route_returns_400_for_invalid_payload(monkeypatch):
    monkeypatch.setattr(
        settings_routes,
        "save_global_settings",
        lambda data: (_ for _ in ()).throw(ValueError("输出文件夹不能为空")),
    )

    response = asyncio.run(settings_routes.handle_global_settings_put(_JsonRequest({"output_root": ""})))
    body = json.loads(response.text)

    assert response.status == 400
    assert body == {"ok": False, "error": "输出文件夹不能为空"}


def test_save_global_settings_persists_history_and_queue_roots(tmp_path, monkeypatch):
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    settings_file.write_text(
        "[global]\noutput_root = \"output/runs\"\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_service, "ROOT", tmp_path)
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", settings_file)
    monkeypatch.setattr(library_env, "project_root", lambda: tmp_path)

    saved = settings_service.save_global_settings(
        {
            "output_root": "output/runs",
            "history_root": "alt-history",
            "queue_root": "alt-queue",
        }
    )

    assert saved["ok"] is True
    override = toml.loads((tmp_path / ".anima-webui-settings.toml").read_text(encoding="utf-8"))
    assert override["paths"]["history_root"] == "alt-history"
    assert override["paths"]["queue_root"] == "alt-queue"
    assert library_env.get_training_history_root() == (tmp_path / "alt-history").resolve()
    assert library_env.get_training_queue_root() == (tmp_path / "alt-queue").resolve()

