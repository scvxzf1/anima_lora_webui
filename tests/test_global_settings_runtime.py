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


def test_set_configs_root_hot_swaps_domain_module_roots(tmp_path, monkeypatch):
    """Hot-swapped configs root must update domain modules, not only the facade."""
    from web.services.config import (
        common as common_mod,
        merge as merge_mod,
        preflight_runtime as preflight_runtime_mod,
        raw_files as raw_files_mod,
        sample_prompts as sample_prompts_mod,
    )

    # Keep originals restorable so this test does not leak root changes.
    for mod in (
        config_service,
        common_mod,
        merge_mod,
        preflight_runtime_mod,
        raw_files_mod,
        sample_prompts_mod,
    ):
        monkeypatch.setattr(mod, "CONFIGS_DIR", getattr(mod, "CONFIGS_DIR"), raising=False)

    resolved = config_service.set_configs_root(tmp_path / "hot-swap-configs")

    assert config_service.CONFIGS_DIR == resolved
    assert common_mod.CONFIGS_DIR == resolved
    assert merge_mod.CONFIGS_DIR == resolved
    assert preflight_runtime_mod.CONFIGS_DIR == resolved
    assert raw_files_mod.CONFIGS_DIR == resolved
    assert sample_prompts_mod.CONFIGS_DIR == resolved
    assert Path(sample_prompts_mod.DEFAULT_SAMPLE_PROMPTS_FILE) == resolved / "sample_prompts.txt"


def test_global_settings_frontend_reload_after_configs_root_switch():
    source = GLOBAL_SETTINGS_JS.read_text(encoding="utf-8")

    assert "if (res.requires_reload)" in source
    assert "location.reload()" in source
    assert "正在切换新的配置根目录" in source


def test_global_settings_tooltips_and_help_describe_configs_root_reload():
    listeners = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            EVENT_LISTENERS_JS,
            EVENT_LISTENERS_JS.with_name("event-listeners-setup.js"),
            EVENT_LISTENERS_JS.with_name("beginner-tooltips.js"),
        )
    )
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


def test_tagging_job_retention_is_persisted_clamped_and_applied_to_runtime(tmp_path, monkeypatch):
    settings_file = tmp_path / "configs" / "web-ui-settings.toml"
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    settings_file.write_text('[global]\noutput_root = "output/runs"\n', encoding="utf-8")
    monkeypatch.setattr(settings_service, "ROOT", tmp_path)
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", settings_file)

    saved = settings_service.save_global_settings({"tagging_max_retained_jobs": 12})
    assert saved["tagging_max_retained_jobs"] == 12
    assert toml.loads(settings_file.read_text(encoding="utf-8"))["global"]["tagging_max_retained_jobs"] == 12

    class TaggingStub:
        retained = None

        def set_job_retention(self, value):
            self.retained = value

    service = TaggingStub()
    monkeypatch.setattr(
        settings_routes,
        "save_global_settings",
        lambda _data: {"ok": True, "tagging_max_retained_jobs": 500},
    )
    response = asyncio.run(
        settings_routes.handle_global_settings_put(
            _JsonRequest({"tagging_max_retained_jobs": 999}, app={"tagging_service": service})
        )
    )
    assert response.status == 200
    assert service.retained == 500


def test_dragon_global_settings_exposes_tagging_job_retention_control() -> None:
    source = (ROOT / "web" / "static" / "js" / "dragon-ui" / "pages" / "global-settings.js").read_text(
        encoding="utf-8"
    )
    assert "tagging_max_retained_jobs" in source
    assert "打标任务保留上限" in source
    assert "retained < 1 || retained > 500" in source


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
    assert saved["path_overrides"] == {
        "configs_root": "configs",
        "history_root": "alt-history",
        "queue_root": "alt-queue",
    }
    assert saved["effective_paths"]["history_root"] == "alt-history"
    assert saved["effective_paths"]["queue_root"] == "alt-queue"
    override = toml.loads((tmp_path / ".anima-webui-settings.toml").read_text(encoding="utf-8"))
    assert override["paths"]["history_root"] == "alt-history"
    assert override["paths"]["queue_root"] == "alt-queue"
    assert library_env.get_training_history_root() == (tmp_path / "alt-history").resolve()
    assert library_env.get_training_queue_root() == (tmp_path / "alt-queue").resolve()


def test_global_settings_exposes_raw_and_effective_config_paths(tmp_path, monkeypatch):
    configs_dir = tmp_path / "external-configs"
    settings_file = configs_dir / "web-ui-settings.toml"
    configs_dir.mkdir(parents=True, exist_ok=True)
    settings_file.write_text('[global]\noutput_root = "output/runs"\n', encoding="utf-8")
    (tmp_path / ".anima-webui-settings.toml").write_text(
        '[paths]\nconfigs_root = "external-configs"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(settings_service, "ROOT", tmp_path)
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", settings_file)
    monkeypatch.setattr(settings_service, "get_training_history_root", lambda: configs_dir / "web-training-history")
    monkeypatch.setattr(settings_service, "get_training_queue_root", lambda: configs_dir / "web-training-queue")

    payload = settings_service.get_global_settings()

    assert payload["path_overrides"] == {
        "configs_root": "external-configs",
        "history_root": "",
        "queue_root": "",
    }
    assert payload["effective_paths"]["history_root"] == "external-configs/web-training-history"
    assert payload["effective_paths"]["queue_root"] == "external-configs/web-training-queue"


def test_training_policy_settings_roundtrip_and_clamp(tmp_path, monkeypatch):
    settings_file = tmp_path / "web-ui-settings.toml"
    monkeypatch.setattr(settings_service, "SETTINGS_FILE", settings_file)
    saved = settings_service.save_training_policy(
        {
            "auto_retry": True,
            "max_attempts": 99,
            "retry_backoff_sec": 99999,
            "max_queue_items": 5,
            "max_history_items": 12,
        }
    )
    assert saved["ok"] is True
    assert saved["auto_retry"] is True
    assert saved["max_attempts"] == 10  # clamp
    assert saved["retry_backoff_sec"] == 3600.0
    assert saved["max_queue_items"] == 10  # floor
    assert saved["max_history_items"] == 12
    loaded = settings_service.get_training_policy()
    assert loaded["auto_retry"] is True
    assert loaded["max_attempts"] == 10



def test_set_configs_root_hot_swaps_file_groups_datasets_output_runs(tmp_path, monkeypatch):
    """Broadcast must cover file_groups/datasets/output_runs family modules."""
    from web.services.config import (
        datasets as datasets_mod,
        file_group_runtime as file_group_runtime_mod,
        file_groups as file_groups_mod,
        output_runs as output_runs_mod,
    )

    modules = (
        config_service,
        datasets_mod,
        file_groups_mod,
        file_group_runtime_mod,
        output_runs_mod,
    )
    for mod in modules:
        if hasattr(mod, "CONFIGS_DIR"):
            monkeypatch.setattr(mod, "CONFIGS_DIR", getattr(mod, "CONFIGS_DIR"), raising=False)
        if hasattr(mod, "IMPORTED_CONFIGS_DIR"):
            monkeypatch.setattr(mod, "IMPORTED_CONFIGS_DIR", getattr(mod, "IMPORTED_CONFIGS_DIR"), raising=False)

    resolved = config_service.set_configs_root(tmp_path / "broadcast-more")
    assert config_service.CONFIGS_DIR == resolved
    assert file_groups_mod.CONFIGS_DIR == resolved
    assert file_group_runtime_mod.CONFIGS_DIR == resolved
    assert datasets_mod.CONFIGS_DIR == resolved
    assert output_runs_mod.CONFIGS_DIR == resolved
    if hasattr(file_groups_mod, "IMPORTED_CONFIGS_DIR"):
        assert file_groups_mod.IMPORTED_CONFIGS_DIR == resolved / "imported"
    if hasattr(datasets_mod, "IMPORTED_CONFIGS_DIR"):
        assert datasets_mod.IMPORTED_CONFIGS_DIR == resolved / "imported"
    if hasattr(output_runs_mod, "IMPORTED_CONFIGS_DIR"):
        assert output_runs_mod.IMPORTED_CONFIGS_DIR == resolved / "imported"
