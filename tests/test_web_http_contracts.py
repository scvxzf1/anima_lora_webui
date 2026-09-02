"""Minimal HTTP contract smoke for backend WebUI routes."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from aiohttp import web

from web.routes import config as config_routes
from web.routes import image_test as image_test_routes
from web.routes import preview as preview_routes
from web.routes import settings as settings_routes
from web.routes import training as training_routes


class _FakeTrainingService:
    def get_status_snapshot(self) -> dict[str, Any]:
        return {"ok": True, "status": "idle", "running": False}

    def get_queue_snapshot(self) -> dict[str, Any]:
        return {
            "ok": True,
            "paused": False,
            "failure_policy": "pause",
            "auto_retry": False,
            "max_attempts": 1,
            "retry_backoff_sec": 0,
            "items": [],
            "summary": {
                "total": 0,
                "queued": 0,
                "running": 0,
                "done": 0,
                "error": 0,
                "canceled": 0,
            },
        }


class _FakeRequest:
    def __init__(
        self,
        *,
        app: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
        query: dict[str, str] | None = None,
        match_info: dict[str, str] | None = None,
    ) -> None:
        self.app = app or {}
        self._payload = payload or {}
        self.query = query or {}
        self.match_info = match_info or {}

    async def json(self) -> dict[str, Any]:
        return self._payload


def _json_payload(response: web.Response) -> dict[str, Any]:
    return json.loads(response.text or "{}")


def test_http_training_status_contract():
    app = {"training_service": _FakeTrainingService()}
    response = asyncio.run(training_routes.handle_status(_FakeRequest(app=app)))  # type: ignore[arg-type]
    assert response.status == 200
    payload = _json_payload(response)
    assert payload["ok"] is True
    assert payload["status"] == "idle"
    assert payload["running"] is False


def test_http_training_queue_contract():
    app = {"training_service": _FakeTrainingService()}
    response = asyncio.run(training_routes.handle_queue_status(_FakeRequest(app=app)))  # type: ignore[arg-type]
    assert response.status == 200
    payload = _json_payload(response)
    assert payload["ok"] is True
    assert isinstance(payload["items"], list)
    assert "failure_policy" in payload
    assert "summary" in payload
    assert set(payload["summary"]) >= {"total", "queued", "running", "done", "error", "canceled"}


def test_http_preflight_contract(monkeypatch):
    captured = {}

    def fake_preflight(*args, **kwargs):
        captured.update(kwargs)
        return {
            "ok": True,
            "variant": "lora",
            "preset": "default",
            "methods_subdir": "gui-methods",
            "summary": {"errors": 0, "warnings": 0, "checks": 0},
            "checks": [],
            "errors": [],
            "warnings": [],
        }

    monkeypatch.setattr(
        training_routes,
        "preflight_training_config",
        fake_preflight,
    )
    response = asyncio.run(
        training_routes.handle_preflight(
            _FakeRequest(
                payload={
                    "variant": "lora",
                    "preset": "default",
                    "methods_subdir": "gui-methods",
                    "gpu_whitelist": [0, 1],
                }
            )  # type: ignore[arg-type]
        )
    )
    assert response.status == 200
    payload = _json_payload(response)
    assert payload["ok"] is True
    assert "checks" in payload
    assert "summary" in payload
    assert captured["world_size"] == 2


def test_http_global_settings_contract(monkeypatch):
    monkeypatch.setattr(
        settings_routes,
        "get_global_settings",
        lambda: {
            "ok": True,
            "output_root": "output/runs",
            "configs_root": "configs",
            "history_root": "configs/web-training-history",
            "queue_root": "configs/web-training-queue",
        },
    )
    response = asyncio.run(settings_routes.handle_global_settings_get(_FakeRequest()))  # type: ignore[arg-type]
    assert response.status == 200
    payload = _json_payload(response)
    assert payload["ok"] is True
    assert "output_root" in payload
    assert "configs_root" in payload


class _FakeStopService(_FakeTrainingService):
    def __init__(self) -> None:
        self.stopped = False

    async def stop(self) -> dict[str, Any]:
        self.stopped = True
        return {"ok": True, "message": "stopped"}


def test_http_training_stop_contract():
    svc = _FakeStopService()
    app = {"training_service": svc}
    response = asyncio.run(training_routes.handle_stop(_FakeRequest(app=app)))  # type: ignore[arg-type]
    assert response.status == 200
    payload = _json_payload(response)
    assert payload["ok"] is True
    assert "message" in payload
    assert svc.stopped is True


class _FakeHistoryDeleteService(_FakeTrainingService):
    def __init__(self, *, error: Exception | None = None, result: dict[str, Any] | None = None) -> None:
        self.error = error
        self.result = result or {"ok": True, "deleted_task_ids": ["task-1"]}
        self.seen_task_id: str | None = None

    def delete_history_task(self, task_id: str) -> dict[str, Any]:
        self.seen_task_id = task_id
        if self.error is not None:
            raise self.error
        return self.result


def test_http_history_delete_conflict_409_shape():
    svc = _FakeHistoryDeleteService(error=RuntimeError("当前运行中的任务不能删除"))
    response = asyncio.run(
        training_routes.handle_history_delete(
            _FakeRequest(app={"training_service": svc}, match_info={"task_id": "task-running"})  # type: ignore[arg-type]
        )
    )
    assert response.status == 409
    payload = _json_payload(response)
    assert payload["ok"] is False
    assert "不能删除" in payload["error"]
    assert svc.seen_task_id == "task-running"


def test_http_history_delete_success_shape():
    svc = _FakeHistoryDeleteService(result={"ok": True, "deleted_task_ids": ["task-ok"]})
    response = asyncio.run(
        training_routes.handle_history_delete(
            _FakeRequest(app={"training_service": svc}, match_info={"task_id": "task-ok"})  # type: ignore[arg-type]
        )
    )
    assert response.status == 200
    payload = _json_payload(response)
    assert payload["ok"] is True
    assert payload["deleted_task_ids"] == ["task-ok"]


def test_http_image_test_delete_rejects_path_escape(monkeypatch):
    class _FakeImageTestService:
        pass

    def _raise_escape(*args, **kwargs):
        raise ValueError("只允许删除当前推理预览目录中的图片")

    monkeypatch.setattr(image_test_routes, "delete_preview_images", _raise_escape)
    response = asyncio.run(
        image_test_routes.handle_image_test_images_delete(
            _FakeRequest(
                app={"image_test_service": _FakeImageTestService()},
                payload={"files": ["../../etc/passwd.png"]},
            )  # type: ignore[arg-type]
        )
    )
    assert response.status == 400
    payload = _json_payload(response)
    assert payload["ok"] is False
    assert "只允许删除" in payload["error"] or "路径" in payload["error"]


def test_http_image_test_delete_success_envelope(monkeypatch):
    class _FakeImageTestService:
        pass

    monkeypatch.setattr(
        image_test_routes,
        "delete_preview_images",
        lambda source, files: {
            "ok": True,
            "source": source,
            "directory": "output/runs/demo/samples",
            "deleted": list(files or []),
            "deleted_count": len(files or []),
            "missing": [],
            "missing_count": 0,
            "blocked": [],
            "blocked_count": 0,
            "remaining_total": 0,
        },
    )
    response = asyncio.run(
        image_test_routes.handle_image_test_images_delete(
            _FakeRequest(
                app={"image_test_service": _FakeImageTestService()},
                payload={"files": ["a.png"]},
            )  # type: ignore[arg-type]
        )
    )
    assert response.status == 200
    payload = _json_payload(response)
    assert payload["ok"] is True
    assert payload["source"] == "inference"
    assert payload["deleted"] == ["a.png"]


class _FakePreviewTaskService:
    def __init__(self, task: dict[str, Any]) -> None:
        self.task = task

    def get_history_task_summary(self, task_id: str) -> dict[str, Any]:
        assert task_id == self.task["id"]
        return {"ok": True, "task": self.task}


def test_http_preview_delete_mixed_envelope_and_selected_task_context(monkeypatch):
    seen: dict[str, Any] = {}

    def fake_delete(source, files, **kwargs):
        seen.update({"source": source, "files": files, **kwargs})
        return {
            "ok": False,
            "source": source,
            "directory": "output/runs/task-1/sample",
            "deleted": ["a.png"],
            "deleted_count": 1,
            "missing": ["missing.png"],
            "missing_count": 1,
            "blocked": [{"file": "../blocked.png", "error": "blocked"}],
            "blocked_count": 1,
            "remaining_total": 0,
        }

    monkeypatch.setattr(preview_routes, "delete_preview_images", fake_delete)
    task = {
        "id": "task-1",
        "job": "training",
        "sample_dir": "output/runs/task-1/sample",
    }
    response = asyncio.run(
        preview_routes.handle_preview_images_delete(
            _FakeRequest(
                app={"training_service": _FakePreviewTaskService(task)},
                payload={"source": "training", "files": ["a.png", "missing.png", "../blocked.png"]},
                query={"task_id": "task-1"},
            )  # type: ignore[arg-type]
        )
    )

    assert response.status == 200
    payload = _json_payload(response)
    assert payload["ok"] is False
    assert payload["deleted_count"] == 1
    assert payload["missing_count"] == 1
    assert payload["blocked_count"] == 1
    assert seen["current_task_sample_dir"] == "output/runs/task-1/sample"
    assert seen["allow_latest_fallback"] is False


def test_http_preview_delete_without_task_allows_latest_fallback(monkeypatch):
    seen: dict[str, Any] = {}

    def fake_delete(source, files, **kwargs):
        seen.update({"source": source, "files": files, **kwargs})
        return {
            "ok": True,
            "source": source,
            "directory": "output/tests",
            "deleted": [],
            "deleted_count": 0,
            "missing": list(files),
            "missing_count": len(files),
            "blocked": [],
            "blocked_count": 0,
            "remaining_total": 0,
        }

    monkeypatch.setattr(preview_routes, "delete_preview_images", fake_delete)
    response = asyncio.run(
        preview_routes.handle_preview_images_delete(
            _FakeRequest(payload={"source": "inference", "files": ["missing.png"]})  # type: ignore[arg-type]
        )
    )

    assert response.status == 200
    assert seen["allow_latest_fallback"] is True
    assert seen["current_task_sample_dir"] == ""


def test_http_preview_delete_limit_error_is_400(monkeypatch):
    def raise_limit(*args, **kwargs):
        raise ValueError("一次最多删除 500 张图片")

    monkeypatch.setattr(preview_routes, "delete_preview_images", raise_limit)
    response = asyncio.run(
        preview_routes.handle_preview_images_delete(
            _FakeRequest(payload={"source": "inference", "files": [f"{index}.png" for index in range(501)]})  # type: ignore[arg-type]
        )
    )

    assert response.status == 400
    payload = _json_payload(response)
    assert payload == {"ok": False, "error": "一次最多删除 500 张图片"}


def test_http_preview_delete_missing_directory_is_404(monkeypatch):
    def raise_missing(*args, **kwargs):
        raise FileNotFoundError("预览图目录不存在")

    monkeypatch.setattr(preview_routes, "delete_preview_images", raise_missing)
    response = asyncio.run(
        preview_routes.handle_preview_images_delete(
            _FakeRequest(payload={"source": "custom", "files": ["a.png"]})  # type: ignore[arg-type]
        )
    )

    assert response.status == 404
    payload = _json_payload(response)
    assert payload == {"ok": False, "error": "预览图目录不存在"}


def test_http_image_test_delete_service_missing_is_503():
    response = asyncio.run(
        image_test_routes.handle_image_test_images_delete(_FakeRequest())  # type: ignore[arg-type]
    )

    assert response.status == 503
    assert _json_payload(response)["ok"] is False


def test_http_config_methods_envelope(monkeypatch):
    monkeypatch.setattr(config_routes, "list_methods", lambda: ["lora", "hydralora", "spd"])
    response = asyncio.run(config_routes.handle_methods(_FakeRequest()))  # type: ignore[arg-type]
    assert response.status == 200
    payload = _json_payload(response)
    assert payload.get("ok") is True
    assert isinstance(payload.get("items"), list)
    assert "lora" in payload["items"]
    assert "spd" in payload["items"]


def test_http_config_variants_envelope(monkeypatch):
    monkeypatch.setattr(config_routes, "list_variants", lambda method: [method, f"{method}-8gb"])
    response = asyncio.run(
        config_routes.handle_variants(_FakeRequest(match_info={"method": "lora"}))  # type: ignore[arg-type]
    )
    assert response.status == 200
    payload = _json_payload(response)
    assert payload.get("ok") is True
    assert payload.get("items") == ["lora", "lora-8gb"]


def test_http_config_presets_envelope(monkeypatch):
    monkeypatch.setattr(config_routes, "list_presets", lambda: ["default", "low_vram"])
    response = asyncio.run(config_routes.handle_presets(_FakeRequest()))  # type: ignore[arg-type]
    assert response.status == 200
    payload = _json_payload(response)
    assert payload.get("ok") is True
    assert "default" in payload["items"]


def test_http_config_model_families_exposes_pipeline_capabilities():
    response = asyncio.run(config_routes.handle_model_families(_FakeRequest()))  # type: ignore[arg-type]
    assert response.status == 200
    payload = _json_payload(response)
    items = {item["name"]: item for item in payload["items"]}

    assert payload["ok"] is True
    assert set(items) == {"anima", "krea2_raw", "z_image"}
    assert items["anima"]["pipeline_parallel"]["known_num_blocks"] == [28, 40]
    assert items["krea2_raw"]["pipeline_parallel"]["runtime_available"] is False
    assert items["z_image"]["pipeline_parallel"]["block_container"] == "layers"


def test_http_config_merged_envelope(monkeypatch):
    monkeypatch.setattr(
        config_routes,
        "load_merged_config",
        lambda variant, preset, methods_subdir: {
            "variant": variant,
            "preset": preset,
            "methods_subdir": methods_subdir,
            "max_train_steps": 100,
            "network_module": "networks.lora_anima",
        },
    )
    response = asyncio.run(
        config_routes.handle_merged(
            _FakeRequest(
                query={
                    "variant": "lora",
                    "preset": "default",
                    "methods_subdir": "gui-methods",
                }
            )  # type: ignore[arg-type]
        )
    )
    assert response.status == 200
    payload = _json_payload(response)
    assert payload["variant"] == "lora"
    assert payload["preset"] == "default"
    assert payload["methods_subdir"] == "gui-methods"
    assert "max_train_steps" in payload
    assert "error" not in payload


def test_http_config_merged_error_envelope(monkeypatch):
    def _boom(*args, **kwargs):
        raise FileNotFoundError("missing config")

    monkeypatch.setattr(config_routes, "load_merged_config", _boom)
    response = asyncio.run(
        config_routes.handle_merged(
            _FakeRequest(query={"variant": "missing", "preset": "default"})  # type: ignore[arg-type]
        )
    )
    assert response.status == 400
    payload = _json_payload(response)
    assert payload.get("ok") is False
    assert "error" in payload
    assert "missing" in payload["error"]


def test_http_config_raw_envelope(monkeypatch):
    monkeypatch.setattr(config_routes, "load_raw_file", lambda path: 'output_name = "demo"\n')
    monkeypatch.setattr(
        config_routes,
        "get_config_file_meta",
        lambda path: {
            "path": path,
            "locked": False,
            "trainable": True,
            "group_id": "custom",
            "group_label": "自定义",
        },
    )
    response = asyncio.run(
        config_routes.handle_raw_get(
            _FakeRequest(query={"file": "gui-methods/lora.toml"})  # type: ignore[arg-type]
        )
    )
    assert response.status == 200
    payload = _json_payload(response)
    assert payload["file"] == "gui-methods/lora.toml"
    assert "output_name" in payload["content"]
    assert isinstance(payload["meta"], dict)
    assert "locked" in payload["meta"]


def test_http_config_raw_requires_file():
    response = asyncio.run(config_routes.handle_raw_get(_FakeRequest(query={})))  # type: ignore[arg-type]
    assert response.status == 400
    payload = _json_payload(response)
    assert "error" in payload
