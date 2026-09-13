import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

from web.routes.training_artifacts import handle_history_artifact_manifest
from web.services.training.history_artifacts import history_artifact_manifest
from web.services.training.common import HISTORY_ARTIFACT_FILES, HISTORY_RUNTIME_ARTIFACT_FIELDS


def test_manifest_only_uses_validated_whitelist_paths(tmp_path):
    snapshot = tmp_path / "snapshot.toml"
    snapshot.write_text("rank=16", encoding="utf-8")
    service = Mock()

    def resolve(task_id, key):
        assert task_id == "task"
        if key == "config-snapshot":
            return snapshot
        if key == "runtime-config":
            raise ValueError("outside root")
        if key == "metrics":
            raise PermissionError("unreadable")
        raise FileNotFoundError(key)

    service.get_history_artifact_path.side_effect = resolve
    manifest = history_artifact_manifest(service, "task")
    entries = {entry["key"]: entry for entry in manifest["artifacts"]}
    assert set(entries) == set(HISTORY_ARTIFACT_FILES) | set(HISTORY_RUNTIME_ARTIFACT_FIELDS)
    assert entries["config-snapshot"] == {"key": "config-snapshot", "state": "available", "name": "snapshot.toml", "size_bytes": 7}
    assert entries["runtime-config"]["state"] == "blocked"
    assert entries["metrics"]["state"] == "unreadable"
    assert entries["logs"]["state"] == "missing"
    assert str(tmp_path) not in str(manifest)
    service.get_history_task_summary.assert_called_once_with("task")


def test_missing_task_returns_404_without_checking_files():
    service = Mock()
    service.get_history_task_summary.side_effect = FileNotFoundError("任务不存在")
    request = SimpleNamespace(app={"training_service": service}, match_info={"task_id": "missing"})
    assert asyncio.run(handle_history_artifact_manifest(request)).status == 404
    service.get_history_artifact_path.assert_not_called()
