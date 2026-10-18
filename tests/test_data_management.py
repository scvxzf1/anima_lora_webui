"""Bounded, isolated tests for the data-management deletion contract."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from aiohttp import web

from web.routes.data_management import handle_roots
from web.services import data_management_service as module
from web.services.data_management_service import DataManagementError, DataManagementService


def _service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[DataManagementService, Path, Path]:
    project = tmp_path / "project"
    runs = tmp_path / "runs"
    post = project / "post_image_dataset"
    project.mkdir()
    runs.mkdir()
    post.mkdir()
    monkeypatch.setattr(DataManagementService, "_project_root", staticmethod(lambda: project.resolve()))
    monkeypatch.setattr(module.settings_service, "resolve_output_root", lambda: runs)
    monkeypatch.setattr(module, "get_configs_root", lambda: project / "configs")
    monkeypatch.setattr(module, "get_training_history_root", lambda: project / "history")
    monkeypatch.setattr(module, "get_training_queue_root", lambda: project / "queue")
    app = web.Application()
    app["training_service"] = SimpleNamespace(status="idle", current_task_dir=None, current_output_dir="", current_runtime_info={}, _queue={"items": []})
    app["image_test_service"] = None
    app["tagging_service"] = None
    return DataManagementService(app), runs, post


def _runtime_dir(root: Path, name: str = "run-a") -> Path:
    path = root / name
    path.mkdir()
    for child in ("model_cache", "dataset_cache", "training_output"):
        (path / child).mkdir()
    (path / "config.runtime.toml").write_text("[runtime]\n", encoding="utf-8")
    (path / "payload.bin").write_bytes(b"payload")
    return path


def test_runtime_preview_and_permanent_cleanup_are_confirmed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, runs, _ = _service(tmp_path, monkeypatch)
    target = _runtime_dir(runs)

    preview = service.preview_cleanup("training-runs", ["run-a"])
    assert preview["blocked"] == []
    assert preview["items"][0]["relative_path"] == "run-a"

    result = service.cleanup("training-runs", ["run-a"], ["run-a"], preview["plan_revision"], True)
    assert result["deleted"] == ["run-a"]
    assert not target.exists()


def test_cleanup_rejects_non_runtime_and_symlink_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, runs, _ = _service(tmp_path, monkeypatch)
    (runs / "ordinary").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (runs / "escape").symlink_to(outside, target_is_directory=True)

    for name in ("ordinary", "escape"):
        preview = service.preview_cleanup("training-runs", [name])
        assert preview["blocked"]
    assert outside.exists()


def test_generated_data_allows_nested_files_and_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, _, post = _service(tmp_path, monkeypatch)
    (post / "resized").mkdir()
    nested = post / "masks" / "deep"
    nested.mkdir(parents=True)
    (nested / "payload.bin").write_bytes(b"payload")

    allowed = service.preview_cleanup("postprocessed-dataset", ["resized"])
    assert allowed["blocked"] == []
    nested_file = service.preview_cleanup("postprocessed-dataset", ["masks/deep/payload.bin"])
    assert nested_file["blocked"] == []
    nested_dir = service.preview_cleanup("postprocessed-dataset", ["masks/deep"])
    assert nested_dir["blocked"] == []
    with pytest.raises(DataManagementError, match="不允许清理"):
        service.preview_cleanup("models", ["anything"])


def test_plan_revision_uses_full_relative_path_for_same_named_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, _, post = _service(tmp_path, monkeypatch)
    for parent in (post / "first", post / "second"):
        parent.mkdir()
        (parent / "same.bin").write_bytes(b"same")

    first = service.preview_cleanup("postprocessed-dataset", ["first/same.bin"])
    second = service.preview_cleanup("postprocessed-dataset", ["second/same.bin"])

    assert first["plan_revision"] != second["plan_revision"]
    assert "target:first/same.bin:" in first["plan_revision"]
    assert "target:second/same.bin:" in second["plan_revision"]


def test_directory_changes_invalidate_previous_cleanup_preview(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, _, post = _service(tmp_path, monkeypatch)
    target = post / "resized"
    target.mkdir()
    payload = target / "payload.bin"
    payload.write_bytes(b"before")

    preview = service.preview_cleanup("postprocessed-dataset", ["resized"])
    payload.write_bytes(b"after")
    with pytest.raises(DataManagementError, match="重新预览"):
        service.cleanup("postprocessed-dataset", ["resized"], ["resized"], preview["plan_revision"], True)

    refreshed = service.preview_cleanup("postprocessed-dataset", ["resized"])
    (target / "new.bin").write_bytes(b"new")
    with pytest.raises(DataManagementError, match="重新预览"):
        service.cleanup("postprocessed-dataset", ["resized"], ["resized"], refreshed["plan_revision"], True)


def test_scan_is_recursive_complete_and_cached_until_refresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, _, post = _service(tmp_path, monkeypatch)
    (post / "a").mkdir()
    (post / "a" / "one.bin").write_bytes(b"123")
    (post / "a" / "b").mkdir()
    (post / "a" / "b" / "two.bin").write_bytes(b"12345")

    first = service.list_roots(refresh=True)
    root = next(item for item in first["roots"] if item["id"] == "postprocessed-dataset")
    entry = next(item for item in root["entries"] if item["name"] == "a")
    assert (root["file_count"], root["directory_count"], root["size_bytes"]) == (2, 2, 8)
    assert (entry["file_count"], entry["directory_count"], entry["size_bytes"]) == (2, 1, 8)

    (post / "a" / "new.bin").write_bytes(b"new")
    cached = service.list_roots()
    cached_root = next(item for item in cached["roots"] if item["id"] == "postprocessed-dataset")
    assert cached_root["file_count"] == 2
    refreshed = service.list_roots(refresh=True)
    refreshed_root = next(item for item in refreshed["roots"] if item["id"] == "postprocessed-dataset")
    assert (refreshed_root["file_count"], refreshed_root["size_bytes"]) == (3, 11)


def test_project_postprocessed_root_symlink_is_not_a_delete_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, _, _ = _service(tmp_path, monkeypatch)
    external = tmp_path / "external-post"
    external.mkdir()
    (external / "resized").mkdir()
    (tmp_path / "project" / "post_image_dataset").rmdir()
    (tmp_path / "project" / "post_image_dataset").symlink_to(external, target_is_directory=True)

    with pytest.raises(DataManagementError, match="符号链接"):
        service.preview_cleanup("postprocessed-dataset", ["resized"])


def test_active_training_blocks_postprocessed_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, _, post = _service(tmp_path, monkeypatch)
    (post / "resized").mkdir()
    service.app["training_service"].status = "running"

    preview = service.preview_cleanup("postprocessed-dataset", ["resized"])
    assert preview["blocked"]
    assert "训练或预处理" in preview["blocked"][0]["reason"]


def test_project_output_is_cleanup_root_and_accepts_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, _, _ = _service(tmp_path, monkeypatch)
    output = tmp_path / "project" / "output"
    output.mkdir()
    target = output / "nested" / "result.bin"
    target.parent.mkdir()
    target.write_bytes(b"result")

    preview = service.preview_cleanup("project-output", ["nested/result.bin"])
    assert preview["blocked"] == []
    result = service.cleanup("project-output", ["nested/result.bin"], ["nested/result.bin"], preview["plan_revision"], True)
    assert result["deleted"] == ["nested/result.bin"]
    assert not target.exists()


def test_list_roots_builds_specs_once_not_per_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, _, post = _service(tmp_path, monkeypatch)
    for name in ("one", "two", "three"):
        (post / name).write_bytes(name.encode())
    original = service._root_specs
    calls = 0

    def counted_specs():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(service, "_root_specs", counted_specs)
    service.list_roots(refresh=True)
    assert calls == 1


def test_overlapping_roots_share_one_physical_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, _, _ = _service(tmp_path, monkeypatch)
    output = tmp_path / "project" / "output"
    output.mkdir()
    _runtime_dir(output, "run-a")
    (output / "artifact.bin").write_bytes(b"x")
    monkeypatch.setattr(module.settings_service, "resolve_output_root", lambda: output)
    original = service._scan_uncached
    scans: list[Path] = []

    def counted_scan(base: Path, request_cache: dict):
        scans.append(base.resolve())
        return original(base, request_cache)

    monkeypatch.setattr(service, "_scan_uncached", counted_scan)
    roots = service.list_roots(refresh=True)["roots"]
    by_id = {root["id"]: root for root in roots}
    assert scans.count(output.resolve()) == 1
    assert by_id["project-output"]["file_count"] == by_id["training-runs"]["file_count"]
    assert next(item for item in by_id["training-runs"]["entries"] if item["name"] == "run-a")["cleanup_allowed"]


def test_entry_guards_collect_active_blockers_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, _, post = _service(tmp_path, monkeypatch)
    for name in ("a", "b", "c"):
        (post / name).mkdir()
    calls: list[tuple[list[Path], bool]] = []

    def blockers(targets: list[Path], root_id: str, *, inspect_unknown: bool = True):
        calls.append((targets, inspect_unknown))
        return []

    monkeypatch.setattr(service, "_active_blockers", blockers)
    service.list_entries("postprocessed-dataset", "", refresh=True)
    assert len(calls) == 1
    assert len(calls[0][0]) == 3
    assert calls[0][1] is True


def test_unknown_active_tagging_service_blocks_cleanup_listing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, _, post = _service(tmp_path, monkeypatch)
    (post / "resized").mkdir()
    service.app["tagging_service"] = SimpleNamespace()

    listing = service.list_entries("postprocessed-dataset", "", refresh=True)

    entry = next(item for item in listing["entries"] if item["name"] == "resized")
    assert entry["cleanup_allowed"] is False
    assert "打标任务正在运行" in entry["cleanup_reason"]


def test_nested_cleanup_does_not_refresh_root_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, _, post = _service(tmp_path, monkeypatch)
    target = post / "nested" / "leaf.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"payload")
    preview = service.preview_cleanup("postprocessed-dataset", ["nested/leaf.bin"])

    def unexpected_scan(*_args, **_kwargs):
        raise AssertionError("cleanup should not refresh the full root")

    monkeypatch.setattr(service, "_scan", unexpected_scan)
    result = service.cleanup(
        "postprocessed-dataset", ["nested/leaf.bin"], ["nested/leaf.bin"], preview["plan_revision"], True,
    )
    assert result["deleted"] == ["nested/leaf.bin"]
    assert not target.exists()


def test_root_metadata_mode_skips_the_tree_walk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, _, _ = _service(tmp_path, monkeypatch)

    def unexpected_scan(*_args, **_kwargs):
        raise AssertionError("metadata mode must not scan directory trees")

    monkeypatch.setattr(service, "_scan", unexpected_scan)
    roots = service.list_roots(include_scan=False)["roots"]

    assert [root["id"] for root in roots][:3] == ["project-output", "training-runs", "postprocessed-dataset"]
    for root in roots:
        assert root["label"]
        assert root["access"] in {"cleanup", "read_only"}
        assert root["size_state"] == "pending"
        assert root["size_bytes"] is None
        assert (root["file_count"], root["directory_count"]) == (None, None)
        assert root["entries"] == []


def test_roots_route_serves_metadata_without_scanning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, _, _ = _service(tmp_path, monkeypatch)

    def unexpected_scan(*_args, **_kwargs):
        raise AssertionError("metadata request must not scan directory trees")

    monkeypatch.setattr(service, "_scan", unexpected_scan)
    request = SimpleNamespace(app=service.app, query={"mode": "meta"})
    response = asyncio.run(handle_roots(request))
    payload = json.loads(response.body)

    assert payload["ok"] is True
    assert all(root["size_state"] == "pending" for root in payload["roots"])


def test_scan_cache_keys_are_physical_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, _, post = _service(tmp_path, monkeypatch)
    (post / "a" / "b").mkdir(parents=True)
    (post / "a" / "b" / "payload.bin").write_bytes(b"payload")

    service.list_roots(refresh=True)
    prefix = str(post.resolve())
    keys = [key for key in DataManagementService._scan_cache if key.startswith(prefix)]

    assert any(key.endswith("b") for key in keys)
    assert all(str(Path(key).resolve()) == key for key in keys)


def test_overlapping_child_scan_waits_for_parent_instead_of_rewalking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    service, _, _ = _service(tmp_path, monkeypatch)
    output = tmp_path / "project" / "output"
    (output / "runs" / "run-a").mkdir(parents=True)
    (output / "runs" / "run-a" / "payload.bin").write_bytes(b"payload")
    monkeypatch.setattr(module.settings_service, "resolve_output_root", lambda: output)

    scans: list[Path] = []
    entered = threading.Event()
    release = threading.Event()
    original = service._scan_uncached

    def slow_scan(base: Path, request_cache: dict):
        scans.append(base)
        entered.set()
        release.wait(5)
        return original(base, request_cache)

    monkeypatch.setattr(service, "_scan_uncached", slow_scan)
    parent = threading.Thread(target=lambda: service._physical_scan(output, refresh=False))
    parent.start()
    assert entered.wait(5)
    child = threading.Thread(target=lambda: service._physical_scan(output / "runs", refresh=False))
    child.start()
    time.sleep(0.05)
    release.set()
    parent.join(5)
    child.join(5)

    assert not parent.is_alive() and not child.is_alive()
    assert [path for path in scans] == [output]


def test_cache_invalidation_covers_ancestors_and_descendants_only(tmp_path: Path):
    root = tmp_path / "output"
    target = root / "runs" / "run-a"
    sibling = tmp_path / "output-other"
    cache = DataManagementService._scan_cache
    cache.clear()
    try:
        for path in (root, target, sibling):
            cache[str(path.resolve())] = {"entries": []}

        DataManagementService._invalidate_cache(target)

        remaining = set(DataManagementService._scan_cache)
        assert str(target.resolve()) not in remaining
        assert str(root.resolve()) not in remaining
        assert str(sibling.resolve()) in remaining
    finally:
        DataManagementService._scan_cache.clear()
