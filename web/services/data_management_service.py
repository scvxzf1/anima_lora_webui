"""Bounded data inspection and explicitly-scoped generated-data cleanup.

This service deliberately does not use ``settings_paths._effective_roots`` for
deletion authorization.  The read-only roots and the two cleanup policies are
fixed here so adding a configured path cannot silently add a new delete target.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
import shutil
import stat
import threading
from typing import Any

from library.env import (
    anima_home,
    get_configs_root,
    get_training_history_root,
    get_training_queue_root,
)
from web.services import settings_service
from web.services.data_cleanup_gate import data_cleanup_lease

MAX_CLEANUP_ITEMS = 32
ACTIVE_TRAINING_STATES = frozenset({"running", "preprocessing", "compiling", "stopping"})


class DataManagementError(Exception):
    """An expected API error with a safe, user-facing message."""

    def __init__(self, message: str, status: int = 400, *, extra: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.extra = extra or {}


class DataManagementService:
    _cleanup_lock = threading.Lock()
    _cache_lock = threading.Lock()
    _scan_cache: dict[str, dict[str, Any]] = {}
    _scan_inflight: dict[str, threading.Event] = {}

    def __init__(self, app: Any):
        self.app = app

    @staticmethod
    def _project_root() -> Path:
        return Path(anima_home()).resolve()

    @staticmethod
    def _display_path(path: Path) -> str:
        try:
            return settings_service.display_path(path)
        except (OSError, RuntimeError, ValueError):
            return str(path)

    def _root_specs(self) -> dict[str, dict[str, Any]]:
        project_root = self._project_root()
        values: dict[str, tuple[str, Path | None, str]] = {
            "project-output": ("项目输出", project_root / "output", "cleanup"),
            "training-runs": ("训练运行目录", None, "cleanup"),
            "postprocessed-dataset": ("后处理数据集", project_root / "post_image_dataset", "cleanup"),
            "models": ("模型", project_root / "models", "read_only"),
            "image-dataset": ("原始图像数据集", project_root / "image_dataset", "read_only"),
            "history": ("训练历史", None, "read_only"),
            "queue": ("训练队列", None, "read_only"),
            "configs": ("配置", None, "read_only"),
        }
        resolvers = {
            "training-runs": self._resolve_output_root,
            "history": get_training_history_root,
            "queue": get_training_queue_root,
            "configs": get_configs_root,
        }
        specs: dict[str, dict[str, Any]] = {}
        for root_id, (label, path, access) in values.items():
            if path is None:
                try:
                    path = Path(resolvers[root_id]()).resolve()
                except (OSError, RuntimeError, ValueError):
                    path = None
            else:
                # Keep fixed project roots lexical.  Resolving here would turn a
                # project-local symlink into an external delete root.
                path = Path(path)
            specs[root_id] = {
                "id": root_id,
                "label": label,
                "path": path,
                "access": access,
                "fixed_project_root": root_id not in resolvers,
            }
        return specs

    @staticmethod
    def _resolve_output_root() -> Path:
        return Path(settings_service.resolve_output_root()).resolve()

    def _spec(self, root_id: Any, specs: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
        specs = specs if specs is not None else self._root_specs()
        if not isinstance(root_id, str) or root_id not in specs:
            raise DataManagementError("未知的数据根目录", 404)
        return specs[root_id]

    @staticmethod
    def _safe_relative(value: Any, *, allow_root: bool = True) -> tuple[str, ...]:
        if not isinstance(value, str) or len(value) > 1024 or "\x00" in value:
            raise DataManagementError("路径必须是相对路径", 403)
        raw = value.strip()
        if not raw or raw == ".":
            if allow_root:
                return ()
            raise DataManagementError("不能选择根目录", 403)
        if "\\" in raw or raw.startswith("/"):
            raise DataManagementError("路径必须是相对路径", 403)
        windows = PureWindowsPath(raw)
        if windows.drive or raw.startswith(("\\", "//")):
            raise DataManagementError("路径必须是相对路径", 403)
        parts = tuple(part for part in Path(raw).parts if part not in {""})
        if not parts or any(part in {".", ".."} for part in parts):
            raise DataManagementError("路径不能包含 . 或 ..", 403)
        return parts

    @staticmethod
    def _has_symlink_component(root: Path, path: Path) -> bool:
        try:
            relative = path.relative_to(root)
        except ValueError:
            return True
        current = root
        for part in relative.parts:
            current /= part
            try:
                if current.is_symlink():
                    return True
            except OSError:
                return True
        return False

    def _resolve_path(self, spec: dict[str, Any], relative: Any, *, allow_root: bool = True) -> tuple[Path, str]:
        root = spec.get("path")
        if not isinstance(root, Path):
            raise DataManagementError("该数据根目录暂时不可用", 503)
        parts = self._safe_relative(relative, allow_root=allow_root)
        self._validate_root(spec)
        root = root.resolve()
        candidate = root.joinpath(*parts)
        try:
            resolved = candidate.resolve(strict=False)
            if not resolved.is_relative_to(root) or self._has_symlink_component(root, candidate):
                raise DataManagementError("路径不在允许的数据根目录内", 403)
        except OSError as exc:
            raise DataManagementError("路径检查暂时不可用", 503) from exc
        return candidate, "/".join(parts)

    def _validate_root(self, spec: dict[str, Any]) -> None:
        root = spec.get("path")
        if not isinstance(root, Path):
            raise DataManagementError("该数据根目录暂时不可用", 503)
        if not spec.get("fixed_project_root"):
            return
        project_root = self._project_root()
        try:
            if root.resolve() != root or self._has_symlink_component(project_root, root):
                raise DataManagementError("固定数据根目录不能是符号链接", 503)
        except OSError as exc:
            raise DataManagementError("数据根目录状态暂时不可用", 503) from exc

    @staticmethod
    def _modified_at(value: float) -> str | None:
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None

    def _entry(
        self, path: Path, root: Path, spec: dict[str, Any], *, direct: bool, info: os.stat_result | None = None,
    ) -> dict[str, Any]:
        relative = path.relative_to(root).as_posix()
        item: dict[str, Any] = {
            "relative_path": relative,
            "name": path.name,
            "kind": "missing",
            "size_bytes": None,
            "file_count": 0,
            "directory_count": 0,
            "modified_at": None,
            "cleanup_allowed": False,
            "cleanup_reason": "仅允许查看",
            "status": "missing",
        }
        if info is None:
            try:
                info = path.lstat()
            except FileNotFoundError:
                return item
            except OSError:
                item.update(status="unavailable", cleanup_reason="条目状态暂时不可用")
                return item
        mode = info.st_mode
        if stat.S_ISLNK(mode):
            kind = "symlink"
        elif stat.S_ISREG(mode):
            kind = "file"
        elif stat.S_ISDIR(mode):
            kind = "directory"
        else:
            kind = "other"
        item.update(kind=kind, size_bytes=info.st_size if kind == "file" else None,
                    modified_at=self._modified_at(info.st_mtime), status="ready")
        root_id = spec["id"]
        if not direct and root_id not in {"project-output", "postprocessed-dataset"}:
            item["cleanup_reason"] = "只允许清理固定根目录的直接子项"
            return item
        access = spec["access"]
        if access != "cleanup":
            item["cleanup_reason"] = "该根目录仅允许查看"
        elif kind == "symlink":
            item["cleanup_reason"] = "不允许清理符号链接"
        elif kind == "other":
            item["cleanup_reason"] = "不支持清理特殊文件"
        elif root_id in {"project-output", "postprocessed-dataset"}:
            item.update(cleanup_allowed=True, cleanup_reason="项目生成数据可永久删除")
        elif root_id == "training-runs":
            if kind != "directory":
                item["cleanup_reason"] = "训练运行目录只允许清理目录"
            elif self._is_runtime_dir(path):
                item.update(cleanup_allowed=True, cleanup_reason="WebUI runtime 目录可永久删除")
            else:
                item["cleanup_reason"] = "缺少 WebUI runtime 标记"
        else:
            item["cleanup_reason"] = "该根目录仅允许查看"
        return item

    @staticmethod
    def _is_runtime_dir(path: Path) -> bool:
        try:
            from web.services.training.runtime_state import _is_web_runtime_dir

            marker_names = ("config.runtime.toml", "run.meta.json", "model_cache", "dataset_cache", "training_output")
            if any((path / name).is_symlink() for name in marker_names):
                return False
            return bool(_is_web_runtime_dir(path))
        except (OSError, RuntimeError, ValueError, ImportError):
            return False

    @staticmethod
    def _empty_scan() -> dict[str, Any]:
        return {"entries": [], "file_count": 0, "directory_count": 0, "size_bytes": None,
                "size_state": "unavailable", "truncated": False}

    @staticmethod
    def _cache_key(path: Path) -> str:
        return str(path.resolve())

    @classmethod
    def _invalidate_cache(cls, root_path: Path | None) -> None:
        if not isinstance(root_path, Path):
            return
        # Cache keys are absolute resolved strings, so overlap is a plain string
        # prefix check.  Rebuilding Path objects per key cost half a second per
        # deleted target on a warm cache with thousands of directory scans.
        prefix = cls._cache_key(root_path)
        nested_prefix = prefix + os.sep
        with cls._cache_lock:
            cls._scan_cache = {
                key: value for key, value in cls._scan_cache.items()
                if not (
                    key == prefix
                    or key.startswith(nested_prefix)
                    or prefix.startswith(key + os.sep)
                )
            }

    @staticmethod
    def _raw_entry(child: os.DirEntry[str], relative: str) -> dict[str, Any]:
        item = {"relative_path": relative, "name": child.name, "kind": "unavailable",
                "size_bytes": None, "file_count": 0, "directory_count": 0,
                "modified_at": None, "status": "unavailable"}
        try:
            info = child.stat(follow_symlinks=False)
        except OSError:
            return item
        mode = info.st_mode
        kind = "symlink" if stat.S_ISLNK(mode) else "file" if stat.S_ISREG(mode) else "directory" if stat.S_ISDIR(mode) else "other"
        item.update(kind=kind, size_bytes=info.st_size if kind == "file" else None,
                    modified_at=DataManagementService._modified_at(info.st_mtime), status="ready")
        if kind == "file":
            item["file_count"] = 1
        return item

    def _scan_uncached(self, base: Path, request_cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
        collected: dict[str, dict[str, Any]] = {}

        def walk(path: str, key: str) -> dict[str, Any]:
            reused = request_cache.get(key)
            if reused is not None:
                return reused
            entries: list[dict[str, Any]] = []
            size_bytes = file_count = directory_count = 0
            complete = True
            try:
                with os.scandir(path) as children:
                    for child in children:
                        item = self._raw_entry(child, child.name)
                        if item["kind"] == "directory":
                            # Only real directories are recursed into (symlinks
                            # keep their own kind), so a child key is the parent
                            # key plus one name component.  That keeps keys
                            # physically identical without a resolve() per
                            # directory, which cost ~0.2s on a 5k-directory tree.
                            child_scan = walk(child.path, os.path.join(key, child.name))
                            item.update(size_bytes=child_scan["size_bytes"], file_count=child_scan["file_count"],
                                        directory_count=child_scan["directory_count"], size_state=child_scan["size_state"])
                            size_bytes += child_scan["size_bytes"]
                            file_count += child_scan["file_count"]
                            directory_count += child_scan["directory_count"] + 1
                            complete = complete and child_scan["size_state"] == "ready"
                        elif item["kind"] == "file":
                            size_bytes += int(item["size_bytes"] or 0)
                            file_count += 1
                        entries.append(item)
            except OSError:
                complete = False
            entries.sort(key=lambda item: (str(item["name"]).casefold(), str(item["name"])))
            scan = {"entries": entries, "file_count": file_count, "directory_count": directory_count,
                    "size_bytes": size_bytes, "size_state": "ready" if complete else "partial", "truncated": False}
            collected[key] = scan
            request_cache[key] = scan
            return scan

        scan = walk(str(base), self._cache_key(base))
        with self._cache_lock:
            self._scan_cache.update(collected)
        return scan

    def _ancestor_inflight_locked(self, key: str) -> threading.Event | None:
        """Return the in-flight scan of a parent directory, if there is one.

        Caller holds the cache lock.  No scan ever waits on a descendant, so this
        cannot deadlock; it only prevents two overlapping roots from walking the
        same subtree at the same time.
        """
        nested = key + os.sep
        best: tuple[int, threading.Event] | None = None
        for other, event in self._scan_inflight.items():
            if len(other) < len(key) and nested.startswith(other + os.sep):
                if best is None or len(other) > best[0]:
                    best = (len(other), event)
        return best[1] if best is not None else None

    def _physical_scan(
        self, base: Path, *, refresh: bool, request_cache: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        key = self._cache_key(base)
        request_cache = request_cache if request_cache is not None else {}
        if key in request_cache:
            return request_cache[key]
        while True:
            started: threading.Event | None = None
            wait_for: threading.Event | None = None
            with self._cache_lock:
                if not refresh and key in self._scan_cache:
                    return self._scan_cache[key]
                event = self._scan_inflight.get(key)
                if event is None:
                    # A scan started from an overlapping root already walks this
                    # subtree.  Wait for it and reuse its cache instead of
                    # walking the same files twice in parallel (the UI asks for
                    # the output root and its training-runs child at once).
                    wait_for = self._ancestor_inflight_locked(key)
                    if wait_for is None:
                        started = threading.Event()
                        self._scan_inflight[key] = started
                else:
                    wait_for = event
            if started is not None:
                break
            wait_for.wait()
            refresh = False
        try:
            scan = self._scan_uncached(base, request_cache)
            request_cache[key] = scan
            return scan
        finally:
            with self._cache_lock:
                self._scan_inflight.pop(key, None)
                started.set()


    def _apply_entry_policy(self, item: dict[str, Any], path: Path, spec: dict[str, Any], *, direct: bool) -> None:
        item.update(cleanup_allowed=False, cleanup_reason="仅允许查看")
        kind = item["kind"]
        root_id = spec["id"]
        if item["status"] != "ready":
            item["cleanup_reason"] = "条目状态暂时不可用"
        elif not direct and root_id not in {"project-output", "postprocessed-dataset"}:
            item["cleanup_reason"] = "只允许清理固定根目录的直接子项"
        elif spec["access"] != "cleanup":
            item["cleanup_reason"] = "该根目录仅允许查看"
        elif kind == "symlink":
            item["cleanup_reason"] = "不允许清理符号链接"
        elif kind == "other":
            item["cleanup_reason"] = "不支持清理特殊文件"
        elif root_id in {"project-output", "postprocessed-dataset"}:
            item.update(cleanup_allowed=True, cleanup_reason="项目生成数据可永久删除")
        elif root_id == "training-runs" and kind != "directory":
            item["cleanup_reason"] = "训练运行目录只允许清理目录"
        elif root_id == "training-runs" and self._is_runtime_dir(path):
            item.update(cleanup_allowed=True, cleanup_reason="WebUI runtime 目录可永久删除")
        elif root_id == "training-runs":
            item["cleanup_reason"] = "缺少 WebUI runtime 标记"

    def _scan(
        self, spec: dict[str, Any], base: Path, relative: str, *, refresh: bool = False,
        request_cache: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        self._validate_root(spec)
        try:
            info = base.lstat()
        except FileNotFoundError:
            if relative:
                raise DataManagementError("目录不存在", 404)
            return self._empty_scan()
        except OSError as exc:
            raise DataManagementError("目录状态暂时不可用", 503) from exc
        if not stat.S_ISDIR(info.st_mode):
            raise DataManagementError("path 必须指向目录", 400)
        raw = self._physical_scan(base, refresh=refresh, request_cache=request_cache)
        scan = {key: value for key, value in raw.items() if key != "entries"}
        # Snapshot entries are flat records; copy only each record before applying
        # the root-specific cleanup policy.
        scan["entries"] = [item.copy() for item in raw["entries"]]
        root = spec["path"].resolve()
        for item in scan["entries"]:
            local_relative = str(item["relative_path"])
            item["relative_path"] = f"{relative}/{local_relative}" if relative else local_relative
            self._apply_entry_policy(item, root / item["relative_path"], spec, direct=not relative)
        return scan

    def _apply_active_entry_guards(self, spec: dict[str, Any], scan: dict[str, Any]) -> None:
        if spec["access"] != "cleanup" or not isinstance(spec.get("path"), Path):
            return
        root = spec["path"].resolve()
        guarded = [entry for entry in scan["entries"] if entry.get("cleanup_allowed")]
        targets = [root / str(entry["relative_path"]) for entry in guarded]
        # Listing must use the same fail-closed inspection contract as a cleanup
        # plan.  An active service whose references cannot be inspected is not
        # evidence that the entry is safe to remove.
        blockers = self._active_blockers(targets, spec["id"], inspect_unknown=True) if targets else []
        by_path = {str(item["relative_path"]): item["reason"] for item in blockers}
        training = self.app.get("training_service") if self.app is not None else None
        active_dataset = (
            spec["id"] == "postprocessed-dataset" and training is not None
            and str(getattr(training, "status", "") or "").strip().lower() in ACTIVE_TRAINING_STATES
        )
        for entry in guarded:
            reason = by_path.get(str(entry["name"]))
            if reason:
                entry["cleanup_allowed"] = False
                entry["cleanup_reason"] = reason
            elif active_dataset:
                entry["cleanup_allowed"] = False
                entry["cleanup_reason"] = "训练或预处理正在使用数据目录，已阻止清理"

    @staticmethod
    def _refresh_requested(refresh: Any) -> bool:
        return refresh is True or (isinstance(refresh, str) and refresh.strip().lower() in {"1", "true", "yes", "on"})

    def _unscanned_root(self, spec: dict[str, Any]) -> dict[str, Any]:
        path = spec["path"]
        return {
            "id": spec["id"], "label": spec["label"],
            "path": self._display_path(path) if isinstance(path, Path) else "",
            "access": spec["access"],
            "size_bytes": None, "size_state": "pending",
            "file_count": None, "directory_count": None,
            "entries": [], "truncated": False,
        }

    def list_roots(self, refresh: Any = False, include_scan: bool = True) -> dict[str, Any]:
        """List the fixed roots.

        With include_scan=False only labels, paths and access are returned.  The
        UI paints its cards from that immediately and fills sizes per root
        through the entries endpoint instead of blocking first paint on a full
        tree walk.
        """
        roots: list[dict[str, Any]] = []
        generated_at = datetime.now(timezone.utc).isoformat()
        specs = self._root_specs()
        request_cache: dict[str, dict[str, Any]] = {}
        refreshed = self._refresh_requested(refresh)
        for spec in specs.values():
            path = spec["path"]
            if not include_scan:
                roots.append(self._unscanned_root(spec))
                continue
            scan = self._scan(spec, path, "", refresh=refreshed, request_cache=request_cache) if isinstance(path, Path) else self._empty_scan()
            self._apply_active_entry_guards(spec, scan)
            roots.append({
                "id": spec["id"], "label": spec["label"],
                "path": self._display_path(path) if isinstance(path, Path) else "",
                "access": spec["access"],
                "size_bytes": scan["size_bytes"], "size_state": scan["size_state"],
                "file_count": scan["file_count"], "directory_count": scan["directory_count"],
                "entries": scan["entries"], "truncated": scan["truncated"],
            })
        return {"ok": True, "generated_at": generated_at, "roots": roots}

    def list_entries(self, root_id: Any, relative: Any, refresh: Any = False) -> dict[str, Any]:
        spec = self._spec(root_id)
        base, normalized = self._resolve_path(spec, relative)
        scan = self._scan(spec, base, normalized, refresh=self._refresh_requested(refresh))
        self._apply_active_entry_guards(spec, scan)
        return {"ok": True, "root_id": spec["id"], "path": normalized,
                "entries": scan["entries"], "file_count": scan["file_count"],
                "directory_count": scan["directory_count"], "size_bytes": scan["size_bytes"],
                "size_state": scan["size_state"], "truncated": scan["truncated"]}

    def _cleanup_spec(self, root_id: Any) -> dict[str, Any]:
        spec = self._spec(root_id)
        if spec["access"] != "cleanup":
            raise DataManagementError("该根目录不允许清理", 403)
        return spec

    def _normalize_items(self, items: Any, *, limit: int = MAX_CLEANUP_ITEMS) -> list[str]:
        if not isinstance(items, list) or not items or len(items) > limit:
            raise DataManagementError(f"items 必须是 1 到 {limit} 个相对路径", 400)
        out: list[str] = []
        for value in items:
            parts = self._safe_relative(value, allow_root=False)
            normalized = "/".join(parts)
            if normalized in out:
                raise DataManagementError("items 不能重复", 400)
            out.append(normalized)
        return out

    @staticmethod
    def _same_or_nested(left: Path, right: Path) -> bool:
        try:
            return left == right or left.is_relative_to(right) or right.is_relative_to(left)
        except (OSError, RuntimeError):
            return False

    def _known_paths(self, value: Any) -> list[Path]:
        paths: list[Path] = []
        if isinstance(value, Path):
            return [value.resolve()]
        if isinstance(value, str) and value.strip():
            try:
                path = Path(value).expanduser()
                if not path.is_absolute():
                    # Runtime payloads use the project's display-path convention.
                    path = self._project_root() / path
                paths.append(path.resolve())
            except (OSError, RuntimeError, ValueError):
                pass
        return paths

    def _active_blockers(self, targets: list[Path], root_id: str, *, inspect_unknown: bool = True) -> list[dict[str, str]]:
        blocked: list[dict[str, str]] = []
        training = self.app.get("training_service") if self.app is not None else None
        if training is None and inspect_unknown:
            blocked.extend({"relative_path": target.name, "reason": "训练服务不可用，无法确认当前任务或队列引用"} for target in targets)
        if training is not None:
            state = str(getattr(training, "status", "") or "").strip().lower()
            if root_id == "postprocessed-dataset" and state not in {"", "idle", "stopped", "finished", "failed"}:
                blocked.extend({"relative_path": target.name, "reason": "训练或预处理正在使用数据目录，已阻止清理"} for target in targets)
            current_paths: list[Path] = []
            current_paths.extend(self._known_paths(getattr(training, "current_task_dir", None)))
            current_paths.extend(self._known_paths(getattr(training, "current_output_dir", None)))
            runtime = getattr(training, "current_runtime_info", {})
            if isinstance(runtime, dict):
                for key in ("run_dir", "training_output_dir", "output_dir", "runtime_config_file"):
                    current_paths.extend(self._known_paths(runtime.get(key)))
            for target in targets:
                if any(self._same_or_nested(target, path) for path in current_paths):
                    blocked.append({"relative_path": target.name, "reason": "当前训练任务或 runtime 目录正在使用"})
            if root_id == "training-runs" and state in ACTIVE_TRAINING_STATES and not current_paths:
                blocked.extend({"relative_path": target.name, "reason": "当前训练状态活跃，无法可靠确认运行目录"} for target in targets)

            queue_items: list[dict[str, Any]] = []
            queue_inspection_failed = False
            queue_getter = getattr(training, "_queue_items", None)
            if callable(queue_getter):
                try:
                    queue_items = list(queue_getter())
                except Exception:
                    queue_inspection_failed = True
            elif isinstance(getattr(training, "_queue", None), dict):
                queue_items = list(training._queue.get("items") or [])
            else:
                queue_inspection_failed = True
            if queue_inspection_failed and inspect_unknown:
                blocked.extend({"relative_path": target.name, "reason": "无法可靠读取训练队列引用"} for target in targets)
            for item in queue_items:
                if not isinstance(item, dict) or str(item.get("state") or item.get("status") or "").lower() not in {"queued", "running"}:
                    continue
                item_paths: list[Path] = []
                runtime_info = item.get("runtime_info") if isinstance(item.get("runtime_info"), dict) else {}
                for key in ("run_dir", "training_output_dir", "output_dir", "runtime_config_file"):
                    item_paths.extend(self._known_paths(runtime_info.get(key)))
                    item_paths.extend(self._known_paths(item.get(key)))
                if root_id == "postprocessed-dataset" and not item_paths and inspect_unknown:
                    blocked.extend({"relative_path": target.name, "reason": "等待或运行中的队列项无法可靠确认数据目录引用"} for target in targets)
                for target in targets:
                    if any(self._same_or_nested(target, path) for path in item_paths):
                        blocked.append({"relative_path": target.name, "reason": "等待或运行中的队列项仍引用该 runtime 目录"})

        image_test = self.app.get("image_test_service") if self.app is not None else None
        if image_test is not None and str(getattr(image_test, "status", "") or "").lower() == "running":
            image_paths = self._known_paths(getattr(image_test, "output_dir", None))
            if not image_paths:
                blocked.extend({"relative_path": target.name, "reason": "生图测试目录不可确认"} for target in targets)
            for target in targets:
                if any(self._same_or_nested(target, path) for path in image_paths):
                    blocked.append({"relative_path": target.name, "reason": "生图测试正在使用该目录"})

        tagging = self.app.get("tagging_service") if self.app is not None else None
        jobs = getattr(getattr(tagging, "jobs", None), "jobs", None) if tagging is not None else None
        active_tagging_unknown = False
        if isinstance(jobs, dict):
            for job in jobs.values():
                if not isinstance(job, dict) or str(job.get("state") or "").lower() not in {"queued", "running"}:
                    continue
                paths: list[Path] = []
                for key, value in job.items():
                    if any(token in str(key).lower() for token in ("path", "dir", "root")):
                        paths.extend(self._known_paths(value))
                if paths:
                    for target in targets:
                        if any(self._same_or_nested(target, path) for path in paths):
                            blocked.append({"relative_path": target.name, "reason": "打标任务正在使用该目录"})
                elif root_id == "postprocessed-dataset":
                    active_tagging_unknown = True
        elif tagging is not None and root_id == "postprocessed-dataset" and inspect_unknown:
            # A service without an inspectable job registry is an unknown
            # external/embedded implementation; fail closed for dataset cleanup.
            active_tagging_unknown = True
        if active_tagging_unknown:
            blocked.extend({"relative_path": target.name, "reason": "打标任务正在运行，无法可靠确认数据目录"} for target in targets)
        unique: dict[tuple[str, str], dict[str, str]] = {}
        for item in blocked:
            unique[(item["relative_path"], item["reason"])] = item
        return list(unique.values())

    def _plan(self, root_id: Any, items: Any, *, limit: int = MAX_CLEANUP_ITEMS) -> dict[str, Any]:
        spec = self._cleanup_spec(root_id)
        names = self._normalize_items(items, limit=limit)
        self._validate_root(spec)
        root = spec["path"]
        if not isinstance(root, Path):
            raise DataManagementError("该数据根目录暂时不可用", 503)
        root = root.resolve()
        try:
            root.lstat()
        except FileNotFoundError:
            raise DataManagementError("清理根目录不存在", 503)
        except OSError as exc:
            raise DataManagementError("清理根目录暂时不可用", 503) from exc
        targets: list[Path] = []
        items_out: list[dict[str, Any]] = []
        blocked: list[dict[str, str]] = []
        target_states: list[tuple[str, dict[str, Any]]] = []
        for name in names:
            candidate = root / name
            try:
                resolved = candidate.resolve(strict=False)
                if resolved != candidate or self._has_symlink_component(root, candidate):
                    blocked.append({"relative_path": name, "reason": "不允许清理符号链接或逃逸路径"})
                    continue
            except OSError:
                blocked.append({"relative_path": name, "reason": "路径检查暂时不可用"})
                continue
            entry = self._entry(
                candidate,
                root,
                spec,
                direct=spec["id"] == "training-runs" and len(Path(name).parts) == 1,
            )
            target_stat: os.stat_result | None = None
            if entry["status"] == "ready":
                try:
                    target_stat = candidate.lstat()
                except FileNotFoundError:
                    entry.update(status="missing", kind="missing")
                except OSError:
                    entry.update(status="unavailable", cleanup_allowed=False,
                                 cleanup_reason="条目状态暂时不可用")
            if entry["kind"] == "directory" and entry["status"] == "ready":
                try:
                    size = files = directories = 0
                    latest_mtime_ns = int(target_stat.st_mtime_ns) if target_stat is not None else 0
                    mtime_sum_ns = latest_mtime_ns
                    ctime_sum_ns = int(target_stat.st_ctime_ns) if target_stat is not None else 0
                    pending = [candidate]
                    while pending:
                        current = pending.pop()
                        with os.scandir(current) as children:
                            for child in children:
                                info = child.stat(follow_symlinks=False)
                                latest_mtime_ns = max(latest_mtime_ns, int(info.st_mtime_ns))
                                mtime_sum_ns += int(info.st_mtime_ns)
                                ctime_sum_ns += int(info.st_ctime_ns)
                                if stat.S_ISLNK(info.st_mode):
                                    continue
                                if stat.S_ISREG(info.st_mode):
                                    size += info.st_size
                                    files += 1
                                elif stat.S_ISDIR(info.st_mode):
                                    directories += 1
                                    pending.append(Path(child.path))
                    entry.update(size_bytes=size, file_count=files, directory_count=directories)
                    target_states.append((name, {
                        "file_count": files,
                        "directory_count": directories,
                        "size_bytes": size,
                        "latest_mtime_ns": latest_mtime_ns,
                        "mtime_sum_ns": mtime_sum_ns,
                        "ctime_sum_ns": ctime_sum_ns,
                        "dev": int(target_stat.st_dev) if target_stat is not None else 0,
                        "ino": int(target_stat.st_ino) if target_stat is not None else 0,
                        "mode": int(target_stat.st_mode) if target_stat is not None else 0,
                        "top_ctime_ns": int(target_stat.st_ctime_ns) if target_stat is not None else 0,
                        "top_size": int(target_stat.st_size) if target_stat is not None else 0,
                        "tree_hash": self._fingerprint_tree(candidate),
                    }))
                except OSError:
                    entry.update(status="unavailable", cleanup_allowed=False,
                                 cleanup_reason="删除前目录签名不可用")
            elif target_stat is not None and entry["kind"] == "file":
                target_states.append((name, {
                    "file_count": 1,
                    "directory_count": 0,
                    "size_bytes": int(target_stat.st_size),
                    "latest_mtime_ns": int(target_stat.st_mtime_ns),
                    "mtime_sum_ns": int(target_stat.st_mtime_ns),
                    "ctime_sum_ns": int(target_stat.st_ctime_ns),
                    "dev": int(target_stat.st_dev),
                    "ino": int(target_stat.st_ino),
                    "mode": int(target_stat.st_mode),
                    "top_ctime_ns": int(target_stat.st_ctime_ns),
                    "top_size": int(target_stat.st_size),
                    "tree_hash": self._fingerprint_tree(candidate),
                }))
            if entry["status"] == "missing":
                blocked.append({"relative_path": name, "reason": "条目不存在"})
                continue
            if not entry["cleanup_allowed"]:
                blocked.append({"relative_path": name, "reason": entry["cleanup_reason"]})
                continue
            targets.append(candidate)
            items_out.append({
                "relative_path": name, "name": name, "path": self._display_path(candidate),
                "kind": entry["kind"], "size_bytes": entry["size_bytes"],
                "modified_at": entry["modified_at"],
            })
        blocked.extend(self._active_blockers(targets, spec["id"]))
        try:
            root_stat = root.stat()
            revision_parts = [spec["id"], str(root_stat.st_mtime_ns), str(root_stat.st_ctime_ns)]
            state_by_name = dict(target_states)
            for target in targets:
                try:
                    target_stat = target.lstat()
                    relative_path = target.relative_to(root).as_posix()
                    state = state_by_name.get(relative_path)
                    state_token = ""
                    if state is not None:
                        state_token = (
                            f":files={state['file_count']}:dirs={state['directory_count']}"
                            f":bytes={state['size_bytes']}:mtime_max={state['latest_mtime_ns']}"
                            f":mtime_sum={state['mtime_sum_ns']}:ctime_sum={state['ctime_sum_ns']}"
                        )
                    revision_parts.append(
                        f"target:{relative_path}:{target_stat.st_ino}:{target_stat.st_mtime_ns}:"
                        f"{target_stat.st_ctime_ns}:{target_stat.st_size}{state_token}"
                    )
                except FileNotFoundError:
                    revision_parts.append(f"target:{target.relative_to(root).as_posix()}:missing")
            blocker_keys = sorted(
                f"{item.get('relative_path', '')}:{item.get('reason', '')}"
                for item in blocked
            )
            for blocker in blocker_keys:
                revision_parts.append(f"blocker:{blocker}")
            plan_revision = "v1:" + ":".join(revision_parts)
        except (OSError, DataManagementError) as exc:
            if isinstance(exc, DataManagementError):
                raise
            raise DataManagementError("清理计划暂时不可用", 503) from exc
        return {"spec": spec, "names": names, "items": items_out, "targets": targets,
                "target_states": dict(target_states), "blocked": blocked, "plan_revision": plan_revision}

    @staticmethod
    def _fingerprint_tree(path: Path) -> str:
        """Hash a no-follow tree manifest for plan-to-delete change detection."""
        tokens: list[str] = []
        stack: list[tuple[Path, str]] = [(path, ".")]
        while stack:
            current, relative = stack.pop()
            info = current.lstat()
            tokens.append(
                f"{relative}:{info.st_dev}:{info.st_ino}:{info.st_mode}:{info.st_size}:"
                f"{info.st_mtime_ns}:{info.st_ctime_ns}"
            )
            if not stat.S_ISDIR(info.st_mode):
                continue
            with os.scandir(current) as children:
                for child in children:
                    child_rel = f"{relative}/{child.name}" if relative != "." else child.name
                    child_info = child.stat(follow_symlinks=False)
                    if stat.S_ISDIR(child_info.st_mode):
                        stack.append((Path(child.path), child_rel))
                    else:
                        tokens.append(
                            f"{child_rel}:{child_info.st_dev}:{child_info.st_ino}:{child_info.st_mode}:"
                            f"{child_info.st_size}:{child_info.st_mtime_ns}:{child_info.st_ctime_ns}"
                        )
        return hashlib.sha256("\n".join(sorted(tokens)).encode("utf-8")).hexdigest()

    def _revalidate_target(self, spec: dict[str, Any], target: Path,
                           expected: dict[str, Any] | None = None) -> None:
        """Repeat the no-symlink and in-root check immediately before deletion."""
        root = spec["path"]
        if not isinstance(root, Path):
            raise DataManagementError("清理根目录暂时不可用", 503)
        root = root.resolve()
        try:
            if target.resolve(strict=False) != target or self._has_symlink_component(root, target):
                raise DataManagementError("删除前路径安全检查失败", 409)
            info = target.lstat()
            if expected is not None:
                actual = {
                    "dev": int(info.st_dev), "ino": int(info.st_ino),
                    "mode": int(info.st_mode), "top_ctime_ns": int(info.st_ctime_ns),
                    "top_size": int(info.st_size),
                }
                if any(actual[key] != expected.get(key) for key in actual):
                    raise DataManagementError("条目在清理前已变化，请重新预览", 409)
                if expected.get("tree_hash") and self._fingerprint_tree(target) != expected["tree_hash"]:
                    raise DataManagementError("目录内容在清理前已变化，请重新预览", 409)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise DataManagementError("删除前路径检查暂时不可用", 503) from exc
        if not stat.S_ISDIR(info.st_mode) and not stat.S_ISREG(info.st_mode):
            raise DataManagementError("删除前条目类型不受支持", 409)

    def _execute_plan(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Delete the allowed targets of a validated plan.

        Callers hold ``_cleanup_lock`` and the cleanup lease.  Retention batches
        reuse this loop so path checks, active-task blockers and per-item error
        handling stay identical to the plain cleanup flow.
        """
        deleted: list[str] = []
        missing: list[str] = []
        blocked: list[dict[str, str]] = []
        errors: dict[str, str] = {}
        for item, target in zip(plan["items"], plan["targets"], strict=True):
            try:
                latest_blockers = self._active_blockers([target], plan["spec"]["id"])
                if latest_blockers:
                    blocked.extend(latest_blockers)
                    continue
                expected_state = plan.get("target_states", {}).get(item["relative_path"])
                self._revalidate_target(plan["spec"], target, expected_state)
                if not target.exists():
                    missing.append(item["relative_path"])
                    continue
                if plan["spec"]["id"] == "training-runs":
                    from library.cache_pool.gc import safe_rmtree_run_dir

                    safe_rmtree_run_dir(target)
                else:
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                deleted.append(item["relative_path"])
                self._invalidate_cache(target)
            except FileNotFoundError:
                missing.append(item["relative_path"])
            except DataManagementError as exc:
                blocked.append({"relative_path": item["relative_path"], "reason": exc.message})
            except (OSError, RuntimeError, ValueError):
                errors[item["relative_path"]] = "永久删除失败，文件系统暂时不可用"
        return {"ok": True, "dry_run": False, "root_id": plan["spec"]["id"],
                "deleted": deleted, "missing": missing, "blocked": blocked, "errors": errors,
                "message": f"已永久删除 {len(deleted)} 个生成产物"}

    def preview_cleanup(self, root_id: Any, items: Any) -> dict[str, Any]:
        plan = self._plan(root_id, items)
        total = sum(item["size_bytes"] for item in plan["items"] if isinstance(item["size_bytes"], int))
        if any(item["size_bytes"] is None for item in plan["items"]):
            total_value: int | None = None
        else:
            total_value = total
        return {"ok": True, "dry_run": True, "root_id": plan["spec"]["id"],
                "items": plan["items"], "total_size_bytes": total_value,
                "item_count": len(plan["items"]), "blocked": plan["blocked"],
                "plan_revision": plan["plan_revision"]}

    def cleanup(self, root_id: Any, items: Any, expected_items: Any, plan_revision: Any, confirmed: Any) -> dict[str, Any]:
        if confirmed is not True:
            raise DataManagementError("永久删除需要 confirmed=true", 400)
        expected = self._normalize_items(expected_items)
        if not self._cleanup_lock.acquire(blocking=False):
            raise DataManagementError("已有清理操作正在进行", 409)
        try:
            with data_cleanup_lease():
                plan = self._plan(root_id, items)
                if expected != plan["names"] or not isinstance(plan_revision, str) or plan_revision != plan["plan_revision"]:
                    raise DataManagementError("清理预览已变化，请重新预览", 409)
                if plan["blocked"]:
                    raise DataManagementError("存在阻止清理的条目", 409, extra={"blocked": plan["blocked"], "plan_revision": plan["plan_revision"]})
                return self._execute_plan(plan)
        finally:
            self._cleanup_lock.release()


__all__ = ["DataManagementError", "DataManagementService"]
