"""Delegated training service methods.

This module is a mechanical extraction from ``web.services.training_service``.
The public ``TrainingService`` class keeps the same method names and delegates
here so HTTP routes and WebSocket payloads remain unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import json
    import shutil
    import time
    from datetime import datetime
    from pathlib import Path
    from typing import Any

    from web.services.training_service import (
        HISTORY_COLLECTIONS_FILE,
        HISTORY_DIR,
        _batch_archive_history_tasks,
        _batch_set_history_group,
        _build_config_group_timeline,
        _continue_lora_history_meta,
        _count_jsonl,
        _default_history_archived,
        _default_preprocess_history_name,
        _delete_history_tasks,
        _display_project_path,
        _format_ts,
        _history_artifact_path,
        _history_delete_run_key,
        _history_delete_task_preview,
        _history_group_meta,
        _history_log_path,
        _history_runtime_delete_dirs_for_tasks,
        _history_snapshot_path,
        _history_task_ids_for_delete,
        _clone_frozen_runtime_config,
        _int_or_none,
        _list_history_tasks,
        _list_resume_checkpoints,
        _load_history_collection_settings,
        _load_history_task,
        _normalize_history_collection_settings,
        _normalize_history_task_ids,
        _path_exists,
        _queue_runtime_delete_blockers,
        _read_json,
        _resolve_display_path,
        _resume_checkpoint_diagnostic,
        _runtime_meta,
        _safe_task_id,
        _select_resume_checkpoint,
        _update_history_task,
        _write_config_snapshot,
        _write_json_atomic,
    )


_LOCAL_IMPL_NAMES = {
    "_bind_legacy",
    "resume_from_history_task",
    "_build_resume_payload",
    "_annotate_resume_checkpoints",
    "_clone_resume_runtime",
    "_ensure_resume_checkpoint_available",
    "_resume_checkpoint_estimate",
    "_resume_unavailable_reason",
    "list_history_tasks",
    "get_history_task",
    "get_history_log_path",
    "get_history_artifact_path",
    "get_config_group_timeline",
    "get_history_collection_settings",
    "save_history_collection_settings",
    "get_resume_options",
    "update_history_task",
    "batch_update_history_tasks",
    "delete_history_task",
    "_batch_delete_history_tasks",
    "_plan_history_delete",
    "_reserve_history_task_dir",
    "_start_history_task",
    "_finish_history_task",
    "_append_history_jsonl",
}


def _bind_legacy() -> None:
    """Bind legacy module globals lazily after training_service has loaded."""
    from web.services import training_service as legacy

    for name, value in vars(legacy).items():
        if name.startswith("__") or name in _LOCAL_IMPL_NAMES:
            continue
        globals()[name] = value


async def resume_from_history_task(
    self,
    task_id: str,
    checkpoint: str | None = None,
    *,
    gpu_whitelist: list[Any] | None = None,
) -> dict[str, Any]:
    _bind_legacy()
    task, selected, snapshot_path, resume_info = self._build_resume_payload(task_id, checkpoint)
    runtime = _clone_resume_runtime(task, snapshot_path)
    config_file = str(runtime.get("runtime_config_file") or _display_project_path(str(snapshot_path)))
    source_config_file = str(runtime.get("history_source_config_file") or task.get("history_source_config_file") or "")

    await self.start(
        str(task.get("variant") or ""),
        str(task.get("preset") or "default"),
        ["--resume", selected["path"], "--skip_until_initial_step"],
        str(task.get("methods_subdir") or "gui-methods"),
        config_file=config_file,
        start_message=f"从检查点继续训练: {selected['name']}",
        command_label="续训命令",
        resume_info=resume_info,
        gpu_whitelist=gpu_whitelist,
        source_config_file=source_config_file,
        use_runtime_dir=False,
    )

    return {
        "ok": True,
        "message": "已从检查点继续训练",
        "task_id": self.current_task_id,
        "checkpoint": selected,
    }

def _build_resume_payload(
    self,
    task_id: str,
    checkpoint: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
    _bind_legacy()
    payload = _load_history_task(task_id)
    task = payload.get("task") if isinstance(payload, dict) else {}
    if not isinstance(task, dict):
        raise ValueError("任务不存在")
    if task.get("job") != "training":
        raise ValueError("只能从训练任务继续训练")

    snapshot_path = _history_snapshot_path(task_id)
    if snapshot_path is None:
        raise ValueError("历史任务缺少配置快照，无法安全续训")

    checkpoints = _list_resume_checkpoints(task)
    if not checkpoints:
        raise ValueError("这个训练任务没有可续训的检查点")

    checkpoints = _annotate_resume_checkpoints(task, checkpoints, snapshot_path)
    selected = (
        _select_resume_checkpoint(checkpoints, checkpoint)
        if checkpoint
        else next((item for item in checkpoints if item.get("resume_available") is not False), None)
    )
    if selected is None:
        if not checkpoint:
            first_reason = next(
                (str(item.get("unavailable_reason") or "") for item in checkpoints if item.get("unavailable_reason")),
                "",
            )
            if first_reason:
                raise ValueError(first_reason)
        raise ValueError("未找到指定的检查点")
    _ensure_resume_checkpoint_available(selected)

    resume_info = {
        "source_task_id": task_id,
        "source_task_name": str(task.get("name") or ""),
        "history_group_key": str(task.get("history_group_key") or ""),
        "history_group_label": str(task.get("history_group_label") or ""),
        "history_source_config_file": str(task.get("history_source_config_file") or ""),
        "checkpoint": selected["path"],
        "checkpoint_name": selected["name"],
        "checkpoint_kind": selected["kind"],
        "checkpoint_epoch": selected.get("epoch"),
        "checkpoint_step": selected.get("step"),
        "target_total_steps": selected.get("target_total_steps"),
        "remaining_steps": selected.get("remaining_steps"),
    }
    return task, selected, snapshot_path, resume_info

def _annotate_resume_checkpoints(
    task: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    snapshot_path: Path | None,
) -> list[dict[str, Any]]:
    estimate = _resume_checkpoint_estimate(task, snapshot_path)
    target_total_steps = estimate.get("target_total_steps")
    estimate_error = str(estimate.get("estimate_error") or "")
    out: list[dict[str, Any]] = []
    for raw in checkpoints:
        item = dict(raw)
        step = _int_or_none(item.get("step"))
        item["target_total_steps"] = target_total_steps
        item["estimate_error"] = estimate_error
        if isinstance(target_total_steps, int) and step is not None:
            item["remaining_steps"] = max(0, target_total_steps - step)
        else:
            item["remaining_steps"] = None
        reason = _resume_unavailable_reason(step, target_total_steps)
        item["resume_available"] = not reason
        item["unavailable_reason"] = reason
        out.append(item)
    return out

def _resume_checkpoint_estimate(
    task: dict[str, Any],
    snapshot_path: Path | None,
) -> dict[str, Any]:
    if snapshot_path is None:
        return {"target_total_steps": None, "estimate_error": "历史任务缺少配置快照"}
    try:
        from web.services import config_service

        estimate = config_service.estimate_training_steps(
            str(task.get("variant") or ""),
            str(task.get("preset") or "default"),
            str(task.get("methods_subdir") or "gui-methods"),
            config_file=_display_project_path(str(snapshot_path)),
        )
        total_steps = _int_or_none(estimate.get("total_steps"))
        return {
            "target_total_steps": total_steps if total_steps and total_steps > 0 else None,
            "estimate_error": "",
        }
    except Exception as exc:
        return {"target_total_steps": None, "estimate_error": str(exc)}

def _resume_unavailable_reason(step: int | None, target_total_steps: int | None) -> str:
    if step is None or target_total_steps is None:
        return ""
    if step >= target_total_steps:
        return (
            f"这个检查点已训练到 step {step}，当前配置目标是 {target_total_steps}，"
            "继续训练不会产生新步数。请先增加 max_train_steps / max_train_epochs，或改用权重热启动。"
        )
    return ""

def _ensure_resume_checkpoint_available(selected: dict[str, Any]) -> None:
    reason = str(selected.get("unavailable_reason") or "")
    if reason:
        raise ValueError(reason)

def _clone_resume_runtime(task: dict[str, Any], snapshot_path: Path) -> dict[str, Any]:
    return _clone_frozen_runtime_config(
        _display_project_path(str(snapshot_path)),
        source_config_file=str(task.get("history_source_config_file") or ""),
        reset_data_dirs=False,
    )

def list_history_tasks(self, *, include_archived: bool = False, limit: int | None = None) -> list[dict[str, Any]]:
    _bind_legacy()
    return _list_history_tasks(include_archived=include_archived, limit=limit)

def get_history_task(self, task_id: str) -> dict[str, Any]:
    _bind_legacy()
    return _load_history_task(task_id)

def get_history_log_path(self, task_id: str) -> Path:
    _bind_legacy()
    return _history_log_path(task_id)

def get_history_artifact_path(self, task_id: str, artifact_key: str) -> Path:
    _bind_legacy()
    return _history_artifact_path(task_id, artifact_key)

def get_config_group_timeline(
    self,
    methods_subdir: str,
    variant: str,
    preset: str,
    *,
    group_key: str = "",
    include_archived: bool = False,
    task_ids: list[str] | None = None,
) -> dict[str, Any]:
    _bind_legacy()
    return _build_config_group_timeline(
        methods_subdir,
        variant,
        preset,
        group_key=group_key,
        include_archived=include_archived,
        task_ids=task_ids,
    )

def get_history_collection_settings(self) -> dict[str, Any]:
    _bind_legacy()
    return _load_history_collection_settings()

def save_history_collection_settings(self, payload: dict[str, Any]) -> dict[str, Any]:
    _bind_legacy()
    settings = _normalize_history_collection_settings(payload)
    settings["updated_at"] = datetime.now().timestamp()
    settings["updated_at_text"] = _format_ts(settings["updated_at"])
    _write_json_atomic(HISTORY_COLLECTIONS_FILE, settings)
    return {"ok": True, **settings}

def get_resume_options(self, task_id: str) -> dict[str, Any]:
    _bind_legacy()
    payload = _load_history_task(task_id)
    task = payload.get("task") if isinstance(payload, dict) else {}
    if not isinstance(task, dict):
        raise FileNotFoundError("任务不存在")
    if task.get("job") != "training":
        raise ValueError("只能从训练任务读取续训检查点")
    snapshot_path = _history_snapshot_path(task_id)
    checkpoints = _annotate_resume_checkpoints(task, _list_resume_checkpoints(task), snapshot_path)
    diagnostic = _resume_checkpoint_diagnostic(task, checkpoints)
    default = next((item for item in checkpoints if item.get("resume_available") is not False), None)
    default_checkpoint = default["path"] if default else ""
    message = "选择一个保存了训练状态的目录继续训练。普通权重文件不能恢复优化器和步数。"
    if not checkpoints:
        message = diagnostic.get("reason") or "这个任务没有找到可续训的状态目录。只有保存了 train_state.json 的目录才能继续训练。"
    elif not default_checkpoint:
        message = str(checkpoints[0].get("unavailable_reason") or "") or message
    return {
        "ok": True,
        "task": {
            "id": task.get("id", task_id),
            "name": task.get("name", ""),
            "group": task.get("group", ""),
            "variant": task.get("variant", ""),
            "preset": task.get("preset", ""),
            "methods_subdir": task.get("methods_subdir", ""),
            "output_dir": task.get("output_dir", ""),
            "sample_dir": task.get("sample_dir", ""),
        },
        "checkpoints": checkpoints,
        "default_checkpoint": default_checkpoint,
        "message": message,
        "diagnostic": diagnostic,
    }

def update_history_task(self, task_id: str, patch: dict[str, Any]) -> dict[str, Any]:
    _bind_legacy()
    return _update_history_task(task_id, patch)

def batch_update_history_tasks(self, payload: dict[str, Any]) -> dict[str, Any]:
    _bind_legacy()
    action = str(payload.get("action") or "").strip().lower()
    task_ids = _normalize_history_task_ids(payload.get("task_ids"))
    if not task_ids:
        raise ValueError("请先选择历史任务")
    if action in {"archive", "unarchive"}:
        return _batch_archive_history_tasks(task_ids, archived=(action == "archive"))
    if action == "set_group":
        return _batch_set_history_group(task_ids, payload.get("group"))
    if action == "delete":
        return self._batch_delete_history_tasks(payload, task_ids)
    raise ValueError("不支持的批量操作")

def delete_history_task(self, task_id: str) -> dict[str, Any]:
    _bind_legacy()
    delete_task_ids = _history_task_ids_for_delete(task_id)
    if self.status == "running" and self.current_task_id in delete_task_ids:
        raise RuntimeError("当前运行中的任务不能删除")
    return _delete_history_tasks(delete_task_ids)

def _batch_delete_history_tasks(self, payload: dict[str, Any], task_ids: list[str]) -> dict[str, Any]:
    _bind_legacy()
    delete_runtime_dirs = bool(payload.get("delete_runtime_dirs"))
    plan = self._plan_history_delete(task_ids, delete_runtime_dirs=delete_runtime_dirs)
    if payload.get("dry_run", False):
        return {"ok": True, "dry_run": True, **plan}
    if plan["blocked"]:
        raise RuntimeError("存在不能删除的任务或运行目录，请先处理阻止项")
    if delete_runtime_dirs and payload.get("confirmed") is not True:
        raise ValueError("彻底删除需要完成二次按钮确认")
    result = _delete_history_tasks([item["id"] for item in plan["tasks"]])
    deleted_runtime_dirs: list[str] = []
    runtime_cleanup_errors: dict[str, str] = {}
    if delete_runtime_dirs:
        for item in plan["runtime_dirs"]:
            path = _resolve_display_path(str(item.get("path") or ""))
            if path is None or not _path_exists(path):
                continue
            try:
                shutil.rmtree(path)
                deleted_runtime_dirs.append(str(item.get("path") or ""))
            except OSError as exc:
                runtime_cleanup_errors[str(item.get("path") or "")] = str(exc)
    result.update({
        "dry_run": False,
        "deleted_runtime_dirs": deleted_runtime_dirs,
        "runtime_cleanup_errors": runtime_cleanup_errors,
        "preview": plan,
    })
    if delete_runtime_dirs:
        result["message"] = f"已彻底删除 {len(result.get('deleted_task_ids') or [])} 个历史记录和 {len(deleted_runtime_dirs)} 个运行目录"
    return result

def _plan_history_delete(self, task_ids: list[str], *, delete_runtime_dirs: bool) -> dict[str, Any]:
    _bind_legacy()
    tasks_by_id = {
        str(task.get("id") or ""): task
        for task in _list_history_tasks(include_archived=True, limit=0)
    }
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    blocked: list[dict[str, str]] = []
    for task_id in task_ids:
        task = tasks_by_id.get(task_id)
        if not task:
            blocked.append({"id": task_id, "reason": "任务不存在"})
            continue
        for linked_id in _history_task_ids_for_delete(task_id):
            if linked_id in seen:
                continue
            linked = tasks_by_id.get(linked_id)
            if linked:
                selected.append(linked)
                seen.add(linked_id)
    if self.status == "running" and self.current_task_id in seen:
        blocked.append({"id": self.current_task_id, "reason": "当前运行中的任务不能删除"})

    runtime_dirs: list[dict[str, str]] = []
    if delete_runtime_dirs:
        run_keys = {
            _history_delete_run_key(task)
            for task in selected
            if _history_delete_run_key(task)
        }
        for candidate in tasks_by_id.values():
            candidate_id = str(candidate.get("id") or "")
            if not candidate_id or candidate_id in seen:
                continue
            if _history_delete_run_key(candidate) not in run_keys:
                continue
            selected.append(candidate)
            seen.add(candidate_id)
        if self.status == "running" and self.current_task_id in seen and not any(
            item.get("id") == self.current_task_id for item in blocked
        ):
            blocked.append({"id": self.current_task_id, "reason": "当前运行中的任务不能删除"})
        runtime_dirs, runtime_blocked = _history_runtime_delete_dirs_for_tasks(selected)
        blocked.extend(runtime_blocked)
        queue_refs = _queue_runtime_delete_blockers(self._queue_items(), runtime_dirs)
        blocked.extend(queue_refs)

    return {
        "tasks": [_history_delete_task_preview(task) for task in selected],
        "runtime_dirs": runtime_dirs,
        "blocked": blocked,
        "delete_runtime_dirs": delete_runtime_dirs,
        "task_count": len(selected),
        "runtime_dir_count": len(runtime_dirs),
    }

def _reserve_history_task_dir(self, job: str, methods_subdir: str, variant: str) -> Path:
    _bind_legacy()
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    task_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{job}-{methods_subdir}-{variant}"
    task_id = _safe_task_id(task_id)
    task_dir = HISTORY_DIR / task_id
    suffix = 1
    while task_dir.exists():
        suffix += 1
        task_dir = HISTORY_DIR / f"{task_id}-{suffix}"
    task_dir.mkdir(parents=True, exist_ok=True)
    self.current_task_id = task_dir.name
    self.current_task_dir = task_dir
    return task_dir

def _start_history_task(
    self,
    *,
    job: str,
    variant: str,
    preset: str,
    methods_subdir: str,
    output_dir: str,
    sample_dir: str,
    data_dirs: dict[str, str],
    sample_config: dict[str, Any],
    command: list[str],
    config_file: str | None = None,
    resume_info: dict[str, Any] | None = None,
    continue_info: dict[str, Any] | None = None,
    gpu_whitelist: list[int] | None = None,
    runtime_info: dict[str, str] | None = None,
    queue_info: dict[str, Any] | None = None,
) -> None:
    _bind_legacy()
    task_dir = self.current_task_dir or self._reserve_history_task_dir(job, methods_subdir, variant)
    now = time.time()
    runtime_meta = _runtime_meta(runtime_info)
    history_meta = _history_group_meta(
        methods_subdir,
        variant,
        preset,
        output_dir=output_dir,
        runtime_info=runtime_meta,
        resume_info=resume_info,
        task_id=task_dir.name,
    )
    default_name = _default_preprocess_history_name({
        "id": task_dir.name,
        "job": job,
        "output_dir": output_dir,
        **_continue_lora_history_meta(continue_info),
        **runtime_meta,
        **history_meta,
    })
    continue_meta = _continue_lora_history_meta(continue_info)
    queue_meta = queue_info if isinstance(queue_info, dict) else {}
    meta = {
        "id": task_dir.name,
        "name": default_name,
        "group": "",
        "archived": _default_history_archived(job),
        "job": job,
        "state": "running",
        "variant": variant,
        "preset": preset,
        "methods_subdir": methods_subdir,
        "output_dir": output_dir,
        "sample_dir": sample_dir,
        "source_image_dir": data_dirs.get("source_image_dir", ""),
        "resized_image_dir": data_dirs.get("resized_image_dir", ""),
        "lora_cache_dir": data_dirs.get("lora_cache_dir", ""),
        "data_dirs": data_dirs,
        "sample_config": sample_config,
        "command": command,
        "resume_from": resume_info or {},
        **continue_meta,
        "gpu_whitelist": gpu_whitelist or [],
        **runtime_meta,
        **history_meta,
        **queue_meta,
        "started_at": now,
        "started_at_text": _format_ts(now),
        "finished_at": None,
        "finished_at_text": "",
        "message": "",
        "returncode": None,
        "log_count": 0,
        "metric_count": 0,
    }
    _write_json_atomic(task_dir / "meta.json", meta)
    _write_config_snapshot(
        task_dir / "config.snapshot.toml",
        variant,
        preset,
        methods_subdir,
        config_file=config_file,
        continue_info=continue_info,
    )

def _finish_history_task(self, *, state: str, message: str, returncode: int) -> None:
    _bind_legacy()
    if not self.current_task_dir:
        return
    meta = _read_json(self.current_task_dir / "meta.json")
    now = time.time()
    meta.update({
        "state": state,
        "finished_at": now,
        "finished_at_text": _format_ts(now),
        "message": message,
        "returncode": returncode,
        "log_count": _count_jsonl(self.current_task_dir / "logs.jsonl"),
        "metric_count": _count_jsonl(self.current_task_dir / "metrics.jsonl"),
    })
    _write_json_atomic(self.current_task_dir / "meta.json", meta)

def _append_history_jsonl(self, filename: str, payload: dict[str, Any]) -> None:
    _bind_legacy()
    if not self.current_task_dir:
        return
    try:
        with (self.current_task_dir / filename).open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
