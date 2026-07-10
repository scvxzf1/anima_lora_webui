"""Resume helpers for WebUI training history."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from web.services.training.common import _int_or_none
from web.services.training.history_meta import (
    _history_group_meta,
    _history_snapshot_path,
    _load_history_task,
)
from web.services.training.history_runtime import (
    _clone_frozen_runtime_config,
    _display_project_path,
    _list_resume_checkpoints,
    _path_exists,
    _resolve_display_path,
    _resume_checkpoint_diagnostic,
    _runtime_meta,
    _select_resume_checkpoint,
)
from web.services.training.resume import _resume_state_integrity_unavailable_reason

async def resume_from_history_task(
    self,
    task_id: str,
    checkpoint: str | None = None,
    *,
    duration_overrides: dict[str, Any] | None = None,
    gpu_whitelist: list[Any] | None = None,
) -> dict[str, Any]:
    task, selected, snapshot_path, resume_info = self._build_resume_payload(
        task_id,
        checkpoint,
        duration_overrides=duration_overrides,
    )
    runtime = _clone_resume_runtime(
        task,
        snapshot_path,
        resume_step=_int_or_none(selected.get("step")),
        duration_overrides=duration_overrides,
    )
    if isinstance(runtime.get("resume_duration"), dict) and runtime["resume_duration"]:
        resume_info["duration_overrides"] = runtime["resume_duration"]
        resume_info["target_total_steps"] = runtime["resume_duration"].get("target_total_steps")
        resume_info["remaining_steps"] = runtime["resume_duration"].get("append_steps")
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
    *,
    duration_overrides: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
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
    _ensure_resume_checkpoint_available(
        selected,
        allow_completed_by_duration_override=_resume_duration_override_requested(duration_overrides),
    )

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
        reason = _resume_state_integrity_unavailable_reason(item.get("state_integrity"))
        if not reason:
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

def _ensure_resume_checkpoint_available(
    selected: dict[str, Any],
    *,
    allow_completed_by_duration_override: bool = False,
) -> None:
    reason = str(selected.get("unavailable_reason") or "")
    if (
        reason
        and allow_completed_by_duration_override
        and _int_or_none(selected.get("step")) is not None
    ):
        integrity = selected.get("state_integrity") if isinstance(selected.get("state_integrity"), dict) else {}
        if integrity.get("complete") is False:
            raise ValueError(reason)
        target_total_steps = _int_or_none(selected.get("target_total_steps"))
        if target_total_steps is not None and _int_or_none(selected.get("step")) >= target_total_steps:
            return
    if reason:
        raise ValueError(reason)

def _resume_duration_override_requested(duration_overrides: dict[str, Any] | None) -> bool:
    if not isinstance(duration_overrides, dict):
        return False
    for key in ("max_train_epochs", "max_train_steps"):
        if _positive_resume_duration_int(duration_overrides.get(key)) is not None:
            return True
    return False

def _positive_resume_duration_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    try:
        n = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None

def _clone_resume_runtime(
    task: dict[str, Any],
    snapshot_path: Path,
    *,
    resume_step: int | None = None,
    duration_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _clone_frozen_runtime_config(
        _display_project_path(str(snapshot_path)),
        source_config_file=str(task.get("history_source_config_file") or ""),
        reset_data_dirs=False,
        resume_step=resume_step,
        duration_overrides=duration_overrides,
    )

def get_resume_options(self, task_id: str) -> dict[str, Any]:
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
