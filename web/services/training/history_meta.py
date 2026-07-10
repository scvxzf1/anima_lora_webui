"""History metadata, artifact path, and summary helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from web.services.training.common import (
    HISTORY_ARTIFACT_FILES,
    HISTORY_RUNTIME_ARTIFACT_FIELDS,
    TRAINING_PROGRESS_LOG_RE,
    _clean_history_text,
    _float_or_none,
    _format_ts,
    _int_or_none,
    _safe_task_id,
)
from web.services.training.context import training_facade as _training_facade


def _history_dir() -> Path:
    return _training_facade().HISTORY_DIR


def _history_artifact_files() -> dict[str, str]:
    return HISTORY_ARTIFACT_FILES


def _history_runtime_artifact_fields() -> dict[str, str]:
    return HISTORY_RUNTIME_ARTIFACT_FIELDS


def _max_history_detail_log_records() -> int:
    return _training_facade().MAX_HISTORY_DETAIL_LOG_RECORDS


def _max_history_detail_system_records() -> int:
    return _training_facade().MAX_HISTORY_DETAIL_SYSTEM_RECORDS


def _history_average_speed_version() -> int:
    return _training_facade().HISTORY_AVERAGE_SPEED_VERSION


def _training_progress_log_re():
    return TRAINING_PROGRESS_LOG_RE


def _display_project_path(*args, **kwargs):
    return _training_facade()._display_project_path(*args, **kwargs)


def _display_settings_path(*args, **kwargs):
    return _training_facade()._display_settings_path(*args, **kwargs)


def _resolve_display_path(*args, **kwargs):
    return _training_facade()._resolve_display_path(*args, **kwargs)


def _path_is_relative_to(*args, **kwargs):
    return _training_facade()._path_is_relative_to(*args, **kwargs)


def _path_exists(*args, **kwargs):
    return _training_facade()._path_exists(*args, **kwargs)


def resolve_output_root() -> Path:
    return _training_facade().resolve_output_root()


def _read_json(*args, **kwargs):
    return _training_facade()._read_json(*args, **kwargs)


def _read_jsonl(*args, **kwargs):
    return _training_facade()._read_jsonl(*args, **kwargs)


def _read_jsonl_limited(*args, **kwargs):
    return _training_facade()._read_jsonl_limited(*args, **kwargs)


def _read_text_file(*args, **kwargs):
    return _training_facade()._read_text_file(*args, **kwargs)


def _count_jsonl(*args, **kwargs):
    return _training_facade()._count_jsonl(*args, **kwargs)


def _write_json_atomic(*args, **kwargs):
    return _training_facade()._write_json_atomic(*args, **kwargs)


def _repair_history_meta(*args, **kwargs):
    return _training_facade()._repair_history_meta(*args, **kwargs)


def _history_summary(*args, **kwargs):
    return _training_facade()._history_summary(*args, **kwargs)


def _history_metrics_for_task(*args, **kwargs):
    return _training_facade()._history_metrics_for_task(*args, **kwargs)


def _linked_preprocess_task_for_training(*args, **kwargs):
    return _training_facade()._linked_preprocess_task_for_training(*args, **kwargs)


def _linked_preprocess_tasks_for_training(*args, **kwargs):
    return _training_facade()._linked_preprocess_tasks_for_training(*args, **kwargs)


def _history_config_group(*args, **kwargs):
    return _training_facade()._history_config_group(*args, **kwargs)


def _bound_history_task_ids(*args, **kwargs):
    from web.services.training import history_batch as _history_batch

    return _history_batch.bound_history_task_ids(*args, **kwargs)


def _runtime_meta(*args, **kwargs):
    return _training_facade()._runtime_meta(*args, **kwargs)


def _is_web_runtime_dir(*args, **kwargs):
    return _training_facade()._is_web_runtime_dir(*args, **kwargs)


def _format_step_rate(*args, **kwargs):
    return _training_facade()._format_step_rate(*args, **kwargs)


def _is_finite_number(*args, **kwargs):
    return _training_facade()._is_finite_number(*args, **kwargs)


def _history_meta_paths(*args, **kwargs):
    return _training_facade()._history_meta_paths(*args, **kwargs)


def _is_deleting_history_dir(*args, **kwargs):
    return _training_facade()._is_deleting_history_dir(*args, **kwargs)


def _history_group_meta(
    methods_subdir: str,
    variant: str,
    preset: str,
    *,
    output_dir: str = "",
    runtime_info: dict[str, Any] | None = None,
    resume_info: dict[str, Any] | None = None,
    task_id: str = "",
) -> dict[str, str]:
    runtime = runtime_info if isinstance(runtime_info, dict) else {}
    resume = resume_info if isinstance(resume_info, dict) else {}

    inherited_key = str(resume.get("history_group_key") or "").strip()
    inherited_label = str(resume.get("history_group_label") or "").strip()
    inherited_source = str(resume.get("history_source_config_file") or "").strip()
    if inherited_key:
        return {
            "history_group_key": inherited_key,
            "history_group_label": inherited_label or inherited_source or inherited_key,
            "history_source_config_file": inherited_source,
            "history_run_label": _history_run_label_from_runtime(output_dir, runtime, task_id),
        }

    source_config_file = str(runtime.get("history_source_config_file") or "").strip()
    if source_config_file:
        source_display = _display_project_path(source_config_file)
        key = "source:" + source_display
        return {
            "history_group_key": key,
            "history_group_label": source_display,
            "history_source_config_file": source_display,
            "history_run_label": _history_run_label_from_runtime(output_dir, runtime, task_id),
        }

    group = _history_config_group(methods_subdir, variant, preset)
    return {
        "history_group_key": _legacy_history_group_key(group),
        "history_group_label": _legacy_history_group_label(group),
        "history_source_config_file": "",
        "history_run_label": _history_run_label_from_runtime(output_dir, runtime, task_id),
    }


def _fill_history_group_meta(task: dict[str, Any]) -> None:
    existing_key = str(task.get("history_group_key") or "").strip()
    if existing_key:
        task["history_group_key"] = existing_key
        task["history_group_label"] = str(
            task.get("history_group_label")
            or task.get("history_source_config_file")
            or existing_key
        )
        task["history_source_config_file"] = str(task.get("history_source_config_file") or "")
        if not str(task.get("history_run_label") or "").strip():
            task["history_run_label"] = _history_run_label_from_runtime(
                str(task.get("training_output_dir") or task.get("output_dir") or ""),
                task,
                str(task.get("id") or ""),
            )
        return
    task.update(_history_group_meta(
        str(task.get("methods_subdir") or ""),
        str(task.get("variant") or ""),
        str(task.get("preset") or "default"),
        output_dir=str(task.get("training_output_dir") or task.get("output_dir") or ""),
        runtime_info=task,
        resume_info=task.get("resume_from") if isinstance(task.get("resume_from"), dict) else None,
        task_id=str(task.get("id") or ""),
    ))


def _history_run_label_from_runtime(
    output_dir: str,
    runtime_info: dict[str, Any] | None,
    task_id: str = "",
) -> str:
    runtime = runtime_info if isinstance(runtime_info, dict) else {}
    for key in ("run_dir", "training_output_dir", "output_dir"):
        raw = str(runtime.get(key) or "").strip()
        label = _history_run_label_from_path(raw)
        if label:
            return label
    return _history_run_label_from_path(output_dir) or str(task_id or "").strip()


def _history_run_label_from_path(value: str) -> str:
    path = _resolve_display_path(str(value or ""))
    if path is None:
        return ""
    if path.name == "training_output":
        return path.parent.name
    return path.name


def _legacy_history_group_key(group: dict[str, str]) -> str:
    return "legacy:" + "\u0001".join([
        group.get("methods_subdir") or "",
        group.get("variant") or "",
        group.get("preset") or "default",
    ])


def _legacy_history_group_label(group: dict[str, str]) -> str:
    return f"{group.get('methods_subdir') or '-'} / {group.get('variant') or '-'} / {group.get('preset') or 'default'}"


def _mark_orphaned_running_history_tasks() -> int:
    count = 0
    for meta_path in _history_meta_paths():
        if _is_deleting_history_dir(meta_path.parent):
            continue
        meta = _read_json(meta_path)
        if not meta or meta.get("state") != "running":
            continue
        task_dir = meta_path.parent
        finished_at = _last_history_event_ts(task_dir, meta)
        interrupted_at = time.time()
        meta.update({
            "state": "interrupted",
            "finished_at": finished_at,
            "finished_at_text": _format_ts(finished_at),
            "message": "WebUI 上次退出时任务仍标记为运行中，已自动标记为中断。",
            "returncode": meta.get("returncode"),
            "log_count": _count_jsonl(task_dir / "logs.jsonl"),
            "metric_count": _count_jsonl(task_dir / "metrics.jsonl"),
            "interrupted_at": interrupted_at,
            "interrupted_at_text": _format_ts(interrupted_at),
        })
        try:
            _write_json_atomic(meta_path, meta)
        except OSError:
            continue
        count += 1
    return count


def _last_history_event_ts(task_dir: Path, meta: dict[str, Any]) -> float:
    candidates = [
        _float_or_none(meta.get("finished_at")),
        _float_or_none(meta.get("updated_at")),
    ]
    for filename in ("logs.jsonl", "metrics.jsonl", "system.jsonl"):
        records = _read_jsonl(task_dir / filename)
        for record in reversed(records):
            ts = _float_or_none(record.get("ts"))
            if ts is not None:
                candidates.append(ts)
                break
    candidates.append(_float_or_none(meta.get("started_at")))
    candidates = [value for value in candidates if value is not None]
    return max(candidates) if candidates else time.time()


def _ensure_history_average_speed_meta(meta_path: Path, task_dir: Path, meta: dict[str, Any]) -> None:
    if str(meta.get("job") or "").strip() != "training":
        return
    if str(meta.get("state") or "").strip() in {"running", "compiling", "queued"}:
        return
    current_version = _int_or_none(meta.get("average_step_speed_version"))
    existing_seconds = _float_or_none(meta.get("average_step_seconds"))
    if (
        current_version == _history_average_speed_version()
        and existing_seconds is not None
        and existing_seconds > 0
    ):
        return
    stats = _history_average_speed_from_logs(task_dir / "logs.jsonl")
    if not stats:
        return
    now = time.time()
    meta.update(stats)
    meta["average_step_speed_version"] = _history_average_speed_version()
    meta["average_step_computed_at"] = now
    meta["average_step_computed_at_text"] = _format_ts(now)
    try:
        _write_json_atomic(meta_path, meta)
    except OSError:
        pass


def _history_average_speed_from_logs(logs_path: Path) -> dict[str, Any] | None:
    first: tuple[int, float] | None = None
    last: tuple[int, float] | None = None
    sample_count = 0
    try:
        with logs_path.open("r", encoding="utf-8") as f:
            for raw in f:
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if record.get("kind") != "progress":
                    continue
                sample = _training_progress_log_sample(record)
                if sample is None:
                    continue
                step, ts = sample
                sample_count += 1
                if first is None:
                    first = sample
                    continue
                first_step, first_ts = first
                if step <= first_step or ts <= first_ts:
                    continue
                if last is None or step > last[0] or (step == last[0] and ts > last[1]):
                    last = sample
    except OSError:
        return None
    if first is None or last is None:
        return None
    first_step, first_ts = first
    last_step, last_ts = last
    step_delta = last_step - first_step
    seconds_delta = last_ts - first_ts
    if step_delta <= 0 or seconds_delta <= 0:
        return None
    seconds_per_step = seconds_delta / step_delta
    if not _is_finite_number(seconds_per_step) or seconds_per_step <= 0:
        return None
    return {
        "average_step_seconds": round(seconds_per_step, 4),
        "average_step_rate": _format_step_rate(seconds_per_step),
        "average_step_source": "logs.jsonl",
        "average_step_sample_count": sample_count,
        "average_step_start_step": first_step,
        "average_step_end_step": last_step,
        "average_step_started_at": first_ts,
        "average_step_finished_at": last_ts,
    }


def _training_progress_log_sample(record: dict[str, Any]) -> tuple[int, float] | None:
    ts = _float_or_none(record.get("ts"))
    if ts is None:
        return None
    line = str(record.get("line") or "")
    match = _training_progress_log_re().search(line)
    if not match:
        return None
    step = _int_or_none(match.group("cur"))
    total = _int_or_none(match.group("tot"))
    if step is None or total is None or step < 0 or total <= 0:
        return None
    return step, ts


def _history_task_dir(task_id: str) -> Path:
    safe_id = _safe_task_id(task_id)
    if safe_id != task_id:
        raise ValueError("任务 ID 不合法")
    task_dir = (_history_dir() / safe_id).resolve()
    try:
        task_dir.relative_to(_history_dir().resolve())
    except ValueError as exc:
        raise ValueError("任务 ID 不合法") from exc
    return task_dir


def _load_history_task(task_id: str) -> dict[str, Any]:
    task_dir = _history_task_dir(task_id)
    if not _path_exists(task_dir):
        raise FileNotFoundError("任务不存在")
    meta_path = task_dir / "meta.json"
    meta = _read_json(meta_path)
    if not meta:
        raise FileNotFoundError("任务元信息不存在")
    _repair_history_meta(meta_path, meta)
    _ensure_history_average_speed_meta(meta_path, task_dir, meta)
    snapshot_path = task_dir / "config.snapshot.toml"
    logs, logs_total, logs_truncated = _read_jsonl_limited(
        task_dir / "logs.jsonl",
        limit=_max_history_detail_log_records(),
    )
    system, system_total, system_truncated = _read_jsonl_limited(
        task_dir / "system.jsonl",
        limit=_max_history_detail_system_records(),
    )
    metrics = _history_metrics_for_task(task_dir, logs=logs)
    task = _history_summary(meta, task_dir)
    if str(task.get("job") or "").strip() == "training":
        task["linked_preprocess_task"] = _linked_preprocess_task_for_training(task)
    return {
        "ok": True,
        "task": task,
        "logs": logs,
        "metrics": metrics,
        "system": system,
        "limits": {
            "logs_total": logs_total,
            "logs_returned": len(logs),
            "logs_truncated": logs_truncated,
            "system_total": system_total,
            "system_returned": len(system),
            "system_truncated": system_truncated,
            "metrics_total": len(metrics),
            "metrics_returned": len(metrics),
            "metrics_truncated": False,
        },
        "config_toml": _read_text_file(snapshot_path),
    }


def _load_history_task_summary(task_id: str) -> dict[str, Any]:
    task_dir = _history_task_dir(task_id)
    if not _path_exists(task_dir):
        raise FileNotFoundError("任务不存在")
    meta_path = task_dir / "meta.json"
    meta = _read_json(meta_path)
    if not meta:
        raise FileNotFoundError("任务元信息不存在")
    _repair_history_meta(meta_path, meta)
    return {
        "ok": True,
        "task": _history_summary(meta, task_dir),
    }


def _history_log_path(task_id: str) -> Path:
    task_dir = _history_task_dir(task_id)
    if not _path_exists(task_dir):
        raise FileNotFoundError("任务不存在")
    path = (task_dir / "logs.jsonl").resolve()
    try:
        path.relative_to(task_dir.resolve())
    except ValueError as exc:
        raise ValueError("任务日志路径不合法") from exc
    if not _path_exists(path) or not path.is_file():
        raise FileNotFoundError("任务日志不存在")
    return path


def _history_artifact_path(task_id: str, artifact_key: str) -> Path:
    key = str(artifact_key or "").strip()
    if key in _history_artifact_files():
        return _history_task_file_artifact_path(task_id, _history_artifact_files()[key])
    if key in _history_runtime_artifact_fields():
        return _history_runtime_artifact_path(task_id, _history_runtime_artifact_fields()[key])
    raise ValueError("历史文件类型不支持")


def _history_task_file_artifact_path(task_id: str, filename: str) -> Path:
    task_dir = _history_task_dir(task_id)
    if not _path_exists(task_dir):
        raise FileNotFoundError("任务不存在")
    path = (task_dir / filename).resolve()
    try:
        path.relative_to(task_dir.resolve())
    except ValueError as exc:
        raise ValueError("历史文件路径不合法") from exc
    if not _path_exists(path) or not path.is_file():
        raise FileNotFoundError("历史文件不存在")
    return path


def _history_runtime_artifact_path(task_id: str, field: str) -> Path:
    task_dir = _history_task_dir(task_id)
    if not _path_exists(task_dir):
        raise FileNotFoundError("任务不存在")
    meta = _read_json(task_dir / "meta.json")
    if not meta:
        raise FileNotFoundError("任务元信息不存在")
    task = _history_summary(meta, task_dir)
    run_dir = _resolve_display_path(str(task.get("run_dir") or ""))
    path = _resolve_display_path(str(task.get(field) or ""))
    if run_dir is None or path is None:
        raise FileNotFoundError("运行文件不存在")
    run_dir = run_dir.resolve()
    path = path.resolve()
    try:
        path.relative_to(run_dir)
    except ValueError as exc:
        raise ValueError("运行文件路径不合法") from exc
    output_root = resolve_output_root().resolve()
    if not _path_is_relative_to(run_dir, output_root) and not _is_web_runtime_dir(run_dir):
        raise ValueError("运行文件路径不合法")
    if not _path_exists(path) or not path.is_file():
        raise FileNotFoundError("运行文件不存在")
    return path


def _update_history_task(task_id: str, patch: dict[str, Any], *, bind_group: bool = True) -> dict[str, Any]:
    if bind_group and set(patch.keys()) == {"group"}:
        expanded_task_ids = _bound_history_task_ids([task_id])
        tasks = [
            _update_history_task(bound_id, patch, bind_group=False)["task"]
            for bound_id in expanded_task_ids
        ]
        primary = next((task for task in tasks if task.get("id") == task_id), tasks[0] if tasks else {})
        return {
            "ok": True,
            "task": primary,
            "tasks": tasks,
            "updated": len(tasks),
        }

    task_dir = _history_task_dir(task_id)
    if not task_dir.exists():
        raise FileNotFoundError("任务不存在")
    meta_path = task_dir / "meta.json"
    meta = _read_json(meta_path)
    if not meta:
        raise FileNotFoundError("任务元信息不存在")

    if "name" in patch:
        meta["name"] = _clean_history_text(patch.get("name"), max_len=80)
    if "group" in patch:
        meta["group"] = _clean_history_text(patch.get("group"), max_len=48)
    if "archived" in patch:
        meta["archived"] = bool(patch.get("archived"))

    meta["updated_at"] = time.time()
    meta["updated_at_text"] = _format_ts(meta["updated_at"])
    _write_json_atomic(meta_path, meta)
    return {"ok": True, "task": _history_summary(meta, task_dir)}


def _history_task_ids_for_delete(task_id: str) -> list[str]:
    task_dir = _history_task_dir(task_id)
    if not _path_exists(task_dir):
        raise FileNotFoundError("任务不存在")
    meta = _read_json(task_dir / "meta.json")
    if not meta:
        return [task_id]
    _repair_history_meta(task_dir / "meta.json", meta)
    task = _history_summary(meta, task_dir)
    task_ids = [task_id]
    if str(task.get("job") or "").strip() != "training":
        return task_ids

    seen = {task_id}
    for candidate in _linked_preprocess_tasks_for_training(task):
        candidate_id = str(candidate.get("id") or "").strip()
        if not candidate_id or candidate_id in seen:
            continue
        task_ids.append(candidate_id)
        seen.add(candidate_id)
    return task_ids


def _default_history_archived(job: str) -> bool:
    return str(job or "").strip() == "preprocess"


def _history_task_archived(task: dict[str, Any]) -> bool:
    archived = bool(task.get("archived", False))
    if archived:
        return True
    if str(task.get("job") or "").strip() != "preprocess":
        return False
    return "updated_at" not in task


def _default_preprocess_history_name(task: dict[str, Any]) -> str:
    if str(task.get("job") or "").strip() == "training" and str(task.get("training_mode") or "") == "continue_lora":
        kind = str(task.get("continue_from_weight_kind") or "LoRA").strip() or "LoRA"
        name = str(task.get("continue_from_weight_name") or "").strip()
        suffix = f" · {name}" if name else ""
        return f"权重热启动 {kind}{suffix}"
    if str(task.get("job") or "").strip() != "preprocess":
        return ""
    label = str(task.get("history_run_label") or "").strip()
    if not label:
        label = _history_run_label_from_runtime(
            str(task.get("output_dir") or ""),
            _runtime_meta(task),
            str(task.get("id") or ""),
        )
    label = label or str(task.get("id") or "").strip()
    if not label:
        return "预处理"
    return label


def _is_legacy_auto_preprocess_name(value: Any, default_name: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    return bool(default_name) and text == f"预处理 {default_name}"


def _fill_history_runtime_meta(task: dict[str, Any]) -> None:
    run_dir_raw = str(task.get("run_dir") or "").strip()
    if not run_dir_raw:
        output_dir = _resolve_display_path(str(task.get("training_output_dir") or task.get("output_dir") or ""))
        if output_dir and output_dir.name == "training_output":
            run_dir_raw = _display_project_path(str(output_dir.parent))
            task["run_dir"] = run_dir_raw
    run_dir = _resolve_display_path(run_dir_raw)
    if not run_dir:
        return

    defaults = {
        "runtime_config_file": run_dir / "config.runtime.toml",
        "original_config_file": run_dir / "config.original.toml",
        "dataset_config_file": run_dir / "dataset.runtime.toml",
        "model_cache_dir": run_dir / "model_cache",
        "dataset_cache_dir": run_dir / "dataset_cache",
        "training_output_dir": run_dir / "training_output",
        "logs_dir": run_dir / "model_cache" / "logs",
    }
    for key, path in defaults.items():
        if not str(task.get(key) or "").strip():
            task[key] = _display_project_path(str(path))


def _history_snapshot_path(task_id: str) -> Path | None:
    task_dir = _history_task_dir(task_id)
    snapshot = task_dir / "config.snapshot.toml"
    if _path_exists(snapshot):
        return snapshot
    return None
