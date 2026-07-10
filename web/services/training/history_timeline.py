"""History timeline aggregation helpers."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

from web.services.training.common import _float_or_none, _int_or_none
from web.services.training.context import training_facade as _training_facade
from web.services.training import progress_parser as _progress_parser
from web.services.training.history_meta import (
    _history_task_dir,
    _legacy_history_group_key,
    _legacy_history_group_label,
)
from web.services.training.history_store import (
    _list_history_tasks as _history_store_list_history_tasks,
)
from web.services.training.storage import (
    _read_json as _storage_read_json,
    _read_jsonl as _storage_read_jsonl,
)


def _list_history_tasks(*args, **kwargs):
    reader = getattr(_training_facade(), "_list_history_tasks", _history_store_list_history_tasks)
    return reader(*args, **kwargs)


def _max_timeline_log_records() -> int:
    return _training_facade().MAX_TIMELINE_LOG_RECORDS


def _max_timeline_metric_records() -> int:
    return _training_facade().MAX_TIMELINE_METRIC_RECORDS


def _progress_rate_sample_window() -> int:
    return _training_facade().PROGRESS_RATE_SAMPLE_WINDOW


def _metric_from_progress_jsonl_event(*args, **kwargs):
    return _training_facade()._metric_from_progress_jsonl_event(*args, **kwargs)


def _progress_event_wall_ts_from_started_at(*args, **kwargs):
    return _training_facade()._progress_event_wall_ts_from_started_at(*args, **kwargs)


def _read_json(*args, **kwargs):
    reader = getattr(_training_facade(), "_read_json", _storage_read_json)
    return reader(*args, **kwargs)


def _read_jsonl(*args, **kwargs):
    reader = getattr(_training_facade(), "_read_jsonl", _storage_read_jsonl)
    return reader(*args, **kwargs)


def _build_config_group_timeline(
    methods_subdir: str,
    variant: str,
    preset: str,
    *,
    group_key: str = "",
    include_archived: bool = False,
    task_ids: list[str] | None = None,
) -> dict[str, Any]:
    group = _history_config_group(methods_subdir, variant, preset)
    group_key = str(group_key or "").strip()
    selected_ids = _normalize_timeline_task_ids(task_ids)
    all_tasks = _list_history_tasks(include_archived=True)
    if selected_ids:
        tasks = _select_timeline_tasks_by_id(
            all_tasks,
            selected_ids,
            include_archived=include_archived,
        )
        groups = _timeline_groups_for_tasks(tasks)
        if len(groups) == 1:
            group = groups[0]
        else:
            group = {
                "methods_subdir": "手动选择",
                "variant": f"{len(groups)} 个配置分组",
                "preset": "selected",
            }
    else:
        tasks = [
            task for task in all_tasks
            if task.get("job") == "training"
            and (
                _task_history_group_matches(task, group_key)
                if group_key
                else _task_config_group_matches(task, group)
            )
            and (include_archived or not task.get("archived"))
        ]
        if group_key and tasks:
            group = _history_group_from_task(tasks[0])
    tasks.sort(key=lambda item: (float(item.get("started_at") or 0), str(item.get("id") or "")))
    if not tasks:
        if selected_ids:
            raise FileNotFoundError("没有找到可合并的已选训练任务")
        raise FileNotFoundError("这个配置文件分组没有可合并的训练任务")

    logs: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    next_visual_step = 1

    for index, task in enumerate(tasks, start=1):
        task_id = str(task.get("id") or "")
        task_dir = _history_task_dir(task_id)
        task_logs = _read_jsonl(task_dir / "logs.jsonl")
        visible_logs = [record for record in task_logs if record.get("kind") != "progress"]
        task_metrics = _timeline_training_metrics(_history_metrics_for_task(task_dir, logs=task_logs))
        if task_metrics:
            start_visual_step = next_visual_step
            next_visual_step = _assign_visual_steps(task_metrics, next_visual_step)
            end_visual_step = next_visual_step - 1
            display_step_offset = _timeline_resume_step_offset(task)
            start_display_step, end_display_step = _assign_display_steps(task_metrics, display_step_offset)
            raw_steps = [_int_or_none(item.get("step")) for item in task_metrics]
            raw_steps = [step for step in raw_steps if step is not None]
            start_raw_step = raw_steps[0] if raw_steps else None
            end_raw_step = raw_steps[-1] if raw_steps else None
        else:
            start_visual_step = None
            end_visual_step = None
            display_step_offset = _timeline_resume_step_offset(task)
            start_display_step = None
            end_display_step = None
            start_raw_step = None
            end_raw_step = None

        source_label = _timeline_task_label(task)
        for record in visible_logs:
            item = dict(record)
            item["source_task_id"] = task_id
            item["source_task_index"] = index
            item["source_task_label"] = source_label
            logs.append(item)

        for metric_offset, metric in enumerate(task_metrics):
            item = dict(metric)
            item["source_task_id"] = task_id
            item["source_task_index"] = index
            item["source_task_label"] = source_label
            item["stage_break_before"] = index > 1 and metric_offset == 0
            metrics.append(item)

        segments.append({
            "task": _timeline_task_brief(task),
            "index": index,
            "log_count": len(visible_logs),
            "raw_log_count": len(task_logs),
            "progress_count": max(0, len(task_logs) - len(visible_logs)),
            "metric_count": len(task_metrics),
            "loss_count": sum(1 for item in task_metrics if item.get("loss") is not None),
            "start_visual_step": start_visual_step,
            "end_visual_step": end_visual_step,
            "start_display_step": start_display_step,
            "end_display_step": end_display_step,
            "display_step_offset": display_step_offset,
            "start_raw_step": start_raw_step,
            "end_raw_step": end_raw_step,
        })

    logs.sort(key=lambda item: (
        float(item.get("ts") or 0),
        int(item.get("source_task_index") or 0),
        int(item.get("id") or 0),
    ))
    metrics.sort(key=lambda item: (
        float(item.get("ts") or 0),
        int(item.get("source_task_index") or 0),
        int(item.get("visual_step") or 0),
    ))

    max_logs = _max_timeline_log_records()
    max_metrics = _max_timeline_metric_records()
    if len(logs) > max_logs:
        logs = logs[-max_logs:]
    if len(metrics) > max_metrics:
        metrics = metrics[-max_metrics:]

    return {
        "ok": True,
        "mode": "config_group",
        "group": group,
        "tasks": [_timeline_task_brief(task) for task in tasks],
        "segments": segments,
        "logs": logs,
        "metrics": metrics,
        "summary": {
            "task_count": len(tasks),
            "log_count": len(logs),
            "raw_log_count": sum(segment["raw_log_count"] for segment in segments),
            "progress_count": sum(segment["progress_count"] for segment in segments),
            "metric_count": len(metrics),
            "loss_count": sum(1 for item in metrics if item.get("loss") is not None),
            "started_at": tasks[0].get("started_at") if tasks else None,
            "started_at_text": tasks[0].get("started_at_text") if tasks else "",
            "finished_at": tasks[-1].get("finished_at") if tasks and tasks[-1].get("finished_at") else None,
            "finished_at_text": tasks[-1].get("finished_at_text") if tasks and tasks[-1].get("finished_at") else "",
            "start_display_step": next((segment["start_display_step"] for segment in segments if segment["start_display_step"] is not None), None),
            "end_display_step": next((segment["end_display_step"] for segment in reversed(segments) if segment["end_display_step"] is not None), None),
            "include_archived": include_archived,
            "selection_mode": "manual" if selected_ids else "config_group",
            "selected_task_ids": [str(task.get("id") or "") for task in tasks],
            "group_count": len(_timeline_groups_for_tasks(tasks)),
        },
    }


def _normalize_timeline_task_ids(task_ids: list[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in task_ids or []:
        task_id = str(raw or "").strip()
        if not task_id or task_id in seen:
            continue
        out.append(task_id)
        seen.add(task_id)
    return out


def _select_timeline_tasks_by_id(
    tasks: list[dict[str, Any]],
    task_ids: list[str],
    *,
    include_archived: bool,
) -> list[dict[str, Any]]:
    by_id = {str(task.get("id") or ""): task for task in tasks}
    selected: list[dict[str, Any]] = []
    invalid: list[str] = []
    for task_id in task_ids:
        task = by_id.get(task_id)
        if (
            not task
            or task.get("job") != "training"
            or (task.get("archived") and not include_archived)
        ):
            invalid.append(task_id)
            continue
        selected.append(task)
    if invalid:
        raise ValueError("所选训练任务不存在、已隐藏或不能参与合并: " + ", ".join(invalid))
    return selected


def _timeline_groups_for_tasks(tasks: list[dict[str, Any]]) -> list[dict[str, str]]:
    groups: list[dict[str, str]] = []
    seen: set[str] = set()
    for task in tasks:
        group = _history_group_from_task(task)
        key = str(group.get("history_group_key") or "")
        if key in seen:
            continue
        seen.add(key)
        groups.append(group)
    return groups


def _history_config_group(methods_subdir: str, variant: str, preset: str) -> dict[str, str]:
    return {
        "methods_subdir": str(methods_subdir or "").strip(),
        "variant": str(variant or "").strip(),
        "preset": str(preset or "default").strip() or "default",
    }


def _task_config_group_matches(task: dict[str, Any], group: dict[str, str]) -> bool:
    task_group = _history_config_group(
        str(task.get("methods_subdir") or ""),
        str(task.get("variant") or ""),
        str(task.get("preset") or "default"),
    )
    return task_group == group


def _task_history_group_matches(task: dict[str, Any], group_key: str) -> bool:
    return str(task.get("history_group_key") or "").strip() == str(group_key or "").strip()


def _history_group_from_task(task: dict[str, Any]) -> dict[str, str]:
    group = _history_config_group(
        str(task.get("methods_subdir") or ""),
        str(task.get("variant") or ""),
        str(task.get("preset") or "default"),
    )
    history_key = str(task.get("history_group_key") or "").strip() or _legacy_history_group_key(group)
    history_label = str(task.get("history_group_label") or "").strip() or _legacy_history_group_label(group)
    source_config = str(task.get("history_source_config_file") or "").strip()
    run_label = str(task.get("history_run_label") or "").strip()
    return {
        **group,
        "key": history_key,
        "history_group_key": history_key,
        "history_group_label": history_label,
        "history_source_config_file": source_config,
        "history_run_label": run_label,
        "label": history_label,
    }


def _history_metrics_for_task(
    task_dir: Path,
    *,
    logs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    task_logs = logs if logs is not None else _read_jsonl(task_dir / "logs.jsonl")
    metrics = _read_jsonl(task_dir / "metrics.jsonl")
    progress_metrics = _metrics_from_progress_jsonl(task_dir / "progress.jsonl", task_dir)
    return _metrics_from_history(task_logs, metrics, progress_metrics)


def _metrics_from_progress_jsonl(progress_path: Path, task_dir: Path) -> list[dict[str, Any]]:
    events = _read_jsonl(progress_path)
    if not events:
        return []
    meta = _read_json(task_dir / "meta.json")
    started_at = _float_or_none(meta.get("started_at")) if isinstance(meta, dict) else None
    out: list[dict[str, Any]] = []
    rate_last: tuple[float, int] | None = None
    rate_samples: deque[float] = deque(maxlen=_progress_rate_sample_window())
    for event in events:
        ev = str(event.get("ev") or "")
        if ev not in {"step", "val"}:
            continue
        ts = _progress_event_wall_ts_from_started_at(event, started_at)
        step = _int_or_none(event.get("global_step"))
        rate = ""
        if ev == "step" and step is not None:
            rate, rate_last = _step_rate_text_from_sample(rate_last, rate_samples, step, ts)
        metric = _metric_from_progress_jsonl_event(
            event,
            ts,
            rate=rate,
        )
        if metric:
            out.append(metric)
    return out


def _step_rate_text_from_sample(
    last: tuple[float, int] | None,
    samples: deque[float],
    step: int,
    timestamp: float,
) -> tuple[str, tuple[float, int] | None]:
    return _progress_parser.step_rate_text_from_sample(last, samples, step, timestamp)


def _median_or_none(values: deque[float] | list[float]) -> float | None:
    return _progress_parser.median_or_none(values)


def _format_step_rate(seconds_per_step: float | None) -> str:
    return _progress_parser.format_step_rate(seconds_per_step)


def _is_finite_number(value: Any) -> bool:
    return _progress_parser.is_finite_number(value)


def _metrics_from_history(
    logs: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    progress_metrics: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[int | None, float | None, float | None, float | None, str]] = set()

    def add_metric(item: dict[str, Any]) -> None:
        normalized = _normalize_metric_record(item)
        if normalized is None:
            return
        key = _metric_seen_key(normalized)
        if key in seen:
            return
        seen.add(key)
        out.append(normalized)

    normalized_progress = [
        item
        for item in (_normalize_metric_record(record) for record in (progress_metrics or []))
        if item is not None
    ]
    has_structured_loss = any(
        str(item.get("kind") or "") != "val" and _float_or_none(item.get("loss")) is not None
        for item in normalized_progress
    )

    if has_structured_loss:
        for item in normalized_progress:
            add_metric(item)
        for item in metrics:
            normalized = _normalize_metric_record(item)
            if normalized is not None and _is_validation_metric(normalized):
                add_metric(normalized)
    else:
        for item in metrics:
            add_metric(item)
        for item in normalized_progress:
            add_metric(item)

    for record in logs:
        if record.get("kind") != "progress":
            continue
        parsed = _metric_from_progress_line(str(record.get("line") or ""))
        if parsed is None:
            continue
        if record.get("ts") is not None:
            parsed["ts"] = record.get("ts")
        if has_structured_loss and not _is_validation_metric(parsed):
            continue
        add_metric(parsed)

    out.sort(key=lambda item: (float(item.get("ts") or 0), int(item.get("step") or 0)))
    return out


def _is_validation_metric(item: dict[str, Any]) -> bool:
    return str(item.get("kind") or "") == "val" or _float_or_none(item.get("cmmd")) is not None


def _timeline_training_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _progress_parser.timeline_training_metrics(metrics)


def _normalize_metric_record(item: dict[str, Any]) -> dict[str, Any] | None:
    return _progress_parser.normalize_metric_record(item)


def _metric_from_progress_line(line: str) -> dict[str, Any] | None:
    return _progress_parser.metric_from_progress_line(line)


def _metric_seen_key(item: dict[str, Any]) -> tuple[int | None, float | None, float | None, float | None, str]:
    return _progress_parser.metric_seen_key(item)


def _assign_visual_steps(metrics: list[dict[str, Any]], next_step: int) -> int:
    for item in metrics:
        item["visual_step"] = next_step
        next_step += 1
    return next_step


def _timeline_resume_step_offset(task: dict[str, Any]) -> int:
    resume_from = task.get("resume_from")
    if not isinstance(resume_from, dict):
        return 0
    checkpoint_step = _int_or_none(resume_from.get("checkpoint_step"))
    return checkpoint_step if checkpoint_step is not None and checkpoint_step > 0 else 0


def _assign_display_steps(metrics: list[dict[str, Any]], offset: int) -> tuple[int | None, int | None]:
    return _progress_parser.assign_display_steps(metrics, offset)


def _timeline_task_brief(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": task.get("id", ""),
        "name": task.get("name", ""),
        "label": _timeline_task_label(task),
        "training_mode": task.get("training_mode", ""),
        "continue_from_weight_abs_path": task.get("continue_from_weight_abs_path", ""),
        "continue_from_weight_name": task.get("continue_from_weight_name", ""),
        "continue_from_weight_kind": task.get("continue_from_weight_kind", ""),
        "state": task.get("state", ""),
        "variant": task.get("variant", ""),
        "preset": task.get("preset", ""),
        "methods_subdir": task.get("methods_subdir", ""),
        "output_dir": task.get("output_dir", ""),
        "run_dir": task.get("run_dir", ""),
        "history_dir": task.get("history_dir", ""),
        "history_group_key": task.get("history_group_key", ""),
        "history_group_label": task.get("history_group_label", ""),
        "history_source_config_file": task.get("history_source_config_file", ""),
        "history_run_label": task.get("history_run_label", ""),
        "resume_from": task.get("resume_from") if isinstance(task.get("resume_from"), dict) else {},
        "started_at": task.get("started_at"),
        "started_at_text": task.get("started_at_text", ""),
        "finished_at": task.get("finished_at"),
        "finished_at_text": task.get("finished_at_text", ""),
        "log_count": int(task.get("log_count") or 0),
        "metric_count": int(task.get("metric_count") or 0),
        "archived": bool(task.get("archived", False)),
    }


def _timeline_task_label(task: dict[str, Any]) -> str:
    return str(
        task.get("name")
        or task.get("history_run_label")
        or f"{task.get('methods_subdir') or '-'} / {task.get('variant') or task.get('id') or '-'}"
    )
