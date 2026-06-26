"""Progress JSONL and metric text parsing helpers."""

from __future__ import annotations

from collections import deque
import re
import time
from typing import Any, Callable


def step_rate_text_from_sample(
    last: tuple[float, int] | None,
    samples: deque[float],
    step: int,
    timestamp: float,
) -> tuple[str, tuple[float, int] | None]:
    current_step = int(step)
    current_ts = float(timestamp)
    if current_step <= 0 or not is_finite_number(current_ts):
        return (format_step_rate(median_or_none(samples)) if samples else ""), last
    if last is None:
        return "", (current_ts, current_step)
    last_ts, last_step = last
    if current_step == last_step:
        return (format_step_rate(median_or_none(samples)) if samples else ""), last
    if current_step < last_step or current_ts <= last_ts:
        samples.clear()
        return "", (current_ts, current_step)
    step_delta = current_step - last_step
    seconds_per_step = (current_ts - last_ts) / step_delta
    if is_finite_number(seconds_per_step) and seconds_per_step > 0:
        samples.append(seconds_per_step)
    return format_step_rate(median_or_none(samples)), (current_ts, current_step)


def median_or_none(values: deque[float] | list[float]) -> float | None:
    finite = sorted(value for value in values if is_finite_number(value) and value > 0)
    if not finite:
        return None
    mid = len(finite) // 2
    if len(finite) % 2:
        return finite[mid]
    return (finite[mid - 1] + finite[mid]) / 2


def format_step_rate(seconds_per_step: float | None) -> str:
    if seconds_per_step is None or not is_finite_number(seconds_per_step) or seconds_per_step <= 0:
        return ""
    return f"{seconds_per_step:.2f}s/step"


def is_finite_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number == number and number not in {float("inf"), float("-inf")}


def timeline_training_metrics(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    max_step: int | None = None
    for item in metrics:
        step = int_or_none(item.get("step"))
        if step is not None:
            if max_step is not None and step < max_step:
                continue
            max_step = step if max_step is None else max(max_step, step)
        out.append(item)
    return out


def normalize_metric_record(item: dict[str, Any]) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    step = int_or_none(item.get("step"))
    if step is not None:
        out["step"] = step
    for key in ("loss", "lr", "cmmd"):
        value = float_or_none(item.get(key))
        if value is not None:
            out[key] = value
    if item.get("kind"):
        out["kind"] = str(item.get("kind"))
    if item.get("rate"):
        out["rate"] = str(item.get("rate"))
    ts = float_or_none(item.get("ts"))
    if ts is not None:
        out["ts"] = ts
    if not any(key in out for key in ("loss", "lr", "cmmd")):
        return None
    return out


def metric_from_progress_line(line: str) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    step_match = re.search(r"\|\s*(\d+)\/\d+\s*\[", line) or re.search(r"step[=:/\s]+(\d+)", line, re.IGNORECASE)
    if step_match:
        out["step"] = int(step_match.group(1))
    loss = extract_float_metric(line, ("avr_loss", "loss"))
    if loss is not None:
        out["loss"] = loss
    lr = extract_float_metric(line, ("lr", "learning_rate"))
    if lr is not None:
        out["lr"] = lr
    rate_match = re.search(r"([\d.]+\s*(?:s/it|it/s|s/step))", line, re.IGNORECASE)
    if rate_match:
        out["rate"] = rate_match.group(1).replace(" ", "")
    return out if any(key in out for key in ("loss", "lr")) else None


def metric_seen_key(item: dict[str, Any]) -> tuple[int | None, float | None, float | None, float | None, str]:
    step = int_or_none(item.get("step"))
    loss = float_or_none(item.get("loss"))
    lr = float_or_none(item.get("lr"))
    cmmd = float_or_none(item.get("cmmd"))
    return (
        step,
        round(loss, 8) if loss is not None else None,
        round(lr, 12) if lr is not None else None,
        round(cmmd, 8) if cmmd is not None else None,
        str(item.get("kind") or ""),
    )


def assign_display_steps(metrics: list[dict[str, Any]], offset: int) -> tuple[int | None, int | None]:
    start_step: int | None = None
    last_step: int | None = None
    for item in metrics:
        raw_step = int_or_none(item.get("step"))
        display_step = (offset + raw_step) if raw_step is not None else ((last_step or offset) + 1)
        if last_step is not None and display_step <= last_step:
            display_step = last_step + 1
        item["display_step"] = display_step
        item["display_step_offset"] = offset
        if start_step is None:
            start_step = display_step
        last_step = display_step
    return start_step, last_step


def live_metric_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int_or_none(item.get("step")),
        int_or_none(item.get("epoch")),
        round(float_or_none(item.get("loss")) or 0.0, 8) if float_or_none(item.get("loss")) is not None else None,
        round(float_or_none(item.get("lr")) or 0.0, 12) if float_or_none(item.get("lr")) is not None else None,
        round(float_or_none(item.get("cmmd")) or 0.0, 8) if float_or_none(item.get("cmmd")) is not None else None,
        str(item.get("kind") or ""),
    )


def progress_event_key(event: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(event.get("ev") or ""),
        event.get("ts"),
        event.get("global_step"),
        event.get("epoch"),
        event.get("val_step"),
        event.get("path"),
        event.get("status"),
        event.get("final_step"),
    )


def progress_event_wall_ts_from_started_at(
    event: dict[str, Any],
    started_at: float | None,
    *,
    now_fn: Callable[[], float] | None = None,
) -> float:
    rel_ts = float_or_none(event.get("ts"))
    if rel_ts is not None and started_at is not None:
        return started_at + rel_ts
    if rel_ts is not None and rel_ts > 1_000_000_000:
        return rel_ts
    return (now_fn or time.time)()


def metric_from_progress_jsonl_event(event: dict[str, Any], ts: float, *, rate: str = "") -> dict[str, Any] | None:
    metric: dict[str, Any] = {"ts": ts}
    step = int_or_none(event.get("global_step"))
    if step is not None:
        metric["step"] = step
    epoch = int_or_none(event.get("epoch"))
    if epoch is not None:
        metric["epoch"] = epoch
    if rate:
        metric["rate"] = rate

    if str(event.get("ev") or "") == "val":
        metric["kind"] = "val"
        cmmd = float_or_none(event.get("cmmd"))
        if cmmd is not None:
            metric["cmmd"] = cmmd
            metric["loss"] = cmmd
        val_step = int_or_none(event.get("val_step"))
        if val_step is not None:
            metric["val_step"] = val_step
    else:
        loss = progress_event_loss(event)
        if loss is not None:
            metric["loss"] = loss
        lr = progress_event_lr(event)
        if lr is not None:
            metric["lr"] = lr

    return metric if any(key in metric for key in ("loss", "lr", "cmmd")) else None


def progress_event_loss(event: dict[str, Any]) -> float | None:
    return first_float_field(event, ("loss", "loss/average", "loss/current"))


def progress_event_lr(event: dict[str, Any]) -> float | None:
    direct = first_float_field(
        event,
        ("lr", "learning_rate", "lr/unet", "lr/group0", "lr/textencoder"),
    )
    if direct is not None:
        return direct
    for key, value in event.items():
        key_text = str(key)
        if key_text.startswith("lr/") and not key_text.startswith("lr/d*lr/"):
            lr = float_or_none(value)
            if lr is not None:
                return lr
    for key, value in event.items():
        if str(key).startswith("lr/d*lr"):
            lr = float_or_none(value)
            if lr is not None:
                return lr
    return None


def first_float_field(record: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = float_or_none(record.get(key))
        if value is not None:
            return value
    return None


def extract_float_metric(text: str, names: tuple[str, ...]) -> float | None:
    for name in names:
        match = re.search(
            rf"{re.escape(name)}[=:/\s]+([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?|[+\-]?nan|[+\-]?inf(?:inity)?)",
            text,
            re.IGNORECASE,
        )
        if match:
            raw = match.group(1)
            value = float(raw)
            return value
    return None


def int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
