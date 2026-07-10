"""Stage schedule validation for WebUI preflight/runtime gates."""

from __future__ import annotations

from typing import Any, Callable

from library.training.stage_schedule import (
    normalize_stage_dicts,
    parse_stage_specs,
    validate_stage_specs,
)


def check_stage_schedule(
    cfg: dict[str, Any],
    *,
    dataset_rows: list[dict[str, Any]] | None,
    add: Callable[..., None],
) -> None:
    """Validate stage_schedule when enabled; report errors via ``add``."""
    enabled = bool(cfg.get("stage_schedule_enabled"))
    raw_stages = cfg.get("stage_schedule")
    stages = normalize_stage_dicts(raw_stages)
    if not enabled:
        return
    if not stages:
        add("error", "stage_schedule", "已启用分阶段调度，但 stage_schedule 为空")
        return

    subset_count = len(dataset_rows or [])
    specs = parse_stage_specs(stages)
    problems = validate_stage_specs(
        specs,
        subset_count=subset_count if subset_count > 0 else None,
    )
    if subset_count <= 0:
        add("error", "stage_schedule", "已启用分阶段调度，但当前配置没有数据集行")
    for msg in problems:
        add("error", "stage_schedule", msg)
    if not problems and subset_count > 0:
        add("ok", "stage_schedule", f"分阶段调度有效，共 {len(specs)} 个阶段")


def validate_stage_schedule_or_raise(
    cfg: dict[str, Any],
    *,
    dataset_rows: list[dict[str, Any]] | None,
) -> None:
    """Raise ValueError when stage_schedule is enabled but invalid."""
    errors: list[str] = []

    def add(level: str, key: str, message: str, path=None) -> None:
        if level == "error":
            errors.append(str(message))

    check_stage_schedule(cfg, dataset_rows=dataset_rows, add=add)
    if errors:
        raise ValueError("stage_schedule invalid: " + "; ".join(errors))
