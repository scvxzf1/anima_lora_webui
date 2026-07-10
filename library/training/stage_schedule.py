"""Percent-based multi-dataset stage schedule for curriculum training.

Stages bind to subset indices in the training dataset blueprint and cover
``[start_pct, end_pct)`` of ``max_train_steps`` (last stage includes 100%).

Training assumes every stage's VAE/TE caches were prepared before start;
switching only rebuilds bucket indices (no preprocess).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageSpec:
    """One curriculum stage."""

    subset_index: int
    start_pct: float
    end_pct: float
    name: str = ""

    def contains_progress(self, progress: float) -> bool:
        """Return True if ``progress`` in [0, 1] falls in this stage."""
        p = max(0.0, min(1.0, float(progress)))
        start = float(self.start_pct)
        end = float(self.end_pct)
        if end >= 1.0 - 1e-12:
            return p >= start
        return start <= p < end


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_stage_dicts(raw: Any) -> list[dict[str, Any]]:
    """Coerce config/UI payloads into a list of plain stage dicts."""
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        import json

        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("stage_schedule string is not valid JSON")
            return []
    if isinstance(raw, Mapping):
        # Allow {enabled, stages: [...]} wrappers.
        if "stages" in raw:
            raw = raw.get("stages")
        else:
            raw = [raw]
    if not isinstance(raw, Sequence) or isinstance(raw, (bytes, bytearray)):
        return []
    out: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, Mapping):
            continue
        start = _as_float(item.get("start_pct", item.get("startPct", 0.0)))
        end = _as_float(item.get("end_pct", item.get("endPct", 1.0)))
        # Accept 0-100 UI values.
        if start > 1.0 or end > 1.0:
            start = start / 100.0
            end = end / 100.0
        start = max(0.0, min(1.0, start))
        end = max(0.0, min(1.0, end))
        subset_index = _as_int(
            item.get("subset_index", item.get("subsetIndex", item.get("dataset_index", i))),
            i,
        )
        name = str(item.get("name") or f"阶段{i + 1}").strip() or f"阶段{i + 1}"
        out.append(
            {
                "name": name,
                "subset_index": max(0, subset_index),
                "start_pct": start,
                "end_pct": end,
            }
        )
    return out


def parse_stage_specs(raw: Any) -> list[StageSpec]:
    return [
        StageSpec(
            subset_index=int(d["subset_index"]),
            start_pct=float(d["start_pct"]),
            end_pct=float(d["end_pct"]),
            name=str(d.get("name") or ""),
        )
        for d in normalize_stage_dicts(raw)
    ]


def validate_stage_specs(
    stages: Sequence[StageSpec],
    *,
    subset_count: Optional[int] = None,
) -> list[str]:
    """Return human-readable problems; empty means OK."""
    problems: list[str] = []
    if not stages:
        problems.append("阶段表为空")
        return problems
    if abs(stages[0].start_pct - 0.0) > 1e-6:
        problems.append("第一阶段必须从 0% 开始")
    if abs(stages[-1].end_pct - 1.0) > 1e-6:
        problems.append("最后一阶段必须到 100%")
    for i, stage in enumerate(stages):
        if stage.end_pct <= stage.start_pct + 1e-9:
            problems.append(f"阶段{i + 1} 区间为空")
        if subset_count is not None and not (0 <= stage.subset_index < subset_count):
            problems.append(
                f"阶段{i + 1} 的 subset_index={stage.subset_index} 超出范围 0..{subset_count - 1}"
            )
        if i > 0:
            prev = stages[i - 1]
            if abs(prev.end_pct - stage.start_pct) > 1e-6:
                problems.append(
                    f"阶段{i} 与阶段{i + 1} 未贴齐 ({prev.end_pct:.4f} → {stage.start_pct:.4f})"
                )
            if stage.start_pct + 1e-9 < prev.end_pct and abs(prev.end_pct - stage.start_pct) > 1e-6:
                # Overlap beyond floating noise when not equal.
                if stage.start_pct < prev.end_pct - 1e-6:
                    problems.append(f"阶段{i} 与阶段{i + 1} 区间重叠")
    return problems


def resolve_stage_index(stages: Sequence[StageSpec], progress: float) -> int:
    """Pick active stage for progress in [0, 1]. Falls back to last stage."""
    if not stages:
        return 0
    p = max(0.0, min(1.0, float(progress)))
    for i, stage in enumerate(stages):
        if stage.contains_progress(p):
            return i
    return len(stages) - 1


def progress_from_steps(global_step: int, max_train_steps: int) -> float:
    total = max(1, int(max_train_steps))
    return max(0.0, min(1.0, float(global_step) / float(total)))


def stage_schedule_enabled(args: Any) -> bool:
    if not bool(getattr(args, "stage_schedule_enabled", False)):
        return False
    stages = parse_stage_specs(getattr(args, "stage_schedule", None))
    return len(stages) > 0


def active_subset_indices_for_progress(args: Any, progress: float) -> Optional[set[int]]:
    """Return active subset indices, or None when schedule is off."""
    if not stage_schedule_enabled(args):
        return None
    stages = parse_stage_specs(getattr(args, "stage_schedule", None))
    idx = resolve_stage_index(stages, progress)
    return {int(stages[idx].subset_index)}


def active_subset_indices_for_step(args: Any, global_step: int) -> Optional[set[int]]:
    max_steps = int(getattr(args, "max_train_steps", 0) or 0)
    return active_subset_indices_for_progress(args, progress_from_steps(global_step, max_steps))


def attach_stage_schedule_from_config(args: Any, config: Mapping[str, Any] | None) -> None:
    """Copy stage schedule fields from merged TOML onto argparse namespace."""
    if not config:
        return
    if "stage_schedule_enabled" in config:
        args.stage_schedule_enabled = bool(config.get("stage_schedule_enabled"))
    if "stage_schedule" in config:
        args.stage_schedule = normalize_stage_dicts(config.get("stage_schedule"))


def snapshot_full_image_data(dataset: Any, *, force: bool = False) -> None:
    """Freeze full image maps on every leaf dataset before stage filtering.

    Call this while the dataset still contains every subset. Filtering without
    a prior snapshot freezes the first active stage as "full" and later stages
    can never recover other subsets.
    """
    if dataset is None:
        return
    if hasattr(dataset, "datasets"):
        for member in dataset.datasets:
            snapshot_full_image_data(member, force=force)
        return
    snap = getattr(dataset, "snapshot_full_image_data", None)
    if callable(snap):
        snap(force=force)


def apply_active_subsets_to_dataset(dataset: Any, active_subset_indices: Optional[Iterable[int]]) -> bool:
    """Filter ``dataset`` buckets to subsets in ``active_subset_indices``.

    Returns True when the filter was applied successfully on at least one leaf.
    ``active_subset_indices is None`` restores the full training set.
    """
    if dataset is None:
        return False
    if hasattr(dataset, "datasets"):
        # DatasetGroup: apply to each member with the same index space on that
        # member's own subsets (WebUI single-dataset multi-subset is the common case).
        any_ok = False
        for member in dataset.datasets:
            if apply_active_subsets_to_dataset(member, active_subset_indices):
                any_ok = True
        if any_ok and hasattr(dataset, "refresh_concat_state"):
            dataset.refresh_concat_state()
        return any_ok

    rebuild = getattr(dataset, "rebuild_buckets_for_subsets", None)
    if callable(rebuild):
        return bool(rebuild(active_subset_indices))
    return False


def log_stage_switch(stage: StageSpec, index: int, global_step: int, max_steps: int) -> None:
    logger.info(
        "stage schedule → #%s %s subset=%s progress=%.1f%% (step %s/%s)",
        index + 1,
        stage.name or f"stage{index + 1}",
        stage.subset_index,
        100.0 * progress_from_steps(global_step, max_steps),
        global_step,
        max_steps,
    )
