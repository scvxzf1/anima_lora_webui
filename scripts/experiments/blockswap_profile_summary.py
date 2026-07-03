#!/usr/bin/env python3
"""Summarize block-swap JSONL timing and slot lifecycle fields."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


SUMMARY_FIELDS = (
    "wait_ms",
    "host_wait_ms",
    "gpu_wait_ms",
    "h2d_ms",
    "enqueue_ms",
    "host_queue_ms",
    "prefetch_runway_ms",
    "enqueue_to_wait_ms",
    "estimated_ready_slack_ms",
    "slot_reuse_age_ms",
    "submit_lag_ms",
)


def _load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("ev") == "block_swap":
            events.append(event)
    return events


def _numbers(events: list[dict[str, Any]], field: str) -> list[float]:
    values: list[float] = []
    for event in events:
        value = event.get(field)
        if value is None:
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = round((len(ordered) - 1) * q)
    return ordered[max(0, min(len(ordered) - 1, idx))]


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _print_phase_summary(events: list[dict[str, Any]]) -> None:
    phases = sorted({str(event.get("phase") or "") for event in events})
    for phase in phases:
        rows = [event for event in events if str(event.get("phase") or "") == phase]
        print(f"\n## {phase or '(no phase)'} n={len(rows)}")
        print("| field | avg | p50 | p90 | max |")
        print("| --- | ---: | ---: | ---: | ---: |")
        for field in SUMMARY_FIELDS:
            values = _numbers(rows, field)
            if not values:
                continue
            print(
                "| "
                f"{field} | "
                f"{_fmt(statistics.fmean(values))} | "
                f"{_fmt(statistics.median(values))} | "
                f"{_fmt(_percentile(values, 0.90))} | "
                f"{_fmt(max(values))} |"
            )


def _print_top_events(events: list[dict[str, Any]], *, field: str, limit: int) -> None:
    rows = sorted(
        [event for event in events if event.get(field) is not None],
        key=lambda event: float(event.get(field) or 0.0),
        reverse=True,
    )[:limit]
    print(f"\n## top {limit} by {field}")
    print(
        "| phase | step | block | cpu | slot | lead | "
        "runway_ms | slack_ms | slot_reuse_ms | wait_ms | h2d_ms |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for event in rows:
        print(
            "| "
            f"{event.get('phase') or ''} | "
            f"{event.get('step') if event.get('step') is not None else ''} | "
            f"{event.get('block_idx') if event.get('block_idx') is not None else ''} | "
            f"{event.get('block_idx_to_cpu') if event.get('block_idx_to_cpu') is not None else ''} | "
            f"{event.get('slot_id') if event.get('slot_id') is not None else ''} | "
            f"{event.get('prefetch_lead_blocks') if event.get('prefetch_lead_blocks') is not None else ''} | "
            f"{_fmt(float(event.get('prefetch_runway_ms') or 0.0))} | "
            f"{_fmt(float(event.get('estimated_ready_slack_ms') or 0.0))} | "
            f"{_fmt(float(event.get('slot_reuse_age_ms') or 0.0))} | "
            f"{_fmt(float(event.get('wait_ms') or 0.0))} | "
            f"{_fmt(float(event.get('h2d_ms') or 0.0))} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Anima block_swap_profile.jsonl runway and slot metrics."
    )
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--top-field", default="estimated_ready_slack_ms")
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    events = _load_events(args.jsonl)
    print(f"# {args.jsonl}")
    print(f"events={len(events)}")
    _print_phase_summary(events)
    _print_top_events(events, field=args.top_field, limit=max(0, args.top))


if __name__ == "__main__":
    main()
