"""Aggregate per-row hot-test results into summary.json / summary.md."""
from __future__ import annotations

import json
from pathlib import Path

from bench.moss_tier_hot.matrix import TierRow


def _row_result(root: Path, row: TierRow) -> dict | None:
    path = root / row.name / "result.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def summarize(root: Path, rows: list[TierRow]) -> dict:
    collected = []
    for row in rows:
        result = _row_result(root, row)
        if result is None:
            continue
        result["tier"] = {
            "method": row.method, "resolution": row.resolution, "caps": row.caps,
            "tokens": row.tokens, "max_token_length": row.max_token_length,
        }
        collected.append(result)
    summary = {"root": str(root), "rows": collected}
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (root / "summary.md").write_text(_markdown(collected), encoding="utf-8")
    return summary


def _markdown(rows: list[dict]) -> str:
    lines = [
        "# MOSS-Image tier hot test summary",
        "",
        "| row | image tokens | cap | first s | tail mean s | tail med s | tail min-max s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        tier = row.get("tier", {})
        span = f"{_fmt(row.get('tail_min_seconds'))}-{_fmt(row.get('tail_max_seconds'))}"
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            row.get("row"), tier.get("tokens", "-"), tier.get("max_token_length", "-"),
            _fmt(row.get("first_step_seconds")), _fmt(row.get("tail_mean_seconds")),
            _fmt(row.get("tail_median_seconds")), span))
    lines += ["", "## GPU / memory", "",
              "| row | card mem peak MiB | temp max C | power max W | sm MHz min-max | exit |",
              "| --- | ---: | ---: | ---: | --- | ---: |"]
    for row in rows:
        gpu = row.get("gpu", {})
        lines.append("| {} | {} | {} | {} | {}-{} | {} |".format(
            row.get("row"), _fmt(gpu.get("mem_used_mib_peak"), 1),
            _fmt(gpu.get("temp_c_max"), 1), _fmt(gpu.get("power_w_max"), 2),
            _fmt(gpu.get("sm_mhz_min"), 0), _fmt(gpu.get("sm_mhz_max"), 0),
            row.get("train_exit_code")))
    lines.append("")
    return "\n".join(lines)
