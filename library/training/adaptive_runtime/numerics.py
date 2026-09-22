"""Bounded finite-value plan discovery; NOT a numerical accuracy certificate."""

from __future__ import annotations


def promotion_candidate(result, excluded=()):
    """Only the first overflowing plain Linear with finite inputs is eligible."""
    if result.get("status") != "error" or result.get("failure_kind") != "nonfinite":
        return None
    event = result.get("first_nonfinite")
    if not isinstance(event, dict):
        return None
    name = event.get("module")
    inputs, outputs = event.get("inputs"), event.get("outputs")
    if not all(isinstance(items, list) and items and all(isinstance(t, dict) for t in items)
               for items in (inputs, outputs)):
        return None
    if (event.get("type") == "Linear" and isinstance(name, str) and name and name not in excluded
            and all(t.get("finite") is True for t in inputs)
            and any(t.get("finite") is False for t in outputs)):
        return name
    return None


def discover_finite_plan(runner, *, max_attempts=8, record):
    if max_attempts < 1:
        raise ValueError("At least one attempt is required")
    promoted = []
    attempts = []
    report = {"status": "searching", "precision_calibrated": False, "attempts": attempts}
    for attempt in range(max_attempts):
        result = runner(tuple(promoted), attempt=attempt)
        if not isinstance(result, dict):
            result = {"status": "error", "error": "invalid structured worker result"}
        attempts.append({"fp32_modules": list(promoted), "result": result})
        record(report)
        if result.get("status") == "ok":
            report.update(status="finite_only", fp32_modules=list(promoted))
            record(report)
            return report
        name = promotion_candidate(result, promoted)
        if name is None or attempt + 1 == max_attempts:
            break
        promoted.append(name)
    report.update(status="failed", fp32_modules=list(promoted))
    record(report)
    return report
