"""Joint finite-value and memory discovery using disposable preflight workers."""

from dataclasses import asdict

from library.training.adaptive_runtime.numerics import promotion_candidate
from library.training.adaptive_runtime.retry import next_plan


def discover_preflight_plan(initial, limits, runner, *, record, max_promotions=8,
                            allow_precision_promotions=True):
    """Runner(plan, promoted, attempt=...) starts each trial from base state.

    All worker updates are disposable calibration work, never user progress.
    Result must explicitly attest disposable_probe=True. No checkpoint is
    transferred across precision changes. Success means finite_only, not an
    accuracy certificate; validate the assembled policy separately.
    """
    if max_promotions < 0 or initial.blocks_to_swap > limits.max_blocks:
        raise ValueError("Invalid joint search bounds")
    plan, promoted, history = initial, [], []
    report = {"status": "searching", "precision_calibrated": False,
              "calibration_only": True, "production_ready": False, "attempts": history}
    for attempt in range(limits.max_attempts):
        result = runner(plan, tuple(promoted), attempt=attempt)
        if not isinstance(result, dict):
            result = {"status": "error", "error": "invalid structured worker result"}
        item = {"attempt": attempt, "plan": asdict(plan),
                "fp32_modules": list(promoted), "result": result}
        history.append(item)
        record(report)
        if result.get("disposable_probe") is not True:
            item["stop_reason"] = "worker_not_disposable"
            break
        if result.get("status") == "ok":
            resolved = result.get("fp32_modules")
            if (not isinstance(resolved, list) or not all(isinstance(n, str) for n in resolved)
                    or not set(promoted).issubset(resolved)):
                item["stop_reason"] = "precision_profile_not_applied"
                break
            report.update(status="finite_only", selected=asdict(plan), fp32_modules=list(promoted),
                          resolved_fp32_modules=list(resolved))
            record(report)
            return report
        if attempt + 1 == limits.max_attempts:
            item["stop_reason"] = "attempt_limit"
            break
        if result.get("status") == "cuda_oom":
            change = next_plan(plan, limits, stage=result.get("stage", "unknown"))
            if change is None:
                item["stop_reason"] = "no_authorized_memory_adjustment"
                break
            plan, item["next_reason"] = change
        else:
            resolved = result.get("fp32_modules", [])
            if not isinstance(resolved, list) or not all(isinstance(n, str) for n in resolved):
                item["stop_reason"] = "invalid_precision_profile"
                break
            name = promotion_candidate(result, [*promoted, *resolved])
            if name is None or not allow_precision_promotions:
                item["stop_reason"] = "no_eligible_precision_promotion"
                break
            if len(promoted) >= max_promotions:
                item["stop_reason"] = "promotion_limit"
                break
            promoted.append(name)
            item.update(next_reason="promote_linear_fp32", promoted_module=name)
        # Persist the decision before starting the next expensive worker.
        record(report)
    report.update(status="failed", fp32_modules=list(promoted))
    record(report)
    return report
