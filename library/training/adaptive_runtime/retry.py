"""Pure, bounded OOM recovery decisions; never alter numerical precision."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class MemoryPlan:
    blocks_to_swap: int = 0
    gradient_checkpointing: bool = False
    micro_batch: int = 1
    accumulation: int = 1

    def __post_init__(self):
        if self.blocks_to_swap < 0 or min(self.micro_batch, self.accumulation) < 1:
            raise ValueError("Invalid memory plan")


@dataclass(frozen=True)
class RetryLimits:
    max_blocks: int
    max_attempts: int = 4
    swap_increment: int = 2
    allow_batch_change: bool = False
    allow_checkpoint_change: bool = False

    def __post_init__(self):
        if self.max_blocks < 0 or min(self.max_attempts, self.swap_increment) < 1:
            raise ValueError("Invalid recovery limits")


def next_plan(current: MemoryPlan, limits: RetryLimits, *, stage: str):
    """Return (plan, reason), or None when no authorized adjustment remains."""
    if current.blocks_to_swap > limits.max_blocks:
        raise ValueError("Current swap exceeds the model/host-memory limit")
    if stage in ("model_load", "forward", "backward"):
        if current.blocks_to_swap < limits.max_blocks:
            count = min(limits.max_blocks, current.blocks_to_swap + limits.swap_increment)
            return replace(current, blocks_to_swap=count), "increase_block_swap"
    if stage in ("forward", "backward"):
        if limits.allow_checkpoint_change and not current.gradient_checkpointing:
            return replace(current, gradient_checkpointing=True), "enable_checkpointing"
        if limits.allow_batch_change:
            # Only exact divisors preserve the configured effective batch.
            divisor = next((n for n in range(2, current.micro_batch + 1)
                            if current.micro_batch % n == 0), None)
            if divisor:
                return replace(current, micro_batch=current.micro_batch // divisor,
                               accumulation=current.accumulation * divisor), "reduce_micro_batch"
    return None


def run_recovery(initial, limits, runner, *, record, expected_precision_contract=None):
    """Runner must start a FRESH process and return a structured worker result.

    Only explicit CUDA OOM results are retried. A started optimizer requires a
    committed checkpoint path supplied by the worker; its validation/loading is
    the training adapter's responsibility. No checkpoint means fail closed.
    When supplied, ``expected_precision_contract`` must match every structured
    ``ok`` or ``cuda_oom`` result before any next plan is considered.
    """
    plan, resume = initial, None
    seen = set()
    history = []
    realized_manifest = None
    for attempt in range(limits.max_attempts):
        if plan in seen:
            history.append({"attempt": attempt, "plan": asdict(plan),
                            "stop_reason": "repeated_plan"})
            final = {"status": "failed", "attempts": history}
            record(final)
            return final
        seen.add(plan)
        result = runner(plan, attempt=attempt, resume=resume)
        if not isinstance(result, dict):
            result = {"status": "error", "error": "invalid structured worker result"}
        history.append({"attempt": attempt, "plan": asdict(plan), "result": result})
        record({"status": "running", "attempts": history})
        status = result.get("status")
        if expected_precision_contract is not None:
            actual = result.get("precision_contract_id")
            if status in {"ok", "cuda_oom"} and actual != expected_precision_contract:
                history[-1]["stop_reason"] = "precision_contract_mismatch"
                break
        manifest = result.get("precision_manifest_id")
        if expected_precision_contract is not None and status in {"ok", "cuda_oom"}:
            if manifest is None:
                # No DiT exists yet when loading itself OOMs; its dtype
                # partition is checked as soon as a later attempt installs it.
                if not (status == "cuda_oom" and result.get("stage") == "model_load"
                        and result.get("optimizer_started") is False):
                    history[-1]["stop_reason"] = "missing_precision_manifest"
                    break
            elif realized_manifest is None:
                realized_manifest = manifest
            elif manifest != realized_manifest:
                history[-1]["stop_reason"] = "realized_precision_manifest_mismatch"
                break
        if status == "ok":
            final = {"status": "ok", "selected": asdict(plan), "attempts": history}
            if expected_precision_contract is not None:
                final["precision_contract_id"] = expected_precision_contract
            record(final)
            return final
        if status != "cuda_oom":
            break
        progress = result.get("optimizer_started")
        if not isinstance(progress, bool):
            history[-1]["stop_reason"] = "unknown_optimizer_progress"
            break
        if progress:
            resume = result.get("committed_checkpoint")
            if not resume:
                history[-1]["stop_reason"] = "no_committed_checkpoint"
                break
        change = next_plan(plan, limits, stage=result.get("stage", "unknown"))
        if change is None:
            history[-1]["stop_reason"] = "no_authorized_adjustment"
            break
        plan, reason = change
        history[-1]["next_reason"] = reason
    final = {"status": "failed", "attempts": history}
    record(final)
    return final
