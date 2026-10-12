"""Fresh-process training supervisor; only structured CUDA OOM permits retry."""

from argparse import Namespace
import copy
import json
import os
from pathlib import Path
import sys

from library.training.auto_block_swap.process import write_result
from library.training.auto_block_swap.capacity import preload_model_capacity
from .process import IsolatedRunner
from .contract import precision_contract_id, freeze_native_sources
from .retry import MemoryPlan, RetryLimits, run_recovery
from .training_config import require_training_contract, resolved_mode


def worker_arguments(args, request):
    data = json.loads(request.read_text(encoding="utf-8"))
    recovery = getattr(args, "adaptive_oom_retry_resume", False)
    private_resume = None
    if data["resume"] is not None:
        if not recovery:
            raise ValueError("Training data-cursor resume is not enabled")
        from .resume_checkpoint import validate_recovery_state
        owner = request.parent.parent.resolve()
        validate_recovery_state(data["resume"], owner_root=owner,
                                precision_contract_id=precision_contract_id(args))
        private_resume = {"path": data["resume"], "owner_root": str(owner)}
    worker = copy.deepcopy(vars(args))
    worker["_adaptive_precision_contract_id"] = precision_contract_id(args)
    capacity = getattr(args, "_adaptive_swap_capacity", None)
    if capacity is not None:
        worker["_adaptive_swap_capacity"] = capacity
    worker.update(adaptive_oom_retry=False, blocks_to_swap=data["plan"]["blocks_to_swap"],
                  resume=None, skip_until_initial_step=False,
                  _adaptive_training_resume=private_resume,
                  _adaptive_training_worker={"directory": str(request.parent.resolve())})
    if resolved_mode(args) != "native_fp16":
        worker.update(save_state=False, save_state_on_train_end=False,
                      save_every_n_steps=None, save_every_n_epochs=None, checkpointing_epochs=None)
    path = request.parent / "training-args.json"
    # The child starts only after close; do not create a world-readable JSON
    # temporary file containing the full merged configuration.
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                   "w", encoding="utf-8") as handle:
        json.dump(worker, handle, allow_nan=False)
    return [sys.executable, "-m", "library.training.adaptive_runtime.training_worker", str(path)]


def run_supervised(args):
    from .precision import resolve_adaptive_precision

    resolve_adaptive_precision(args)
    require_training_contract(args, world_size=int(os.environ.get("WORLD_SIZE", "1")))
    # Establish the identity before creating an owned recovery directory. An
    # unreadable CUDA device identity must not strand an unusable run there.
    frozen_args = Namespace(**copy.deepcopy(vars(args)))
    native = resolved_mode(frozen_args) == "native_fp16"
    recovery = getattr(frozen_args, "adaptive_oom_retry_resume", False)
    if native:
        freeze_native_sources(frozen_args)
    expected_precision_contract = precision_contract_id(frozen_args)
    capacity = preload_model_capacity(frozen_args)
    requested = args.adaptive_oom_retry_max_swap
    if requested > capacity["model_limit"]:
        raise ValueError(f"adaptive_oom_retry_max_swap={requested} exceeds model limit "
                         f"{capacity['model_limit']}")
    effective = min(requested or capacity["model_limit"], capacity["model_limit"])
    if args.blocks_to_swap > effective:
        raise ValueError(f"blocks_to_swap={args.blocks_to_swap} exceeds effective retry limit "
                         f"{effective}")
    capacity.update(requested_cap=requested, effective_cap=effective)
    frozen_args._adaptive_swap_capacity = capacity
    directory = Path(args.output_dir).resolve() / ".adaptive-recovery"
    if directory.exists():
        raise FileExistsError(f"Refusing to reuse recovery directory: {directory}")
    directory.mkdir(parents=True)
    # Freeze merged config instead of reloading files that could undo the
    # memory plan or change precision between fresh workers.
    isolated = IsolatedRunner(directory, lambda path: worker_arguments(frozen_args, path),
                              timeout=args.adaptive_oom_retry_timeout)

    def runner(plan, *, attempt, resume):
        try:
            return isolated(plan, attempt=attempt, resume=resume)
        finally:
            (directory / f"attempt-{attempt:03d}" / "training-args.json").unlink(missing_ok=True)

    parent_report = None
    if native:
        from library.training.precision_report import PrecisionReport
        parent_report = PrecisionReport(frozen_args)

    def record(report):
        write_result(directory / "summary.json", {
            **report, "scope": "experimental_training_committed_oom_resume" if recovery else
            "native_training_startup_oom_retry" if native else "experimental_training_startup_oom_retry",
            "production_ready": False, "adaptive_precision": args.adaptive_precision,
            "precision_contract_id": expected_precision_contract,
            "data_cursor_resume_supported": recovery,
            "native_scope": "committed_resume" if recovery and native else "startup_only" if native else None,
            "validation": "cpu_static_not_quality_certification",
        })
        if parent_report is not None:
            from library.training.precision_report import report_supervisor
            report_supervisor(parent_report, report, directory / "summary.json")

    return run_recovery(
        MemoryPlan(blocks_to_swap=args.blocks_to_swap, gradient_checkpointing=True),
        RetryLimits(max_blocks=effective,
                    max_attempts=args.adaptive_oom_retry_max_attempts,
                    swap_increment=args.adaptive_oom_retry_swap_increment,
                    swap_strategy="exponential"),
        runner, record=record, expected_precision_contract=expected_precision_contract,
        capacity=capacity, startup_only=native and not recovery,
    )
