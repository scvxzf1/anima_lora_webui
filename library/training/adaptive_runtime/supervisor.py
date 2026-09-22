"""Fresh-process training supervisor; only structured CUDA OOM permits retry."""

from argparse import Namespace
import copy
import json
import os
from pathlib import Path
import sys

from library.training.auto_block_swap.process import write_result
from .process import IsolatedRunner
from .retry import MemoryPlan, RetryLimits, run_recovery
from .training_config import require_training_contract


def worker_arguments(args, request):
    data = json.loads(request.read_text(encoding="utf-8"))
    if data["resume"] is not None:
        raise ValueError("Training data-cursor resume is not supported yet")
    worker = copy.deepcopy(vars(args))
    worker.update(adaptive_oom_retry=False, blocks_to_swap=data["plan"]["blocks_to_swap"],
                  resume=data["resume"], skip_until_initial_step=bool(data["resume"]),
                  save_state=False, save_state_on_train_end=False,
                  save_every_n_steps=None, save_every_n_epochs=None, checkpointing_epochs=None,
                  _adaptive_training_worker={"directory": str(request.parent.resolve())})
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
    directory = Path(args.output_dir).resolve() / ".adaptive-recovery"
    if directory.exists():
        raise FileExistsError(f"Refusing to reuse recovery directory: {directory}")
    directory.mkdir(parents=True)
    # Freeze merged config instead of reloading files that could undo the
    # memory plan or change precision between fresh workers.
    frozen_args = Namespace(**copy.deepcopy(vars(args)))
    isolated = IsolatedRunner(directory, lambda path: worker_arguments(frozen_args, path),
                              timeout=args.adaptive_oom_retry_timeout)

    def runner(plan, *, attempt, resume):
        try:
            return isolated(plan, attempt=attempt, resume=resume)
        finally:
            (directory / f"attempt-{attempt:03d}" / "training-args.json").unlink(missing_ok=True)

    def record(report):
        write_result(directory / "summary.json", {
            **report, "scope": "experimental_training_startup_oom_retry",
            "production_ready": False, "adaptive_precision": args.adaptive_precision,
            "data_cursor_resume_supported": False,
        })

    return run_recovery(
        MemoryPlan(blocks_to_swap=args.blocks_to_swap, gradient_checkpointing=True),
        RetryLimits(max_blocks=args.adaptive_oom_retry_max_swap,
                    max_attempts=args.adaptive_oom_retry_max_attempts,
                    swap_increment=args.adaptive_oom_retry_swap_increment),
        runner, record=record,
    )
