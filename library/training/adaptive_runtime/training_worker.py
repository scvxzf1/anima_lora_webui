"""Private frozen-config worker for real training with structured CUDA OOM."""

from argparse import Namespace
import json
import os
from pathlib import Path
import sys
import traceback

import torch

from library.training.auto_block_swap.process import write_result
from .training_config import islands_enabled, require_training_contract, retry_configuration_errors
from .training_runtime import TrainingRuntime


def execute(args, directory, trainer_factory):
    runtime = TrainingRuntime(directory)
    try:
        require_training_contract(args, world_size=int(os.environ.get("WORLD_SIZE", "1")))
        errors = retry_configuration_errors(lambda key, default=None: getattr(args, key, default))
        private = getattr(args, "_adaptive_training_worker", {})
        if (errors or getattr(args, "adaptive_oom_retry", False)
                or not islands_enabled(args)
                or Path(private.get("directory", "")).resolve() != Path(directory).resolve()):
            raise ValueError("Invalid isolated training worker contract: " + "; ".join(errors))
        trainer = trainer_factory()
        trainer._adaptive_training_runtime = runtime
        trainer.train(args)
        if not runtime.finished:
            raise RuntimeError("Training returned without a completed adaptive training result")
        result = runtime.result("ok")
    except torch.cuda.OutOfMemoryError as exc:
        result = {**runtime.result("cuda_oom"), "error_type": type(exc).__name__, "error": str(exc)}
    except Exception as exc:
        traceback.print_exc()
        result = {**runtime.result("error"), "error_type": type(exc).__name__, "error": str(exc)}
    write_result(Path(directory) / "result.json", result)
    return result


def main(request):
    from library.training.train_bootstrap import install_stop_signal_handlers

    install_stop_signal_handlers()
    try:
        args = Namespace(**json.loads(request.read_text(encoding="utf-8")))
    finally:
        request.unlink(missing_ok=True)
    from train import AnimaTrainer

    execute(args, request.parent, AnimaTrainer)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
