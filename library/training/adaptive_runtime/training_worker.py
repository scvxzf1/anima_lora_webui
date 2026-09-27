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


def _possible_oom(exc):
    message = str(exc).lower()
    return "out of memory" in message or "cublas_status_alloc_failed" in message


_TORCH_OOM_TYPES = tuple(
    dict.fromkeys(
        cls for cls in (getattr(torch, "OutOfMemoryError", None),
                        torch.cuda.OutOfMemoryError)
        if isinstance(cls, type)
    )
)


def _error_result(runtime, exc):
    traceback.print_exc()
    possible_oom = _possible_oom(exc)
    mismatch = "precision_contract_mismatch" in str(exc)
    return {
        **runtime.result("error"),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "failure_kind": "precision_contract_mismatch" if mismatch
        else "possible_oom" if possible_oom else "runtime",
        "reason": "precision_contract_mismatch" if mismatch
        else "possible_oom" if possible_oom else "runtime_error",
    }


def execute(args, directory, trainer_factory):
    runtime = TrainingRuntime(
        directory,
        precision_contract_id=getattr(args, "_adaptive_precision_contract_id", None),
        args=args,
    )
    trainer = None
    try:
        require_training_contract(args, world_size=int(os.environ.get("WORLD_SIZE", "1")))
        errors = retry_configuration_errors(lambda key, default=None: getattr(args, key, default))
        private = getattr(args, "_adaptive_training_worker", {})
        if (errors or getattr(args, "adaptive_oom_retry", False)
                or not islands_enabled(args)
                or Path(private.get("directory", "")).resolve() != Path(directory).resolve()):
            raise ValueError("Invalid isolated training worker contract: " + "; ".join(errors))
        # Validate H before model loading, which can OOM before any dtype
        # manifest exists and must remain eligible for a fresh-process retry.
        runtime.validate_precision()
        trainer = trainer_factory()
        trainer._adaptive_training_runtime = runtime
        trainer.train(args)
        if not runtime.finished:
            raise RuntimeError("Training returned without a completed adaptive training result")
        result = runtime.result("ok")
    except _TORCH_OOM_TYPES as exc:
        result = {**runtime.result("cuda_oom"), "error_type": type(exc).__name__, "error": str(exc)}
    except Exception as exc:
        result = _error_result(runtime, exc)
    if result["status"] in {"ok", "cuda_oom"}:
        try:
            runtime.validate_precision(
                getattr(trainer, "_adaptive_precision_manifest_model", None)
            )
            result["precision_manifest_id"] = runtime.precision_manifest_id
        except Exception as exc:
            result = _error_result(runtime, exc)
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
