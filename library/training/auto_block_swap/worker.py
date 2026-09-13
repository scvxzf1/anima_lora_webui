"""Private JSON-argument worker; not a second public training CLI."""

from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import sys
import traceback

from .process import ProbeComplete, write_result


def main(request: Path) -> None:
    import torch
    from train import AnimaTrainer
    from library.training.train_bootstrap import install_stop_signal_handlers

    install_stop_signal_handlers()
    args = Namespace(**json.loads(request.read_text(encoding="utf-8")))
    if not getattr(args, "_auto_swap_probe", None) or args.auto_block_swap:
        raise ValueError("Expected an isolated AUTO probe request")
    try:
        AnimaTrainer().train(args)
    except ProbeComplete as complete:
        result = complete.result
    except torch.cuda.OutOfMemoryError as exc:
        result = {"status": "cuda_oom", "error": str(exc)}
    except Exception as exc:
        traceback.print_exc()
        result = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    else:
        result = {
            "status": "error",
            "error": "Training returned without a complete probe result",
        }
    write_result(request.parent / "result.json", result)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
