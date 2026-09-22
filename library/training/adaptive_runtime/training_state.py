"""Atomic diagnostic training snapshots, not a data-cursor resume protocol."""

import json
from pathlib import Path

from library.training.auto_block_swap.process import write_result


def save_training_state(state, directory):
    from accelerate.utils.constants import (
        OPTIMIZER_NAME, RNG_STATE_NAME, SAFE_WEIGHTS_NAME, SCALER_NAME, SCHEDULER_NAME,
    )

    directory = Path(directory)
    destination = directory / f"state-{state.global_step:08d}"
    temporary = directory / f".state-{state.global_step:08d}.partial"
    if destination.exists() or temporary.exists():
        raise FileExistsError("Refusing to overwrite an existing adaptive state")
    temporary.mkdir()
    state.accelerator.save_state(str(temporary), safe_serialization=True)
    required = (SAFE_WEIGHTS_NAME, f"{OPTIMIZER_NAME}.bin", f"{SCHEDULER_NAME}.bin",
                f"{RNG_STATE_NAME}_0.pkl", SCALER_NAME, "train_state.json", "adaptive_precision.json")
    for name in required:
        path = temporary / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise ValueError(f"Incomplete adaptive training snapshot: {name}")
    progress = json.loads((temporary / "train_state.json").read_text(encoding="utf-8"))
    if progress.get("current_step") != state.global_step:
        raise ValueError("Adaptive snapshot step disagrees with training loop")
    inventory = {path.name: path.stat().st_size for path in temporary.iterdir() if path.is_file()}
    write_result(temporary / "snapshot.json", {
        "schema": "adaptive_training_snapshot_v1", "global_step": state.global_step,
        "data_cursor_resume_supported": False, "files": inventory,
    })
    temporary.rename(destination)
    return destination
