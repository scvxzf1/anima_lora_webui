"""Training launch helpers delegated from ``TrainingService``.

Compatibility facade. Implementation lives in:

- ``launcher_runtime``: shared facade accessors / accelerate helper
- ``launcher_start``: start / preprocess / pending training orchestration
- ``launcher_job``: job launch lifecycle, stop, launch guards
"""

from __future__ import annotations

from web.services.training.launcher_job import (
    _ensure_launch_allowed,
    _launch_job,
    _write_terminal,
    stop,
)
from web.services.training.launcher_runtime import (
    _accelerate_mixed_precision_for_training,
    _apply_gpu_whitelist,
    _apply_runtime_env,
    _clone_frozen_runtime_config,
    _display_project_path,
    _ensure_training_data_dirs,
    _load_config_file_config,
    _normalize_gpu_whitelist,
    _prepare_web_runtime_config,
    _resolve_display_path,
    _resolve_training_runtime_info,
    _root,
    _runtime_from_config_file,
    _runtime_meta,
    _sample_config_from_cfg,
    _training_facade,
    preflight_training_config,
)
from web.services.training.launcher_start import (
    _start_pending_training,
    _start_preprocess_unlocked,
    _start_unlocked,
    start,
    start_preprocess,
)

__all__ = [
    "start",
    "_start_unlocked",
    "start_preprocess",
    "_start_preprocess_unlocked",
    "_launch_job",
    "stop",
    "_start_pending_training",
    "_ensure_launch_allowed",
    "_write_terminal",
]
