"""Pipeline-parallel compatibility gate shared by resume entrypoints."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import toml

from library.env import anima_home
from library.models.krea2_raw.pipeline_parallel import Krea2PipelineParallelConfig
from library.runtime.launch import resolve_training_world_size_for_gpu_selection
from library.training.compat_matrix import check_training_compat
from web.services.training.gpu import normalize_gpu_whitelist


def ensure_resume_pipeline_compatible(
    config_file: str,
    gpu_whitelist: list[Any] | None,
) -> None:
    """Reject PP resume requests before they launch or enter the queue."""

    expanded = Path(os.path.expandvars(config_file)).expanduser()
    config_path = expanded if expanded.is_absolute() else anima_home() / expanded
    try:
        config = toml.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, toml.TomlDecodeError) as exc:
        raise ValueError(
            f"续训流水线配置无法读取或解析: {config_file}: {exc}"
        ) from exc

    try:
        pipeline_config = Krea2PipelineParallelConfig.from_config(config)
    except ValueError as exc:
        raise ValueError(f"续训流水线预检测失败: {exc}") from exc
    if not pipeline_config.enabled:
        return

    gpu_selection = normalize_gpu_whitelist(gpu_whitelist)
    world_size = resolve_training_world_size_for_gpu_selection(gpu_selection)
    compatibility = check_training_compat(config, world_size=world_size)
    errors = [
        item.message
        for item in compatibility.errors
        if item.key in {"model_family", "pipeline_parallel"}
    ]
    if errors:
        raise ValueError("续训流水线预检测失败: " + "; ".join(errors))
