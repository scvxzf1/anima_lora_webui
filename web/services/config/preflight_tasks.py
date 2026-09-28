"""Resolve the effective dataset before checking model training capabilities."""

from pathlib import Path

import toml

from library.env import expand_env_vars_in_obj
from library.models.family_registry import get_model_family_spec
from library.training.task_contracts import (
    configured_training_task,
    dataset_requires_edit,
)


def task_check_config(cfg: dict, dataset_path: Path | None, add) -> dict:
    if not cfg.get("dataset_config"):
        return cfg
    if dataset_path is None or not dataset_path.is_file():
        spec = get_model_family_spec(cfg["model_family"])
        level = (
            "error"
            if configured_training_task(cfg, spec) == "edit"
            or dataset_requires_edit(cfg)
            else "warning"
        )
        add(
            level,
            "dataset_config",
            "数据集配置尚未生成，无法确认数据集任务；训练前将重新检查",
            dataset_path,
        )
        return {**cfg, "general": {}, "datasets": [], "_dataset_task_unknown": True}
    try:
        data = expand_env_vars_in_obj(
            toml.loads(dataset_path.read_text(encoding="utf-8"))
        )
    except (OSError, ValueError) as exc:
        add("error", "dataset_config", f"数据集配置无法读取: {exc}", dataset_path)
        return {**cfg, "general": {}, "datasets": []}
    if not isinstance(data.get("datasets"), list) or not data["datasets"]:
        add("error", "dataset_config", "数据集配置缺少训练子集", dataset_path)
    return {
        **cfg,
        "general": data.get("general", {}),
        "datasets": data.get("datasets", []),
    }


def training_task_summary(cfg: dict) -> dict:
    spec = get_model_family_spec(cfg["model_family"])
    return {
        "model_family": spec.name,
        "model_name": spec.display_name,
        "supported_tasks": sorted(spec.supported_tasks),
        "configured_task": configured_training_task(cfg, spec),
        "dataset_task": "unknown"
        if cfg.get("_dataset_task_unknown")
        else "edit"
        if dataset_requires_edit(cfg)
        else "t2i",
    }
