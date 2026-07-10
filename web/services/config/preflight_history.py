"""History output-dir reuse checks for training preflight."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from web.services.config.preflight_paths import _has_web_runtime_dirs
from web.services.config.preflight_runtime import (
    CONFIGS_DIR,
    _resolve_project_path,
)

def _check_output_dir_history_reuse(cfg: dict[str, Any], add) -> None:
    raw = str(cfg.get("output_dir") or "").strip()
    if not raw:
        return
    output_dir = _resolve_project_path(raw)
    if not _is_web_runtime_training_output_dir(output_dir):
        return

    matches = _history_training_tasks_for_output_dir(output_dir)
    if not matches:
        return
    labels = "、".join(_history_output_match_label(item) for item in matches[:3])
    if len(matches) > 3:
        labels += f" 等 {len(matches)} 个历史训练任务"
    add(
        "error",
        "output_dir",
        (
            f"当前运行配置的 output_dir 指向已有历史训练输出目录（{labels}）。"
            "从零训练或权重热启动继续写入这里，可能触发 save_last_n_epochs / "
            "checkpointing_last_n_epochs 清理旧权重或完整续训点。请改用“完整续训”，"
            "或从配置页重新预处理生成新的运行目录。"
        ),
        output_dir,
    )

def _is_web_runtime_training_output_dir(path: Path) -> bool:
    if path.name != "training_output":
        return False
    run_dir = path.parent
    return _has_web_runtime_dirs(run_dir) or (run_dir / "config.runtime.toml").is_file()

def _history_training_tasks_for_output_dir(output_dir: Path) -> list[dict[str, Any]]:
    history_root = CONFIGS_DIR / "web-training-history"
    if not history_root.is_dir():
        return []
    try:
        target = output_dir.resolve()
    except OSError:
        target = output_dir
    matches: list[dict[str, Any]] = []
    for meta_path in sorted(history_root.glob("*/meta.json")):
        task = _read_history_meta_for_output_reuse(meta_path)
        if not task or str(task.get("job") or "") != "training":
            continue
        if _history_task_reuses_output_dir(task, target):
            matches.append(task)
    return matches

def _read_history_meta_for_output_reuse(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}

def _history_task_reuses_output_dir(task: dict[str, Any], target: Path) -> bool:
    for candidate in _history_task_output_candidates(task):
        try:
            if candidate.resolve() == target:
                return True
        except OSError:
            if candidate == target:
                return True
    return False

def _history_task_output_candidates(task: dict[str, Any]) -> list[Path]:
    out: list[Path] = []
    for key in ("output_dir", "training_output_dir"):
        raw = str(task.get(key) or "").strip()
        if raw:
            out.append(_resolve_project_path(raw))
    run_dir_raw = str(task.get("run_dir") or "").strip()
    if run_dir_raw:
        out.append(_resolve_project_path(run_dir_raw) / "training_output")
    return out

def _history_output_match_label(task: dict[str, Any]) -> str:
    task_id = str(task.get("id") or "").strip()
    name = str(task.get("name") or task.get("history_run_label") or "").strip()
    if name and task_id:
        return f"{name} / {task_id}"
    return name or task_id or "未命名任务"
