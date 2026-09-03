"""Resume-state discovery and diagnostics for WebUI training history."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any, Callable


ResolveDisplayPath = Callable[[str], Path | None]
DisplayProjectPath = Callable[[str], str]
PathExists = Callable[[Path], bool]
ReadJson = Callable[[Path], dict[str, Any]]
IntOrNone = Callable[[Any], int | None]
FloatOrNone = Callable[[Any], float | None]
FormatTs = Callable[[float | None], str]
ResolveOutputRoot = Callable[[], Path]
PathIsRelativeTo = Callable[[Path, Path], bool]
IsWebRuntimeDir = Callable[[Path], bool]


def _list_resume_checkpoints(
    task: dict[str, Any],
    *,
    scheduler_required: bool = True,
    resolve_display_path: ResolveDisplayPath,
    display_project_path: DisplayProjectPath,
    path_exists: PathExists,
    read_json: ReadJson,
    int_or_none: IntOrNone,
    float_or_none: FloatOrNone,
    format_ts: FormatTs,
) -> list[dict[str, Any]]:
    output_dir = resolve_display_path(str(task.get("output_dir") or ""))
    if output_dir is None or not path_exists(output_dir) or not output_dir.is_dir():
        return []

    started_at = float_or_none(task.get("started_at"))
    finished_at = float_or_none(task.get("finished_at"))
    lower = started_at - 180 if started_at is not None else None
    upper = (finished_at + 180) if finished_at is not None else (datetime.now().timestamp() + 180)

    items: list[dict[str, Any]] = []
    for child in sorted(output_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        if _is_transient_resume_state_dir(child.name):
            continue
        state_file = child / "train_state.json"
        if not path_exists(state_file):
            continue
        state = read_json(state_file)
        step = int_or_none(state.get("current_step"))
        if step is None:
            continue
        epoch = int_or_none(state.get("current_epoch"))
        mtime = _state_mtime(child, state_file)
        scope = "task" if lower is not None and lower <= mtime <= upper else "other"
        if scope != "task":
            continue
        kind = _resume_state_kind(child.name)
        paired_weight = _paired_resume_weight(
            child,
            output_dir,
            path_exists=path_exists,
            display_project_path=display_project_path,
        )
        integrity = _resume_state_integrity(
            child,
            scheduler_required=scheduler_required,
        )
        items.append({
            "id": display_project_path(str(child)),
            "path": display_project_path(str(child)),
            "name": child.name,
            "kind": kind,
            "kind_label": _resume_state_kind_label(kind),
            "scope": scope,
            "scope_label": "本任务" if scope == "task" else "同目录其他训练",
            "epoch": epoch,
            "step": step,
            "current_epoch": epoch,
            "current_step": step,
            "mtime": mtime,
            "mtime_text": format_ts(mtime),
            "train_state_file": display_project_path(str(state_file)),
            "paired_weight": paired_weight,
            "state_integrity": integrity,
            "state_complete": bool(integrity.get("ok")),
            "missing_state_files": list(integrity.get("missing") or []),
        })

    items.sort(key=_resume_state_sort_key)
    return items


def _resume_checkpoint_diagnostic(
    task: dict[str, Any],
    checkpoints: list[dict[str, Any]] | None = None,
    *,
    resolve_display_path: ResolveDisplayPath,
    display_project_path: DisplayProjectPath,
    path_exists: PathExists,
    float_or_none: FloatOrNone,
    resolve_output_root: ResolveOutputRoot,
    path_is_relative_to: PathIsRelativeTo,
    is_web_runtime_dir: IsWebRuntimeDir,
) -> dict[str, Any]:
    raw_output_dir = str(task.get("output_dir") or "")
    output_dir = resolve_display_path(raw_output_dir)
    diagnostic: dict[str, Any] = {
        "output_dir": raw_output_dir,
        "output_dir_resolved": display_project_path(str(output_dir)) if output_dir is not None else "",
        "output_dir_valid": output_dir is not None,
        "output_dir_exists": bool(output_dir is not None and path_exists(output_dir)),
        "output_dir_is_dir": bool(output_dir is not None and path_exists(output_dir) and output_dir.is_dir()),
        "all_subdir_count": 0,
        "state_dir_count": 0,
        "train_state_count": 0,
        "complete_state_count": 0,
        "incomplete_state_count": 0,
        "missing_state_files": [],
        "checkpoint_count": len(checkpoints or []),
        "reason": "",
        "recommendation": "如需权重热启动，可回到配置页选择这个任务导出的 LoRA/LoHa/LoKr/GLoRA 权重；热启动不会恢复 optimizer、scheduler 和已完成步数。",
    }
    if output_dir is None:
        diagnostic["reason"] = "这个历史任务记录的输出目录不合法，无法扫描完整续训状态。"
        return diagnostic
    if not path_exists(output_dir):
        diagnostic["reason"] = "这个历史任务记录的输出目录不存在，完整续训所需的 train_state.json 状态目录无法读取。"
        return diagnostic
    if not output_dir.is_dir():
        diagnostic["reason"] = "这个历史任务记录的输出路径不是目录，无法扫描完整续训状态。"
        return diagnostic

    all_subdirs = [
        child
        for child in output_dir.iterdir()
        if child.is_dir() and not _is_transient_resume_state_dir(child.name)
    ]
    state_dirs = [child for child in all_subdirs if child.name.endswith("-state")]
    diagnostic["all_subdir_count"] = len(all_subdirs)
    diagnostic["state_dir_count"] = len(state_dirs)
    diagnostic["train_state_count"] = sum(1 for child in state_dirs if path_exists(child / "train_state.json"))
    integrity_items = [
        item.get("state_integrity")
        for item in (checkpoints or [])
        if isinstance(item.get("state_integrity"), dict)
    ]
    diagnostic["complete_state_count"] = sum(1 for item in integrity_items if item.get("ok"))
    diagnostic["incomplete_state_count"] = sum(1 for item in integrity_items if not item.get("ok"))
    missing_files: list[str] = []
    for item in integrity_items:
        missing_files.extend(str(name) for name in (item.get("missing") or []))
    diagnostic["missing_state_files"] = sorted(set(missing_files))
    resume_from = task.get("resume_from") if isinstance(task.get("resume_from"), dict) else {}
    resume_checkpoint = str(resume_from.get("checkpoint") or "").strip()
    resume_checkpoint_path = resolve_display_path(resume_checkpoint) if resume_checkpoint else None
    resume_train_state_exists = bool(
        resume_checkpoint_path is not None and path_exists(resume_checkpoint_path / "train_state.json")
    )
    diagnostic["resume_source_checkpoint"] = resume_checkpoint
    diagnostic["resume_source_train_state_exists"] = resume_train_state_exists
    if diagnostic["complete_state_count"]:
        diagnostic["reason"] = "已找到可完整续训的状态目录。"
        return diagnostic
    if checkpoints and diagnostic["incomplete_state_count"]:
        diagnostic["reason"] = (
            "找到包含 train_state.json 的状态目录，但缺少完整续训必需的 "
            f"{'、'.join(diagnostic['missing_state_files'])}，无法恢复 optimizer/scheduler 状态。"
        )
        return diagnostic
    if resume_checkpoint and not resume_train_state_exists and int(task.get("metric_count") or 0) == 0:
        diagnostic["reason"] = "这次续训没有产生训练步，且完整续训状态目录已不存在；可用缓存/权重仍可能存在，但 optimizer/scheduler 状态无法恢复。"
        return diagnostic
    if diagnostic["train_state_count"]:
        diagnostic["reason"] = "输出目录里存在 train_state.json 状态目录，但不属于当前历史任务时间范围。"
        return diagnostic
    if diagnostic["state_dir_count"]:
        diagnostic["reason"] = "输出目录里有子目录，但没有包含 train_state.json 的完整续训状态目录。"
        return diagnostic

    try:
        output_root = resolve_output_root().resolve()
        if not path_is_relative_to(output_dir, output_root) and not is_web_runtime_dir(output_dir):
            diagnostic["reason"] = "输出目录存在，但不在当前 WebUI 输出根目录内，也没有 WebUI runtime 标记。"
            return diagnostic
    except Exception:
        pass

    diagnostic["reason"] = "输出目录里没有完整续训状态目录；旧版本训练完成时 checkpoint-state 可能已被清理，或该配置未写出训练状态。"
    return diagnostic


def _is_transient_resume_state_dir(name: str) -> bool:
    return name.endswith((".tmp", ".backup"))


def _optimizer_arg_bool(
    optimizer_args: Any,
    name: str,
    *,
    default: bool,
) -> bool:
    if not isinstance(optimizer_args, (list, tuple)):
        return default
    for item in optimizer_args:
        key, separator, raw_value = str(item).partition("=")
        if not separator or key.strip().lower() != name.lower():
            continue
        value = raw_value.strip().strip("'\"").lower()
        if value in {"false", "0", "no", "off", "none"}:
            return False
        if value in {"true", "1", "yes", "on"}:
            return True
    return default


def _resume_scheduler_state_required(config: dict[str, Any] | None) -> bool:
    config = config if isinstance(config, dict) else {}
    optimizer_type = str(config.get("optimizer_type") or "AdamW").strip().lower()
    if optimizer_type == "automagic":
        return False
    if optimizer_type == "prodigyplusschedulefree":
        schedule_free = _optimizer_arg_bool(
            config.get("optimizer_args"),
            "use_schedulefree",
            default=True,
        )
        return not schedule_free
    return not optimizer_type.endswith("schedulefree")


def _resume_state_integrity(
    state_dir: Path,
    *,
    scheduler_required: bool = True,
) -> dict[str, Any]:
    """Check the minimum Accelerate files needed for a real full resume."""
    checks = {
        "train_state": (state_dir / "train_state.json").exists(),
        "model": _state_has_any_file(state_dir, ("model.safetensors", "model_*.safetensors", "pytorch_model*.bin")),
        "optimizer": _state_has_any_file(state_dir, ("optimizer.bin", "optimizer_*.bin", "optimizer*.bin")),
        "scheduler": _state_has_any_file(state_dir, ("scheduler.bin", "scheduler_*.bin", "scheduler*.bin")),
        "random_state": _state_has_any_file(state_dir, ("random_states_*.pkl", "random_state*.pkl")),
    }
    labels = {
        "train_state": "train_state.json",
        "model": "model.safetensors",
        "optimizer": "optimizer.bin",
    }
    if scheduler_required:
        labels["scheduler"] = "scheduler.bin"
    missing = [label for key, label in labels.items() if not checks.get(key)]
    return {
        "ok": not missing,
        "missing": missing,
        "scheduler_required": scheduler_required,
        **checks,
    }


def _resume_state_integrity_unavailable_reason(integrity: dict[str, Any] | None) -> str:
    if not isinstance(integrity, dict) or integrity.get("ok"):
        return ""
    missing = [str(name) for name in (integrity.get("missing") or []) if str(name)]
    if not missing:
        return ""
    return (
        "这个状态目录不完整，缺少 "
        f"{'、'.join(missing)}，无法完整恢复 optimizer、scheduler 和步数。"
    )


def _select_resume_checkpoint(
    checkpoints: list[dict[str, Any]],
    checkpoint: str | None,
    *,
    resolve_display_path: ResolveDisplayPath,
    display_project_path: DisplayProjectPath,
) -> dict[str, Any] | None:
    if not checkpoints:
        return None
    if not checkpoint:
        return checkpoints[0]

    target = resolve_display_path(checkpoint)
    if target is None:
        return None
    target_text = display_project_path(str(target))
    for item in checkpoints:
        if display_project_path(str(item.get("path") or "")) == target_text:
            return item
    return None


def _resume_state_kind(name: str) -> str:
    if re.search(r"-checkpoint-\d{6}-state$", name):
        return "checkpoint"
    if name.endswith("-checkpoint-state"):
        return "checkpoint"
    if re.search(r"-step\d+-state$", name):
        return "step"
    if re.search(r"-\d{6}-state$", name):
        return "epoch"
    if name.endswith("-state"):
        return "last"
    return "state"


def _resume_state_kind_label(kind: str) -> str:
    return {
        "checkpoint": "自动续训检查点",
        "step": "按步保存状态",
        "epoch": "按轮保存状态",
        "last": "训练结束状态",
        "state": "训练状态",
    }.get(kind, "训练状态")


def _resume_state_sort_key(item: dict[str, Any]) -> tuple[int, int, int, float, str]:
    scope_rank = {"task": 0, "other": 1}
    kind_rank = {"checkpoint": 0, "last": 1, "epoch": 2, "step": 3, "state": 4}
    step = int(item.get("step") or -1)
    return (
        int(scope_rank.get(str(item.get("scope")), 9)),
        int(kind_rank.get(str(item.get("kind")), 9)),
        -step,
        -float(item.get("mtime") or 0),
        str(item.get("name") or ""),
    )


def _state_has_any_file(state_dir: Path, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
        if any(state_dir.glob(pattern)):
            return True
    return False


def _state_mtime(state_dir: Path, state_file: Path) -> float:
    for path in (state_file, state_dir):
        try:
            return float(path.stat().st_mtime)
        except OSError:
            continue
    return datetime.now().timestamp()


def _paired_resume_weight(
    state_dir: Path,
    output_dir: Path,
    *,
    path_exists: PathExists,
    display_project_path: DisplayProjectPath,
) -> str:
    name = state_dir.name
    if not name.endswith("-state"):
        return ""
    base_name = name[:-6]
    weight = output_dir / f"{base_name}.safetensors"
    if path_exists(weight):
        return display_project_path(str(weight))
    return ""
