"""Checkpoint and weight retention planning for generated training artifacts.

The planner is read-only.  It scans bounded scopes, classifies every artifact
into a role, applies a retention policy *per experiment unit* (never across
units) and returns the exact keep / delete decisions for a dry-run.  Deletion is
delegated to :class:`DataManagementService` so path, symlink, active-task and
plan-revision guards stay the single authority for what may be removed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from web.services.data_cleanup_gate import data_cleanup_lease
from web.services.data_management_service import DataManagementError, DataManagementService

MAX_RETENTION_ITEMS = 400
MAX_SCOPE_DIRS = 64
MAX_SCAN_FILES = 60000
MAX_SCAN_DIRS = 12000
MAX_SCAN_DEPTH = 10
MAX_ATOMIC_ENTRIES = 200000
MAX_PATTERNS = 16
MAX_PATTERN_LENGTH = 64
MAX_COUNT = 200

WEIGHT_EXTS = frozenset({".safetensors", ".sft", ".pt", ".pth", ".ckpt", ".bin", ".gguf"})
PROTECTED_EXTS = frozenset({
    ".json", ".jsonl", ".tsv", ".csv", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".log", ".txt", ".md", ".rst", ".py", ".sh", ".ipynb", ".pdf",
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg", ".html",
})
# A checkpoint directory that still holds these files is treated as a user
# artifact and is never deleted wholesale.
BLOCKING_EXTS = frozenset({
    ".log", ".txt", ".tsv", ".csv", ".md", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".pdf",
})

STATE_DIR_SUFFIXES = ("-state", "_state", ".state", "-states", "_states")
STATE_DIR_NAMES = frozenset({
    "optimizer", "optimizer_state", "optimizer_states", "optim_state", "optim_states",
    "scheduler", "scheduler_state", "lr_scheduler", "rng_state", "random_states",
    "zero_pp", "fsdp_state", "accelerator_state",
})
STATE_FILE_TOKENS = (
    "optimizer", "optim_state", "scheduler", "lr_scheduler", "grad_scaler", "scaler",
    "rng_state", "random_states", "random_state", "trainer_state", "train_state",
    "accelerator_state", "fsdp_state", "zero_pp", "state_dict",
)

PROGRESS_RE = re.compile(
    r"(?:^|[^a-z0-9])(step|epoch|iter|iteration|ckpt|checkpoint|save)[-_ .]?(\d{1,9})(?![0-9])",
    re.IGNORECASE,
)
TRAILING_NUM_RE = re.compile(r"(?:^|[^0-9])(\d{3,9})(?=[^0-9]*$)")
DATE_LIKE_RE = re.compile(r"^(19|20)\d{6}$")

CACHE_DIR_NAMES = frozenset({
    "cache", "caches", "latents", "latent", "latent_cache", "data_cache", "vae_cache", "te_cache",
})
ANALYSIS_DIR_NAMES = frozenset({
    "dump", "dumps", "activation", "activations", "activation_dump", "tensor_dump", "tensors",
})
CACHE_EXTS = frozenset({".npz"})
ANALYSIS_EXTS = frozenset({".npy", ".ptdump", ".bin_dump"})

ROLE_CHECKPOINT = "checkpoint"
ROLE_FINAL = "final_weight"
ROLE_STATE = "optimizer_state"
ROLE_CACHE = "data_cache"
ROLE_ANALYSIS = "analysis_tensor"
ROLE_PROTECTED = "protected"
ROLE_OTHER = "other"
ROLE_SYMLINK = "symlink"

SORT_MODES = frozenset({"progress", "mtime", "size"})
SORT_ORDERS = frozenset({"desc", "asc"})
RETENTION_MODES = frozenset({"keep_top_n", "delete_top_m", "keep_first_last", "delete_all_checkpoints"})

SORT_LABELS = {"progress": "训练步数", "mtime": "修改时间", "size": "文件体积"}


def _extension(name: str) -> str:
    return os.path.splitext(name)[1].lower()


def progress_of(name: str, *, allow_fallback: bool = True) -> int | None:
    """Return the checkpoint progress number of a name, using natural order.

    Explicit step/epoch/iter markers win. A bare trailing number is only used
    for weight files (adapter-000500.safetensors) and never for directory
    names, so date-stamped folders such as default_20260904-1315 stay
    ordinary directories; date-like 8-digit numbers are rejected too.
    """
    matches = PROGRESS_RE.findall(name)
    if matches:
        try:
            return int(matches[-1][1])
        except (TypeError, ValueError):
            return None
    if not allow_fallback:
        return None
    fallback = TRAILING_NUM_RE.search(os.path.splitext(name)[0])
    if fallback is None:
        return None
    digits = fallback.group(1)
    if DATE_LIKE_RE.match(digits):
        return None
    try:
        return int(digits)
    except (TypeError, ValueError):
        return None


def progress_token_count(name: str) -> int:
    """How many explicit step/epoch markers a name carries.

    A single marker on a weight file is the run exported final artifact
    ("model-...-step1500.safetensors"); two markers mean an intermediate
    checkpoint ("...-step1500-step00001000.safetensors").
    """
    return len(PROGRESS_RE.findall(name))


def _matches_patterns(name: str, patterns: tuple[str, ...]) -> bool:
    lowered = name.casefold()
    for pattern in patterns:
        token = pattern.strip().casefold()
        if not token:
            continue
        if any(char in token for char in "*?["):
            if fnmatchcase(lowered, token):
                return True
        elif token in lowered:
            return True
    return False


def _is_state_file(lowered_name: str) -> bool:
    return any(token in lowered_name for token in STATE_FILE_TOKENS)


def _is_state_dir(name: str) -> bool:
    lowered = name.casefold()
    if lowered.endswith(STATE_DIR_SUFFIXES):
        return True
    return lowered in STATE_DIR_NAMES


def _is_cache_dir(name: str) -> bool:
    return name.casefold() in CACHE_DIR_NAMES


def _is_analysis_dir(name: str) -> bool:
    return name.casefold() in ANALYSIS_DIR_NAMES


def _is_cache_path(name: str, directory_parts: tuple[str, ...]) -> bool:
    if any(part.casefold() in CACHE_DIR_NAMES for part in directory_parts):
        return True
    return "cache" in name.casefold()


def _is_analysis_path(name: str, directory_parts: tuple[str, ...]) -> bool:
    if any(part.casefold() in ANALYSIS_DIR_NAMES for part in directory_parts):
        return True
    lowered = name.casefold()
    return any(token in lowered for token in ("dump", "activation", "tensor", "matrix", "evidence"))


def classify_file(name: str, directory_parts: tuple[str, ...] = ()) -> tuple[str, int | None]:
    """Classify a regular file into a retention role."""
    ext = _extension(name)
    if ext in PROTECTED_EXTS:
        return ROLE_PROTECTED, None
    lowered = name.casefold()
    if _is_state_file(lowered):
        return ROLE_STATE, None
    if ext in CACHE_EXTS:
        if _is_cache_path(name, directory_parts):
            return ROLE_CACHE, None
        if _is_analysis_path(name, directory_parts):
            return ROLE_ANALYSIS, None
        # A standalone npz can be a user artifact or a latent cache; do not
        # delete it without a cache/analysis ownership signal.
        return ROLE_OTHER, None
    if ext in ANALYSIS_EXTS:
        return ROLE_ANALYSIS, None
    if ext in WEIGHT_EXTS:
        progress = progress_of(name)
        if progress is None:
            return ROLE_FINAL, None
        return ROLE_CHECKPOINT, progress
    return ROLE_OTHER, None


@dataclass(frozen=True)
class RetentionPolicy:
    sort_by: str = "progress"
    sort_order: str = "desc"
    mode: str = "keep_top_n"
    keep_count: int = 2
    delete_count: int = 3
    keep_first: int = 1
    keep_last: int = 1
    keep_final_weights: bool = True
    keep_weight_patterns: tuple[str, ...] = ()
    ensure_at_least_one_weight: bool = True
    clean_optimizer_states: bool = True
    clean_data_cache: bool = False
    clean_analysis_tensors: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "sort_by": self.sort_by,
            "sort_order": self.sort_order,
            "mode": self.mode,
            "keep_count": self.keep_count,
            "delete_count": self.delete_count,
            "keep_first": self.keep_first,
            "keep_last": self.keep_last,
            "keep_final_weights": self.keep_final_weights,
            "keep_weight_patterns": list(self.keep_weight_patterns),
            "ensure_at_least_one_weight": self.ensure_at_least_one_weight,
            "clean_optimizer_states": self.clean_optimizer_states,
            "clean_data_cache": self.clean_data_cache,
            "clean_analysis_tensors": self.clean_analysis_tensors,
        }

    @classmethod
    def from_payload(cls, payload: Any) -> "RetentionPolicy":
        if payload is None:
            payload = {}
        if not isinstance(payload, dict):
            raise DataManagementError("策略参数必须是 JSON 对象", 400)
        patterns_raw = payload.get("keep_weight_patterns", ())
        if isinstance(patterns_raw, str):
            patterns_raw = [part for part in re.split(r"[,;\s]+", patterns_raw) if part]
        if not isinstance(patterns_raw, (list, tuple)) or len(patterns_raw) > MAX_PATTERNS:
            raise DataManagementError(f"保留规则最多 {MAX_PATTERNS} 条", 400)
        patterns: list[str] = []
        for value in patterns_raw:
            if not isinstance(value, str):
                raise DataManagementError("保留规则必须是字符串", 400)
            token = value.strip()
            if not token:
                continue
            if len(token) > MAX_PATTERN_LENGTH:
                raise DataManagementError(f"保留规则不能超过 {MAX_PATTERN_LENGTH} 个字符", 400)
            patterns.append(token)
        return cls(
            sort_by=_choice(payload, "sort_by", SORT_MODES, "progress"),
            sort_order=_choice(payload, "sort_order", SORT_ORDERS, "desc"),
            mode=_choice(payload, "mode", RETENTION_MODES, "keep_top_n"),
            keep_count=_count(payload, "keep_count", 2),
            delete_count=_count(payload, "delete_count", 3),
            keep_first=_count(payload, "keep_first", 1),
            keep_last=_count(payload, "keep_last", 1),
            keep_final_weights=_flag(payload, "keep_final_weights", True),
            keep_weight_patterns=tuple(patterns),
            ensure_at_least_one_weight=_flag(payload, "ensure_at_least_one_weight", True),
            clean_optimizer_states=_flag(payload, "clean_optimizer_states", True),
            clean_data_cache=_flag(payload, "clean_data_cache", False),
            clean_analysis_tensors=_flag(payload, "clean_analysis_tensors", False),
        )


def _choice(payload: dict[str, Any], key: str, allowed: frozenset[str], default: str) -> str:
    value = payload.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str) or value not in allowed:
        raise DataManagementError(f"策略参数无效：{key} 只能是 {'/'.join(sorted(allowed))}", 400)
    return value


def _count(payload: dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool):
        raise DataManagementError(f"策略参数无效：{key} 必须是整数", 400)
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value.strip())
    if not isinstance(value, int) or value < 0 or value > MAX_COUNT:
        raise DataManagementError(f"策略参数无效：{key} 必须是 0 到 {MAX_COUNT} 的整数", 400)
    return value


def _flag(payload: dict[str, Any], key: str, default: bool) -> bool:
    value = payload.get(key, default)
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"0", "false", "no", "off"}:
        return False
    raise DataManagementError(f"策略参数无效：{key} 必须是布尔值", 400)


@dataclass
class Record:
    relative_path: str
    name: str
    kind: str
    role: str
    size_bytes: int
    mtime_ns: int
    progress: int | None
    unit: str
    has_foreign: bool = False


class _Budget:
    def __init__(self) -> None:
        self.files = 0
        self.dirs = 0
        self.truncated = False


def _atomic_dir_stats(path: Path, budget: _Budget, *, state_bundle: bool = False) -> tuple[int, bool]:
    """Return byte size plus whether an atomic directory is safe to erase.

    State, checkpoint and cache directories are candidates for whole-directory
    deletion. Any nested directory, symlink, protected summary artifact or
    unknown regular file makes that candidate fail closed. A user weight or a
    binary research artifact must never become deletable merely because its
    parent is named cache.
    """
    total = 0
    has_foreign = False
    stack = [path]
    seen = 0
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as children:
                for child in children:
                    seen += 1
                    if seen > MAX_ATOMIC_ENTRIES:
                        budget.truncated = True
                        return total, True
                    try:
                        info = child.stat(follow_symlinks=False)
                    except OSError:
                        has_foreign = True
                        continue
                    if stat.S_ISLNK(info.st_mode):
                        has_foreign = True
                        continue
                    if stat.S_ISDIR(info.st_mode):
                        # Nested directories need their own explicit scan and
                        # ownership classification; never erase them as a side
                        # effect of a generic parent directory name.
                        has_foreign = True
                        continue
                    if stat.S_ISREG(info.st_mode):
                        total += int(info.st_size)
                        ext = _extension(child.name)
                        resume_metadata = state_bundle and child.name == "train_state.json"
                        known_state_model = state_bundle and child.name == "model.safetensors"
                        known_payload = known_state_model or _is_state_file(child.name.casefold())
                        if not resume_metadata and (ext in PROTECTED_EXTS or not known_payload):
                            has_foreign = True
        except OSError:
            has_foreign = True
    return total, has_foreign


def _cache_dir_stats(path: Path, budget: _Budget, allowed_exts: frozenset[str]) -> tuple[int, bool]:
    """Inspect cache/tensor directories with an allow-list of disposable files."""
    total = 0
    has_foreign = False
    seen = 0
    try:
        with os.scandir(path) as children:
            for child in children:
                seen += 1
                if seen > MAX_ATOMIC_ENTRIES:
                    budget.truncated = True
                    return total, True
                try:
                    info = child.stat(follow_symlinks=False)
                except OSError:
                    has_foreign = True
                    continue
                if not stat.S_ISREG(info.st_mode):
                    has_foreign = True
                    continue
                total += int(info.st_size)
                if _extension(child.name) not in allowed_exts:
                    has_foreign = True
    except OSError:
        has_foreign = True
    return total, has_foreign


def scan_scope(scope: list[tuple[str, Path]]) -> tuple[list[Record], dict[str, Any]]:
    """Walk the selected directories with hard bounds; never follow symlinks."""
    records: list[Record] = []
    budget = _Budget()
    for scope_rel, scope_path in scope:
        if budget.truncated:
            break
        stack: list[tuple[Path, str, int]] = [(scope_path, scope_rel, 0)]
        while stack:
            current, rel, depth = stack.pop()
            if budget.dirs >= MAX_SCAN_DIRS:
                budget.truncated = True
                break
            budget.dirs += 1
            try:
                children = list(os.scandir(current))
            except OSError:
                continue
            for child in children:
                if budget.files >= MAX_SCAN_FILES:
                    budget.truncated = True
                    break
                try:
                    info = child.stat(follow_symlinks=False)
                except OSError:
                    continue
                child_rel = f"{rel}/{child.name}" if rel else child.name
                if stat.S_ISLNK(info.st_mode):
                    budget.files += 1
                    records.append(Record(child_rel, child.name, "symlink", ROLE_SYMLINK,
                                          0, int(info.st_mtime_ns), None, rel))
                    continue
                if stat.S_ISDIR(info.st_mode):
                    if child.name.startswith("."):
                        continue
                    if _is_state_dir(child.name):
                        budget.files += 1
                        size, foreign = _atomic_dir_stats(Path(child.path), budget, state_bundle=True)
                        records.append(Record(child_rel, child.name, "directory", ROLE_STATE,
                                              size, int(info.st_mtime_ns), progress_of(child.name), rel, foreign))
                        continue
                    if _is_cache_dir(child.name) or _is_analysis_dir(child.name):
                        budget.files += 1
                        role = ROLE_CACHE if _is_cache_dir(child.name) else ROLE_ANALYSIS
                        allowed_exts = CACHE_EXTS if role == ROLE_CACHE else CACHE_EXTS | ANALYSIS_EXTS
                        size, foreign = _cache_dir_stats(Path(child.path), budget, allowed_exts)
                        records.append(Record(child_rel, child.name, "directory", role,
                                              size, int(info.st_mtime_ns), None, rel, foreign))
                        continue
                    progress = progress_of(child.name, allow_fallback=False)
                    if progress is not None:
                        budget.files += 1
                        size, foreign = _atomic_dir_stats(Path(child.path), budget)
                        records.append(Record(child_rel, child.name, "directory", ROLE_CHECKPOINT,
                                              size, int(info.st_mtime_ns), progress, rel, foreign))
                        continue
                    if depth + 1 > MAX_SCAN_DEPTH:
                        budget.truncated = True
                        budget.files += 1
                        records.append(Record(child_rel, child.name, "directory", ROLE_OTHER,
                                              int(info.st_size), int(info.st_mtime_ns), None, rel))
                        continue
                    stack.append((Path(child.path), child_rel, depth + 1))
                    continue
                if not stat.S_ISREG(info.st_mode):
                    budget.files += 1
                    records.append(Record(child_rel, child.name, "other", ROLE_OTHER,
                                          0, int(info.st_mtime_ns), None, rel))
                    continue
                budget.files += 1
                directory_parts = tuple(part for part in rel.split("/") if part) if rel else ()
                role, progress = classify_file(child.name, directory_parts)
                records.append(Record(child_rel, child.name, "file", role,
                                      int(info.st_size), int(info.st_mtime_ns), progress, rel))
    return records, {
        "truncated": budget.truncated,
        "scanned_files": budget.files,
        "scanned_dirs": budget.dirs,
    }



ROLE_LABELS = {
    ROLE_CHECKPOINT: "中间检查点",
    ROLE_FINAL: "最终 / 主权重",
    ROLE_STATE: "优化器 / 恢复状态",
    ROLE_CACHE: "实验内缓存",
    ROLE_ANALYSIS: "分析中间张量",
    ROLE_PROTECTED: "配置 / 日志 / 指标",
    ROLE_OTHER: "其他文件",
    ROLE_SYMLINK: "符号链接",
}


def _iso(mtime_ns: int) -> str | None:
    try:
        return datetime.fromtimestamp(mtime_ns / 1_000_000_000, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _sort_key(record: Record, sort_by: str) -> tuple[Any, ...]:
    if sort_by == "mtime":
        return (record.mtime_ns, record.name)
    if sort_by == "size":
        return (record.size_bytes, record.name)
    return (record.progress if record.progress is not None else -1, record.mtime_ns, record.name)


def rank_records(records: list[Record], policy: RetentionPolicy) -> list[Record]:
    """Rank artifacts by the configured key; numbers are compared as numbers."""
    return sorted(records, key=lambda record: _sort_key(record, policy.sort_by),
                  reverse=policy.sort_order == "desc")


def _retention_decisions(pool: list[Record], policy: RetentionPolicy,
                         ranks: dict[str, int]) -> list[tuple[Record, str, str]]:
    label = SORT_LABELS.get(policy.sort_by, "训练步数")
    if policy.mode == "keep_top_n":
        count = policy.keep_count
        return [
            (record, "keep" if index < count else "delete",
             (f"保留检查点 #{ranks.get(record.relative_path, index + 1)}（保留前 {count} 个）"
              if index < count else f"清理检查点（按{label}排序，只保留前 {count} 个）"))
            for index, record in enumerate(pool)
        ]
    if policy.mode == "delete_top_m":
        count = policy.delete_count
        return [
            (record, "delete" if index < count else "keep",
             (f"清理检查点（按{label}排序，清理前 {count} 个）"
              if index < count else "保留检查点（未进入清理名次）"))
            for index, record in enumerate(pool)
        ]
    if policy.mode == "keep_first_last":
        ascending = sorted(pool, key=lambda record: _sort_key(record, policy.sort_by))
        descending = list(reversed(ascending))
        keep_oldest = {record.relative_path for record in ascending[:policy.keep_first]}
        keep_newest = {record.relative_path for record in descending[:policy.keep_last]}
        decisions: list[tuple[Record, str, str]] = []
        for record in pool:
            if record.relative_path in keep_oldest:
                decisions.append((record, "keep", f"保留最早 {policy.keep_first} 个检查点之一"))
            elif record.relative_path in keep_newest:
                decisions.append((record, "keep", f"保留最新 {policy.keep_last} 个检查点之一"))
            else:
                decisions.append((record, "delete", "清理首尾之外的中间检查点"))
        return decisions
    return [(record, "delete", "清理全部中间检查点") for record in pool]


def _apply_safety_lock(records: list[Record], actions: dict[str, tuple[str, str]],
                       policy: RetentionPolicy) -> None:
    if not policy.ensure_at_least_one_weight:
        return
    weights = [record for record in records if record.role in {ROLE_CHECKPOINT, ROLE_FINAL}]
    if not weights:
        return
    if any(actions.get(record.relative_path, ("keep", ""))[0] == "keep" for record in weights):
        return
    for record in rank_records(weights, policy):
        if actions.get(record.relative_path, ("keep", ""))[0] == "delete":
            actions[record.relative_path] = ("keep", "保底安全锁：该实验至少保留 1 个权重")
            return


def _group_item(record: Record, actions: dict[str, tuple[str, str]], ranks: dict[str, int]) -> dict[str, Any]:
    action, reason = actions.get(record.relative_path, ("keep", "默认保留"))
    return {
        "relative_path": record.relative_path,
        "name": record.name,
        "kind": record.kind,
        "role": record.role,
        "role_label": ROLE_LABELS.get(record.role, record.role),
        "size_bytes": record.size_bytes,
        "modified_at": _iso(record.mtime_ns),
        "progress": record.progress,
        "action": action,
        "reason": reason,
        "rank": ranks.get(record.relative_path),
    }


def _promote_final_exports(records: list[Record]) -> None:
    """Treat the newest single-marker weight of a unit as its final export.

    Training runs write both "base-step1500.safetensors" (the exported final
    weight) and "base-step1500-step00001000.safetensors" (a resumable
    checkpoint). Both carry a step number, so the exported artifact is the one
    with a single marker at the unit highest progress.
    """
    candidates = [
        record for record in records
        if record.role == ROLE_CHECKPOINT and record.kind == "file" and record.progress is not None
    ]
    if not candidates:
        return
    # Only a run that writes double-marker checkpoints has a separate exported
    # final weight; otherwise every single-marker file is just a checkpoint and
    # retention must be free to rank them all.
    if not any(progress_token_count(record.name) >= 2 for record in candidates):
        return
    top = max(record.progress for record in candidates)
    for record in records:
        if record.role != ROLE_CHECKPOINT or record.kind != "file":
            continue
        if record.progress == top and progress_token_count(record.name) == 1:
            record.role = ROLE_FINAL


def plan_unit(unit_rel: str, records: list[Record],
              policy: RetentionPolicy) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Apply the retention policy inside one experiment unit only."""
    _promote_final_exports(records)
    blocked: list[dict[str, str]] = []
    actions: dict[str, tuple[str, str]] = {}
    for record in records:
        if record.role in {ROLE_PROTECTED, ROLE_OTHER}:
            actions[record.relative_path] = ("keep", "配置 / 日志 / 指标默认受保护")
        elif record.role == ROLE_SYMLINK:
            # 软链接在扫描阶段就被识别；默认静默跳过，不跟随、不删除、不报错。
            actions[record.relative_path] = ("keep", "符号链接已跳过（不跟随、不删除）")

    for record in records:
        if record.role not in {ROLE_CACHE, ROLE_ANALYSIS}:
            continue
        is_cache = record.role == ROLE_CACHE
        enabled = policy.clean_data_cache if is_cache else policy.clean_analysis_tensors
        label = "实验内预处理缓存" if is_cache else "分析中间张量"
        if record.has_foreign:
            actions[record.relative_path] = ("keep", f"{label}目录包含日志 / 预览等受保护文件")
            blocked.append({"relative_path": record.relative_path,
                            "reason": f"{label}目录包含受保护文件，已跳过"})
        elif enabled:
            actions[record.relative_path] = ("delete", f"清理{label}")
        else:
            actions[record.relative_path] = ("keep", f"{label}（未启用清理）")

    for record in records:
        if record.role != ROLE_STATE:
            continue
        if record.has_foreign:
            actions[record.relative_path] = ("keep", "恢复状态目录包含受保护或未知产物")
            blocked.append({"relative_path": record.relative_path,
                            "reason": "恢复状态目录包含受保护或未知产物，已跳过"})
        else:
            actions[record.relative_path] = (
                ("delete", "清理优化器 / 断点恢复状态") if policy.clean_optimizer_states
                else ("keep", "按策略保留优化器状态")
            )

    for record in records:
        if record.role != ROLE_FINAL:
            continue
        if _matches_patterns(record.name, policy.keep_weight_patterns):
            actions[record.relative_path] = ("keep", "匹配保留规则")
        elif policy.keep_final_weights:
            actions[record.relative_path] = ("keep", "最终 / 主权重（默认保护）")
        else:
            actions[record.relative_path] = ("delete", "主权重保护已关闭")

    checkpoints = [record for record in records if record.role == ROLE_CHECKPOINT]
    ordered = rank_records(checkpoints, policy)
    ranks = {record.relative_path: index + 1 for index, record in enumerate(ordered)}
    pool: list[Record] = []
    for record in ordered:
        if record.has_foreign:
            actions[record.relative_path] = ("keep", "检查点目录包含日志 / 预览等受保护文件")
            blocked.append({"relative_path": record.relative_path,
                            "reason": "检查点目录包含受保护文件，已跳过"})
        elif _matches_patterns(record.name, policy.keep_weight_patterns):
            actions[record.relative_path] = ("keep", "匹配保留规则")
        else:
            pool.append(record)

    for record, action, reason in _retention_decisions(pool, policy, ranks):
        actions[record.relative_path] = (action, reason)

    _apply_safety_lock(records, actions, policy)

    items = [_group_item(record, actions, ranks) for record in records]
    items.sort(key=lambda item: (item["action"] == "delete", item["rank"] or 9999, item["name"].casefold()))
    group = {
        "relative_path": unit_rel,
        "name": unit_rel.rsplit("/", 1)[-1] if unit_rel else ".",
        "checkpoint_count": len(checkpoints),
        "items": items,
        "truncated": False,
        "excluded": False,
    }
    _refresh_group_counts(group)
    return group, blocked


def _refresh_group_counts(group: dict[str, Any]) -> None:
    keep = [item for item in group["items"] if item["action"] == "keep"]
    delete = [item for item in group["items"] if item["action"] == "delete"]
    group["keep_count"] = len(keep)
    group["delete_count"] = len(delete)
    group["keep_size_bytes"] = sum(item["size_bytes"] for item in keep if isinstance(item["size_bytes"], int))
    group["delete_size_bytes"] = sum(item["size_bytes"] for item in delete if isinstance(item["size_bytes"], int))



def _revision_token(digest: str, plan_revision: str) -> str:
    """Collapse the (very long) per-target signature into a fixed-size token.

    The raw plan revision embeds one stat signature per target and easily grows
    past 60 KB for a few hundred items, which does not belong in a client-facing
    token.  The digest still covers policy, scope, exclusions and the raw plan
    signature, so any filesystem change invalidates the token exactly as before.
    """
    raw = f"{digest}:{plan_revision}".encode("utf-8")
    return "r1:" + hashlib.sha256(raw).hexdigest()


def _dedupe_blocked(items: list[dict[str, str]]) -> list[dict[str, str]]:
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for item in items:
        unique[(item.get("relative_path", ""), item.get("reason", ""))] = item
    return list(unique.values())


class DataManagementRetentionService:
    """Dry-run and execute a checkpoint / weight retention policy."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.base = DataManagementService(app)

    def preview(self, root_id: Any, dirs: Any, policy_payload: Any,
                exclude_units: Any = None) -> dict[str, Any]:
        policy = RetentionPolicy.from_payload(policy_payload)
        spec, scope = self._scope(root_id, dirs)
        excludes = self._normalize_excludes(exclude_units)
        report = self._build(spec, scope, policy, excludes)
        root = spec.get("path")
        return {
            "ok": True,
            "dry_run": True,
            "root_id": spec["id"],
            "path": self.base._display_path(root) if isinstance(root, Path) else "",
            "policy": policy.to_dict(),
            "exclude_units": sorted(excludes),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **report,
        }

    def cleanup(self, root_id: Any, dirs: Any, policy_payload: Any, expected_items: Any,
                plan_revision: Any, confirmed: Any, exclude_units: Any = None) -> dict[str, Any]:
        if confirmed is not True:
            raise DataManagementError("永久删除需要 confirmed=true", 400)
        policy = RetentionPolicy.from_payload(policy_payload)
        spec, scope = self._scope(root_id, dirs)
        excludes = self._normalize_excludes(exclude_units)
        if not isinstance(plan_revision, str) or not plan_revision or len(plan_revision) > 256:
            raise DataManagementError("缺少清理计划版本，请重新预览", 400)
        if expected_items is None:
            expected_items = []
        if not isinstance(expected_items, list) or len(expected_items) > MAX_RETENTION_ITEMS:
            raise DataManagementError("expected_items 必须是相对路径列表", 400)
        expected = [] if not expected_items else sorted(
            self.base._normalize_items(expected_items, limit=MAX_RETENTION_ITEMS)
        )
        if not self.base._cleanup_lock.acquire(blocking=False):
            raise DataManagementError("已有清理操作正在进行", 409)
        try:
            with data_cleanup_lease():
                report = self._build(spec, scope, policy, excludes)
                current = sorted(
                    item["relative_path"]
                    for group in report["groups"]
                    for item in group["items"]
                    if item["action"] == "delete"
                )
                if plan_revision != report["plan_revision"] or expected != current:
                    raise DataManagementError("策略结果已变化，请重新预览", 409)
                if not current:
                    return {"ok": True, "dry_run": False, "deleted": [], "missing": [],
                            "blocked": report["blocked"], "errors": {},
                            "message": "当前策略下没有需要清理的检查点"}
                plan = self.base._plan(spec["id"], current, limit=MAX_RETENTION_ITEMS)
                result = self.base._execute_plan(plan)
                result["blocked"] = _dedupe_blocked(report["blocked"] + list(result.get("blocked") or []))
                result["message"] = (
                    f"已永久删除 {len(result['deleted'])} 项检查点 / 恢复状态产物"
                    if result["deleted"] else "没有条目被删除"
                )
                return result
        finally:
            self.base._cleanup_lock.release()

    @staticmethod
    def _normalize_excludes(value: Any) -> frozenset[str]:
        if value is None:
            return frozenset()
        if not isinstance(value, list) or len(value) > 2000:
            raise DataManagementError("exclude_units 必须是相对路径列表", 400)
        out: set[str] = set()
        for item in value:
            if not isinstance(item, str) or not item or len(item) > 1024 or "\x00" in item:
                raise DataManagementError("exclude_units 包含无效路径", 400)
            normalized = item.strip().strip("/")
            if not normalized or ".." in normalized.split("/"):
                raise DataManagementError("exclude_units 包含无效路径", 400)
            out.add(normalized)
        return frozenset(out)

    def _scope(self, root_id: Any, dirs: Any) -> tuple[dict[str, Any], list[tuple[str, Path]]]:
        spec = self.base._cleanup_spec(root_id)
        if not isinstance(dirs, list) or not dirs:
            raise DataManagementError("请先选择要应用策略的实验目录", 400)
        if len(dirs) > MAX_SCOPE_DIRS:
            raise DataManagementError(f"一次最多处理 {MAX_SCOPE_DIRS} 个目录", 400)
        resolved: list[tuple[str, Path]] = []
        seen: set[str] = set()
        for value in dirs:
            path, normalized = self.base._resolve_path(spec, value, allow_root=False)
            if normalized in seen:
                raise DataManagementError("目录不能重复", 400)
            seen.add(normalized)
            try:
                info = path.lstat()
            except FileNotFoundError:
                raise DataManagementError(f"目录不存在：{normalized}", 404)
            except OSError as exc:
                raise DataManagementError("目录状态暂时不可用", 503) from exc
            if not stat.S_ISDIR(info.st_mode):
                raise DataManagementError("只能对目录应用检查点保留策略", 400)
            resolved.append((normalized, path))
        filtered = [
            (rel, path) for rel, path in resolved
            if not any(other != rel and rel.startswith(other + "/") for other in seen)
        ]
        if not filtered:
            raise DataManagementError("选择的目录互相嵌套，请只保留最外层目录", 400)
        return spec, filtered

    def _build(self, spec: dict[str, Any], scope: list[tuple[str, Path]],
               policy: RetentionPolicy, exclude_units: frozenset[str] = frozenset()) -> dict[str, Any]:
        records, scan = scan_scope(scope)
        units: dict[str, list[Record]] = {}
        for record in records:
            units.setdefault(record.unit, []).append(record)
        groups: list[dict[str, Any]] = []
        blocked: list[dict[str, str]] = []
        for unit_rel in sorted(units):
            unit_records = units[unit_rel]
            roles = {record.role for record in unit_records}
            relevant = roles & {ROLE_CHECKPOINT, ROLE_FINAL, ROLE_STATE}
            if not relevant:
                if not (policy.clean_data_cache and ROLE_CACHE in roles):
                    if not (policy.clean_analysis_tensors and ROLE_ANALYSIS in roles):
                        continue
            group, unit_blocked = plan_unit(unit_rel, unit_records, policy)
            if unit_rel in exclude_units:
                group["excluded"] = True
                for item in group["items"]:
                    if item["action"] == "delete":
                        item["action"] = "keep"
                        item["reason"] = "已豁免：手动跳过该实验"
                _refresh_group_counts(group)
                unit_blocked = []
            groups.append(group)
            blocked.extend(unit_blocked)
        digest = hashlib.sha256(json.dumps(
            {"policy": policy.to_dict(), "dirs": [rel for rel, _ in scope],
             "exclude_units": sorted(exclude_units)},
            sort_keys=True, ensure_ascii=False,
        ).encode("utf-8")).hexdigest()[:24]
        plan_revision = self._apply_server_plan(spec, groups, blocked, digest)
        summary = self._summary(scope, scan, groups, blocked)
        return {
            "scope": {
                "dirs": [rel for rel, _ in scope],
                "truncated": scan["truncated"],
                "scanned_files": scan["scanned_files"],
                "scanned_dirs": scan["scanned_dirs"],
            },
            "groups": groups,
            "summary": summary,
            "blocked": blocked,
            "plan_revision": plan_revision,
        }

    def _apply_server_plan(self, spec: dict[str, Any], groups: list[dict[str, Any]],
                           blocked: list[dict[str, str]], digest: str) -> str:
        candidates = [
            item["relative_path"] for group in groups
            for item in group["items"] if item["action"] == "delete"
        ]
        if len(candidates) > MAX_RETENTION_ITEMS:
            raise DataManagementError(
                f"本次策略将清理 {len(candidates)} 项，超过单次上限 {MAX_RETENTION_ITEMS}；"
                "请缩小范围或提高保留数量", 400)
        if not candidates:
            blocked[:] = _dedupe_blocked(blocked)
            return _revision_token(digest, "empty")
        plan = self.base._plan(spec["id"], candidates, limit=MAX_RETENTION_ITEMS)
        allowed = {item["relative_path"] for item in plan["items"]}
        reasons = {item["relative_path"]: item["reason"] for item in plan["blocked"]}
        for group in groups:
            for item in group["items"]:
                rel = item["relative_path"]
                if item["action"] != "delete" or rel in allowed:
                    continue
                reason = reasons.get(rel, "服务端不允许清理该项目")
                item["action"] = "keep"
                item["reason"] = f"已阻止：{reason}"
                blocked.append({"relative_path": rel, "reason": reason})
            _refresh_group_counts(group)
        blocked[:] = _dedupe_blocked(blocked)
        return _revision_token(digest, str(plan["plan_revision"]))

    @staticmethod
    def _summary(scope: list[tuple[str, Path]], scan: dict[str, Any],
                 groups: list[dict[str, Any]], blocked: list[dict[str, str]]) -> dict[str, Any]:
        keep = [item for group in groups for item in group["items"] if item["action"] == "keep"]
        delete = [item for group in groups for item in group["items"] if item["action"] == "delete"]
        return {
            "scope_dirs": len(scope),
            "group_count": len(groups),
            "matched_group_count": sum(1 for group in groups if group["delete_count"]),
            "keep_count": len(keep),
            "delete_count": len(delete),
            "keep_size_bytes": sum(item["size_bytes"] for item in keep if isinstance(item["size_bytes"], int)),
            "delete_size_bytes": sum(item["size_bytes"] for item in delete if isinstance(item["size_bytes"], int)),
            "blocked_count": len(blocked),
            "excluded_group_count": sum(1 for group in groups if group.get("excluded")),
            "truncated": scan["truncated"],
            "scanned_files": scan["scanned_files"],
        }


__all__ = [
    "DataManagementRetentionService",
    "RetentionPolicy",
    "Record",
    "classify_file",
    "plan_unit",
    "progress_of",
    "rank_records",
    "scan_scope",
]

