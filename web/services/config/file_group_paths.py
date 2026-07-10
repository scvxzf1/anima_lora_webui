"""Path, label, archive-name, and string-list helpers for config file groups."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from web.services.config import paths as _config_paths
from web.services.config.file_group_runtime import (
    ROOT,
    _owner_attr,
    _safe_resolve,
)


def _normalize_config_rel_path(rel_path: str) -> str:
    return _config_paths.normalize_config_rel_path(rel_path)


def _strip_configs_prefix(rel_path: str) -> str:
    return _normalize_config_rel_path(rel_path).removeprefix("configs/")


def _normalize_dataset_preset_path(rel_path: str, *, must_exist: bool) -> str:
    raw = str(rel_path or "").replace("\\", "/").strip()
    if not raw:
        raise ValueError("缺少数据集预设路径")
    path = Path(raw)
    if path.is_absolute():
        try:
            raw = path.resolve().relative_to(_owner_attr("ROOT", ROOT).resolve()).as_posix()
        except ValueError as exc:
            raise ValueError("数据集预设必须在项目目录内") from exc
        path = Path(raw)
    if ".." in path.parts:
        raise ValueError("数据集预设路径不能包含 ..")
    if path.suffix.lower() != ".toml":
        path = path.with_suffix(".toml")
    if len(path.parts) == 1:
        path = Path("datasets") / path
    normalized = path.as_posix().lstrip("/")
    # 支持外部 configs root：归一化后只保留相对于 configs root 的路径
    if not normalized.startswith("datasets/"):
        raise ValueError("数据集预设必须保存在 datasets/ 下")
    safe_path = _safe_resolve(normalized)
    if safe_path is None:
        raise ValueError("数据集预设路径不合法")
    if must_exist and not safe_path.exists():
        raise ValueError("数据集预设不存在")
    return normalized


def _normalize_group_id(group_id: str) -> str:
    return str(group_id or "").strip()


def _normalize_group_label(label: str) -> str:
    return " ".join(str(label or "").strip().split())[:48]


def _slugify_group_label(label: str) -> str:
    chars: list[str] = []
    for ch in label.strip().lower():
        if ch.isascii() and ch.isalnum():
            chars.append(ch)
        elif ch in {"-", "_"}:
            chars.append(ch)
        elif ch.isspace():
            chars.append("_")
    slug = "".join(chars).strip("_-")
    return slug or "custom_group"


def _unique_group_id(base: str, specs: list[dict[str, Any]]) -> str:
    used = {str(spec.get("id") or "") for spec in specs}
    root = base or "custom_group"
    candidate = root
    idx = 2
    while candidate in used:
        candidate = f"{root}_{idx}"
        idx += 1
    return candidate


def _safe_archive_name(name: str) -> str:
    clean = re.sub(r'[\\/:*?"<>|\r\n\t]+', "_", str(name or "").strip())
    clean = re.sub(r"\s+", "_", clean).strip("._")
    return clean or "toml-group"


def _unique_archive_member_name(name: str, used_names: set[str]) -> str:
    clean = _safe_archive_name(name)
    if not clean.lower().endswith(".toml"):
        clean = f"{clean}.toml"
    candidate = clean
    stem = candidate[:-5]
    index = 2
    while candidate in used_names:
        candidate = f"{stem}-{index}.toml"
        index += 1
    used_names.add(candidate)
    return candidate


def _place_index(value: Any | None, length: int) -> int:
    if value is None or value == "":
        return max(0, length)
    try:
        index = int(value)
    except (TypeError, ValueError):
        return max(0, length)
    return max(0, min(index, max(0, length)))


def _string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (set, frozenset)):
        return [str(item) for item in sorted(value) if item]
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if item]


def _config_group_path_list(value: Any) -> list[str]:
    paths: list[str] = []
    for raw in _string_list(value):
        normalized = _normalize_config_rel_path(raw)
        if normalized:
            paths.append(normalized)
    return list(dict.fromkeys(paths))


def _backup_relative_path(rel_path: str) -> Path:
    path = Path(rel_path)
    try:
        return path.relative_to("configs")
    except ValueError:
        return path


def _read_git_head_file(rel_path: str) -> str | None:
    result = subprocess.run(
        ["git", "show", f"HEAD:{rel_path}"],
        cwd=_owner_attr("ROOT", ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout
