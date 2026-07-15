"""List/export/restore public operations for config file groups."""

from __future__ import annotations

import io
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from web.services.atomic_io import atomic_write_text
from web.services.config.file_group_locks import (
    _is_system_preset_path,
    _list_system_preset_files,
)
from web.services.config.file_group_paths import (
    _backup_relative_path,
    _normalize_config_rel_path,
    _normalize_group_id,
    _read_git_head_file,
    _safe_archive_name,
    _unique_archive_member_name,
)
from web.services.config.file_group_runtime import (
    CONFIGS_DIR,
    _display_path,
    _exported,
    _owner_attr,
    _safe_resolve,
)
from web.services.config.file_group_specs import (
    _build_config_file_group,
    _load_config_file_group_specs,
    _normalize_config_file_group_kind_filter,
    _sort_config_file_group_specs_for_display,
)


def restore_system_presets(files: list[str] | None = None) -> dict[str, Any]:
    targets = _list_system_preset_files() if files is None else files
    normalized_targets: list[str] = []
    errors: list[dict[str, str]] = []
    seen: set[str] = set()

    for raw in targets:
        normalized = _normalize_config_rel_path(raw)
        path = _safe_resolve(normalized)
        if path is None or path.suffix != ".toml":
            errors.append({"file": normalized, "reason": "路径不合法"})
            continue
        if not _is_system_preset_path(normalized):
            errors.append({"file": normalized, "reason": "不是系统预设文件"})
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        normalized_targets.append(normalized)

    if errors:
        return {
            "ok": False,
            "error": "还原请求包含不合法文件",
            "restored": [],
            "skipped": [],
            "errors": errors,
            "backup_dir": "",
        }

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    configs_dir = _owner_attr("CONFIGS_DIR", CONFIGS_DIR)
    backup_root = configs_dir / ".restore-backups" / timestamp
    restored: list[str] = []
    skipped: list[dict[str, str]] = []

    for rel_path in normalized_targets:
        path = _safe_resolve(rel_path)
        if path is None or not path.exists():
            skipped.append({"file": rel_path, "reason": "当前文件不存在"})
            continue

        baseline = _read_git_head_file(rel_path)
        if baseline is None:
            skipped.append({"file": rel_path, "reason": "没有可还原的系统基线"})
            continue

        backup_path = backup_root / _backup_relative_path(rel_path)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_path)
        atomic_write_text(path, baseline)
        restored.append(rel_path)

    return {
        "ok": True,
        "restored": restored,
        "skipped": skipped,
        "errors": [],
        "backup_dir": _display_path(backup_root) if restored else "",
    }


def list_config_file_groups(kind: str | None = None) -> list[dict[str, Any]]:
    specs = _sort_config_file_group_specs_for_display(_load_config_file_group_specs())
    groups = [_build_config_file_group(spec) for spec in specs]
    kind_filter = _normalize_config_file_group_kind_filter(kind)
    if kind_filter == "all":
        return groups
    return [group for group in groups if str(group.get("kind") or "training") == kind_filter]


def list_config_files() -> list[str]:
    return [item["path"] for group in list_config_file_groups() for item in group["files"]]


def export_config_file_group_archive(group_id: str, kind: str | None = "training") -> dict[str, Any]:
    normalized_group_id = _normalize_group_id(group_id)
    groups = list_config_file_groups(kind=kind)
    group = next((item for item in groups if item.get("id") == normalized_group_id), None)
    if group is None:
        raise FileNotFoundError("配置分组不存在")

    files = [
        item for item in group.get("files", [])
        if item.get("path") and str(item.get("path")).lower().endswith(".toml")
    ]
    if not files:
        raise ValueError("该分组没有可导出的 TOML 文件")

    archive_stem = _safe_archive_name(str(group.get("label") or group.get("id") or "toml-group"))
    used_names: set[str] = set()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in files:
            rel_path = _normalize_config_rel_path(str(item.get("path") or ""))
            path = _safe_resolve(rel_path)
            if path is None or not path.is_file():
                raise FileNotFoundError(f"配置文件不存在: {rel_path}")
            archive_name = _unique_archive_member_name(
                _safe_archive_name(str(item.get("filename") or Path(rel_path).name)),
                used_names,
            )
            archive.writestr(archive_name, path.read_text(encoding="utf-8"))

    return {
        "filename": f"{archive_stem}.zip",
        "content": buffer.getvalue(),
        "count": len(files),
        "group": group,
    }


restore_system_presets = _exported(restore_system_presets)
list_config_files = _exported(list_config_files)
list_config_file_groups = _exported(list_config_file_groups)
export_config_file_group_archive = _exported(export_config_file_group_archive)
