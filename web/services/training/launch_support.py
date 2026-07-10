"""Launch-time training helpers shared across training service modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import toml

from web.services.continue_lora_service import (
    inspect_continue_lora_weight as _inspect_continue_lora_weight,
)
from web.services.config_service import apply_auto_data_dirs, load_merged_config
from web.services.training.context import training_facade as _training_facade
from web.services.training.runtime_paths import (
    _history_dir,
    _path_exists,
    _project_root,
    _resolve_display_path,
)


def _load_config_file_config(*args, **kwargs):
    return _training_facade()._load_config_file_config(*args, **kwargs)


def toml_dumps_sorted(*args, **kwargs):
    return _training_facade().toml_dumps_sorted(*args, **kwargs)


def inspect_continue_lora_weight(
    path: str,
    *,
    variant: str = "lora",
    preset: str = "default",
    methods_subdir: str = "gui-methods",
    config_file: str | None = None,
) -> dict[str, Any]:
    cfg, config_error = _continue_lora_inspection_config(
        variant=variant,
        preset=preset,
        methods_subdir=methods_subdir,
        config_file=config_file,
    )
    return _inspect_continue_lora_weight(
        path,
        variant=variant,
        preset=preset,
        methods_subdir=methods_subdir,
        cfg=cfg,
        config_error=config_error,
        root=_project_root(),
    )


def _continue_lora_inspection_config(
    *,
    variant: str,
    preset: str,
    methods_subdir: str,
    config_file: str | None,
) -> tuple[dict[str, Any] | None, Exception | None]:
    cfg = _load_config_file_config(config_file) if config_file else {}
    if cfg:
        return cfg, None
    try:
        return load_merged_config(variant, preset, methods_subdir), None
    except Exception as exc:
        return None, exc


def _normalize_continue_lora_info(
    value: Any,
    *,
    variant: str,
    preset: str,
    methods_subdir: str,
    config_file: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    raw_path = str(
        value.get("continue_from_weight_abs_path")
        or value.get("abs_path")
        or value.get("path")
        or ""
    ).strip()
    if not raw_path:
        return None
    inspected = inspect_continue_lora_weight(
        raw_path,
        variant=variant,
        preset=preset,
        methods_subdir=methods_subdir,
        config_file=config_file,
    )
    if not inspected.get("compatible"):
        raise ValueError(inspected.get("message") or "当前训练配置与权重热启动来源不兼容")
    return {
        "continue_from_weight_abs_path": inspected["abs_path"],
        "continue_from_weight_name": inspected["name"],
        "continue_from_weight_kind": inspected["kind"],
    }


def _continue_lora_history_meta(continue_info: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(continue_info, dict) or not continue_info.get("continue_from_weight_abs_path"):
        return {"training_mode": "fresh"}
    return {
        "training_mode": "continue_lora",
        "continue_from_weight_abs_path": str(continue_info.get("continue_from_weight_abs_path") or ""),
        "continue_from_weight_name": str(continue_info.get("continue_from_weight_name") or ""),
        "continue_from_weight_kind": str(continue_info.get("continue_from_weight_kind") or ""),
    }


def _write_config_snapshot(
    path: Path,
    variant: str,
    preset: str,
    methods_subdir: str,
    *,
    config_file: str | None = None,
    continue_info: dict[str, Any] | None = None,
) -> None:
    try:
        if config_file:
            source = _resolve_display_path(config_file)
            if source is None or not _path_exists(source):
                raise FileNotFoundError("续训配置快照不存在")
            text = source.read_text(encoding="utf-8", errors="replace")
            path.write_text(_append_continue_lora_snapshot_note(text, continue_info), encoding="utf-8")
            return
        cfg = apply_auto_data_dirs(load_merged_config(variant, preset, methods_subdir))
        path.write_text(_append_continue_lora_snapshot_note(toml_dumps_sorted(cfg), continue_info), encoding="utf-8")
    except Exception as exc:
        path.write_text(f"# 无法生成配置快照: {exc}\n", encoding="utf-8")


def _append_continue_lora_snapshot_note(text: str, continue_info: dict[str, Any] | None) -> str:
    if not isinstance(continue_info, dict) or not continue_info.get("continue_from_weight_abs_path"):
        return text
    base = text.rstrip()
    lines = [
        "",
        "",
        "# WebUI 权重热启动来源",
        '# training_mode = "continue_lora"',
        f'# continue_from_weight_kind = "{_toml_comment_string(continue_info.get("continue_from_weight_kind"))}"',
        f'# continue_from_weight_name = "{_toml_comment_string(continue_info.get("continue_from_weight_name"))}"',
        f'# continue_from_weight_abs_path = "{_toml_comment_string(continue_info.get("continue_from_weight_abs_path"))}"',
        "",
    ]
    return base + "\n".join(lines)


def _toml_comment_string(value: Any) -> str:
    return str(value or "").replace("\\", "\\\\").replace('"', '\\"')


def _command_has_option(args: list[str], option: str) -> bool:
    prefix = f"{option}="
    return any(str(arg) == option or str(arg).startswith(prefix) for arg in args)


def _command_option_value(args: list[str], option: str) -> str | None:
    prefix = f"{option}="
    for idx, arg in enumerate(args):
        if arg == option and idx + 1 < len(args):
            return str(args[idx + 1])
        if str(arg).startswith(prefix):
            return str(arg).split("=", 1)[1]
    return None


def _resolve_block_swap_profile_auto_arg(args: list[str], path: Path) -> list[str]:
    return _resolve_auto_path_arg(args, "--block_swap_profile_jsonl", path)


def _resolve_block_swap_profile_auto_config(config_file: str | None, path: Path) -> bool:
    return _resolve_auto_path_config(
        config_file,
        config_key="block_swap_profile_jsonl",
        path=path,
        is_history_path_fn=_is_history_block_swap_profile_path,
    )


def _resolve_memory_probe_auto_arg(args: list[str], path: Path) -> list[str]:
    return _resolve_auto_path_arg(args, "--memory_probe_jsonl", path)


def _resolve_memory_probe_auto_config(config_file: str | None, path: Path) -> bool:
    return _resolve_auto_path_config(
        config_file,
        config_key="memory_probe_jsonl",
        path=path,
        is_history_path_fn=_is_history_memory_probe_path,
    )


def _resolve_peak_probe_auto_arg(args: list[str], path: Path) -> list[str]:
    return _resolve_auto_path_arg(args, "--peak_probe_jsonl", path)


def _resolve_peak_probe_auto_config(config_file: str | None, path: Path) -> bool:
    return _resolve_auto_path_config(
        config_file,
        config_key="peak_probe_jsonl",
        path=path,
        is_history_path_fn=_is_history_peak_probe_path,
    )


def _resolve_auto_path_arg(args: list[str], option: str, path: Path) -> list[str]:
    out = list(args)
    prefix = f"{option}="
    replacement = str(path)
    idx = 0
    while idx < len(out):
        arg = str(out[idx])
        if arg == option and idx + 1 < len(out):
            if str(out[idx + 1]).strip().lower() == "auto":
                out[idx + 1] = replacement
            idx += 2
            continue
        if arg.startswith(prefix) and arg.split("=", 1)[1].strip().lower() == "auto":
            out[idx] = f"{option}={replacement}"
        idx += 1
    return out


def _resolve_auto_path_config(
    config_file: str | None,
    *,
    config_key: str,
    path: Path,
    is_history_path_fn,
) -> bool:
    config_path = _resolve_display_path(str(config_file or ""))
    if config_path is None or not _path_exists(config_path) or not config_path.is_file():
        return False
    try:
        cfg = toml.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    value = str(cfg.get(config_key) or "").strip()
    if value.lower() != "auto" and not is_history_path_fn(value):
        return False
    if config_path.name == "config.runtime.toml":
        cfg[config_key] = str(path)
        config_path.write_text(toml_dumps_sorted(cfg), encoding="utf-8")
    return True


def _is_history_block_swap_profile_path(value: str) -> bool:
    return _is_history_artifact_path(value, "block_swap_profile.jsonl")


def _is_history_memory_probe_path(value: str) -> bool:
    return _is_history_artifact_path(value, "memory_probe.jsonl")


def _is_history_peak_probe_path(value: str) -> bool:
    return _is_history_artifact_path(value, "peak_probe.jsonl")


def _is_history_artifact_path(value: str, filename: str) -> bool:
    artifact_path = _resolve_display_path(value)
    if artifact_path is None or artifact_path.name != filename:
        return False
    try:
        artifact_path.resolve().relative_to(_history_dir().resolve())
    except ValueError:
        return False
    return True
