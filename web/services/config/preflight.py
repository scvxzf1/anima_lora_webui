"""Training preflight checks and runtime config path validation.

This module is loaded by ``web.services.config_service`` as part of the
compatibility facade.  It snapshots legacy globals at import time and syncs
mutable path settings from the facade before exported calls so existing tests
and callers that monkeypatch ``config_service.ROOT`` continue to work.
"""

from __future__ import annotations

import json
from functools import wraps

from web.services import config_service as _facade

for _name, _value in _facade.__dict__.items():
    if _name.startswith("__") and _name.endswith("__"):
        continue
    globals().setdefault(_name, _value)

_SYNC_NAMES = (
    "ROOT",
    "CONFIGS_DIR",
    "GUI_METHODS_DIR",
    "IMPORTED_CONFIGS_DIR",
    "PRESETS_FILE",
    "WEB_FILE_GROUPS_FILE",
    "WEB_USER_LOCKS_FILE",
    "DATASET_PRESETS_DIR",
    "resolve_output_root",
    "_display_settings_path",
    "save_raw_file",
    "load_raw_file",
    "delete_raw_file",
    "patch_raw_file_values",
    "preview_raw_file_patch",
    "get_config_file_meta",
    "list_config_file_groups",
    "move_config_file_to_group",
    "_inspect_network_weight",
    "LOGGER",
)


def _sync_from_facade() -> None:
    _exported_names = set(globals().get("__all__", ()))
    _legacy_module = getattr(_facade, "_legacy", None)
    for _name in _SYNC_NAMES:
        if not hasattr(_facade, _name):
            continue
        _value = getattr(_facade, _name)
        if _name not in _exported_names:
            globals()[_name] = _value
        if _legacy_module is not None:
            setattr(_legacy_module, _name, _value)


def _exported(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        _sync_from_facade()
        return fn(*args, **kwargs)

    return wrapper

GLOBAL_MODEL_PATH_KEYS = (
    "pretrained_model_name_or_path",
    "qwen3",
    "vae",
)

__all__ = ['preflight_training_config', '_load_training_config_for_web_run', '_config_file_path', 'is_web_runtime_config', 'training_sample_sampler_status', 'apply_global_model_path_defaults', '_check_training_images', '_check_dataset_source_paths', '_check_dataset_paths', '_check_cache_sidecars']

def preflight_training_config(
    variant: str,
    preset: str,
    methods_subdir: str = "gui-methods",
    *,
    config_file: str | None = None,
) -> dict[str, Any]:
    cfg = _load_training_config_for_web_run(variant, preset, methods_subdir, config_file=config_file)
    checks: list[dict[str, str]] = []
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    runtime_config = is_web_runtime_config(config_file) or _looks_like_web_runtime_config(cfg)

    def add(level: str, key: str, message: str, path: Path | None = None) -> None:
        item = {
            "level": level,
            "key": key,
            "message": message,
        }
        if path is not None:
            item["path"] = _display_path(path)
        checks.append(item)
        if level == "error":
            errors.append(item)
        elif level == "warning":
            warnings.append(item)

    def check_file(key: str, label: str, suffixes: tuple[str, ...] = ()) -> None:
        raw = cfg.get(key)
        if not raw:
            add("error", key, f"{label} 未填写")
            return
        path = _resolve_project_path(str(raw))
        if not path.exists():
            add("error", key, f"{label} 不存在", path)
            return
        if not path.is_file():
            add("error", key, f"{label} 不是文件", path)
            return
        if suffixes and path.suffix.lower() not in suffixes:
            add("warning", key, f"{label} 后缀不是常见格式 {', '.join(suffixes)}", path)
            return
        add("ok", key, f"{label} 存在", path)

    def check_dir(key: str, label: str, *, must_exist: bool, warn_empty: bool = False) -> None:
        raw = cfg.get(key)
        if not raw:
            add("error", key, f"{label} 未填写")
            return
        path = _resolve_project_path(str(raw))
        if not path.exists():
            if must_exist:
                add("error", key, f"{label} 不存在", path)
            else:
                add("warning", key, f"{label} 不存在，训练/预处理可能会创建它", path)
            return
        if not path.is_dir():
            add("error", key, f"{label} 不是目录", path)
            return
        if warn_empty and not any(path.iterdir()):
            add("warning", key, f"{label} 为空", path)
            return
        add("ok", key, f"{label} 存在", path)

    if "output_name" in cfg and _is_blank_output_name(cfg.get("output_name")):
        add("error", "output_name", "输出名称未填写")
    _check_checkpointing_config(cfg, add)
    _check_no_dataset_regularization_config(cfg, add)
    _check_output_dir_history_reuse(cfg, add)
    check_file("pretrained_model_name_or_path", "基础 DiT 模型", (".safetensors", ".pt", ".pth", ".ckpt"))
    check_file("qwen3", "Qwen3 文本编码器", (".safetensors", ".pt", ".pth", ".bin"))
    check_file("vae", "VAE 模型", (".safetensors", ".pt", ".pth", ".ckpt"))
    _check_network_weights(cfg, add, variant, preset, methods_subdir, config_file)
    dataset_config_path = _dataset_config_path_from_cfg(cfg)
    if cfg.get("dataset_config") and (runtime_config or (dataset_config_path and dataset_config_path.exists())):
        check_file("dataset_config", "数据集配置", (".toml",))

    _check_dataset_source_paths(cfg, add)
    _check_dataset_paths(cfg, add, check_runtime_dirs=runtime_config)
    _check_training_sample_config(cfg, add)
    if not runtime_config:
        _check_web_preprocess_environment(add)
    if runtime_config:
        _check_training_images(cfg, add)
        _check_cache_sidecars(cfg, add)

    return {
        "ok": not errors,
        "variant": variant,
        "preset": preset,
        "methods_subdir": methods_subdir,
        "summary": {
            "errors": len(errors),
            "warnings": len(warnings),
            "checks": len(checks),
        },
        "checks": checks,
        "errors": errors,
        "warnings": warnings,
    }


def _inspect_network_weight(
    path: str,
    *,
    variant: str,
    preset: str,
    methods_subdir: str,
    config_file: str | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del config_file
    return _inspect_continue_lora_weight(
        path,
        variant=variant,
        preset=preset,
        methods_subdir=methods_subdir,
        cfg=cfg,
        root=ROOT,
    )


def _check_checkpointing_config(cfg: dict[str, Any], add) -> None:
    selective_checkpoint = str(cfg.get("selective_checkpoint") or "off").strip().lower()
    gradient_checkpointing = _bool_value(cfg.get("gradient_checkpointing"), False)
    cpu_offload_checkpointing = _bool_value(cfg.get("cpu_offload_checkpointing"), False)
    unsloth_offload_checkpointing = _bool_value(cfg.get("unsloth_offload_checkpointing"), False)
    blocks_to_swap = _nonnegative_int_value(cfg.get("blocks_to_swap"), 0)

    if selective_checkpoint != "off" and gradient_checkpointing:
        add(
            "error",
            "gradient_checkpointing",
            (
                "selective_checkpoint 是 DiT 选择性检查点模式，不能同时开启完整 "
                "gradient_checkpointing；请关闭完整检查点，保留 selective_checkpoint。"
            ),
        )
    if selective_checkpoint != "off" and cpu_offload_checkpointing:
        add(
            "error",
            "cpu_offload_checkpointing",
            "selective_checkpoint 不支持 CPU activation offload；请关闭 cpu_offload_checkpointing。",
        )
    if selective_checkpoint != "off" and unsloth_offload_checkpointing:
        add(
            "error",
            "unsloth_offload_checkpointing",
            "selective_checkpoint 不能和 unsloth_offload_checkpointing 同时开启。",
        )
    if blocks_to_swap > 0 and cpu_offload_checkpointing:
        add(
            "error",
            "cpu_offload_checkpointing",
            (
                "blocks_to_swap 可以和普通 gradient_checkpointing 同用，但不能和 "
                "cpu_offload_checkpointing 同时开启。"
            ),
        )
    if blocks_to_swap > 0 and unsloth_offload_checkpointing:
        add(
            "error",
            "unsloth_offload_checkpointing",
            (
                "blocks_to_swap 可以和普通 gradient_checkpointing 同用，但不能和 "
                "unsloth_offload_checkpointing 同时开启。"
            ),
        )


def _check_no_dataset_regularization_config(cfg: dict[str, Any], add) -> None:
    prior_weight = _nonnegative_float_value(cfg.get("prior_preservation_weight"), 0.0)
    mask_weight = _nonnegative_float_value(cfg.get("inverted_mask_prior_weight"), 0.0)
    blank_enabled = _bool_value(cfg.get("blank_prompt_preservation"), False)
    dop_trigger = str(cfg.get("diff_output_preservation_trigger") or "").strip()
    dop_class = str(cfg.get("diff_output_preservation_class") or "").strip()
    use_text_cache = _bool_value(cfg.get("use_text_cache"), False)
    cache_llm_adapter_outputs = _bool_value(cfg.get("cache_llm_adapter_outputs"), False)

    if prior_weight > 0.0 and dop_trigger and not dop_class and not blank_enabled:
        add(
            "error",
            "diff_output_preservation_class",
            "DOP 已填写触发词，但未填写类提示；请填写 woman / character / style 等类提示，或关闭 prior_preservation_weight。",
        )
    elif prior_weight > 0.0 and not (blank_enabled or dop_class):
        add(
            "error",
            "prior_preservation_weight",
            "prior_preservation_weight 大于 0 时，需要开启 blank_prompt_preservation 或填写 DOP 类提示。",
        )

    if prior_weight > 0.0 and blank_enabled and dop_class:
        add(
            "error",
            "blank_prompt_preservation",
            "blank_prompt_preservation 不能和 DOP 类提示同时使用。",
        )

    if (prior_weight > 0.0 or mask_weight > 0.0) and not use_text_cache:
        add(
            "error",
            "use_text_cache",
            "无数据集正则化需要 use_text_cache=true；请开启文本缓存后重新预处理。",
        )
    prior_needs_adapter_cache = prior_weight > 0.0 and (blank_enabled or dop_class)
    if (prior_needs_adapter_cache or mask_weight > 0.0) and not cache_llm_adapter_outputs:
        add(
            "error",
            "cache_llm_adapter_outputs",
            "无数据集正则化需要 cache_llm_adapter_outputs=true；请开启 LLM adapter 输出缓存后重新预处理。",
        )


def _nonnegative_int_value(value: Any, fallback: int = 0) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed >= 0 else fallback


def _nonnegative_float_value(value: Any, fallback: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed >= 0.0 else fallback


def _check_network_weights(
    cfg: dict[str, Any],
    add,
    variant: str,
    preset: str,
    methods_subdir: str,
    config_file: str | None,
) -> None:
    raw = str(cfg.get("network_weights") or "").strip()
    if not raw:
        return
    try:
        info = _inspect_network_weight(
            raw,
            variant=variant,
            preset=preset,
            methods_subdir=methods_subdir,
            config_file=config_file,
            cfg=cfg,
        )
    except Exception as exc:
        add("error", "network_weights", f"热启动权重不可用：{exc}", _resolve_project_path(raw))
        return

    weight_path = _resolve_project_path(str(info.get("abs_path") or raw))
    if not info.get("compatible", False):
        message = str(info.get("message") or "当前训练配置与 network_weights 不兼容")
        add("error", "network_weights", message, weight_path)
        return

    kind = str(info.get("kind") or "LoRA/LoHa/LoKr/GLoRA")
    message = f"热启动权重可用（{kind}）"
    if _bool_value(cfg.get("dim_from_weights"), False):
        message += "，将从权重读取维度"
    add("ok", "network_weights", message, weight_path)


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


def _check_training_sample_config(cfg: dict[str, Any], add) -> None:
    sample_prompts = str(cfg.get("sample_prompts") or "").strip()
    epoch_freq = _positive_int_or_none(cfg.get("sample_every_n_epochs"))
    step_freq = _positive_int_or_none(cfg.get("sample_every_n_steps"))
    sample_at_first = _bool_value(cfg.get("sample_at_first"), False)

    if sample_prompts and epoch_freq is None and step_freq is None and not sample_at_first:
        add(
            "warning",
            "sample_prompts",
            "已填写 sample_prompts，但未启用训练前、按轮或按步采样，训练不会生成样张",
        )

    if epoch_freq is not None and step_freq is not None:
        add(
            "warning",
            "sample_schedule",
            "已同时启用按轮和按步采样，会分别在轮末和步数命中时生成样张，采样开销会增加",
        )

    raw_sampler = str(cfg.get("sample_sampler") or "euler").strip().lower()
    sampler, sampler_status = training_sample_sampler_status(raw_sampler)
    if sampler_status == "legacy":
        add(
            "warning",
            "sample_sampler",
            f"sample_sampler={raw_sampler} 是旧 Diffusers 采样器名，训练预览会按 {sampler} 兼容处理",
        )
    elif sampler_status == "unknown":
        add(
            "warning",
            "sample_sampler",
            f"sample_sampler={raw_sampler} 当前训练预览不支持，会按 {sampler} 处理",
        )


def _load_training_config_for_web_run(
    variant: str,
    preset: str,
    methods_subdir: str,
    *,
    config_file: str | None = None,
) -> dict[str, Any]:
    fallback_cfg = apply_auto_data_dirs(load_merged_config(variant, preset, methods_subdir))
    cfg = dict(fallback_cfg)
    source = _config_file_path(config_file)
    if source is not None:
        try:
            cfg.update(expand_env_vars_in_obj(toml.loads(source.read_text(encoding="utf-8"))))
        except toml.TomlDecodeError as exc:
            raise ValueError(f"训练配置 TOML 解析失败: {config_file}") from exc
    cfg = apply_global_model_path_defaults(cfg, fallback=fallback_cfg)
    return apply_auto_data_dirs(cfg)


def apply_global_model_path_defaults(
    cfg: dict[str, Any],
    *,
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fill blank model path scalars from Web global settings, preserving explicit config values."""

    defaults = _global_model_path_defaults(fallback or {})
    if not defaults:
        return cfg
    next_cfg = dict(cfg)
    for key in GLOBAL_MODEL_PATH_KEYS:
        if _blank_model_path(next_cfg.get(key)) and not _blank_model_path(defaults.get(key)):
            next_cfg[key] = defaults[key]
    return next_cfg


def _global_model_path_defaults(fallback: dict[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    settings_file = CONFIGS_DIR / "web-ui-settings.toml"
    if settings_file.exists():
        try:
            raw = toml.loads(settings_file.read_text(encoding="utf-8"))
        except toml.TomlDecodeError:
            raw = {}
        section = raw.get("global") if isinstance(raw, dict) else {}
        if isinstance(section, dict):
            for key in GLOBAL_MODEL_PATH_KEYS:
                value = str(section.get(key) or "").strip()
                if value:
                    values[key] = expand_env_vars(value)
    for key in GLOBAL_MODEL_PATH_KEYS:
        if key not in values and not _blank_model_path(fallback.get(key)):
            values[key] = expand_env_vars(str(fallback[key]).strip())
    return values


def _blank_model_path(value: Any) -> bool:
    return str(value or "").strip() == ""


def _config_file_path(config_file: str | None) -> Path | None:
    raw = str(config_file or "").strip()
    if not raw:
        return None
    path = Path(raw.replace("\\", "/"))
    if ".." in path.parts:
        raise ValueError("训练配置路径不能包含 ..")
    if path.is_absolute():
        resolved = path.resolve()
    else:
        normalized = _normalize_config_rel_path(raw)
        resolved = _config_path_from_display_path(normalized)
        if resolved is None:
            resolved = (ROOT / normalized).resolve()
    if _is_output_run_snapshot_config(resolved) and resolved.name != OUTPUT_RUN_CONFIG_FILES["runtime"][0]:
        raise ValueError("训练输出目录只能使用 config.runtime.toml 作为训练配置")
    if not _is_allowed_training_config_path(resolved):
        raise ValueError("训练配置必须在项目目录或全局输出文件夹内")
    if not resolved.is_file():
        raise FileNotFoundError(f"训练配置不存在: {config_file}")
    if resolved.suffix.lower() != ".toml":
        raise ValueError("训练配置必须是 TOML 文件")
    return resolved


def _config_path_from_display_path(normalized: str) -> Path | None:
    if normalized == "configs" or normalized.startswith("configs/"):
        return _safe_resolve(normalized)
    return None


def _is_allowed_training_config_path(path: Path) -> bool:
    resolved = path.resolve()

    # 1. 检查是否在项目根目录下
    try:
        resolved.relative_to(ROOT.resolve())
        return True
    except ValueError:
        pass

    # 2. 检查是否在配置目录下（支持外置配置）
    # 使用同步后的 CONFIGS_DIR，它会在 _exported 装饰器中更新
    # 注意：即使 configs/ 是指向外部目录的符号链接，resolve() 也会返回相同的真实路径
    # 因此这个检查能覆盖符号链接和真实外部目录两种情况
    try:
        configs_root = CONFIGS_DIR.resolve()
        resolved.relative_to(configs_root)
        return True
    except (ValueError, AttributeError):
        pass

    # 3. 检查是否在全局输出目录下
    try:
        rel_to_output = resolved.relative_to(resolve_output_root().resolve())
    except ValueError:
        return _is_web_runtime_config_tree(resolved)
    return (
        len(rel_to_output.parts) == 2
        and rel_to_output.name == "config.runtime.toml"
        and _is_web_runtime_config_tree(resolved)
    )


def _is_web_runtime_config_tree(path: Path) -> bool:
    run_dir = path.parent
    return (
        path.name == "config.runtime.toml"
        and path.is_file()
        and _has_web_runtime_dirs(run_dir)
    )


def _is_output_run_snapshot_config(path: Path) -> bool:
    resolved = path.resolve()
    if resolved.name not in {filename for filename, _label in OUTPUT_RUN_CONFIG_FILES.values()}:
        return False
    try:
        rel_to_output = resolved.relative_to(resolve_output_root().resolve())
        if len(rel_to_output.parts) == 2:
            return True
    except ValueError:
        pass
    return _has_web_runtime_dirs(resolved.parent)


def _has_web_runtime_dirs(run_dir: Path) -> bool:
    return (
        (run_dir / "model_cache").is_dir()
        and (run_dir / "dataset_cache").is_dir()
        and (run_dir / "training_output").is_dir()
    )


def is_web_runtime_config(config_file: str | None) -> bool:
    path = _config_file_path(config_file)
    if path is None:
        return False
    run_dir = path.parent
    return (
        path.name == "config.runtime.toml"
        and (run_dir / "model_cache").is_dir()
        and (run_dir / "dataset_cache").is_dir()
        and (run_dir / "training_output").is_dir()
    )


def _looks_like_web_runtime_config(cfg: dict[str, Any]) -> bool:
    output_root = resolve_output_root().resolve()
    for key in ("output_dir", "logging_dir", "dataset_config", "resized_image_dir", "lora_cache_dir"):
        raw = str(cfg.get(key) or "").strip()
        if not raw:
            continue
        path = _resolve_project_path(raw)
        try:
            path.relative_to(output_root)
            return True
        except ValueError:
            continue
    return False


def _check_web_preprocess_environment(add) -> None:
    python_exe = Path(_web_python_executable())
    if not python_exe.is_file():
        add(
            "error",
            PREPROCESS_ENV_CHECK_KEY,
            f"预处理启动环境异常: Python 解释器不存在 {python_exe}",
            python_exe,
        )
        return
    missing = [rel for rel in PREPROCESS_ENV_REQUIRED_FILES if not (ROOT / rel).is_file()]
    if missing:
        add(
            "error",
            PREPROCESS_ENV_CHECK_KEY,
            f"预处理启动环境异常: 缺少 {', '.join(missing)}",
            ROOT / missing[0],
        )
        return
    add("ok", PREPROCESS_ENV_CHECK_KEY, "预处理启动环境文件检查通过", ROOT)


def _web_python_executable() -> str:
    from web.services.project_python import resolve_web_python_executable

    return resolve_web_python_executable(ROOT)


def training_sample_sampler_status(value: Any) -> tuple[str, str]:
    sampler = str(value or "euler").strip().lower()
    if sampler in SUPPORTED_TRAINING_SAMPLE_SAMPLERS:
        return sampler, "supported"
    if sampler in LEGACY_TRAINING_SAMPLE_SAMPLERS:
        return "euler", "legacy"
    return "euler", "unknown"

def _check_training_images(cfg: dict[str, Any], add) -> None:
    rows = _dataset_rows_for_estimate(cfg)
    if not rows:
        rows = [{
            "source_dir": str(cfg.get("source_image_dir") or ""),
            "image_dir": str(cfg.get("resized_image_dir") or cfg.get("source_image_dir") or ""),
        }]
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    all_missing_captions: list[str] = []
    detected_caption_modes: dict[str, int] = {}
    detected_caption_total = 0
    checked_groups = 0
    for idx, row in enumerate(rows, start=1):
        image_dir = _resolve_project_path(str(row.get("image_dir") or row.get("source_dir") or ""))
        source_dir = _resolve_project_path(str(row.get("source_dir") or ""))
        settings = row.get("settings") if isinstance(row.get("settings"), dict) else {}
        recursive = _bool_value(row.get("recursive"), True)
        path_pattern = _normalize_path_pattern(row.get("path_pattern"))
        caption_extension = str(settings.get("caption_extension") or cfg.get("caption_extension") or ".txt")
        if not caption_extension.startswith("."):
            caption_extension = f".{caption_extension}"
        prefer_json_caption = _bool_value(
            settings.get("prefer_json_caption", cfg.get("prefer_json_caption")),
            False,
        )
        caption_source_mode = normalize_caption_source_mode(
            settings.get("caption_source_mode", cfg.get("caption_source_mode")),
            prefer_json_caption,
        )
        if not image_dir.is_dir():
            continue
        checked_groups += 1
        key = "training_images" if idx == 1 else f"dataset_{idx}_training_images"
        label = "缩放图像目录" if idx == 1 else f"第 {idx} 组缩放图像目录"
        try:
            images = _dataset_image_files(
                image_dir,
                image_exts,
                recursive=recursive,
                path_pattern=path_pattern,
            )
        except ValueError as exc:
            add("error", key, str(exc), image_dir)
            continue
        if not images:
            add("error", key, f"{label}里没有可训练图片，请先预处理生成训练图", image_dir)
            continue
        for image in images[:50]:
            source = read_caption_source_from_dirs(
                image,
                [source_dir, image.parent],
                prefer_json_caption=prefer_json_caption,
                caption_source_mode=caption_source_mode,
                caption_extension=caption_extension,
            )
            if source.path is None:
                all_missing_captions.append(image.name)
            else:
                detected_caption_modes[source.detected_mode] = (
                    detected_caption_modes.get(source.detected_mode, 0) + 1
                )
                detected_caption_total += len(source.caption_texts())
    if checked_groups == 0:
        return
    if all_missing_captions:
        sample = ", ".join(all_missing_captions[:3])
        add("warning", "captions", f"部分图片未找到同名标注，例如 {sample}")
    else:
        summary = _caption_detection_counts_text(detected_caption_modes, detected_caption_total)
        add("ok", "captions", f"抽样图片均找到标注；{summary}" if summary else "抽样图片均找到标注")


def _check_dataset_source_paths(cfg: dict[str, Any], add) -> None:
    rows = _dataset_rows_for_estimate(cfg)
    if not rows:
        return
    for idx, row in enumerate(rows, start=1):
        source = _resolve_project_path(str(row.get("source_dir") or ""))
        key = "source_image_dir" if idx == 1 else f"dataset_{idx}_source_dir"
        label = "源图像目录" if idx == 1 else f"第 {idx} 组原始数据集目录"
        recursive = _bool_value(row.get("recursive"), True)
        path_pattern = _normalize_path_pattern(row.get("path_pattern"))
        trigger_clone = _normalize_trigger_clone(row.get("trigger_clone"))
        if trigger_clone["enabled"] and not trigger_clone["prompt"]:
            add(
                "error",
                f"{key}_trigger_clone_prompt",
                f"{label} 的触发提示词图像克隆已开启，但触发提示词为空",
                source,
            )
        if not str(row.get("source_dir") or "").strip():
            add("error", key, f"{label} 未填写")
        elif not source.exists():
            add("error", key, f"{label} 不存在", source)
        elif not source.is_dir():
            add("error", key, f"{label} 不是目录", source)
        elif trigger_clone["enabled"] and _count_source_images(
            source,
            DATASET_IMAGE_EXTS,
            recursive=recursive,
            path_pattern=path_pattern,
        ) <= 0:
            add("error", f"{key}_trigger_clone_images", f"{label} 中没有可克隆的训练图片", source)
        elif _nl_tag_mix_enabled(row):
            settings = row.get("settings") if isinstance(row.get("settings"), dict) else {}
            caption_extension = str(settings.get("caption_extension") or cfg.get("caption_extension") or ".txt")
            if not caption_extension.startswith("."):
                caption_extension = f".{caption_extension}"
            prefer_json_caption = _bool_value(
                settings.get("prefer_json_caption", cfg.get("prefer_json_caption")),
                False,
            )
            caption_source_mode = normalize_caption_source_mode(
                settings.get("caption_source_mode", cfg.get("caption_source_mode")),
                prefer_json_caption,
            )
            image_count, captioned_count = _nl_tag_mix_caption_counts(
                source,
                caption_source_mode=caption_source_mode,
                caption_extension=caption_extension,
                prefer_json_caption=prefer_json_caption,
                recursive=recursive,
                path_pattern=path_pattern,
            )
            if image_count <= 0:
                add("error", f"{key}_nl_tag_mix", f"{label} 中没有可训练图片", source)
            else:
                if captioned_count <= 0:
                    add(
                        "warning",
                        f"{key}_nl_tag_mix_captions",
                        f"{label} 未找到可读取标注，captions格式nl/tag权重调整会全部按 tag 处理",
                        source,
                    )
                add("ok", key, f"{label} 存在", source)
        elif not any(source.iterdir()):
            add("warning", key, f"{label} 为空", source)
        else:
            add("ok", key, f"{label} 存在", source)


def _check_dataset_paths(cfg: dict[str, Any], add, *, check_runtime_dirs: bool = True) -> None:
    rows = _dataset_rows_for_estimate(cfg)
    if not check_runtime_dirs:
        return
    for idx, row in enumerate(rows, start=1):
        image_dir = _resolve_project_path(str(row.get("image_dir") or ""))
        cache_dir = _resolve_project_path(str(row.get("cache_dir") or ""))
        prefix = f"dataset_{idx}"
        if not image_dir.exists():
            add("error", f"{prefix}_image_dir", f"第 {idx} 组缩放图路径不存在", image_dir)
        elif not image_dir.is_dir():
            add("error", f"{prefix}_image_dir", f"第 {idx} 组缩放图路径不是目录", image_dir)
        if not cache_dir.exists():
            add("error", f"{prefix}_cache_dir", f"第 {idx} 组缓存路径不存在", cache_dir)
        elif not cache_dir.is_dir():
            add("error", f"{prefix}_cache_dir", f"第 {idx} 组缓存路径不是目录", cache_dir)


def _check_cache_sidecars(cfg: dict[str, Any], add) -> None:
    cache_dirs: list[tuple[int, Path, bool]] = []
    for idx, row in enumerate(_dataset_rows_for_estimate(cfg), start=1):
        raw = str(row.get("cache_dir") or "").strip()
        if not raw:
            continue
        cache_dirs.append((idx, _resolve_project_path(raw), _bool_value(row.get("recursive"), True)))
    if not cache_dirs:
        raw = str(cfg.get("lora_cache_dir") or "").strip()
        if raw:
            cache_dirs = [(1, _resolve_project_path(raw), True)]

    cache_dirs = [(idx, path, recursive) for idx, path, recursive in cache_dirs if path.is_dir()]
    if not cache_dirs:
        return

    if cfg.get("use_vae_cache", cfg.get("cache_latents_to_disk", False)):
        _check_cache_sidecar_pattern(add, cache_dirs, "*.npz", "latent_cache", "VAE latent 缓存", "未找到 .npz latent 缓存，可能需要先预处理")
    if cfg.get("use_text_cache", cfg.get("cache_text_encoder_outputs_to_disk", False)):
        _check_cache_sidecar_pattern(add, cache_dirs, "*_anima_te.safetensors", "text_cache", "文本编码器缓存", "未找到文本编码器缓存，可能需要先预处理")
    if cfg.get("ip_features_cache_to_disk", False) or cfg.get("use_ip_adapter", False):
        _check_cache_sidecar_pattern(add, cache_dirs, "*_anima_pe.safetensors", "pe_cache", "PE 图像特征缓存", "未找到 PE 图像特征缓存，IP-Adapter 可能需要先 preprocess-pe")


def _check_cache_sidecar_pattern(
    add,
    cache_dirs: list[tuple[int, Path, bool]],
    pattern: str,
    key: str,
    label: str,
    missing_message: str,
) -> None:
    for idx, cache_dir, recursive in cache_dirs:
        matches = cache_dir.rglob(pattern) if recursive else cache_dir.glob(pattern)
        count = sum(1 for path in matches if path.is_file())
        item_key = key if idx == 1 else f"dataset_{idx}_{key}"
        if count:
            add("ok", item_key, f"第 {idx} 组找到 {count} 个{label}", cache_dir)
        else:
            add("warning", item_key, f"第 {idx} 组{missing_message}", cache_dir)


for _name in __all__:
    globals()[_name] = _exported(globals()[_name])
