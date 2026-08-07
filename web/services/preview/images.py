"""Preview image list/delete helpers for WebUI preview service."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote
import re

import toml

from web.services.image_size import probe_image_size
from web.services.preview.context import call, get


def list_preview_images(
    source: str,
    *,
    current_task_sample_dir: str | None = None,
    sample_config: dict[str, Any] | None = None,
    task: dict[str, Any] | None = None,
    task_id: str | None = None,
    task_label: str | None = None,
    allow_latest_fallback: bool = True,
    limit: int = 200,
    days: int | None = None,
) -> dict[str, Any]:
    source = (source or "training").strip().lower()
    if source not in {"training", "inference", "custom"}:
        raise ValueError("source 只能是 training、inference 或 custom")

    settings = call("get_preview_settings", 
        current_task_sample_dir,
        allow_latest_fallback=allow_latest_fallback,
    )
    if source == "training":
        rel_dir = settings["effective_training_dir"]
        label = _training_preview_label(settings, task_id=task_id, task_label=task_label)
    elif source == "inference":
        rel_dir = settings["inference_dir"]
        label = "推理预览"
    else:
        rel_dir = settings["custom_dir"]
        label = "自定义路径"

    if not rel_dir:
        message = (
            "这个历史训练任务没有记录样张目录"
            if source == "training" and not allow_latest_fallback and task_id
            else "尚未设置自定义预览图路径"
        )
        listing = _empty_listing(source, label, "", exists=False, message=message)
        listing["sample_config"] = sample_config or {}
        listing["task_id"] = task_id or ""
        listing["task_label"] = task_label or ""
        listing["preview_settings"] = _preview_settings_meta(settings)
        return listing

    resolved = call("_resolve_preview_dir", rel_dir, current_task_sample_dir=current_task_sample_dir if source == "training" else None)
    if resolved is None:
        raise ValueError("预览图路径不合法")

    display_dir = call("_display_path", resolved)
    if not resolved.exists():
        listing = _empty_listing(
            source,
            label,
            display_dir,
            exists=False,
            message=_preview_empty_message(source, "目录不存在", sample_config, settings=settings),
        )
        listing["sample_config"] = sample_config or {}
        listing["task_id"] = task_id or ""
        listing["task_label"] = task_label or ""
        listing["preview_settings"] = _preview_settings_meta(settings)
        return listing
    if not resolved.is_dir():
        return _empty_listing(source, label, display_dir, exists=False, message="路径不是目录")

    limit = max(1, min(int(limit or 200), get("MAX_IMAGE_LIMIT")))
    days = _normalize_preview_days(days)
    candidates = [
        p
        for p in resolved.iterdir()
        if p.is_file() and p.suffix.lower() in get("IMAGE_EXTS")
    ]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    candidates = _filter_preview_candidates_by_days(candidates, days)
    prompt_entries = _load_sample_prompt_entries(sample_config) if source == "training" else []
    step_index = call("_training_step_index", task) if source == "training" else {}

    return {
        "ok": True,
        "source": source,
        "label": label,
        "directory": display_dir,
        "directory_exists": True,
        "count": len(candidates[:limit]),
        "total": len(candidates),
        "images": [
            _image_meta(
                path,
                task_id=task_id,
                sample_config=sample_config,
                prompt_entries=prompt_entries,
                step_index=step_index,
            )
            for path in candidates[:limit]
        ],
        "message": "" if candidates else _preview_empty_message(source, "暂无预览图", sample_config, settings=settings),
        "sample_config": sample_config or {},
        "task_id": task_id or "",
        "task_label": task_label or "",
        "preview_settings": _preview_settings_meta(settings),
    }


def delete_preview_images(
    source: str,
    files: list[str] | tuple[str, ...] | set[str],
    *,
    current_task_sample_dir: str | None = None,
    allow_latest_fallback: bool = True,
) -> dict[str, Any]:
    source = (source or "training").strip().lower()
    if source not in {"training", "inference", "custom"}:
        raise ValueError("source 只能是 training、inference 或 custom")

    settings = call(
        "get_preview_settings",
        current_task_sample_dir,
        allow_latest_fallback=allow_latest_fallback,
    )
    if source == "training":
        rel_dir = settings["effective_training_dir"]
    elif source == "inference":
        rel_dir = settings["inference_dir"]
    else:
        rel_dir = settings["custom_dir"]

    if not rel_dir:
        raise ValueError("当前来源没有可删除的预览目录")

    resolved_dir = call(
        "_resolve_preview_dir",
        rel_dir,
        current_task_sample_dir=current_task_sample_dir if source == "training" else None,
    )
    if resolved_dir is None:
        raise ValueError("预览图路径不合法")
    if not resolved_dir.exists() or not resolved_dir.is_dir():
        raise FileNotFoundError("预览图目录不存在")

    targets = _normalize_preview_delete_files(files)
    deleted: list[str] = []
    missing: list[str] = []
    blocked: list[dict[str, str]] = []
    allowed_sample_dir = current_task_sample_dir if source == "training" else None
    image_exts = get("IMAGE_EXTS")

    for raw in targets:
        try:
            path = call("_resolve_preview_file", raw, allowed_sample_dir=allowed_sample_dir)
            _ensure_preview_delete_target(path, resolved_dir, source=source)
            if path.suffix.lower() not in image_exts:
                raise ValueError("只允许删除预览图片文件")
            if not path.exists() or not path.is_file():
                missing.append(call("_display_path", path))
                continue
            path.unlink()
            deleted.append(call("_display_path", path))
        except ValueError as exc:
            blocked.append({
                "file": str(raw),
                "error": str(exc),
            })
        except OSError as exc:
            blocked.append({
                "file": str(raw),
                "error": f"删除失败: {exc}",
            })

    try:
        remaining_total = sum(
            1
            for path in resolved_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_exts
        )
    except OSError:
        remaining_total = 0

    return {
        "ok": len(blocked) == 0,
        "source": source,
        "directory": call("_display_path", resolved_dir),
        "deleted": deleted,
        "deleted_count": len(deleted),
        "missing": missing,
        "missing_count": len(missing),
        "blocked": blocked,
        "blocked_count": len(blocked),
        "remaining_total": remaining_total,
        "message": _preview_delete_message(deleted, missing, blocked),
    }


def list_config_group_preview_images(
    tasks: list[dict[str, Any]],
    *,
    methods_subdir: str,
    variant: str,
    preset: str,
    limit: int = 200,
) -> dict[str, Any]:
    group_label = f"{methods_subdir} / {variant} / {preset or 'default'}"
    label = f"训练分组合并采样结果 · {group_label} · {len(tasks)} 次训练"
    limit = max(1, min(int(limit or 200), get("MAX_IMAGE_LIMIT")))
    images_by_path: dict[str, dict[str, Any]] = {}
    directories: list[str] = []

    for task in tasks:
        sample_dir = str(task.get("sample_dir") or "")
        if not sample_dir:
            continue
        resolved = call("_resolve_preview_dir", sample_dir, current_task_sample_dir=sample_dir)
        if resolved is None or not resolved.exists() or not resolved.is_dir():
            continue
        display_dir = call("_display_path", resolved)
        if display_dir not in directories:
            directories.append(display_dir)
        sample_config = task.get("sample_config") if isinstance(task.get("sample_config"), dict) else {}
        prompt_entries = _load_sample_prompt_entries(sample_config)
        step_index = call("_training_step_index", task)
        task_id = str(task.get("id") or "")
        task_label = call("_preview_task_label", task)
        for path in resolved.iterdir():
            if not path.is_file() or path.suffix.lower() not in get("IMAGE_EXTS"):
                continue
            meta = _image_meta(
                path,
                task_id=task_id,
                sample_config=sample_config,
                prompt_entries=prompt_entries,
                step_index=step_index,
            )
            meta["source_task"] = {
                "id": task_id,
                "label": task_label,
                "state": task.get("state", ""),
                "started_at": task.get("started_at"),
                "started_at_text": task.get("started_at_text", ""),
                "finished_at": task.get("finished_at"),
                "finished_at_text": task.get("finished_at_text", ""),
                "sample_dir": sample_dir,
            }
            key = str(path.resolve())
            previous = images_by_path.get(key)
            if previous is None or _task_image_match_score(task, meta) > _task_image_match_score(previous.get("source_task") or {}, previous):
                images_by_path[key] = meta

    images = list(images_by_path.values())
    images.sort(key=lambda item: (float(item.get("mtime") or 0), str(item.get("name") or "")), reverse=True)
    limited = images[:limit]
    return {
        "ok": True,
        "source": "training",
        "mode": "config_group",
        "label": label,
        "directory": " · ".join(directories[:2]) + (" · ..." if len(directories) > 2 else ""),
        "directories": directories,
        "directory_exists": bool(directories),
        "count": len(limited),
        "total": len(images),
        "images": limited,
        "message": "" if images else "这个训练分组还没有可显示的样张",
        "sample_config": {},
        "task_id": "",
        "task_label": group_label,
        "group": {
            "methods_subdir": methods_subdir,
            "variant": variant,
            "preset": preset or "default",
        },
        "task_count": len(tasks),
    }


def resolve_preview_image(rel_path: str, allowed_sample_dir: str | None = None) -> Path:
    resolved = call("_resolve_preview_file", rel_path, allowed_sample_dir=allowed_sample_dir)
    if resolved.suffix.lower() not in get("IMAGE_EXTS"):
        raise ValueError("只允许读取预览图片文件")
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError("图片不存在")
    return resolved


def _normalize_preview_days(value: int | None) -> int | None:
    if value in (None, ""):
        return None
    days = int(value)
    if days <= 0:
        raise ValueError("days 必须是正整数")
    return days


def _filter_preview_candidates_by_days(candidates: list[Path], days: int | None) -> list[Path]:
    if days is None:
        return candidates
    cutoff = datetime.now().timestamp() - days * 24 * 60 * 60
    return [path for path in candidates if path.stat().st_mtime >= cutoff]


def _task_image_match_score(task: dict[str, Any], image: dict[str, Any]) -> int:
    generated_at = call("_float_or_none", (image.get("sample") or {}).get("generated_at")) or call("_float_or_none", image.get("mtime"))
    started_at = call("_float_or_none", task.get("started_at"))
    finished_at = call("_float_or_none", task.get("finished_at"))
    if generated_at is None or started_at is None:
        return 0
    if generated_at < started_at - 180:
        return 0
    if finished_at is not None:
        return 3 if generated_at <= finished_at + 180 else 1
    return 2


def _empty_listing(source: str, label: str, directory: str, *, exists: bool, message: str) -> dict[str, Any]:
    return {
        "ok": True,
        "source": source,
        "label": label,
        "directory": directory,
        "directory_exists": exists,
        "count": 0,
        "total": 0,
        "images": [],
        "message": message,
        "sample_config": {},
    }


def _preview_settings_meta(settings: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "training_output_root",
        "current_task_sample_dir",
        "latest_run_dir",
        "latest_run_sample_dir",
        "effective_training_source",
    )
    return {key: settings.get(key, "") for key in keys}


def _image_meta(
    path: Path,
    *,
    task_id: str | None = None,
    sample_config: dict[str, Any] | None = None,
    prompt_entries: list[dict[str, Any]] | None = None,
    step_index: dict[int, int] | None = None,
) -> dict[str, Any]:
    stat = path.stat()
    width, height = probe_image_size(path)
    rel_path = call("_display_path", path)
    url = f"/api/preview/image?file={quote(rel_path)}"
    if task_id:
        url += f"&task_id={quote(str(task_id))}"
    sample_meta = _sample_image_meta(
        path,
        sample_config=sample_config,
        prompt_entries=prompt_entries or [],
        step_index=step_index or {},
    )
    return {
        "file": rel_path,
        "name": path.name,
        "url": url,
        "mtime": stat.st_mtime,
        "mtime_text": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "size_bytes": stat.st_size,
        "width": width,
        "height": height,
        "sample": sample_meta,
    }


def _sample_image_meta(
    path: Path,
    *,
    sample_config: dict[str, Any] | None,
    prompt_entries: list[dict[str, Any]],
    step_index: dict[int, int],
) -> dict[str, Any]:
    parsed = _parse_sample_image_name(path)
    if not parsed:
        return {}

    cfg = sample_config or {}
    prompt_index = parsed.get("prompt_index")
    prompt_entry = (
        prompt_entries[prompt_index]
        if isinstance(prompt_index, int) and 0 <= prompt_index < len(prompt_entries)
        else {}
    )
    parameters = dict(prompt_entry.get("parameters") or {})
    if parsed.get("seed") is not None and "seed" not in parameters:
        parameters["seed"] = parsed["seed"]
    sampler = str(parameters.get("sample_sampler") or cfg.get("sample_sampler") or "")
    if sampler:
        parameters.setdefault("sample_sampler", sampler)

    epoch = parsed.get("epoch")
    step = parsed.get("step")
    if step is None and isinstance(epoch, int):
        step = step_index.get(epoch)

    return {
        "epoch": epoch,
        "step": step,
        "prompt_index": prompt_index,
        "generated_at": parsed.get("generated_at"),
        "generated_at_text": parsed.get("generated_at_text"),
        "seed": parsed.get("seed"),
        "sampler": sampler,
        "prompt": prompt_entry.get("prompt", ""),
        "negative_prompt": prompt_entry.get("negative_prompt", ""),
        "raw_prompt": prompt_entry.get("raw", ""),
        "parameters": parameters,
        "source": {
            "from_filename": True,
            "prompt_file": str(cfg.get("sample_prompts") or ""),
            "step_from_weight": bool(step is not None and parsed.get("step") is None),
        },
    }


def _load_sample_prompt_entries(sample_config: dict[str, Any] | None) -> list[dict[str, Any]]:
    cfg = sample_config or {}
    prompt_file = str(cfg.get("sample_prompts") or "").strip()
    if not prompt_file:
        return []
    path = call("_resolve_display_path", prompt_file)
    if path is None or not path.exists() or not path.is_file():
        return []
    try:
        raw_text = path.read_text(encoding="utf-8")
    except Exception:
        return []
    if path.suffix.lower() == ".toml":
        return _parse_prompt_toml(raw_text)
    entries: list[dict[str, Any]] = []
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        entries.append(_parse_prompt_line(stripped))
    return entries


def _parse_prompt_line(line: str) -> dict[str, Any]:
    prompt_args = line.split(" --")
    prompt = prompt_args[0].strip()
    out: dict[str, Any] = {
        "raw": line,
        "prompt": prompt,
        "parameters": {},
    }
    params = out["parameters"]
    for arg in prompt_args[1:]:
        try:
            if m := re.match(r"w (\d+)", arg, re.IGNORECASE):
                params["width"] = int(m.group(1))
                continue
            if m := re.match(r"h (\d+)", arg, re.IGNORECASE):
                params["height"] = int(m.group(1))
                continue
            if m := re.match(r"d (\-?\d+)", arg, re.IGNORECASE):
                params["seed"] = int(m.group(1))
                continue
            if m := re.match(r"s (\d+)", arg, re.IGNORECASE):
                params["sample_steps"] = max(1, min(1000, int(m.group(1))))
                continue
            if m := re.match(r"l ([\d\.]+)", arg, re.IGNORECASE):
                params["scale"] = float(m.group(1))
                continue
            if m := re.match(r"g ([\d\.]+)", arg, re.IGNORECASE):
                params["guidance_scale"] = float(m.group(1))
                continue
            if m := re.match(r"n (.+)", arg, re.IGNORECASE):
                out["negative_prompt"] = m.group(1)
                continue
            if m := re.match(r"ss (.+)", arg, re.IGNORECASE):
                params["sample_sampler"] = m.group(1)
                continue
            if m := re.match(r"fs (.+)", arg, re.IGNORECASE):
                params["flow_shift"] = m.group(1)
                continue
        except ValueError:
            continue
    return out


def _parse_prompt_toml(text: str) -> list[dict[str, Any]]:
    try:
        data = toml.loads(text)
    except toml.TomlDecodeError:
        return []
    base = data.get("prompt", {}) if isinstance(data, dict) else {}
    subsets = base.get("subset") if isinstance(base, dict) else []
    if not isinstance(subsets, list):
        return []

    entries: list[dict[str, Any]] = []
    for subset in subsets:
        if not isinstance(subset, dict):
            continue
        merged = {**base, **subset}
        merged.pop("subset", None)
        prompt = str(merged.get("prompt") or "")
        params = {
            "width": call("_int_or_none", merged.get("width")),
            "height": call("_int_or_none", merged.get("height")),
            "seed": call("_int_or_none", merged.get("seed")),
            "sample_steps": call("_int_or_none", merged.get("sample_steps")),
            "scale": call("_float_or_none", merged.get("scale")),
            "guidance_scale": call("_float_or_none", merged.get("guidance_scale")),
            "sample_sampler": str(merged.get("sample_sampler") or ""),
            "flow_shift": merged.get("flow_shift"),
        }
        entries.append(
            {
                "raw": prompt,
                "prompt": prompt,
                "negative_prompt": str(merged.get("negative_prompt") or ""),
                "parameters": {k: v for k, v in params.items() if v not in (None, "")},
            }
        )
    return entries


def _parse_sample_image_name(path: Path) -> dict[str, Any] | None:
    stem = path.stem
    match = get("SAMPLE_NAME_RE").match(stem)
    if not match:
        return None

    tag = match.group("tag")
    epoch = call("_int_or_none", tag[1:]) if tag.startswith("e") else None
    step = None if tag.startswith("e") else call("_int_or_none", tag)
    prompt_index = call("_int_or_none", match.group("prompt_index"))
    timestamp = match.group("timestamp")
    generated_at = None
    generated_at_text = ""
    try:
        generated_at = datetime.strptime(timestamp, "%Y%m%d%H%M%S").timestamp()
        generated_at_text = datetime.strptime(timestamp, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        generated_at = None
    seed = call("_int_or_none", match.group("seed"))
    return {
        "epoch": epoch,
        "step": step,
        "prompt_index": prompt_index,
        "generated_at": generated_at,
        "generated_at_text": generated_at_text,
        "seed": seed,
    }


def _training_step_index(task: dict[str, Any] | None) -> dict[int, int]:
    if not task:
        return {}
    output_dir = str(task.get("output_dir") or "")
    variant = str(task.get("variant") or "")
    if not output_dir:
        return {}
    resolved = call("_resolve_training_output_dir", output_dir)
    if resolved is None or not resolved.exists() or not resolved.is_dir():
        return {}

    primary = resolved / f"{variant}.safetensors"
    candidates: list[Path] = []
    if primary.exists():
        candidates.append(primary)
    candidates.extend(
        sorted(
            [
                p
                for p in resolved.iterdir()
                if p.is_file()
                and p.suffix.lower() in get("WEIGHT_EXTS")
                and not p.name.endswith("_moe.safetensors")
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    )

    index: dict[int, int] = {}
    for path in candidates:
        metadata = call("_read_safetensors_metadata", path)
        epoch = call("_int_or_none", metadata.get("ss_epoch"))
        steps = call("_int_or_none", metadata.get("ss_steps"))
        if epoch is None or steps is None:
            continue
        index.setdefault(epoch, steps)
    return index


def _normalize_preview_delete_files(files: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    if not isinstance(files, (list, tuple, set)):
        raise ValueError("files 必须是非空列表")
    seen: set[str] = set()
    normalized: list[str] = []
    for item in files:
        raw = str(item or "").strip()
        if not raw or raw in seen:
            continue
        seen.add(raw)
        normalized.append(raw)
    if not normalized:
        raise ValueError("请至少选择一张图片")
    return normalized


def _ensure_preview_delete_target(path: Path, directory: Path, *, source: str) -> None:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError as exc:
        raise ValueError(f"只允许删除当前{_preview_source_delete_label(source)}目录中的图片") from exc


def _preview_source_delete_label(source: str) -> str:
    mapping = {
        "training": "训练样张",
        "inference": "生图测试",
        "custom": "自定义预览",
    }
    return mapping.get(source, "预览")


def _preview_delete_message(
    deleted: list[str],
    missing: list[str],
    blocked: list[dict[str, str]],
) -> str:
    parts: list[str] = []
    if deleted:
        parts.append(f"已从硬盘永久删除 {len(deleted)} 张图片")
    if missing:
        parts.append(f"{len(missing)} 张图片已不存在")
    if blocked:
        parts.append(f"{len(blocked)} 张图片未删除")
    if not parts:
        return "没有删除任何图片"
    return "，".join(parts) + "。"


def _preview_empty_message(
    source: str,
    fallback: str,
    sample_config: dict[str, Any] | None,
    *,
    settings: dict[str, Any] | None = None,
) -> str:
    if source != "training":
        return fallback
    settings = settings or {}
    training_source = str(settings.get("effective_training_source") or "")
    if training_source == "latest_run":
        cfg = sample_config or {}
        message = str(cfg.get("message") or "")
        if message and message != "训练中采样已配置":
            return f"{fallback}。{message}。"
        return f"{fallback}。最新运行目录里还没有可显示的样张。"
    if training_source == "saved_default" and not settings.get("latest_run_sample_dir"):
        root = str(settings.get("training_output_root") or get("DEFAULT_OUTPUT_ROOT"))
        return f"{fallback}。全局输出目录 {root} 下还没有可读取的 Web 运行样张目录。"
    cfg = sample_config or {}
    message = str(cfg.get("message") or "")
    if message and message != "训练中采样已配置":
        return f"{fallback}。{message}。"
    if cfg.get("enabled"):
        return f"{fallback}。如果训练刚开始，可能还没到达采样频率。"
    return f"{fallback}。未启用训练中采样时不会自动生成样张。"


def _training_preview_label(settings: dict[str, Any], *, task_id: str | None, task_label: str | None) -> str:
    if task_id and task_label:
        return f"训练过程中采样结果 · {task_label}"
    source = str(settings.get("effective_training_source") or "")
    if source == "current_task":
        return "训练过程中采样结果 · 当前任务"
    if source == "latest_run":
        run_dir = str(settings.get("latest_run_dir") or "")
        run_name = Path(run_dir).name if run_dir else "最新运行目录"
        return f"训练过程中采样结果 · {run_name}"
    return "训练过程中采样结果 · 兼容目录"


