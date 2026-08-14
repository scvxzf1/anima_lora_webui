"""Preview image list/delete helpers for WebUI preview service."""

from __future__ import annotations

from datetime import datetime
from heapq import nlargest
import os
from pathlib import Path
import stat as stat_module
from typing import Any
from urllib.parse import quote
import re

import toml
from PIL import Image

from web.services.image_listing import select_recent_files
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
    candidates, total = select_recent_files(
        resolved,
        suffixes=get("IMAGE_EXTS"),
        limit=limit,
        min_mtime=_preview_days_cutoff(days),
    )
    prompt_entries = _load_sample_prompt_entries(sample_config) if source == "training" else []
    step_index = call("_training_step_index", task) if source == "training" else {}
    images = []
    for path, _ in candidates:
        meta = _available_image_meta(
            path,
            task_id=task_id,
            sample_config=sample_config,
            prompt_entries=prompt_entries,
            step_index=step_index,
        )
        if meta is not None:
            images.append(meta)

    return {
        "ok": True,
        "source": source,
        "label": label,
        "directory": display_dir,
        "directory_exists": True,
        "count": len(images),
        "total": total,
        "images": images,
        "message": "" if images else _preview_empty_message(source, "暂无预览图", sample_config, settings=settings),
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
    image_exts = get("IMAGE_EXTS")

    for raw in targets:
        try:
            path = _ensure_preview_delete_target(raw, resolved_dir, source=source)
            if path.suffix.lower() not in image_exts:
                raise ValueError("只允许删除预览图片文件")
            if not _unlink_preview_target(path, resolved_dir, source=source):
                missing.append(call("_display_path", path))
                continue
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

    _, remaining_total = select_recent_files(
        resolved_dir,
        suffixes=image_exts,
        limit=0,
    )

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
    days: int | None = None,
) -> dict[str, Any]:
    group_label = f"{methods_subdir} / {variant} / {preset or 'default'}"
    label = f"训练分组合并采样结果 · {group_label} · {len(tasks)} 次训练"
    limit = max(1, min(int(limit or 200), get("MAX_IMAGE_LIMIT")))
    days = _normalize_preview_days(days)
    cutoff = _preview_days_cutoff(days)
    candidate_contexts: dict[str, dict[str, Any]] = {}
    scanned_directories: dict[str, list[tuple[Path, os.stat_result]]] = {}
    directories: list[str] = []
    total = 0

    for task in tasks:
        sample_dir = str(task.get("sample_dir") or "")
        if not sample_dir:
            continue
        resolved = call("_resolve_preview_dir", sample_dir, current_task_sample_dir=sample_dir)
        if resolved is None:
            continue
        try:
            if not resolved.is_dir():
                continue
            resolved_key = str(resolved.resolve())
        except OSError:
            continue
        if resolved_key not in scanned_directories:
            candidates, directory_total = select_recent_files(
                resolved,
                suffixes=get("IMAGE_EXTS"),
                limit=limit,
                min_mtime=cutoff,
            )
            scanned_directories[resolved_key] = candidates
            total += directory_total
        candidates = scanned_directories[resolved_key]
        display_dir = call("_display_path", resolved)
        if display_dir not in directories:
            directories.append(display_dir)
        sample_config = task.get("sample_config") if isinstance(task.get("sample_config"), dict) else {}
        prompt_entries = _load_sample_prompt_entries(sample_config)
        step_index = call("_training_step_index", task)
        task_id = str(task.get("id") or "")
        task_label = call("_preview_task_label", task)
        source_task = {
            "id": task_id,
            "label": task_label,
            "state": task.get("state", ""),
            "started_at": task.get("started_at"),
            "started_at_text": task.get("started_at_text", ""),
            "finished_at": task.get("finished_at"),
            "finished_at_text": task.get("finished_at_text", ""),
            "sample_dir": sample_dir,
        }
        for path, stat_result in candidates:
            match_meta = _candidate_match_meta(path, stat_result)
            match_score = _task_image_match_score(task, match_meta)
            key = path.as_posix()
            previous = candidate_contexts.get(key)
            if previous is not None and match_score <= previous["match_score"]:
                continue
            candidate_contexts[key] = {
                "path": path,
                "stat": stat_result,
                "task_id": task_id,
                "sample_config": sample_config,
                "prompt_entries": prompt_entries,
                "step_index": step_index,
                "source_task": source_task,
                "match_score": match_score,
            }

    selected = nlargest(
        limit,
        candidate_contexts.values(),
        key=lambda item: (
            float(item["stat"].st_mtime),
            item["path"].name,
            item["path"].as_posix(),
        ),
    )
    images: list[dict[str, Any]] = []
    for item in selected:
        meta = _available_image_meta(
            item["path"],
            task_id=item["task_id"],
            sample_config=item["sample_config"],
            prompt_entries=item["prompt_entries"],
            step_index=item["step_index"],
        )
        if meta is None:
            continue
        meta["source_task"] = item["source_task"]
        images.append(meta)
    return {
        "ok": True,
        "source": "training",
        "mode": "config_group",
        "label": label,
        "directory": " · ".join(directories[:2]) + (" · ..." if len(directories) > 2 else ""),
        "directories": directories,
        "directory_exists": bool(directories),
        "count": len(images),
        "total": total,
        "images": images,
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


def _preview_days_cutoff(days: int | None) -> float | None:
    if days is None:
        return None
    return datetime.now().timestamp() - days * 24 * 60 * 60


def _filter_preview_candidates_by_days(candidates: list[Path], days: int | None) -> list[Path]:
    cutoff = _preview_days_cutoff(days)
    if cutoff is None:
        return candidates
    filtered: list[Path] = []
    for path in candidates:
        try:
            if path.stat().st_mtime >= cutoff:
                filtered.append(path)
        except OSError:
            continue
    return filtered


def _available_image_meta(
    path: Path,
    *,
    task_id: str | None = None,
    sample_config: dict[str, Any] | None = None,
    prompt_entries: list[dict[str, Any]] | None = None,
    step_index: dict[int, int] | None = None,
) -> dict[str, Any] | None:
    """Return image metadata unless the candidate vanished or changed type."""

    try:
        stat_result = path.lstat()
        if not stat_module.S_ISREG(stat_result.st_mode):
            return None
        return call(
            "_image_meta",
            path,
            task_id=task_id,
            sample_config=sample_config,
            prompt_entries=prompt_entries,
            step_index=step_index,
            stat_result=stat_result,
        )
    except OSError:
        return None


def _candidate_match_meta(path: Path, stat_result: os.stat_result) -> dict[str, Any]:
    return {
        "mtime": stat_result.st_mtime,
        "sample": _parse_sample_image_name(path) or {},
    }


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
    stat_result: os.stat_result | None = None,
) -> dict[str, Any]:
    stat = stat_result or path.stat()
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
    if parsed:
        return _sample_meta_from_filename(parsed, sample_config, prompt_entries, step_index)
    png_meta = _read_png_metadata(path)
    if png_meta:
        return _sample_meta_from_png(png_meta, path)
    return {}


def _sample_meta_from_filename(
    parsed: dict[str, Any],
    sample_config: dict[str, Any] | None,
    prompt_entries: list[dict[str, Any]],
    step_index: dict[int, int],
) -> dict[str, Any]:
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


def _read_png_metadata(path: Path) -> dict[str, str]:
    """Read PNG tEXt chunks written by ``library.inference.output._build_png_info``.
    Only PNG is supported (the inference output format); other exts return {}.
    Uses ``Image.open`` which reads tEXt from the header without decoding pixels.
    """
    if path.suffix.lower() != ".png":
        return {}
    try:
        with Image.open(path) as img:
            raw = dict(getattr(img, "text", None) or img.info or {})
    except Exception:
        return {}
    return {str(k): str(v) for k, v in raw.items() if v not in (None, "")}


def _sample_meta_from_png(png_meta: dict[str, str], path: Path) -> dict[str, Any]:
    """Build ``sample`` for inference images from embedded PNG params.

    Inference images carry seed/sampler/steps/cfg/flow_shift/prompt/size/timestamp
    in tEXt (written at generation time). Epoch/step/prompt_index are training
    concepts and have no source here, so they stay None — the dialog renders `-`.
    """
    def _int(key: str) -> int | None:
        return call("_int_or_none", png_meta.get(key))

    def _float(key: str) -> float | None:
        return call("_float_or_none", png_meta.get(key))

    seed = _int("seed")
    width = _int("width")
    height = _int("height")
    infer_steps = _int("infer_steps")
    guidance_scale = _float("guidance_scale")
    flow_shift = _float("flow_shift")
    sampler = str(png_meta.get("sampler") or "")
    prompt = str(png_meta.get("prompt") or "")
    negative_prompt = str(png_meta.get("negative_prompt") or "")

    parameters: dict[str, Any] = {}
    if seed is not None:
        parameters["seed"] = seed
    if width is not None:
        parameters["width"] = width
    if height is not None:
        parameters["height"] = height
    if infer_steps is not None:
        parameters["sample_steps"] = infer_steps
    if guidance_scale is not None:
        parameters["guidance_scale"] = guidance_scale
    if flow_shift is not None:
        parameters["flow_shift"] = flow_shift
    if sampler:
        parameters["sample_sampler"] = sampler

    generated_at, generated_at_text = _png_timestamp(png_meta.get("timestamp"))

    return {
        "epoch": None,
        "step": None,
        "prompt_index": None,
        "generated_at": generated_at,
        "generated_at_text": generated_at_text,
        "seed": seed,
        "sampler": sampler,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "raw_prompt": "",
        "parameters": parameters,
        "source": {
            "from_png": True,
            "prompt_file": "",
        },
    }


def _png_timestamp(value: str | None) -> tuple[float | None, str]:
    """Parse the ``YYYYMMDD-HHMMSS-mmm`` timestamp written by ``get_time_flag``."""
    raw = str(value or "").strip()
    if not raw:
        return None, ""
    try:
        dt = datetime.strptime(raw, "%Y%m%d-%H%M%S-%f")
    except ValueError:
        return None, ""
    return dt.timestamp(), dt.strftime("%Y-%m-%d %H:%M:%S")


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
    max_files = int(get("MAX_IMAGE_LIMIT"))
    if len(normalized) > max_files:
        raise ValueError(f"一次最多删除 {max_files} 张图片")
    return normalized


def _ensure_preview_delete_target(raw: str | Path, directory: Path, *, source: str) -> Path:
    clean = str(raw or "").replace("\\", "/").strip()
    if not clean:
        raise ValueError("图片路径不能为空")
    requested = Path(clean)
    if ".." in requested.parts:
        raise ValueError("图片路径不能包含 ..")

    if requested.is_absolute():
        candidate = requested
    elif len(requested.parts) == 1:
        candidate = directory / requested.name
    else:
        candidate = Path(get("ROOT")) / requested

    try:
        resolved_directory = directory.resolve()
        resolved_parent = candidate.parent.resolve()
    except OSError as exc:
        raise ValueError("无法确认预览图片目录") from exc
    if resolved_parent != resolved_directory or candidate.name in {"", ".", ".."}:
        raise ValueError(f"只允许删除当前{_preview_source_delete_label(source)}目录中的图片")
    return resolved_directory / candidate.name


def _unlink_preview_target(path: Path, directory: Path, *, source: str) -> bool:
    """Delete one direct child without following a symlink target."""

    label = _preview_source_delete_label(source)
    try:
        resolved_directory = directory.resolve()
    except OSError as exc:
        raise ValueError("无法确认预览图片目录") from exc
    if path.parent != resolved_directory or not path.name or "/" in path.name or "\\" in path.name:
        raise ValueError(f"只允许删除当前{label}目录中的图片")

    supports_dir_fd = (
        os.stat in os.supports_dir_fd
        and os.unlink in os.supports_dir_fd
        and os.stat in os.supports_follow_symlinks
        and hasattr(os, "O_NOFOLLOW")
    )
    if supports_dir_fd:
        expected = os.stat(resolved_directory, follow_symlinks=False)
        if not stat_module.S_ISDIR(expected.st_mode):
            raise ValueError("预览图路径不是目录")
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | os.O_NOFOLLOW
        directory_fd = os.open(resolved_directory, flags)
        try:
            opened = os.fstat(directory_fd)
            if (opened.st_dev, opened.st_ino) != (expected.st_dev, expected.st_ino):
                raise ValueError("预览图目录在删除过程中发生变化")
            try:
                stat_result = os.stat(path.name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                return False
            if stat_module.S_ISLNK(stat_result.st_mode):
                raise ValueError("不允许删除符号链接")
            if not stat_module.S_ISREG(stat_result.st_mode):
                raise ValueError("只允许删除普通图片文件")
            os.unlink(path.name, dir_fd=directory_fd)
            return True
        finally:
            os.close(directory_fd)

    try:
        stat_result = path.lstat()
    except FileNotFoundError:
        return False
    if stat_module.S_ISLNK(stat_result.st_mode):
        raise ValueError("不允许删除符号链接")
    if not stat_module.S_ISREG(stat_result.st_mode):
        raise ValueError("只允许删除普通图片文件")
    try:
        if path.parent.resolve() != resolved_directory:
            raise ValueError(f"只允许删除当前{label}目录中的图片")
    except OSError as exc:
        raise ValueError("无法确认预览图片目录") from exc
    path.unlink()
    return True


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
