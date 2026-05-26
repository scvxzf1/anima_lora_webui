"""Preview image discovery and Web UI preview settings."""

from __future__ import annotations

from datetime import datetime
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

import toml
from PIL import Image

from web.services import settings_service

ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = ROOT / "configs"
SETTINGS_FILE = CONFIGS_DIR / "web-ui-settings.toml"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
WEIGHT_EXTS = {".safetensors"}
DEFAULT_TRAINING_DIR = "output/ckpt/sample"
DEFAULT_INFERENCE_DIR = "output/tests"
DEFAULT_OUTPUT_ROOT = settings_service.DEFAULT_OUTPUT_ROOT
MAX_IMAGE_LIMIT = 500
MAX_WEIGHT_LIMIT = 500
SAMPLE_NAME_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<tag>e\d{6}|\d{6})_(?P<prompt_index>\d+)_(?P<timestamp>\d{14})(?:_(?P<seed>-?\d+))?$"
)


def get_preview_settings(
    current_task_sample_dir: str | None = None,
    *,
    allow_latest_fallback: bool = True,
) -> dict[str, Any]:
    settings = _load_settings()
    task_dir = _normalize_optional_preview_dir(current_task_sample_dir)
    latest_run = _latest_runtime_sample_dir()
    if task_dir:
        training_dir = task_dir
        training_source = "current_task"
    elif allow_latest_fallback and latest_run:
        training_dir = latest_run["sample_dir"]
        training_source = "latest_run"
    elif not allow_latest_fallback:
        training_dir = ""
        training_source = "selected_task_missing"
    else:
        training_dir = settings["training_dir"]
        training_source = "saved_default"
    output_root = _resolve_global_output_root()
    return {
        "ok": True,
        "training_dir": settings["training_dir"],
        "inference_dir": settings["inference_dir"],
        "custom_dir": settings["custom_dir"],
        "training_output_root": settings_service.display_path(output_root),
        "current_task_sample_dir": task_dir,
        "latest_run_dir": latest_run["run_dir"] if latest_run else "",
        "latest_run_sample_dir": latest_run["sample_dir"] if latest_run else "",
        "effective_training_dir": training_dir,
        "effective_training_source": training_source,
        "defaults": {
            "training_dir": DEFAULT_TRAINING_DIR,
            "inference_dir": DEFAULT_INFERENCE_DIR,
            "custom_dir": "",
            "training_output_root": DEFAULT_OUTPUT_ROOT,
        },
    }


def save_preview_settings(data: dict[str, Any]) -> dict[str, Any]:
    current = _load_settings()
    next_settings = {
        "training_dir": _normalize_project_dir(
            data.get("training_dir", current["training_dir"]) or DEFAULT_TRAINING_DIR,
            allow_empty=False,
        ),
        "inference_dir": _normalize_preview_dir(
            data.get("inference_dir", current["inference_dir"]) or DEFAULT_INFERENCE_DIR,
            allow_empty=False,
        ),
        "custom_dir": _normalize_preview_dir(data.get("custom_dir", current["custom_dir"]) or "", allow_empty=True),
    }
    raw = _load_raw_settings()
    raw["preview"] = next_settings
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(toml.dumps(raw), encoding="utf-8")
    return {"ok": True, "message": "预览图路径设置已保存", **next_settings}


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
) -> dict[str, Any]:
    source = (source or "training").strip().lower()
    if source not in {"training", "inference", "custom"}:
        raise ValueError("source 只能是 training、inference 或 custom")

    settings = get_preview_settings(
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

    resolved = _resolve_preview_dir(rel_dir, current_task_sample_dir=current_task_sample_dir if source == "training" else None)
    if resolved is None:
        raise ValueError("预览图路径不合法")

    display_dir = _display_path(resolved)
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

    limit = max(1, min(int(limit or 200), MAX_IMAGE_LIMIT))
    candidates = [
        p
        for p in resolved.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    prompt_entries = _load_sample_prompt_entries(sample_config) if source == "training" else []
    step_index = _training_step_index(task) if source == "training" else {}

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
    limit = max(1, min(int(limit or 200), MAX_IMAGE_LIMIT))
    images_by_path: dict[str, dict[str, Any]] = {}
    directories: list[str] = []

    for task in tasks:
        sample_dir = str(task.get("sample_dir") or "")
        if not sample_dir:
            continue
        resolved = _resolve_preview_dir(sample_dir, current_task_sample_dir=sample_dir)
        if resolved is None or not resolved.exists() or not resolved.is_dir():
            continue
        display_dir = _display_path(resolved)
        if display_dir not in directories:
            directories.append(display_dir)
        sample_config = task.get("sample_config") if isinstance(task.get("sample_config"), dict) else {}
        prompt_entries = _load_sample_prompt_entries(sample_config)
        step_index = _training_step_index(task)
        task_id = str(task.get("id") or "")
        task_label = _preview_task_label(task)
        for path in resolved.iterdir():
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
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


def _task_image_match_score(task: dict[str, Any], image: dict[str, Any]) -> int:
    generated_at = _float_or_none((image.get("sample") or {}).get("generated_at")) or _float_or_none(image.get("mtime"))
    started_at = _float_or_none(task.get("started_at"))
    finished_at = _float_or_none(task.get("finished_at"))
    if generated_at is None or started_at is None:
        return 0
    if generated_at < started_at - 180:
        return 0
    if finished_at is not None:
        return 3 if generated_at <= finished_at + 180 else 1
    return 2


def _group_weight_match_score(weight: dict[str, Any]) -> tuple[int, float]:
    source_task = weight.get("source_task") or {}
    scope = str(weight.get("scope") or "")
    score = 0
    mtime = _float_or_none(weight.get("mtime"))
    started_at = _float_or_none(source_task.get("started_at"))
    finished_at = _float_or_none(source_task.get("finished_at"))
    if mtime is not None and started_at is not None:
        if mtime < started_at - 180:
            return (score, started_at)
        if finished_at is not None:
            score += 6 if mtime <= finished_at + 180 else 2
        else:
            score += 3
    if scope == "task":
        score += 2
    if source_task.get("id"):
        score += 1
    return (score, started_at or 0)


def _group_weight_scope_label(weight: dict[str, Any], source_task: dict[str, Any]) -> str:
    base = str(weight.get("scope_label") or "")
    if source_task.get("id"):
        return f"{base} · {source_task.get('label') or source_task.get('id')}"
    return base


def resolve_preview_image(rel_path: str, allowed_sample_dir: str | None = None) -> Path:
    resolved = _resolve_preview_file(rel_path, allowed_sample_dir=allowed_sample_dir)
    if resolved.suffix.lower() not in IMAGE_EXTS:
        raise ValueError("只允许读取预览图片文件")
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError("图片不存在")
    return resolved


def resolve_training_weight(rel_path: str, task: dict[str, Any] | None = None) -> Path:
    resolved = _resolve_weight_file(rel_path, task=task)
    if resolved.suffix.lower() not in WEIGHT_EXTS or resolved.name.endswith("_moe.safetensors"):
        raise ValueError("只允许下载训练权重文件")
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError("权重文件不存在")
    return resolved


def list_training_weights(
    task: dict[str, Any] | None = None,
    *,
    allow_latest_fallback: bool = True,
) -> dict[str, Any]:
    task = task or {}
    output_dir = str(task.get("output_dir") or "")
    if not output_dir:
        if not allow_latest_fallback:
            return _empty_weights_listing("", "这个历史训练任务没有记录输出目录")
        latest = _latest_runtime_sample_dir()
        if not latest:
            return _empty_weights_listing("", "训练任务没有记录输出目录")
        output_dir = str(Path(latest["sample_dir"]).parent)
        task = {**task, "output_dir": output_dir}

    resolved = _resolve_training_output_dir(output_dir)
    if resolved is None:
        raise ValueError("训练输出目录不合法")
    display_dir = _display_path(resolved)
    if not resolved.exists():
        return _empty_weights_listing(display_dir, "输出目录不存在")
    if not resolved.is_dir():
        return _empty_weights_listing(display_dir, "输出路径不是目录")

    output_name = str(task.get("variant") or "")
    candidates = [
        p
        for p in resolved.iterdir()
        if p.is_file()
        and p.suffix.lower() in WEIGHT_EXTS
        and not p.name.endswith("_moe.safetensors")
    ]
    if output_name:
        named = [p for p in candidates if p.name.startswith(output_name)]
        if named:
            candidates = named

    items = [_weight_meta(path, task=task) for path in candidates[:MAX_WEIGHT_LIMIT]]
    items.sort(key=_weight_sort_key)
    task_count = sum(1 for item in items if item.get("scope") == "task")
    return {
        "ok": True,
        "directory": display_dir,
        "directory_exists": True,
        "count": len(items),
        "total": len(candidates),
        "task_count": task_count,
        "weights": items,
        "message": "" if items else "未找到权重文件",
    }


def list_config_group_training_weights(
    tasks: list[dict[str, Any]],
    *,
    methods_subdir: str,
    variant: str,
    preset: str,
) -> dict[str, Any]:
    group_label = f"{methods_subdir} / {variant} / {preset or 'default'}"
    weights_by_path: dict[str, dict[str, Any]] = {}
    directories: list[str] = []

    for task in tasks:
        listing = list_training_weights(task, allow_latest_fallback=False)
        directory = str(listing.get("directory") or "")
        if directory and directory not in directories:
            directories.append(directory)
        task_source = {
            "id": str(task.get("id") or ""),
            "label": _preview_task_label(task),
            "state": task.get("state", ""),
            "started_at": task.get("started_at"),
            "started_at_text": task.get("started_at_text", ""),
            "finished_at": task.get("finished_at"),
            "finished_at_text": task.get("finished_at_text", ""),
            "output_dir": str(task.get("output_dir") or ""),
        }
        for item in listing.get("weights") or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("file") or "")
            if not key:
                continue
            merged = dict(item)
            merged["source_task"] = task_source
            merged["scope_label"] = _group_weight_scope_label(merged, task_source)
            previous = weights_by_path.get(key)
            if previous is None or _group_weight_match_score(merged) > _group_weight_match_score(previous):
                weights_by_path[key] = merged

    weights = list(weights_by_path.values())
    weights.sort(key=_weight_sort_key)
    task_weight_count = sum(1 for item in weights if item.get("scope") == "task")
    return {
        "ok": True,
        "mode": "config_group",
        "label": f"训练分组合并权重 · {group_label} · {len(tasks)} 次训练",
        "directory": " · ".join(directories[:2]) + (" · ..." if len(directories) > 2 else ""),
        "directories": directories,
        "directory_exists": bool(directories),
        "count": len(weights),
        "total": len(weights),
        "task_count": task_weight_count,
        "weights": weights,
        "message": "" if weights else "这个训练分组还没有可显示的权重文件",
        "group": {
            "methods_subdir": methods_subdir,
            "variant": variant,
            "preset": preset or "default",
        },
        "group_task_count": len(tasks),
    }


def _preview_task_label(task: dict[str, Any]) -> str:
    return str(
        task.get("name")
        or f"{task.get('methods_subdir') or '-'} / {task.get('variant') or task.get('id') or '-'}"
    )


def _load_settings() -> dict[str, str]:
    defaults = {
        "training_dir": DEFAULT_TRAINING_DIR,
        "inference_dir": DEFAULT_INFERENCE_DIR,
        "custom_dir": "",
    }
    raw = _load_raw_settings()
    preview = raw.get("preview", {}) if isinstance(raw, dict) else {}
    if not isinstance(preview, dict):
        return defaults
    out = dict(defaults)
    for key in out:
        try:
            normalizer = _normalize_project_dir if key == "training_dir" else _normalize_preview_dir
            out[key] = normalizer(str(preview.get(key, out[key]) or ""), allow_empty=(key == "custom_dir"))
        except ValueError:
            out[key] = defaults[key]
    return out


def _load_raw_settings() -> dict[str, Any]:
    if not SETTINGS_FILE.exists():
        return {}
    try:
        raw = toml.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except toml.TomlDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


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
    width = None
    height = None
    try:
        with Image.open(path) as img:
            width, height = img.size
    except Exception:
        pass
    rel_path = _display_path(path)
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


def _weight_meta(path: Path, *, task: dict[str, Any] | None = None) -> dict[str, Any]:
    stat = path.stat()
    metadata = _read_safetensors_metadata(path)
    epoch = _int_or_none(metadata.get("ss_epoch"))
    steps = _int_or_none(metadata.get("ss_steps"))
    num_epochs = _int_or_none(metadata.get("ss_num_epochs"))
    max_steps = _int_or_none(metadata.get("ss_max_train_steps"))
    output_name = str(metadata.get("ss_output_name") or "")
    kind = _weight_kind(path.name, output_name)
    scope = _weight_scope(stat.st_mtime, metadata, task)
    rel_path = _display_path(path)
    download_url = f"/api/preview/weight?file={quote(rel_path)}"
    task_id = str((task or {}).get("id") or "")
    if task_id:
        download_url += f"&task_id={quote(task_id)}"
    return {
        "file": rel_path,
        "abs_path": str(path.resolve()),
        "name": path.name,
        "download_url": download_url,
        "kind": kind,
        "scope": scope,
        "scope_label": "本任务" if scope == "task" else "同目录其他运行",
        "epoch": epoch,
        "steps": steps,
        "num_epochs": num_epochs,
        "max_steps": max_steps,
        "mtime": stat.st_mtime,
        "mtime_text": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "size_bytes": stat.st_size,
        "output_name": output_name,
    }


def _read_safetensors_metadata(path: Path) -> dict[str, str]:
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
        return {str(k): str(v) for k, v in metadata.items()}
    except Exception:
        return {}


def _load_sample_prompt_entries(sample_config: dict[str, Any] | None) -> list[dict[str, Any]]:
    cfg = sample_config or {}
    prompt_file = str(cfg.get("sample_prompts") or "").strip()
    if not prompt_file:
        return []
    path = _resolve_display_path(prompt_file)
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
            "width": _int_or_none(merged.get("width")),
            "height": _int_or_none(merged.get("height")),
            "seed": _int_or_none(merged.get("seed")),
            "sample_steps": _int_or_none(merged.get("sample_steps")),
            "scale": _float_or_none(merged.get("scale")),
            "guidance_scale": _float_or_none(merged.get("guidance_scale")),
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
    match = SAMPLE_NAME_RE.match(stem)
    if not match:
        return None

    tag = match.group("tag")
    epoch = _int_or_none(tag[1:]) if tag.startswith("e") else None
    step = None if tag.startswith("e") else _int_or_none(tag)
    prompt_index = _int_or_none(match.group("prompt_index"))
    timestamp = match.group("timestamp")
    generated_at = None
    generated_at_text = ""
    try:
        generated_at = datetime.strptime(timestamp, "%Y%m%d%H%M%S").timestamp()
        generated_at_text = datetime.strptime(timestamp, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        generated_at = None
    seed = _int_or_none(match.group("seed"))
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
    resolved = _resolve_training_output_dir(output_dir)
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
                and p.suffix.lower() in WEIGHT_EXTS
                and not p.name.endswith("_moe.safetensors")
            ],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    )

    index: dict[int, int] = {}
    for path in candidates:
        metadata = _read_safetensors_metadata(path)
        epoch = _int_or_none(metadata.get("ss_epoch"))
        steps = _int_or_none(metadata.get("ss_steps"))
        if epoch is None or steps is None:
            continue
        index.setdefault(epoch, steps)
    return index


def _weight_kind(name: str, output_name: str) -> str:
    if output_name and name == f"{output_name}.safetensors":
        return "final"
    if output_name and name == f"{output_name}-checkpoint.safetensors":
        return "resume"
    if re.search(r"-step\d+\.safetensors$", name):
        return "step"
    if re.search(r"-\d{6}\.safetensors$", name):
        return "epoch"
    return "weight"


def _weight_sort_key(item: dict[str, Any]) -> tuple[int, int, float, str]:
    scope_rank = {"task": 0, "other": 1}
    kind_rank = {"epoch": 0, "step": 1, "resume": 2, "final": 3, "weight": 4}
    primary = item.get("steps") if item.get("steps") is not None else -1
    epoch = item.get("epoch") if item.get("epoch") is not None else -1
    return (
        int(scope_rank.get(str(item.get("scope")), 9)),
        int(kind_rank.get(str(item.get("kind")), 9)),
        int(primary),
        float(item.get("mtime") or 0),
        str(item.get("name") or ""),
    )


def _weight_scope(mtime: float, metadata: dict[str, str], task: dict[str, Any] | None) -> str:
    if not task:
        return "other"
    started = _float_or_none(task.get("started_at"))
    finished = _float_or_none(task.get("finished_at"))
    if started is None:
        return "other"
    lower = started - 180
    upper = (finished + 180) if finished is not None else (datetime.now().timestamp() + 180)
    meta_started = _float_or_none(metadata.get("ss_training_started_at"))
    if meta_started is not None and lower <= meta_started <= upper:
        return "task"
    if lower <= float(mtime) <= upper:
        return "task"
    return "other"


def _empty_weights_listing(directory: str, message: str) -> dict[str, Any]:
    return {
        "ok": True,
        "directory": directory,
        "directory_exists": False,
        "count": 0,
        "total": 0,
        "weights": [],
        "message": message,
    }


def _normalize_optional_preview_dir(value: str | None) -> str:
    if not value:
        return ""
    try:
        return _normalize_preview_dir(value, allow_empty=True)
    except ValueError:
        return ""


def _normalize_preview_dir(value: str, *, allow_empty: bool) -> str:
    clean = str(value or "").replace("\\", "/").strip()
    if not clean:
        if allow_empty:
            return ""
        raise ValueError("路径不能为空")
    path = Path(clean)
    if path.is_absolute():
        return path.resolve().as_posix()
    return _normalize_project_dir(clean, allow_empty=allow_empty)


def _normalize_project_dir(value: str, *, allow_empty: bool) -> str:
    clean = str(value or "").replace("\\", "/").strip()
    if not clean:
        if allow_empty:
            return ""
        raise ValueError("路径不能为空")
    normalized = _normalize_project_file(clean)
    return normalized.rstrip("/")


def _normalize_project_file(value: str) -> str:
    clean = str(value or "").replace("\\", "/").strip()
    if not clean:
        raise ValueError("路径不能为空")
    path = Path(clean)
    if path.is_absolute():
        resolved = path.resolve()
        try:
            return resolved.relative_to(ROOT.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError("路径必须在项目目录内") from exc
    if ".." in path.parts:
        raise ValueError("路径不能包含 ..")
    return path.as_posix().lstrip("/")


def _resolve_project_path(value: str) -> Path | None:
    try:
        rel = _normalize_project_dir(value, allow_empty=False)
    except ValueError:
        return None
    resolved = (ROOT / rel).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError:
        return None
    return resolved


def _resolve_preview_dir(value: str, *, current_task_sample_dir: str | None = None) -> Path | None:
    path = Path(str(value or "").replace("\\", "/").strip())
    if path.is_absolute():
        resolved = path.resolve()
        allowed = _resolve_allowed_sample_dir(current_task_sample_dir)
        if allowed is not None:
            try:
                resolved.relative_to(allowed)
            except ValueError:
                return None
        return resolved
    return _resolve_project_path(value)


def _resolve_preview_file(value: str, *, allowed_sample_dir: str | None = None) -> Path:
    clean = str(value or "").replace("\\", "/").strip()
    if not clean:
        raise ValueError("路径不能为空")
    path = Path(clean)
    if path.is_absolute():
        resolved = path.resolve()
        for allowed in _allowed_external_preview_dirs(allowed_sample_dir):
            try:
                resolved.relative_to(allowed)
                return resolved
            except ValueError:
                continue
        raise ValueError("项目外图片只允许读取当前任务样张目录或已保存的预览目录")

    normalized = _normalize_project_file(clean)
    resolved = (ROOT / normalized).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError("图片路径必须在项目目录内") from exc
    return resolved


def _resolve_weight_file(value: str, *, task: dict[str, Any] | None = None) -> Path:
    clean = str(value or "").replace("\\", "/").strip()
    if not clean:
        raise ValueError("路径不能为空")
    path = Path(clean)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        normalized = _normalize_project_file(clean)
        resolved = (ROOT / normalized).resolve()
        try:
            resolved.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError("权重路径必须在项目目录内") from exc

    for allowed in _allowed_weight_dirs(task):
        try:
            resolved.relative_to(allowed)
            return resolved
        except ValueError:
            continue
    raise ValueError("权重文件只允许从训练输出目录或全局输出目录下载")


def _resolve_allowed_sample_dir(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(str(value).replace("\\", "/").strip())
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _allowed_external_preview_dirs(allowed_sample_dir: str | None) -> list[Path]:
    dirs: list[Path] = []
    sample_dir = _resolve_allowed_sample_dir(allowed_sample_dir)
    if sample_dir is not None:
        dirs.append(sample_dir)
    dirs.append(_resolve_global_output_root())
    settings = _load_settings()
    for key in ("inference_dir", "custom_dir"):
        resolved = _resolve_display_path(settings.get(key, ""))
        if resolved is not None:
            dirs.append(resolved)
    return dirs


def _allowed_weight_dirs(task: dict[str, Any] | None = None) -> list[Path]:
    dirs = [_resolve_global_output_root()]
    output_dir = str((task or {}).get("output_dir") or "")
    if output_dir:
        resolved = _resolve_training_output_dir(output_dir)
        if resolved is not None:
            dirs.append(resolved)
    settings = _load_settings()
    training_dir = _resolve_display_path(settings.get("training_dir", ""))
    if training_dir is not None:
        dirs.append(training_dir.parent if training_dir.name == "sample" else training_dir)
    return dirs


def _resolve_training_output_dir(value: str) -> Path | None:
    raw = str(value or "").replace("\\", "/").strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / raw
    return path.resolve()


def _resolve_display_path(value: str) -> Path | None:
    raw = str(value or "").replace("\\", "/").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def _int_or_none(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


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
        root = str(settings.get("training_output_root") or DEFAULT_OUTPUT_ROOT)
        return f"{fallback}。全局输出目录 {root} 下还没有可读取的 Web 运行样张目录。"
    cfg = sample_config or {}
    message = str(cfg.get("message") or "")
    if message and message != "训练中采样已配置":
        return f"{fallback}。{message}。"
    if cfg.get("enabled"):
        return f"{fallback}。如果训练刚开始，可能还没到达采样频率。"
    return f"{fallback}。未启用训练中采样时不会自动生成样张。"


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


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


def _latest_runtime_sample_dir() -> dict[str, str] | None:
    output_root = _resolve_global_output_root()
    if not output_root.exists() or not output_root.is_dir():
        return None

    candidates: list[tuple[float, str, Path, Path]] = []
    try:
        children = list(output_root.iterdir())
    except OSError:
        return None
    for run_dir in children:
        if not run_dir.is_dir():
            continue
        sample_dir = run_dir / "training_output" / "sample"
        if not sample_dir.is_dir():
            continue
        candidates.append((_runtime_sample_sort_ts(run_dir, sample_dir), run_dir.name, run_dir, sample_dir))
    if not candidates:
        return None
    _, _, run_dir, sample_dir = max(candidates, key=lambda item: (item[0], item[1]))
    return {
        "run_dir": _display_path(run_dir),
        "sample_dir": _display_path(sample_dir),
    }


def _runtime_sample_sort_ts(run_dir: Path, sample_dir: Path) -> float:
    timestamps: list[float] = []
    for path in (run_dir, run_dir / "training_output", sample_dir):
        try:
            timestamps.append(path.stat().st_mtime)
        except OSError:
            continue
    try:
        latest_image = max(
            (
                path.stat().st_mtime
                for path in sample_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS
            ),
            default=None,
        )
        if latest_image is not None:
            timestamps.append(latest_image)
    except OSError:
        pass
    return max(timestamps, default=0.0)


def _resolve_global_output_root() -> Path:
    return settings_service.resolve_output_root()
