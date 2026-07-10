"""Dataset / image / cache sidecar checks for training preflight."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from library.preprocess.captions import normalize_caption_source_mode, read_caption_source_from_dirs
from web.services.config.metadata import DATASET_IMAGE_EXTS
from web.services.config.preflight_runtime import (
    _bool_value,
    _caption_detection_counts_text,
    _count_source_images,
    _dataset_image_files,
    _dataset_rows_for_estimate,
    _nl_tag_mix_caption_counts,
    _nl_tag_mix_enabled,
    _normalize_path_pattern,
    _normalize_trigger_clone,
    _resolve_project_path,
)

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
