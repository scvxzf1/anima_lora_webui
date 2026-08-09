"""Runtime dataset materialization helpers for WebUI training runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from typing import Any

from library.preprocess.captions import (
    CAPTION_SOURCE_CAPTIONS_JSON,
    CAPTIONS_JSON_FILE,
    normalize_caption_source_mode,
)
from web.services import config_service
from web.services.config.metadata import DATASET_CAPTION_EXTS, DATASET_IMAGE_EXTS
from web.services.training.runtime_paths import (
    _display_settings_path,
    _path_exists,
    _resolve_display_path,
)


def _display_logical_path(path: Path) -> str:
    """Project-relative display path without following symlinks.

    ``settings_service.display_path`` uses ``Path.resolve()``, which would turn a
    run ``dataset_cache/.../resized`` symlink into the shared pool path and break
    runtime dataset path contracts / tests.
    """
    path = Path(path)
    root = Path(config_service.ROOT).absolute()
    abs_path = path.absolute()
    try:
        return abs_path.relative_to(root).as_posix()
    except ValueError:
        return abs_path.as_posix()




def _normalize_path_pattern(*args, **kwargs):
    return config_service._normalize_path_pattern(*args, **kwargs)


def _normalize_trigger_clone(*args, **kwargs):
    return config_service._normalize_trigger_clone(*args, **kwargs)


def _normalize_nl_tag_mix(*args, **kwargs):
    return config_service._normalize_nl_tag_mix(*args, **kwargs)


def _nl_tag_mix_image_files(*args, **kwargs):
    return config_service._nl_tag_mix_image_files(*args, **kwargs)


def _nl_tag_mix_caption_source(*args, **kwargs):
    return config_service._nl_tag_mix_caption_source(*args, **kwargs)


def _classify_nl_tag_caption_text(*args, **kwargs):
    return config_service._classify_nl_tag_caption_text(*args, **kwargs)


def _clone_runtime_dataset_rows(
    runtime_rows: list[dict[str, Any]],
    dataset_cache_dir: Path,
    *,
    copy_existing: bool,
) -> list[dict[str, Any]]:
    cloned_rows: list[dict[str, Any]] = []
    for index, row in enumerate(runtime_rows, start=1):
        group_dir = dataset_cache_dir / f"dataset-{index:02d}"
        resized_dir = group_dir / _runtime_dataset_child_name(
            str(row.get("image_dir") or row.get("resized_image_dir") or ""),
            default="resized",
            allowed={"resized", "trigger-clone-resized"},
        )
        lora_dir = group_dir / _runtime_dataset_child_name(
            str(row.get("cache_dir") or row.get("lora_cache_dir") or ""),
            default="lora",
            allowed={"lora", "trigger-clone-lora"},
        )
        resized_dir.mkdir(parents=True, exist_ok=True)
        lora_dir.mkdir(parents=True, exist_ok=True)
        if copy_existing:
            _copy_runtime_dataset_dir(str(row.get("image_dir") or row.get("resized_image_dir") or ""), resized_dir)
            _copy_runtime_dataset_dir(str(row.get("cache_dir") or row.get("lora_cache_dir") or ""), lora_dir)
        source_dir = str(row.get("source_dir") or row.get("source_image_dir") or row.get("image_dir") or "")
        source_path = _resolve_display_path(source_dir)
        source_target = group_dir / _runtime_dataset_child_name(
            source_dir,
            default="source",
            allowed={"source", "trigger-clone-source"},
        )
        if (
            copy_existing
            and source_path
            and _is_materialized_runtime_source_dir(source_path)
            and source_path.resolve() != source_target.resolve()
        ):
            _copy_runtime_dataset_dir(source_dir, source_target)
            source_dir = _display_settings_path(source_target)
        cloned_rows.append({
            "source_dir": source_dir,
            "image_dir": _display_settings_path(resized_dir),
            "cache_dir": _display_settings_path(lora_dir),
            "num_repeats": row.get("num_repeats") or 1,
            "recursive": _bool_value_for_row(row.get("recursive"), True),
            "path_pattern": _normalize_path_pattern(row.get("path_pattern")),
            "settings": row.get("settings") if isinstance(row.get("settings"), dict) else {},
        })
    return cloned_rows


def _runtime_dataset_child_name(value: str, *, default: str, allowed: set[str]) -> str:
    path = _resolve_display_path(value)
    name = path.name if path is not None else ""
    return name if name in allowed else default


def _bool_value_for_row(value: Any, fallback: bool = False) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _prepare_runtime_nl_tag_mix_source(row: dict[str, Any], group_dir: Path, source_dir: str) -> str:
    mix = _normalize_nl_tag_mix(row.get("nl_tag_mix"))
    if not mix.get("enabled"):
        return source_dir
    source_path = _resolve_display_path(source_dir)
    if source_path is None:
        raise ValueError("captions格式nl/tag权重调整需要填写原始数据集路径")
    if not source_path.is_dir():
        raise ValueError(f"captions格式nl/tag权重调整失败: {source_dir} 不是目录")
    target_dir = group_dir / "source"
    target_dir.mkdir(parents=True, exist_ok=True)
    caption_settings = _nl_tag_mix_caption_settings(row)
    manifest = _build_nl_tag_mix_source(
        source_path,
        target_dir,
        tag_ratio=float(mix.get("tag_ratio") or 0.0),
        recursive=_bool_value_for_row(row.get("recursive"), True),
        path_pattern=_normalize_path_pattern(row.get("path_pattern")),
        **caption_settings,
    )
    (target_dir / "results.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return _display_settings_path(target_dir)


def _prepare_runtime_trigger_clone_source(row: dict[str, Any], group_dir: Path, source_dir: str) -> str:
    clone = _normalize_trigger_clone(row.get("trigger_clone"))
    if not clone["enabled"]:
        return source_dir
    prompt = clone["prompt"]
    if not prompt:
        raise ValueError("触发提示词图像克隆需要填写触发提示词")
    source_path = _resolve_display_path(source_dir)
    if source_path is None:
        raise ValueError("触发提示词图像克隆需要填写原始数据集路径")
    if not source_path.is_dir():
        raise ValueError(f"触发提示词图像克隆失败: {source_dir} 不是目录")
    target_dir = group_dir / "trigger-clone-source"
    target_dir.mkdir(parents=True, exist_ok=True)
    recursive = _bool_value_for_row(row.get("recursive"), True)
    path_pattern = _normalize_path_pattern(row.get("path_pattern"))
    images = _nl_tag_mix_image_files(
        source_path,
        DATASET_IMAGE_EXTS,
        recursive=recursive,
        path_pattern=path_pattern,
    )
    if not images:
        raise ValueError("触发提示词图像克隆失败: 数据集目录里没有可训练图片")

    captions_json: dict[str, list[str]] = {}
    items: list[dict[str, str]] = []
    for image_path in images:
        rel_image = _nl_tag_mix_relative_image_path(image_path, source_path)
        target_image = target_dir / rel_image
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_image)
        rel_key = rel_image.as_posix()
        captions_json[rel_key] = [prompt]
        items.append({
            "image": _display_settings_path(image_path),
            "target": _display_settings_path(target_image),
            "caption_key": rel_key,
        })

    (target_dir / CAPTIONS_JSON_FILE).write_text(
        json.dumps(captions_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "prompt": prompt,
        "num_repeats": clone["num_repeats"],
        "recursive": recursive,
        "source_dir": _display_settings_path(source_path),
        "target_dir": _display_settings_path(target_dir),
        "caption_source_mode": CAPTION_SOURCE_CAPTIONS_JSON,
        "total": len(items),
        "items": items,
    }
    (target_dir / "trigger-clone-results.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return _display_settings_path(target_dir)


def _nl_tag_mix_caption_settings(row: dict[str, Any]) -> dict[str, Any]:
    settings = row.get("settings") if isinstance(row.get("settings"), dict) else {}
    prefer_json_caption = bool(settings.get("prefer_json_caption"))
    return {
        "caption_source_mode": normalize_caption_source_mode(
            settings.get("caption_source_mode"),
            prefer_json_caption,
        ),
        "caption_extension": str(settings.get("caption_extension") or ".txt"),
        "prefer_json_caption": prefer_json_caption,
    }


def _build_nl_tag_mix_source(
    source_dir: Path,
    target_dir: Path,
    *,
    tag_ratio: float,
    recursive: bool = True,
    path_pattern: str = "*",
    caption_source_mode: str = "auto",
    caption_extension: str = ".txt",
    prefer_json_caption: bool = False,
) -> dict[str, Any]:
    samples = _classify_nl_tag_mix_samples(
        source_dir,
        recursive=recursive,
        path_pattern=path_pattern,
        caption_source_mode=caption_source_mode,
        caption_extension=caption_extension,
        prefer_json_caption=prefer_json_caption,
    )
    if not samples:
        raise ValueError("captions格式nl/tag权重调整失败: 数据集目录里没有可训练图片")
    captions_json_samples = [sample for sample in samples if sample.get("caption_entries")]
    plain_samples = [sample for sample in samples if not sample.get("caption_entries")]
    selected = [
        *captions_json_samples,
        *_select_nl_tag_mix_samples(plain_samples, tag_ratio=tag_ratio),
    ]
    items: list[dict[str, Any]] = []
    counts = {"tag": 0, "nl": 0}
    available_counts = {"tag": 0, "nl": 0}
    caption_available_counts = {"tag": 0, "nl": 0}
    caption_counts = {"tag": 0, "nl": 0}
    missing_caption_count = 0
    captions_json: dict[str, list[str]] = {}
    captions_json_target = target_dir / CAPTIONS_JSON_FILE
    for sample in samples:
        available_counts[sample["source"]] += 1
        available_entry_counts = _nl_tag_mix_source_counts(sample.get("caption_entries") or [])
        if available_entry_counts["tag"] or available_entry_counts["nl"]:
            caption_available_counts["tag"] += available_entry_counts["tag"]
            caption_available_counts["nl"] += available_entry_counts["nl"]
        elif sample.get("caption_path"):
            caption_available_counts[sample["source"]] += 1
        if not sample.get("caption_path"):
            missing_caption_count += 1
    for sample in sorted(selected, key=lambda item: item["image_path"].as_posix()):
        image_path = sample["image_path"]
        rel_image = _nl_tag_mix_relative_image_path(image_path, source_dir)
        target_image = target_dir / rel_image
        target_image.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_image)
        caption_source = sample.get("caption_source")
        selected_entries: list[dict[str, Any]] = []
        selected_entry_counts = {"tag": 0, "nl": 0}
        if getattr(caption_source, "from_captions_json", False):
            selected_entries = _select_nl_tag_caption_entries(
                sample.get("caption_entries") or [],
                tag_ratio=tag_ratio,
            )
            selected_entry_counts = _nl_tag_mix_source_counts(selected_entries)
            caption_counts["tag"] += selected_entry_counts["tag"]
            caption_counts["nl"] += selected_entry_counts["nl"]
            captions_json[rel_image.as_posix()] = [
                str(entry.get("text") or "")
                for entry in selected_entries
                if str(entry.get("text") or "").strip()
            ]
        elif sample.get("caption_path"):
            caption_counts[sample["source"]] += 1
        copied_captions = _copy_nl_tag_caption_sidecars(
            image_path,
            target_image,
            target_dir,
            caption_source,
            captions_json_path=captions_json_target,
        )
        source_kind = (
            _nl_tag_mix_dominant_source(selected_entry_counts)
            if selected_entries
            else sample["source"]
        )
        counts[source_kind] += 1
        item = {
            "stem": image_path.stem,
            "source": source_kind,
            "classification": sample["classification"],
            "image": _display_settings_path(image_path),
            "target": _display_settings_path(target_image),
            "caption": _display_settings_path(sample["caption_path"]) if sample.get("caption_path") else "",
            "captions": copied_captions,
            "caption_source_mode": sample.get("caption_source_mode", ""),
        }
        if selected_entries:
            item["caption_entry_count"] = len(sample.get("caption_entries") or [])
            item["weighted_caption_count"] = len(selected_entries)
            item["available_caption_counts"] = _nl_tag_mix_source_counts(sample.get("caption_entries") or [])
            item["actual_caption_counts"] = selected_entry_counts
            item["selected_caption_indices"] = [int(entry.get("index", 0)) for entry in selected_entries]
        items.append(item)
    if captions_json:
        captions_json_target.write_text(
            json.dumps(captions_json, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return {
        "tag_ratio": min(1.0, max(0.0, tag_ratio)),
        "classification_method": config_service.NL_TAG_MIX_CLASSIFICATION_METHOD,
        "caption_source_mode": caption_source_mode,
        "recursive": bool(recursive),
        "path_pattern": _normalize_path_pattern(path_pattern),
        "source_dir": _display_settings_path(source_dir),
        "available_tag_count": available_counts["tag"],
        "available_nl_count": available_counts["nl"],
        "actual_tag_count": counts["tag"],
        "actual_nl_count": counts["nl"],
        "available_tag_caption_count": caption_available_counts["tag"],
        "available_nl_caption_count": caption_available_counts["nl"],
        "actual_tag_caption_count": caption_counts["tag"],
        "actual_nl_caption_count": caption_counts["nl"],
        "total": len(items),
        "missing_caption_count": missing_caption_count,
        "items": items,
    }


def _classify_nl_tag_mix_samples(
    source_dir: Path,
    *,
    recursive: bool = True,
    path_pattern: str = "*",
    caption_source_mode: str = "auto",
    caption_extension: str = ".txt",
    prefer_json_caption: bool = False,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for image_path in _nl_tag_mix_image_files(
        source_dir,
        DATASET_IMAGE_EXTS,
        recursive=recursive,
        path_pattern=path_pattern,
    ):
        caption_source = _nl_tag_mix_caption_source(
            image_path,
            caption_source_mode=caption_source_mode,
            caption_extension=caption_extension,
            prefer_json_caption=prefer_json_caption,
            captions_root=source_dir,
        )
        caption_texts = caption_source.caption_texts()
        caption_entries = (
            _nl_tag_mix_caption_entries(caption_texts)
            if getattr(caption_source, "from_captions_json", False)
            else []
        )
        if caption_entries:
            entry_counts = _nl_tag_mix_source_counts(caption_entries)
            source_kind = _nl_tag_mix_dominant_source(entry_counts)
            classification = {
                "kind": source_kind,
                "reason": "captions_json_caption_entries_majority",
                "method": config_service.NL_TAG_MIX_CLASSIFICATION_METHOD,
                "metrics": {
                    "caption_count": len(caption_entries),
                    "tag_caption_count": entry_counts["tag"],
                    "nl_caption_count": entry_counts["nl"],
                },
            }
        else:
            caption_text = "\n".join(caption_texts)
            classification = _classify_nl_tag_caption_text(caption_text)
            source_kind = classification["kind"]
        samples.append({
            "image_path": image_path,
            "caption_path": caption_source.path,
            "caption_source": caption_source,
            "caption_source_mode": caption_source.detected_mode,
            "caption_texts": caption_texts,
            "caption_entries": caption_entries,
            "source": source_kind,
            "classification": classification,
        })
    return samples


def _nl_tag_mix_caption_entries(caption_texts: list[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, text in enumerate(caption_texts):
        clean_text = str(text or "").strip()
        if not clean_text:
            continue
        classification = _classify_nl_tag_caption_text(clean_text)
        entries.append({
            "index": index,
            "text": clean_text,
            "source": classification["kind"],
            "classification": classification,
        })
    return entries


def _nl_tag_mix_source_counts(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"tag": 0, "nl": 0}
    for entry in entries:
        source = str(entry.get("source") or "")
        if source in counts:
            counts[source] += 1
    return counts


def _nl_tag_mix_dominant_source(counts: dict[str, int]) -> str:
    return "nl" if int(counts.get("nl") or 0) > int(counts.get("tag") or 0) else "tag"


def _cycle_nl_tag_entries(entries: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count <= 0 or not entries:
        return []
    return [entries[index % len(entries)] for index in range(count)]


def _select_nl_tag_caption_entries(
    entries: list[dict[str, Any]],
    *,
    tag_ratio: float,
) -> list[dict[str, Any]]:
    if not entries:
        return []
    ratio = min(1.0, max(0.0, tag_ratio))
    total = len(entries)
    tag_entries = [entry for entry in entries if entry["source"] == "tag"]
    nl_entries = [entry for entry in entries if entry["source"] == "nl"]
    tag_quota = int(round(total * ratio))
    nl_quota = total - tag_quota
    selected = [
        *_cycle_nl_tag_entries(tag_entries, tag_quota),
        *_cycle_nl_tag_entries(nl_entries, nl_quota),
    ]
    if len(selected) < total:
        fallback = tag_entries + nl_entries or entries
        selected.extend(_cycle_nl_tag_entries(fallback, total - len(selected)))
    return selected[:total]


def _nl_tag_mix_relative_image_path(image_path: Path, source_dir: Path) -> Path:
    try:
        return image_path.resolve().relative_to(source_dir.resolve())
    except ValueError:
        return Path(image_path.name)


def _select_nl_tag_mix_samples(samples: list[dict[str, Any]], *, tag_ratio: float) -> list[dict[str, Any]]:
    ratio = min(1.0, max(0.0, tag_ratio))
    tag_samples = [sample for sample in samples if sample["source"] == "tag"]
    nl_samples = [sample for sample in samples if sample["source"] == "nl"]
    tag_quota = int(round(len(samples) * ratio))
    nl_quota = len(samples) - tag_quota
    selected = [*tag_samples[:tag_quota], *nl_samples[:nl_quota]]
    if len(selected) < len(samples):
        selected_ids = {id(sample) for sample in selected}
        fill = [sample for sample in samples if id(sample) not in selected_ids]
        selected.extend(fill[:len(samples) - len(selected)])
    return sorted(selected, key=lambda sample: sample["image_path"].name)


def _copy_nl_tag_caption_sidecars(
    image_path: Path,
    target_image: Path,
    target_dir: Path,
    caption_source=None,
    *,
    captions_json_path: Path | None = None,
) -> list[str]:
    if getattr(caption_source, "from_captions_json", False):
        return [_display_settings_path(captions_json_path)] if captions_json_path is not None else []
    copied: list[str] = []
    copied_sources: set[Path] = set()
    if getattr(caption_source, "path", None) is not None:
        source_path = caption_source.path
        if source_path.is_file():
            target = target_image.with_suffix(source_path.suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)
            copied.append(_display_settings_path(target))
            copied_sources.add(source_path.resolve())
    for ext in DATASET_CAPTION_EXTS:
        source = image_path.with_suffix(ext)
        if not source.is_file() or source.resolve() in copied_sources:
            continue
        target = target_image.with_suffix(ext)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(_display_settings_path(target))
    return copied


def _copy_runtime_dataset_dir(source: str, target: Path) -> None:
    source_path = _resolve_display_path(source)
    if source_path is None or not _path_exists(source_path) or not source_path.is_dir():
        return
    if source_path.resolve() == target.resolve():
        return
    shutil.copytree(source_path, target, dirs_exist_ok=True)


def _is_materialized_runtime_source_dir(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return path.name in {"source", "trigger-clone-source"} and "dataset_cache" in parts


def _caption_fingerprint_settings(
    cfg: dict[str, Any],
    settings: dict[str, Any],
    preprocess_settings: dict[str, Any],
    *,
    environ=None,
) -> tuple[dict[str, Any], str, str]:
    from library.preprocess.caption_cache_settings import resolve_caption_cache_settings

    environ = os.environ if environ is None else environ
    signature_settings = dict(preprocess_settings)
    caption_mode_value = environ.get("CAPTION_SOURCE_MODE")
    if caption_mode_value is None:
        caption_mode_value = signature_settings.get(
            "caption_source_mode", settings.get("caption_source_mode")
        )
    prefer_json_value = environ.get("CAPTION_PREFER_JSON")
    if prefer_json_value is None:
        prefer_json_value = signature_settings.get(
            "prefer_json_caption", settings.get("prefer_json_caption")
        )
    caption_mode = normalize_caption_source_mode(
        caption_mode_value,
        _bool_value_for_row(prefer_json_value),
    )
    caption_extension = str(
        signature_settings.get("caption_extension")
        or settings.get("caption_extension")
        or ".txt"
    )
    caption_variants, caption_tag_dropout_rate = resolve_caption_cache_settings(
        {**cfg, **signature_settings},
        environ,
    )
    signature_settings.update(
        {
            "caption_source_mode": caption_mode,
            "caption_extension": caption_extension,
            "caption_shuffle_variants": caption_variants,
            "caption_tag_dropout_rate": caption_tag_dropout_rate,
        }
    )
    return signature_settings, caption_mode, caption_extension


def _bind_subset_to_cache_pool(
    *,
    cfg: dict[str, Any],
    row: dict[str, Any],
    group_dir: Path,
    pool_root: Path,
    run_id: str,
    source_dir: str,
    resized_name: str = "resized",
    lora_name: str = "lora",
) -> dict[str, Any]:
    """Compute fingerprint and mount/copy shared pool into run dataset_cache.

    Returns binding metadata for run.meta.json and the display paths to use.
    """
    from library.cache_pool.fingerprint import (
        build_preprocess_signature,
        compute_fingerprint,
        scan_input_inventory,
    )
    from library.cache_pool.mount import mount_dir
    from library.cache_pool.policy import parse_cache_reuse_policy
    from library.cache_pool.refs import acquire_ref
    from library.cache_pool.store import (
        pool_entry_dir,
        publish_pool_entry,
        read_manifest,
        write_manifest,
    )
    policy = parse_cache_reuse_policy(cfg)
    source_path = _resolve_display_path(source_dir)
    resized_dst = group_dir / resized_name
    lora_dst = group_dir / lora_name

    if source_path is None or not source_path.is_dir():
        resized_dst.mkdir(parents=True, exist_ok=True)
        lora_dst.mkdir(parents=True, exist_ok=True)
        return {
            "fingerprint": "",
            "pool_path": "",
            "link_mode": "private",
            "reuse_flags": {
                "A": policy.reuse_dataset_cache_copy,
                "B": policy.reuse_vae_latents,
                "C": policy.reuse_text_encoder_cache,
            },
            "fingerprint_mode": policy.fingerprint_mode,
            "image_dir": _display_logical_path(resized_dst),
            "cache_dir": _display_logical_path(lora_dst),
        }

    settings = row.get("settings") if isinstance(row.get("settings"), dict) else {}
    preprocess_settings = (
        settings.get("preprocess")
        if isinstance(settings.get("preprocess"), dict)
        else settings
    )
    signature_settings, caption_mode, caption_extension = (
        _caption_fingerprint_settings(
            cfg,
            settings,
            preprocess_settings if isinstance(preprocess_settings, dict) else {},
        )
    )
    sig = build_preprocess_signature(cfg, signature_settings)
    inv = scan_input_inventory(
        source_path,
        recursive=_bool_value_for_row(row.get("recursive"), True),
        path_pattern=_normalize_path_pattern(row.get("path_pattern")),
        caption_mode=caption_mode,
        caption_extension=caption_extension,
    )
    fp = compute_fingerprint(
        mode=policy.fingerprint_mode,
        source_dir=source_path,
        inventory=inv,
        preprocess_signature=sig,
        normalized_source=str(source_path.resolve()),
    )
    entry = pool_entry_dir(pool_root, fp)
    link_mode = "copy"

    if policy.force_rebuild:
        if resized_dst.exists() or resized_dst.is_symlink():
            if resized_dst.is_symlink() or resized_dst.is_file():
                resized_dst.unlink()
            else:
                shutil.rmtree(resized_dst)
        if lora_dst.exists() or lora_dst.is_symlink():
            if lora_dst.is_symlink() or lora_dst.is_file():
                lora_dst.unlink()
            else:
                shutil.rmtree(lora_dst)
        resized_dst.mkdir(parents=True, exist_ok=True)
        lora_dst.mkdir(parents=True, exist_ok=True)
        link_mode = "private"
    else:
        manifest = read_manifest(entry)
        if manifest is None:
            staging = pool_root / f".staging-{fp}-{run_id}"
            if staging.exists():
                shutil.rmtree(staging)
            (staging / "resized").mkdir(parents=True)
            (staging / "lora").mkdir(parents=True)
            write_manifest(
                staging,
                {
                    "schema_version": "1",
                    "fingerprint": fp,
                    "mode": policy.fingerprint_mode,
                    "preprocess_signature": sig,
                },
            )
            entry = publish_pool_entry(
                pool_root,
                fp,
                staging_dir=staging,
                manifest=read_manifest(staging) or {
                    "schema_version": "1",
                    "fingerprint": fp,
                    "mode": policy.fingerprint_mode,
                },
            )
        if policy.reuse_dataset_cache_copy:
            link_mode = mount_dir(entry / "resized", resized_dst)
            mount_dir(entry / "lora", lora_dst)
        else:
            if resized_dst.exists() or resized_dst.is_symlink():
                if resized_dst.is_symlink() or resized_dst.is_file():
                    resized_dst.unlink()
                else:
                    shutil.rmtree(resized_dst)
            if lora_dst.exists() or lora_dst.is_symlink():
                if lora_dst.is_symlink() or lora_dst.is_file():
                    lora_dst.unlink()
                else:
                    shutil.rmtree(lora_dst)
            resized_dst.mkdir(parents=True, exist_ok=True)
            lora_dst.mkdir(parents=True, exist_ok=True)
            if (entry / "resized").exists():
                _copy_runtime_dataset_dir(str(entry / "resized"), resized_dst)
            if (entry / "lora").exists():
                _copy_runtime_dataset_dir(str(entry / "lora"), lora_dst)
            link_mode = "copy"
        acquire_ref(entry, run_id)

    return {
        "fingerprint": fp,
        "pool_path": _display_logical_path(entry),
        "link_mode": link_mode,
        "reuse_flags": {
            "A": policy.reuse_dataset_cache_copy,
            "B": policy.reuse_vae_latents,
            "C": policy.reuse_text_encoder_cache,
        },
        "fingerprint_mode": policy.fingerprint_mode,
        "image_dir": _display_logical_path(resized_dst),
        "cache_dir": _display_logical_path(lora_dst),
    }
