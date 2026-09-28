"""Read-only cache completeness audit matching training dataset semantics."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

from PIL import Image

from library.datasets.mask_mode import (
    LEGACY_AUTO_MASK_DIRS,
    MASK_MODE_AUTO,
    MASK_MODE_EXTERNAL,
    normalize_mask_mode,
)
from library.io.cache_names import pe_cache_suffix
from library.datasets.buckets import BucketManager
from library.datasets.qwen_image_geometry import align_qwen_bucket_manager, align_qwen_resolution
from library.datasets.qwen_image_edit import (
    edit_cache_suffix, edit_condition_fingerprint, inspect_edit_pairs,
)
from library.datasets.dreambooth import read_caption
from library.models.family_registry import (
    get_model_family_spec,
    normalize_registered_family,
)


def audit_dataset_row_caches(
    cfg: dict[str, Any],
    row: dict[str, Any],
    images: Iterable[Path],
    *,
    resolve_path: Callable[[str], Path],
) -> list[dict[str, Any]]:
    """Audit every runtime image against the cache modes training will use."""

    images = list(images)
    image_dir = resolve_path(str(row.get("image_dir") or row.get("source_dir") or ""))
    cache_dir = _optional_path(row.get("cache_dir"), resolve_path)
    text_cache_dir = _optional_path(row.get("text_cache_dir"), resolve_path) or cache_dir
    cond_cache_dir = _optional_path(row.get("cond_cache_dir"), resolve_path)
    family = normalize_registered_family(
        cfg.get("model_family") or "anima",
        allow_aliases=True,
    )
    family_spec = get_model_family_spec(family)
    skip_check = _bool_value(cfg.get("skip_cache_check"), False)
    mask_config = _resolved_mask_config(row, resolve_path)

    results: list[dict[str, Any]] = []
    if _cache_flag(cfg, "use_vae_cache", "cache_latents_to_disk"):
        latent_strategy = _latent_strategy(family, skip_check)
        results.append(
            _audit_files(
                "latent_cache",
                "VAE latent 缓存",
                images,
                cache_dir or image_dir,
                lambda image: _validate_latent(
                    image,
                    image_dir=image_dir,
                    cache_dir=cache_dir,
                    suffix=family_spec.latent_space.cache_suffix,
                    strategy=latent_strategy,
                    flip_aug=_bool_value(row.get("flip_aug"), False),
                    embedded_alpha=mask_config.mode == "embedded",
                    bucket_reso=_bucket_reso(image, row, family),
                ),
            )
        )

    if cond_cache_dir is not None:
        latent_strategy = _latent_strategy(family, skip_check)
        results.append(
            _audit_files(
                "condition_latent_cache",
                "条件 latent 缓存",
                images,
                cond_cache_dir,
                lambda image: _validate_condition_latent(
                    image,
                    image_dir=image_dir,
                    cache_dir=cond_cache_dir,
                    suffix=family_spec.latent_space.cache_suffix,
                    strategy=latent_strategy,
                    flip_aug=_bool_value(row.get("flip_aug"), False),
                    bucket_reso=_bucket_reso(image, row, family),
                ),
            )
        )

    if _cache_flag(
        cfg,
        "use_text_cache",
        "cache_text_encoder_outputs_to_disk",
    ):
        text_strategy = _text_strategy(family, skip_check)
        if _is_qwen_edit(cfg, family):
            from library.models.qwen_image_2_1.strategy import QwenImage21EditTextCache

            text_strategy = QwenImage21EditTextCache(True, 1, skip_check)
            edit_infos, _reference_root = _edit_infos(cfg, row, images, image_dir, resolve_path)
            results.append(
                _audit_files(
                    "text_cache",
                    "文本编码器缓存",
                    images,
                    text_cache_dir or image_dir,
                    lambda image: _validate_edit_text(
                        image, edit_infos.get(image), text_cache_dir, image_dir, text_strategy
                    ),
                )
            )
            latent_strategy = _latent_strategy(family, skip_check)
            results.append(
                _audit_files(
                    "edit_reference_latent_cache",
                    "Edit 参考图 latent 缓存",
                    images,
                    cache_dir or image_dir,
                    lambda image: _validate_edit_reference(
                        image, edit_infos.get(image), cache_dir, image_dir, latent_strategy
                    ),
                )
            )
        else:
            results.append(
                _audit_files(
                    "text_cache",
                    "文本编码器缓存",
                    images,
                    text_cache_dir or image_dir,
                    lambda image: _validate_text(
                        image,
                        image_dir=image_dir,
                        cache_dir=text_cache_dir,
                        suffix=family_spec.text_cache.suffix,
                        strategy=text_strategy,
                    ),
                )
            )

    if _bool_value(cfg.get("ip_features_cache_to_disk"), False):
        encoder = str(cfg.get("ip_encoder") or "pe").strip() or "pe"
        results.append(
            _audit_files(
                "pe_cache",
                f"{encoder} 图像特征缓存",
                images,
                cache_dir or image_dir,
                lambda image: _validate_pe(
                    image,
                    image_dir=image_dir,
                    cache_dir=cache_dir,
                    suffix=pe_cache_suffix(encoder),
                ),
            )
        )

    if mask_config.mode == MASK_MODE_EXTERNAL:
        mask_dir = Path(mask_config.mask_dir or "")
        results.append(
            _audit_files(
                "external_masks",
                "外部遮罩",
                images,
                mask_dir,
                lambda image: _validate_external_mask(
                    image,
                    image_dir=image_dir,
                    mask_dir=mask_dir,
                ),
                missing_level="warning",
            )
        )
    return results


def _cache_flag(cfg: dict[str, Any], primary: str, legacy: str) -> bool:
    key = primary if primary in cfg else legacy
    return _bool_value(cfg.get(key), False)


def _optional_path(value: Any, resolve_path: Callable[[str], Path]) -> Path | None:
    text = str(value or "").strip()
    return resolve_path(text) if text else None


def _resolved_mask_config(
    row: dict[str, Any], resolve_path: Callable[[str], Path]
):
    raw_dir = str(row.get("mask_dir") or "").strip() or None
    config = normalize_mask_mode(
        row.get("mask_mode"),
        alpha_mask=_bool_value(row.get("alpha_mask"), False),
        mask_dir=raw_dir,
    )
    if config.mode == MASK_MODE_AUTO and config.mask_dir is None:
        default_dir = next(
            (
                resolve_path(candidate)
                for candidate in LEGACY_AUTO_MASK_DIRS
                if resolve_path(candidate).is_dir()
            ),
            None,
        )
        if default_dir is not None:
            config = normalize_mask_mode(
                MASK_MODE_EXTERNAL,
                alpha_mask=True,
                mask_dir=os.fspath(default_dir),
            )
    elif config.mask_dir:
        config = normalize_mask_mode(
            config.mode,
            alpha_mask=config.alpha_mask,
            mask_dir=os.fspath(resolve_path(config.mask_dir)),
        )
    return config


def _latent_strategy(family: str, skip_check: bool):
    if family in {"anima", "krea2_raw"}:
        from library.anima.strategy import AnimaLatentsCachingStrategy

        cls = AnimaLatentsCachingStrategy
    elif family == "z_image":
        from library.models.z_image.strategy import ZImageLatentsCachingStrategy

        cls = ZImageLatentsCachingStrategy
    elif family == "qwen_image_2_1":
        from library.models.qwen_image_2_1.strategy import QwenImage21LatentCache

        cls = QwenImage21LatentCache
    else:
        raise ValueError(f"Unsupported latent cache audit family: {family}")
    return cls(True, 1, skip_check)


def _text_strategy(family: str, skip_check: bool):
    if family == "anima":
        from library.anima.strategy import AnimaTextEncoderOutputsCachingStrategy

        return AnimaTextEncoderOutputsCachingStrategy(True, 1, skip_check)
    if family == "krea2_raw":
        from library.models.krea2_raw.strategy import Krea2TextEncoderOutputsCachingStrategy

        return Krea2TextEncoderOutputsCachingStrategy(True, 1, skip_check)
    if family == "z_image":
        from library.models.z_image.strategy import ZImageTextEncoderOutputsCachingStrategy

        return ZImageTextEncoderOutputsCachingStrategy(True, 1, skip_check)
    if family == "qwen_image_2_1":
        from library.models.qwen_image_2_1.strategy import QwenImage21TextCache

        return QwenImage21TextCache(True, 1, skip_check)
    raise ValueError(f"Unsupported cache audit family: {family}")


def _is_qwen_edit(cfg: dict[str, Any], family: str) -> bool:
    return family == "qwen_image_2_1" and str(
        cfg.get("qwen_image_2_1_task") or cfg.get("task") or "t2i"
    ).strip().lower() == "edit"


def _edit_infos(
    cfg: dict[str, Any],
    row: dict[str, Any],
    images: list[Path],
    image_dir: Path,
    resolve_path: Callable[[str], Path],
) -> tuple[dict[Path, Any], Path]:
    reference_root = resolve_path(str(row.get("reference_image_dir") or ""))
    report = inspect_edit_pairs(
        [os.fspath(path) for path in images],
        target_dir=os.fspath(image_dir),
        reference_dir=os.fspath(reference_root),
        recursive=_bool_value(row.get("recursive"), True),
    )
    settings = row.get("settings") if isinstance(row.get("settings"), dict) else {}
    extension = str(settings.get("caption_extension") or cfg.get("caption_extension") or ".txt")
    if not extension.startswith("."):
        extension = f".{extension}"
    infos: dict[Path, Any] = {}
    for image in images:
        reference = report.pairs.get(os.fspath(image))
        caption = read_caption(
            os.fspath(image), extension,
            _bool_value(settings.get("enable_wildcard", cfg.get("enable_wildcard")), False),
        )
        if caption is None:
            caption = row.get("class_tokens") or ""
        infos[image] = SimpleNamespace(
            absolute_path=os.fspath(image),
            reference_image_path=reference,
            caption=caption,
            bucket_reso=_bucket_reso(image, row, cfg.get("model_family") or "anima"),
        )
    return infos, reference_root


def _audit_files(
    kind: str,
    label: str,
    images: list[Path],
    root: Path,
    validator,
    *,
    missing_level: str = "error",
) -> dict[str, Any]:
    missing: list[str] = []
    invalid: list[str] = []
    for image in images:
        status, path = validator(image)
        if status == "missing":
            missing.append(path)
        elif status == "invalid":
            invalid.append(path)
    return {
        "kind": kind,
        "label": label,
        "root": root,
        "total": len(images),
        "valid": len(images) - len(missing) - len(invalid),
        "missing": missing,
        "invalid": invalid,
        "missing_level": missing_level,
    }


def _validate_latent(
    image: Path,
    *,
    image_dir: Path,
    cache_dir: Path | None,
    suffix: str,
    strategy,
    flip_aug: bool,
    embedded_alpha: bool,
    bucket_reso: tuple[int, int],
) -> tuple[str, str]:
    try:
        with Image.open(image) as opened:
            size = opened.size
    except Exception:
        return "invalid", os.fspath(image)
    resolution_suffix = f"_{size[0]:04d}x{size[1]:04d}{suffix}"
    path = _mirrored_cache_path(image, resolution_suffix, cache_dir, image_dir)
    if not path.is_file():
        return "missing", os.fspath(path)
    try:
        valid = strategy.is_disk_cached_latents_expected(
            bucket_reso,
            os.fspath(path),
            flip_aug,
            embedded_alpha,
        )
    except Exception:
        valid = False
    return ("valid" if valid else "invalid"), os.fspath(path)


def _validate_condition_latent(
    image: Path,
    *,
    image_dir: Path,
    cache_dir: Path,
    suffix: str,
    strategy,
    flip_aug: bool,
    bucket_reso: tuple[int, int],
) -> tuple[str, str]:
    resolution_suffix = (
        f"_{bucket_reso[0]:04d}x{bucket_reso[1]:04d}{suffix}"
    )
    path = _mirrored_cache_path(image, resolution_suffix, cache_dir, image_dir)
    if not path.is_file():
        return "missing", os.fspath(path)
    try:
        valid = strategy.is_disk_cached_latents_expected(
            bucket_reso,
            os.fspath(path),
            flip_aug,
            False,
        )
    except Exception:
        valid = False
    return ("valid" if valid else "invalid"), os.fspath(path)


def _bucket_reso(
    image: Path, row: dict[str, Any], model_family: str = "anima"
) -> tuple[int, int]:
    """Mirror DatasetBucketsMixin for raw image dirs when preprocess settings exist."""

    with Image.open(image) as opened:
        width, height = opened.size
    qwen = normalize_registered_family(model_family, allow_aliases=True) == "qwen_image_2_1"
    settings = row.get("settings") if isinstance(row.get("settings"), dict) else {}
    if not settings:
        return align_qwen_resolution((width, height)) if qwen else (width, height)
    resolution = _positive_int(settings.get("resolution"), max(width, height))
    if not _bool_value(settings.get("enable_bucket"), True):
        return align_qwen_resolution((resolution, resolution)) if qwen else (resolution, resolution)
    if _bool_value(settings.get("bucket_no_upscale"), False):
        return align_qwen_resolution((width, height)) if qwen else (width, height)
    min_size = _positive_int(settings.get("min_bucket_reso"), 256)
    max_size = _positive_int(settings.get("max_bucket_reso"), max(resolution, 2048))
    max_size = max(max_size, resolution)
    steps = _positive_int(settings.get("bucket_reso_steps"), 64)
    manager = BucketManager(
        max_reso=(resolution, resolution),
        min_size=min_size,
        max_size=max_size,
        reso_steps=steps,
    )
    manager.make_buckets(constant_token_buckets=True)
    if qwen:
        align_qwen_bucket_manager(manager)
    bucket, _resized, _error = manager.select_bucket(width, height)
    return bucket


def _positive_int(value: Any, fallback: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def _validate_text(
    image: Path,
    *,
    image_dir: Path,
    cache_dir: Path | None,
    suffix: str,
    strategy,
) -> tuple[str, str]:
    path = _mirrored_cache_path(image, suffix, cache_dir, image_dir)
    if not path.is_file():
        return "missing", os.fspath(path)
    try:
        valid = strategy.is_disk_cached_outputs_expected(os.fspath(path))
    except Exception:
        valid = False
    return ("valid" if valid else "invalid"), os.fspath(path)


def _validate_edit_text(
    image: Path,
    info: Any,
    cache_dir: Path | None,
    image_dir: Path,
    strategy: Any,
) -> tuple[str, str]:
    if info is None or not info.reference_image_path:
        return "invalid", os.fspath(image)
    try:
        path = _edit_cache_path(image, info, "te", cache_dir, image_dir)
        if not Path(path).is_file():
            return "missing", os.fspath(path)
        valid = strategy.is_expected_for_info(path, info)
    except Exception:
        path = os.fspath(image)
        valid = False
    return ("valid" if valid else "invalid"), os.fspath(path)


def _validate_edit_reference(
    image: Path,
    info: Any,
    cache_dir: Path | None,
    image_dir: Path,
    strategy: Any,
) -> tuple[str, str]:
    if info is None or not info.reference_image_path:
        return "invalid", os.fspath(image)
    subset = SimpleNamespace(
        cache_dir=os.fspath(cache_dir) if cache_dir else None,
        image_dir=os.fspath(image_dir),
    )
    try:
        path = _edit_cache_path(image, info, "ref", cache_dir, image_dir)
        if not Path(path).is_file():
            return "missing", os.fspath(path)
        valid = strategy.is_edit_reference_cache_expected(info, subset)
    except Exception:
        path = os.fspath(image)
        valid = False
    return ("valid" if valid else "invalid"), os.fspath(path)


def _edit_cache_path(
    image: Path, info: Any, task: str, cache_dir: Path | None, image_dir: Path,
) -> Path:
    # Training's path resolver creates directories; a preflight must stay read-only.
    fingerprint = edit_condition_fingerprint(
        info.reference_image_path, info.caption, info.bucket_reso,
    )
    suffix = edit_cache_suffix(
        task=task, fingerprint=fingerprint, target_bucket=info.bucket_reso,
    )
    return _mirrored_cache_path(image, suffix, cache_dir, image_dir)


def _validate_pe(
    image: Path,
    *,
    image_dir: Path,
    cache_dir: Path | None,
    suffix: str,
) -> tuple[str, str]:
    expected = _mirrored_cache_path(image, suffix, cache_dir, image_dir)
    candidates = [expected]
    if cache_dir is not None:
        flat = cache_dir / f"{image.stem}{suffix}"
        if flat != expected:
            candidates.append(flat)
    source_sidecar = image.with_name(f"{image.stem}{suffix}")
    if source_sidecar not in candidates:
        candidates.append(source_sidecar)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return "missing", os.fspath(expected)
    try:
        from safetensors import safe_open

        with safe_open(os.fspath(path), framework="pt") as handle:
            valid = "image_features" in set(handle.keys())
    except Exception:
        valid = False
    return ("valid" if valid else "invalid"), os.fspath(path)


def _validate_external_mask(
    image: Path,
    *,
    image_dir: Path,
    mask_dir: Path,
) -> tuple[str, str]:
    expected = _mirrored_cache_path(image, "_mask.png", mask_dir, image_dir)
    candidates = [expected]
    flat = mask_dir / f"{image.stem}_mask.png"
    if flat != expected:
        candidates.append(flat)
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    return ("valid", os.fspath(path)) if path else ("missing", os.fspath(expected))


def _mirrored_cache_path(
    image: Path,
    suffix: str,
    cache_dir: Path | None,
    image_dir: Path,
) -> Path:
    if cache_dir is None:
        return image.with_name(f"{image.stem}{suffix}")
    try:
        rel_parent = image.parent.relative_to(image_dir)
    except ValueError:
        rel_parent = Path()
    return cache_dir / rel_parent / f"{image.stem}{suffix}"


def _bool_value(value: Any, fallback: bool) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
