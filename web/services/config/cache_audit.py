"""Read-only cache completeness audit matching training dataset semantics."""

from __future__ import annotations

import os
from pathlib import Path
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
                    bucket_reso=_bucket_reso(image, row),
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
                    bucket_reso=_bucket_reso(image, row),
                ),
            )
        )

    if _cache_flag(
        cfg,
        "use_text_cache",
        "cache_text_encoder_outputs_to_disk",
    ):
        text_strategy = _text_strategy(family, skip_check)
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
    raise ValueError(f"Unsupported cache audit family: {family}")


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


def _bucket_reso(image: Path, row: dict[str, Any]) -> tuple[int, int]:
    """Mirror DatasetBucketsMixin for raw image dirs when preprocess settings exist."""

    with Image.open(image) as opened:
        width, height = opened.size
    settings = row.get("settings") if isinstance(row.get("settings"), dict) else {}
    if not settings:
        return width, height
    resolution = _positive_int(settings.get("resolution"), max(width, height))
    if not _bool_value(settings.get("enable_bucket"), True):
        return resolution, resolution
    if _bool_value(settings.get("bucket_no_upscale"), False):
        return width, height
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
