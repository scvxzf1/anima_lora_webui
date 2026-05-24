"""Default-dataset preprocessing: resize → VAE latents → text-embedding caches."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    tomllib = None
    import toml as _toml  # type: ignore[no-redef]
else:
    _toml = None

from ._common import PY, ROOT, _path, run


# Subfolders under the source dir are walked by default — matches the
# `recursive = true` subset default in configs/base.toml. Stems must stay
# unique across the tree (cache filenames are stem-keyed and flat). Pass
# `--no_recursive` (or edit configs) to opt out.
def _min_pixels_args() -> list[str]:
    """``--min_pixels <N>`` derived from the variant TOML's
    ``drop_lowres_images`` + ``min_pixels`` keys (resolved through the same
    base → preset → method merge chain training uses, via ``_path_overrides``
    in scripts/tasks/_common.py).

    Returns ``[]`` when both keys are absent so plain CLI use keeps each
    script's own argparse default (500_000 = 0.5MP). ``drop_lowres_images
    = false`` forces ``--min_pixels 0`` even when ``min_pixels`` is set, so
    the user can flip a single boolean to disable the filter."""
    from ._common import _path_overrides  # local import: avoids unused circular

    overrides = _path_overrides()
    if "drop_lowres_images" not in overrides and "min_pixels" not in overrides:
        return []
    if overrides.get("drop_lowres_images") is False:
        return ["--min_pixels", "0"]
    raw = overrides.get("min_pixels", 500_000)
    try:
        n = max(0, int(raw))
    except (TypeError, ValueError):
        return []
    return ["--min_pixels", str(n)]


def _config_min_pixels() -> int:
    """The configured ``min_pixels`` threshold, default 0.5MP."""
    from ._common import _path_overrides

    raw = _path_overrides().get("min_pixels", 500_000)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 500_000


def _resolve_lowres_filter(extra: list[str]) -> tuple[list[str], list[str]]:
    """Reconcile low-res filtering convenience flags with forwarded args.

    ``--min_pixels`` is authoritative when passed explicitly. ``--no_drop_lowres``
    keeps every image; ``--drop_lowres`` forces the configured threshold.
    """
    cleaned = list(extra)
    no_drop = "--no_drop_lowres" in cleaned
    drop = "--drop_lowres" in cleaned
    cleaned = [arg for arg in cleaned if arg not in ("--no_drop_lowres", "--drop_lowres")]

    if "--min_pixels" in cleaned:
        return [], cleaned
    if no_drop:
        return ["--min_pixels", "0"], cleaned
    if drop:
        return ["--min_pixels", str(_config_min_pixels())], cleaned
    return _min_pixels_args(), cleaned


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _positive_int(value: Any) -> int | None:
    if isinstance(value, (list, tuple)):
        nums = [_positive_int(item) for item in value]
        nums = [item for item in nums if item is not None]
        return max(nums) if nums else None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _resolve_project_path(value: Any) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def _load_toml_file(path: Path) -> dict[str, Any]:
    if tomllib is not None:
        with path.open("rb") as f:
            return tomllib.load(f)
    return _toml.load(path)


def _dataset_path_value(value: Any, overrides: dict[str, Any]) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return text.format_map({key: str(val) for key, val in overrides.items()})
    except (KeyError, ValueError, IndexError):
        return text


def _same_path_text(left: Any, right: Any) -> bool:
    def normalize(value: Any) -> str:
        return str(value or "").replace("\\", "/").rstrip("/")

    return bool(normalize(left)) and normalize(left) == normalize(right)


def _has_extra_path_override(extra: list[str], *flags: str) -> bool:
    prefixes = tuple(f"{flag}=" for flag in flags)
    return any(arg in flags or arg.startswith(prefixes) for arg in extra)


def _dataset_rows(dataset_config: Any, overrides: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    overrides = overrides or {}
    path = _resolve_project_path(dataset_config)
    if path is None or not path.is_file():
        return []
    try:
        data = _load_toml_file(path)
    except Exception as e:  # noqa: BLE001
        print(f"warn: could not read dataset_config for preprocess rows: {e}")
        return []
    datasets = data.get("datasets") or []
    if not isinstance(datasets, list):
        return []

    rows: list[dict[str, Any]] = []
    fallback_source = str(overrides.get("source_image_dir") or "")
    fallback_image = str(overrides.get("resized_image_dir") or fallback_source)
    fallback_cache = str(overrides.get("lora_cache_dir") or "")
    for dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        subsets = dataset.get("subsets") or []
        if not isinstance(subsets, list):
            continue
        for subset in subsets:
            if not isinstance(subset, dict):
                continue
            attrs = subset.get("custom_attributes")
            if not isinstance(attrs, dict):
                attrs = {}
            image_dir = _dataset_path_value(subset.get("image_dir") or fallback_image, overrides)
            cache_dir = _dataset_path_value(subset.get("cache_dir") or fallback_cache, overrides)
            if attrs.get("source_dir"):
                source_dir = _dataset_path_value(attrs.get("source_dir"), overrides)
            elif image_dir and _same_path_text(image_dir, fallback_image):
                source_dir = _dataset_path_value(fallback_source or image_dir, overrides)
            else:
                source_dir = image_dir or _dataset_path_value(fallback_source, overrides)
            if not image_dir and not source_dir:
                continue
            row = dict(dataset)
            row.pop("subsets", None)
            row.update(
                {
                    "source_image_dir": source_dir,
                    "resized_image_dir": image_dir,
                    "lora_cache_dir": cache_dir,
                    "recursive": subset.get("recursive", dataset.get("recursive", True)),
                }
            )
            rows.append(row)
    return rows


def _first_dataset_settings(dataset_config: Any) -> dict[str, Any]:
    path = _resolve_project_path(dataset_config)
    if path is None or not path.is_file():
        return {}
    try:
        data = _load_toml_file(path)
    except Exception as e:  # noqa: BLE001
        print(f"warn: could not read dataset_config for resize settings: {e}")
        return {}
    datasets = data.get("datasets") or []
    first = datasets[0] if datasets and isinstance(datasets[0], dict) else {}
    if not isinstance(first, dict):
        return {}
    first = dict(first)
    first.pop("subsets", None)
    return first


def _preprocess_rows() -> list[dict[str, Any]]:
    """Return one preprocess row per dataset_config subset, or legacy top-level paths."""
    from ._common import _path_overrides  # local import: avoids unused circular

    overrides = _path_overrides()
    rows = _dataset_rows(overrides.get("dataset_config"), overrides)
    if rows:
        return rows
    return [
        {
            **dict(overrides),
            "source_image_dir": _path("source_image_dir", "image_dataset"),
            "resized_image_dir": _path("resized_image_dir", "post_image_dataset/resized"),
            "lora_cache_dir": _path("lora_cache_dir", "post_image_dataset/lora"),
            "recursive": True,
        }
    ]


def _resize_bucket_args(settings: dict[str, Any] | None = None) -> list[str]:
    """Forward dataset bucket settings to resize_images.py.

    WebUI stores these in dataset_config, while tasks.py preprocess historically
    only read top-level path keys. Callers pass one dataset row at a time so
    multi-dataset WebUI presets get matching resized images and caches.
    """
    from ._common import _path_overrides  # local import: avoids unused circular

    overrides = _path_overrides()
    if settings is None:
        settings = dict(overrides)
        rows = _preprocess_rows()
        if rows:
            settings.update(rows[0])
        settings.update(_first_dataset_settings(overrides.get("dataset_config")))
    else:
        merged = dict(overrides)
        merged.update(settings)
        settings = merged

    out: list[str] = []
    mapping = (
        ("resolution", "--resolution"),
        ("min_bucket_reso", "--min_bucket_reso"),
        ("max_bucket_reso", "--max_bucket_reso"),
        ("bucket_reso_steps", "--bucket_reso_steps"),
    )
    for key, flag in mapping:
        n = _positive_int(settings.get(key))
        if n is not None:
            out.extend([flag, str(n)])
    if _truthy(settings.get("bucket_no_upscale")):
        out.append("--bucket_no_upscale")
    return out


def _recursive_args(row: dict[str, Any]) -> list[str]:
    return ["--recursive"] if _truthy(row.get("recursive", True)) else []


def _run_preprocess_resize(row: dict[str, Any], extra: list[str]) -> None:
    mp_args, extra = _resolve_lowres_filter(extra)
    src = str(row.get("source_image_dir") or _path("source_image_dir", "image_dataset"))
    dst = str(row.get("resized_image_dir") or _path("resized_image_dir", "post_image_dataset/resized"))
    if _same_path_text(src, dst) and not _has_extra_path_override(extra, "--src", "--dst"):
        print(f"skip resize: source_image_dir and resized_image_dir are the same: {src}")
        return
    run(
        [
            PY,
            "preprocess/resize_images.py",
            "--src",
            src,
            "--dst",
            dst,
            "--no_copy_captions",
            *_recursive_args(row),
            *_resize_bucket_args(row),
            *mp_args,
            *extra,
        ]
    )


def _run_preprocess_vae(row: dict[str, Any], extra: list[str]) -> None:
    run(
        [
            PY,
            "preprocess/cache_latents.py",
            "--dir",
            str(row.get("resized_image_dir") or _path("resized_image_dir", "post_image_dataset/resized")),
            "--cache_dir",
            str(row.get("lora_cache_dir") or _path("lora_cache_dir", "post_image_dataset/lora")),
            "--vae",
            _path("vae", "models/vae/qwen_image_vae.safetensors"),
            "--batch_size",
            "4",
            "--chunk_size",
            "64",
            *_recursive_args(row),
            *extra,
        ]
    )


def cmd_preprocess_te(extra):
    # CAPTION_SHUFFLE_VARIANTS / CAPTION_TAG_DROPOUT_RATE let the GUI's
    # Preprocessing tab control these without editing this file. Defaults
    # match the historical hardcoded values so non-GUI invocations are
    # unchanged.
    shuffle_variants = os.environ.get("CAPTION_SHUFFLE_VARIANTS", "4")
    tag_dropout_rate = os.environ.get("CAPTION_TAG_DROPOUT_RATE", "0.1")
    for row in _preprocess_rows():
        _run_preprocess_te(row, extra, shuffle_variants, tag_dropout_rate)


def _run_preprocess_te(
    row: dict[str, Any],
    extra: list[str],
    shuffle_variants: str | None = None,
    tag_dropout_rate: str | None = None,
) -> None:
    shuffle_variants = shuffle_variants or os.environ.get("CAPTION_SHUFFLE_VARIANTS", "4")
    tag_dropout_rate = tag_dropout_rate or os.environ.get("CAPTION_TAG_DROPOUT_RATE", "0.1")
    mp_args, extra = _resolve_lowres_filter(extra)
    run(
        [
            PY,
            "preprocess/cache_text_embeddings.py",
            "--dir",
            str(row.get("source_image_dir") or _path("source_image_dir", "image_dataset")),
            "--cache_dir",
            str(row.get("lora_cache_dir") or _path("lora_cache_dir", "post_image_dataset/lora")),
            "--qwen3",
            _path("qwen3", "models/text_encoders/qwen_3_06b_base.safetensors"),
            "--dit",
            _path(
                "pretrained_model_name_or_path",
                "models/diffusion_models/anima-base-v1.0.safetensors",
            ),
            "--caption_shuffle_variants",
            shuffle_variants,
            "--caption_tag_dropout_rate",
            tag_dropout_rate,
            *_recursive_args(row),
            *mp_args,
            *extra,
        ]
    )


def cmd_preprocess_resize(extra):
    for row in _preprocess_rows():
        _run_preprocess_resize(row, extra)


def cmd_preprocess_vae(extra):
    for row in _preprocess_rows():
        _run_preprocess_vae(row, extra)


def cmd_preprocess_pooled(extra):
    """Cache pooled text embeddings (max over seq dim) from existing TE caches.

    Reads ``{stem}_anima_te.safetensors`` from the LoRA cache dir and writes
    ``{stem}_anima_pooled.safetensors`` sidecars next to them. Consumed by
    ``make distill-mod`` to skip a redundant ``.max(dim=1)`` per training
    microstep / val sigma. No GPU needed.
    """
    for row in _preprocess_rows():
        run(
            [
                PY,
                "preprocess/cache_pooled_text.py",
                "--dir",
                str(row.get("lora_cache_dir") or _path("lora_cache_dir", "post_image_dataset/lora")),
                *extra,
            ]
        )


def cmd_preprocess_pe(extra):
    """Cache PE-Core-L14-336 vision-encoder features.

    Reads pre-resized images from ``post_image_dataset/resized/`` (the
    standard LoRA pipeline source) and writes
    ``{stem}_anima_pe.safetensors`` sidecars into the LoRA cache dir so the
    dataset's existing ``cache_dir`` lookup finds them.

    Consumed by IP-Adapter when reading PE features off disk and by the
    DCW v4 fusion head's pooled-image-feature input channel.
    """
    for row in _preprocess_rows():
        run(
            [
                PY,
                "preprocess/cache_pe_encoder.py",
                "--dir",
                str(row.get("resized_image_dir") or _path("resized_image_dir", "post_image_dataset/resized")),
                "--cache_dir",
                str(row.get("lora_cache_dir") or _path("lora_cache_dir", "post_image_dataset/lora")),
                "--encoder",
                "pe",
                *_recursive_args(row),
                *extra,
            ]
        )


def cmd_caption_index(extra):
    """Build the method-agnostic typed-tag caption index."""
    run(
        [
            PY,
            "preprocess/build_caption_index.py",
            "--src",
            _path("source_image_dir", "image_dataset"),
            *extra,
        ]
    )


def cmd_preprocess(extra):
    # PE features are intentionally NOT cached here — only IP-Adapter / CMMD /
    # DCW v4 need them, and those paths chain `preprocess-pe` explicitly (see
    # `exp-ip-adapter-preprocess`). Leaving PE out keeps the default LoRA
    # preprocess fast on machines that won't ever use the vision tower.
    _, vae_extra = _resolve_lowres_filter(extra)
    for row in _preprocess_rows():
        _run_preprocess_resize(row, extra)
        _run_preprocess_vae(row, vae_extra)
        _run_preprocess_te(row, extra)


def cmd_preprocess_config(extra):
    """Preprocess the exact directories named in a ``--dataset_config`` TOML."""
    import time

    import toml

    args = list(extra)
    cfg_path: str | None = None
    src_dir: str | None = None
    vae_path = _path("vae", "models/vae/qwen_image_vae.safetensors")
    qwen3_path = _path("qwen3", "models/text_encoders/qwen_3_06b_base.safetensors")
    dit_path = _path(
        "pretrained_model_name_or_path",
        "models/diffusion_models/anima-base-v1.0.safetensors",
    )
    rest: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--dataset_config" and i + 1 < len(args):
            cfg_path = args[i + 1]
            i += 2
        elif args[i] == "--src" and i + 1 < len(args):
            src_dir = args[i + 1]
            i += 2
        elif args[i] == "--vae" and i + 1 < len(args):
            vae_path = args[i + 1]
            i += 2
        elif args[i] == "--qwen3" and i + 1 < len(args):
            qwen3_path = args[i + 1]
            i += 2
        elif args[i] == "--dit" and i + 1 < len(args):
            dit_path = args[i + 1]
            i += 2
        else:
            rest.append(args[i])
            i += 1
    if not cfg_path or not src_dir:
        raise SystemExit("preprocess-config requires --dataset_config <path> and --src <dir>")

    last_err: OSError | None = None
    for attempt in range(10):
        try:
            cfg = toml.load(cfg_path)
            break
        except PermissionError as e:
            last_err = e
            time.sleep(0.2 * (attempt + 1))
    else:
        raise SystemExit(
            f"could not read {cfg_path} after retrying (last error: {last_err}). "
            "If this persists, exclude the dataset/temp dir from your antivirus."
        )

    subsets = [
        sub
        for ds in (cfg.get("datasets") or [])
        for sub in (ds.get("subsets") or [])
        if sub.get("image_dir")
    ]
    if not subsets:
        raise SystemExit(f"no [[datasets.subsets]] with image_dir in {cfg_path}")

    for sub in subsets:
        image_dir = sub["image_dir"]
        cache_dir = sub.get("cache_dir") or image_dir
        run(
            [
                PY,
                "preprocess/resize_images.py",
                "--src",
                src_dir,
                "--dst",
                image_dir,
                "--no_copy_captions",
                "--min_pixels",
                "0",
                "--bucket_reso_steps",
                "64",
                "--recursive",
                *rest,
            ]
        )
        run(
            [
                PY,
                "preprocess/cache_latents.py",
                "--dir",
                image_dir,
                "--cache_dir",
                cache_dir,
                "--vae",
                vae_path,
                "--batch_size",
                "4",
                "--chunk_size",
                "64",
                "--recursive",
            ]
        )
        run(
            [
                PY,
                "preprocess/cache_text_embeddings.py",
                "--dir",
                src_dir,
                "--cache_dir",
                cache_dir,
                "--qwen3",
                qwen3_path,
                "--dit",
                dit_path,
                "--recursive",
            ]
        )
