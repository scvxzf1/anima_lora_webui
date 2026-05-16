"""Default-dataset preprocessing: resize → VAE latents → text-embedding caches."""

from __future__ import annotations

import os
from pathlib import Path

import toml

from ._common import PY, ROOT, _path, _path_overrides, run


def _project_path(value: str) -> str:
    text = os.path.expandvars(str(value or ""))
    for key, raw in _path_overrides().items():
        if isinstance(raw, str):
            text = text.replace("{" + key + "}", raw)
    path = Path(text)
    if path.is_absolute():
        return str(path)
    return str(path)


def _dataset_rows():
    overrides = _path_overrides()
    dataset_config = str(overrides.get("dataset_config") or "").strip()
    rows = []
    if dataset_config:
        path = Path(dataset_config)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            data = toml.loads(path.read_text(encoding="utf-8"))
            for dataset in data.get("datasets") or []:
                for subset in dataset.get("subsets") or []:
                    attrs = subset.get("custom_attributes") or {}
                    source = (
                        attrs.get("source_dir")
                        or overrides.get("source_image_dir")
                        or subset.get("image_dir")
                    )
                    rows.append(
                        {
                            "source": _project_path(str(source or "image_dataset")),
                            "resized": _project_path(
                                str(subset.get("image_dir") or "post_image_dataset/resized")
                            ),
                            "cache": _project_path(
                                str(subset.get("cache_dir") or "post_image_dataset/lora")
                            ),
                        }
                    )
    if rows:
        return rows
    return [
        {
            "source": _path("source_image_dir", "image_dataset"),
            "resized": _path("resized_image_dir", "post_image_dataset/resized"),
            "cache": _path("lora_cache_dir", "post_image_dataset/lora"),
        }
    ]


def cmd_preprocess_resize(extra):
    for row in _dataset_rows():
        run(
            [
                PY,
                "preprocess/resize_images.py",
                "--src",
                row["source"],
                "--dst",
                row["resized"],
                "--no_copy_captions",
                *extra,
            ]
        )


def cmd_preprocess_vae(extra):
    for row in _dataset_rows():
        run(
            [
                PY,
                "preprocess/cache_latents.py",
                "--dir",
                row["resized"],
                "--cache_dir",
                row["cache"],
                "--vae",
                "models/vae/qwen_image_vae.safetensors",
                "--batch_size",
                "4",
                "--chunk_size",
                "64",
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
    for row in _dataset_rows():
        run(
            [
                PY,
                "preprocess/cache_text_embeddings.py",
                "--dir",
                row["source"],
                "--cache_dir",
                row["cache"],
                "--qwen3",
                "models/text_encoders/qwen_3_06b_base.safetensors",
                "--dit",
                "models/diffusion_models/anima-base-v1.0.safetensors",
                "--caption_shuffle_variants",
                shuffle_variants,
                "--caption_tag_dropout_rate",
                tag_dropout_rate,
                *extra,
            ]
        )


def cmd_preprocess_pooled(extra):
    """Cache pooled text embeddings (max over seq dim) from existing TE caches.

    Reads ``{stem}_anima_te.safetensors`` from the LoRA cache dir and writes
    ``{stem}_anima_pooled.safetensors`` sidecars next to them. Consumed by
    ``make distill-mod`` to skip a redundant ``.max(dim=1)`` per training
    microstep / val sigma. No GPU needed.
    """
    for row in _dataset_rows():
        run(
            [
                PY,
                "preprocess/cache_pooled_text.py",
                "--dir",
                row["cache"],
                *extra,
            ]
        )


def cmd_preprocess_pe(extra):
    """Cache PE-Core-L14-336 vision-encoder features.

    Reads pre-resized images from ``post_image_dataset/resized/`` (the
    standard LoRA pipeline source) and writes
    ``{stem}_anima_pe.safetensors`` sidecars into the LoRA cache dir so the
    dataset's existing ``cache_dir`` lookup finds them.

    Consumed by methods that align against frozen vision features —
    currently REPA (--use_repa) and IP-Adapter when reading PE features off
    disk.
    """
    for row in _dataset_rows():
        run(
            [
                PY,
                "preprocess/cache_pe_encoder.py",
                "--dir",
                row["resized"],
                "--cache_dir",
                row["cache"],
                "--encoder",
                "pe",
                *extra,
            ]
        )


def cmd_preprocess(extra):
    cmd_preprocess_resize(extra)
    cmd_preprocess_vae(extra)
    cmd_preprocess_te(extra)
    cmd_preprocess_pe(extra)
