"""Small task-runner adapter for paired Qwen Image 2.1 Edit caching."""

from __future__ import annotations

from pathlib import Path
import shutil

from library.models.qwen_image_2_1.cache_policy import validate_cache_policy


def snapshot_edit_captions(source: str, resized: str, extension: str) -> None:
    """Mirror the configured sidecar suffix for images retained by resize."""
    src, dst = Path(source), Path(resized)
    if src.resolve() == dst.resolve():
        return
    extension = extension if extension.startswith(".") else f".{extension}"
    for image in dst.rglob("*.png"):
        relative = image.relative_to(dst).with_suffix(extension)
        caption = src / relative
        if not caption.is_file():
            raise ValueError(f"Qwen Image 2.1 Edit target instruction is missing: {caption}")
        shutil.copy2(caption, dst / relative)


def uses_qwen_edit(family: str, overrides: dict) -> bool:
    if family != "qwen_image_2_1":
        return False
    config_path = overrides.get("dataset_config")
    if not config_path:
        if str(overrides.get("qwen_image_2_1_task") or "").strip().lower() == "edit":
            raise ValueError("Qwen Image 2.1 Edit preprocessing requires dataset_config")
        return False
    from library.config.loader import load_user_config

    config = load_user_config(str(config_path))
    references = [bool(subset.get("reference_image_dir"))
        for dataset in config.get("datasets", [])
        for subset in dataset.get("subsets", [])
    ]
    explicit_edit = str(overrides.get("qwen_image_2_1_task") or "").strip().lower() == "edit"
    if (explicit_edit or any(references)) and (not references or not all(references)):
        raise ValueError("Qwen Image 2.1 Edit preprocessing requires reference_image_dir in every subset")
    return any(references)


def split_edit_options(extra: list[str]) -> tuple[list[str], list[str]]:
    """Keep Edit-only runtime options away from the resize CLI."""
    resize: list[str] = []
    edit: list[str] = []
    choices = {"--device": {"auto", "cpu", "cuda"}, "--offload": {"auto", "on", "off"}}
    i = 0
    while i < len(extra):
        arg = extra[i]
        flag, separator, inline = arg.partition("=")
        if flag in choices:
            if separator:
                value = inline
                i += 1
            else:
                if i + 1 >= len(extra):
                    raise ValueError(f"{flag} requires a value for Qwen Image 2.1 Edit preprocessing")
                value = extra[i + 1]
                i += 2
            if value not in choices[flag]:
                raise ValueError(f"invalid {flag} value for Qwen Image 2.1 Edit preprocessing: {value}")
            edit.extend([flag, value])
        elif arg == "--overwrite":
            # This belongs to cache generation, not resize_images argparse.
            i += 1
        else:
            resize.append(arg)
            i += 1
    return resize, edit


def run_edit_preprocess(overrides: dict, *, run, python: str, path, dtype: str,
                        extra: list[str]) -> None:
    config_path = str(overrides["dataset_config"])
    policy = validate_cache_policy(overrides.get("qwen_text_encoder_cache_policy", "auto"))
    if not Path(config_path).is_file():
        raise FileNotFoundError(config_path)
    _, forwarded = split_edit_options(extra)
    if "--overwrite" in extra:
        forwarded.append("--overwrite")
    run([
        python, "-m", "scripts.qwen_image_2_1.preprocess_edit_cache",
        "--dataset_config", config_path,
        "--vae", path("vae", "models/vae/qwen_image_vae.safetensors"),
        "--qwen3", path("qwen3", "models/text_encoders/Qwen-Image-2.1"),
        "--dtype", dtype,
        "--cache_policy", policy,
        *forwarded,
    ])
