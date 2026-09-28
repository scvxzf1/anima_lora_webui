"""Training-preview task validation and bounded reference image IO."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path

from PIL import Image, ImageOps

from library.datasets.qwen_image_edit import reference_size_for_bucket
from library.env import resolve_under_home
from library.models.family_registry import get_model_family_spec, normalize_registered_family

MAX_REFERENCE_BYTES = 20 * 1024 * 1024
MAX_REFERENCE_PIXELS = 32 * 1024 * 1024
MAX_PREVIEW_REFERENCE_PIXELS = 4 * 1024 * 1024
MAX_PREVIEW_FILE_REFERENCE_PIXELS = 16 * 1024 * 1024


def reference_path(value: str) -> Path:
    path = Path(os.path.expandvars(str(value).strip())).expanduser()
    if not str(value).strip() or ".." in path.parts:
        raise ValueError("参考图路径不能为空或包含 ..")
    path = resolve_under_home(str(path)).resolve()
    if not path.is_file() or path.stat().st_size > MAX_REFERENCE_BYTES:
        raise ValueError("参考图不存在或超过 20 MiB")
    return path


def read_reference(source) -> Image.Image:
    try:
        with Image.open(source) as image:
            # Some Pillow plugins expose ``n_frames`` through a lazy attribute
            # that can raise AttributeError while the image is still open
            # (notably JPEGs). Missing it means a normal single-frame image.
            try:
                frame_count = image.n_frames
            except (AttributeError, KeyError, TypeError):
                frame_count = 1
            if image.width * image.height > MAX_REFERENCE_PIXELS or frame_count != 1:
                raise ValueError("参考图须为不超过 3200 万像素的单帧图片")
            return ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError("无法读取参考图片") from exc


def parse_structured_prompt(line: str) -> dict | None:
    if not re.match(r'^\s*\{\s*"[^"\n]+"\s*:', line):
        return None
    value = json.loads(line)
    if not isinstance(value, dict) or not isinstance(value.get("prompt"), str):
        raise ValueError("采样记录必须包含字符串 prompt")
    return value


def publish_reference(path: Path, content: bytes) -> None:
    """Publish a complete image atomically across threads and training ranks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("参考图目标不能是符号链接")
    if path.exists():
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def preview_task(prompt: dict) -> str:
    return str(prompt.get("sample_task") or (
        "edit" if prompt.get("reference_image") or "reference_images" in prompt else "t2i"
    ))


def preview_references(prompt: dict) -> list[str]:
    """Return ordered edit references, accepting the legacy single-path field."""
    if "reference_images" in prompt:
        references = prompt["reference_images"]
        if not isinstance(references, list) or not references or any(
            not isinstance(path, str) or not path.strip() for path in references
        ):
            raise ValueError("reference_images 须为非空参考图路径列表")
        if prompt.get("reference_image"):
            raise ValueError("reference_image 与 reference_images 不能同时使用")
        return references
    reference = prompt.get("reference_image")
    if reference is None or reference == "":
        return []
    if not isinstance(reference, str):
        raise ValueError("reference_image 须为参考图路径字符串")
    return [reference]


def validate_preview_prompt(prompt: dict, family: str, sampler: str = "euler") -> int:
    spec = get_model_family_spec(normalize_registered_family(family, allow_aliases=True))
    task = preview_task(prompt)
    if task not in spec.supported_preview_tasks:
        raise ValueError(f"{spec.display_name} 不支持采样任务 {task}")
    references = preview_references(prompt)
    if task == "edit" and not references:
        raise ValueError("编辑样张须包含参考图")
    if task != "edit" and (references or "reference_images" in prompt):
        raise ValueError("文生图样张不能包含编辑参考图")
    if len(references) > spec.max_preview_references:
        raise ValueError(f"{spec.display_name} 采样最多支持 {spec.max_preview_references} 张参考图")
    if spec.name == "qwen_image_2_1":
        if str(prompt.get("sample_sampler") or sampler) != "euler":
            raise ValueError("Qwen Image 2.1 采样仅支持 Euler")
        for key in ("width", "height"):
            raw = prompt.get(key, 512)
            value = int(raw)
            if isinstance(raw, bool) or float(raw) != value:
                raise ValueError("Qwen 采样宽高须为整数")
            if not 64 <= value <= 2048 or value % 32:
                raise ValueError("Qwen 采样宽高须为 64..2048 范围内的 32 倍数")
        raw_steps = prompt.get("sample_steps", 28)
        steps = int(raw_steps)
        if isinstance(raw_steps, bool) or float(raw_steps) != steps or not 1 <= steps <= 1000:
            raise ValueError("采样步数须为 1..1000")
        scale = float(prompt.get("guidance_scale", prompt.get("scale", 1.0)))
        if not math.isfinite(scale) or scale < 1:
            raise ValueError("Qwen 采样 CFG 须大于等于 1")
        if prompt.get("flow_shift") not in (None, ""):
            raise ValueError("Qwen 采样使用按分辨率自动计算的 Flow shift")
    total_pixels = 0
    if references:
        target = (int(prompt.get("width", 512)), int(prompt.get("height", 512)))
        for reference in references:
            image = read_reference(reference_path(reference))
            resized = reference_size_for_bucket(image.size, target)
            total_pixels += resized[0] * resized[1]
            if total_pixels > MAX_PREVIEW_REFERENCE_PIXELS:
                raise ValueError("采样参考图缩放后总像素不得超过 419 万")
    return total_pixels


def validate_preview_file(config) -> None:
    from library import train_util
    from library.training.train_bootstrap import resolve_sample_prompts_path

    get = config.get if isinstance(config, dict) else lambda key, default=None: getattr(config, key, default)
    if not any(get(key) for key in ("sample_at_first", "sample_every_n_steps", "sample_every_n_epochs")):
        return
    path = resolve_sample_prompts_path(get("sample_prompts"))
    if path is None:
        raise ValueError("已启用采样，但样张文件不存在")
    prompts = train_util.load_prompts(path)
    if not prompts:
        raise ValueError("采样样张文件为空")
    family = normalize_registered_family(get("model_family") or "anima", allow_aliases=True)
    file_reference_pixels = 0
    for index, prompt in enumerate(prompts):
        try:
            file_reference_pixels += validate_preview_prompt(prompt, family, get("sample_sampler") or "euler")
            if file_reference_pixels > MAX_PREVIEW_FILE_REFERENCE_PIXELS:
                raise ValueError("样张文件参考图缩放后总像素不得超过 16 Mi（16777216 像素）")
        except (ValueError, TypeError, OSError) as exc:
            raise ValueError(f"样张 {index + 1}: {exc}") from exc
