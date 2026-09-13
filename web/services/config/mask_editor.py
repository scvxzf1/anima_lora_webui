"""Dataset-scoped manual masks; never accept a client-provided output path."""

from __future__ import annotations

import base64
import hashlib
import os
import tempfile
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from threading import RLock

import tomlkit
from PIL import Image

from web.services.config import dataset_presets_api as presets
from web.services.config.mask_editor_geometry import check_size, prepare_image

_LOCK = RLock()
MAX_PNG_BYTES = 24 * 1024 * 1024


class MaskConflict(ValueError):
    pass


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@contextmanager
def _write_lock(file: str):
    # Serialize compare-and-replace across local WebUI processes as well as threads.
    key = _digest(str(presets._safe_resolve(file)).encode())
    with _LOCK, (Path(tempfile.gettempdir()) / f"anima-mask-{key}.lock").open("a") as lock:
        if os.name == "nt":
            import msvcrt
            lock.write("0")
            lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


def _context(file: str, index: int):
    preset = presets.load_dataset_preset(file)
    rows = preset["datasets"]
    if index < 0 or index >= len(rows):
        raise ValueError("数据集序号不在范围内")
    row = rows[index]
    source = presets._resolve_project_path(row["source_dir"])
    training = presets._resolve_project_path(row["image_dir"])
    configured = str(row.get("mask_dir") or "")
    if row["mask_mode"] == "auto" and not configured:
        from library.datasets.mask_mode import LEGACY_AUTO_MASK_DIRS
        configured = next((value for value in LEGACY_AUTO_MASK_DIRS
                           if presets._resolve_project_path(value).is_dir()), "")
    if ".." in Path(configured).parts:
        raise ValueError("蒙版目录不能包含 ..")
    key = _digest(f"{preset['file']}:{source}".encode())[:12]
    root = presets._resolve_project_path(configured) if configured else (
        training.parent / f"{training.name}_masks" / key
    )
    root = root.resolve()
    if root.is_relative_to(source) or root.is_relative_to(training):
        raise ValueError("蒙版目录不能位于原图或训练图目录内，请在数据集设置中使用独立目录")
    return preset, row, source, training, root


def _contained(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("蒙版路径不合法")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("蒙版路径越界")
    # Do not overwrite an alias, even when its destination remains inside root.
    if any(p.is_symlink() for p in [path, *path.parents] if p != root and p.is_relative_to(root)):
        raise ValueError("蒙版路径不能包含符号链接")
    return path


def _target(file: str, index: int, image_file: str):
    preset, row, source, training, root = _context(file, index)
    path = presets.resolve_dataset_preview_image(file, index, image_file, source="source")
    duplicates = [p for p in path.parent.iterdir()
                  if p.stem == path.stem and p.suffix.lower() in presets.DATASET_IMAGE_EXTS]
    if len(duplicates) > 1:
        raise ValueError("同一目录存在同名不同格式图片，请先消除重名再编辑蒙版")
    relative = path.relative_to(source)
    target = _contained(root, relative.with_name(f"{relative.stem}_mask.png"))
    flat = _contained(root, Path(target.name))
    existing = target if target.is_file() else flat if flat.is_file() else None
    train_path = _contained(training, relative.with_suffix(".png"))
    if source == training:
        train_path = path
    elif not train_path.is_file():
        train_path = _contained(training, relative)
    actual = train_path.is_file()
    prepared = prepare_image(train_path if actual else path, project=not actual,
                             settings=row.get("settings") or preset["defaults"])
    stamp = (train_path if actual else path).stat()
    if existing and existing.stat().st_size > MAX_PNG_BYTES:
        raise ValueError("现有蒙版文件过大")
    mask_bytes = existing.read_bytes() if existing else b""
    revision = _digest(preset["content"].encode() + str((stamp.st_mtime_ns, stamp.st_size,
                       prepared.size, str(existing))).encode() + mask_bytes)
    return preset, row, root, target, prepared, mask_bytes, revision, actual


def _png_data(image: Image.Image) -> bytes:
    output = BytesIO()
    image.info.clear()
    image.save(output, format="PNG")
    return output.getvalue()


def _data_url(image: Image.Image) -> str:
    return "data:image/png;base64," + base64.b64encode(_png_data(image)).decode("ascii")


def list_masks(file: str, index: int, offset: int = 0) -> dict:
    preset, row, source, _training, root = _context(file, index)
    page = presets.list_dataset_preset_images(file, index, source="source", limit=48, offset=offset)
    for item in page["images"]:
        path = presets.resolve_dataset_preview_image(file, index, item["file"], source="source")
        relative = path.relative_to(source)
        target = _contained(root, relative.with_name(f"{relative.stem}_mask.png"))
        item["has_mask"] = target.is_file() or _contained(root, Path(target.name)).is_file()
    return {**page, "mask_dir": presets._display_path(root), "mask_mode": row["mask_mode"],
            "readonly": bool(preset["readonly"] or preset.get("meta", {}).get("locked")),
            "config_revision": _digest(preset["content"].encode())}


def read_mask(file: str, index: int, image: str) -> dict:
    preset, row, root, target, prepared, saved, revision, actual = _target(file, index, image)
    if saved:
        with Image.open(BytesIO(saved)) as source_mask:
            check_size(source_mask.size)
            mask = source_mask.convert("L").resize(prepared.size, Image.Resampling.NEAREST)
    elif row["mask_mode"] == "embedded":
        mask = prepared.getchannel("A")
    else:
        mask = Image.new("L", prepared.size, 255)
    return {"ok": True, "revision": revision, "width": prepared.width, "height": prepared.height,
            "image_url": _data_url(prepared.convert("RGB")), "mask_url": _data_url(mask),
            "has_mask": bool(saved), "basis": "training" if actual else "projected",
            "mask_dir": presets._display_path(root), "mask_file": presets._display_path(target),
            "readonly": bool(preset["readonly"] or preset.get("meta", {}).get("locked"))}


def _check_writable(preset: dict) -> None:
    if preset["readonly"] or preset.get("meta", {}).get("locked"):
        raise PermissionError("此数据集已锁定或只读，请复制或解锁后编辑蒙版")


def _atomic_png(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".mask-", suffix=".tmp", delete=False) as tmp:
            name = tmp.name
            tmp.write(content)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(name, path)
        name = None
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def save_mask(file: str, index: int, image: str, revision: str, png: bytes) -> dict:
    if len(png) > MAX_PNG_BYTES:
        raise ValueError("蒙版 PNG 过大")
    with _write_lock(file):
        preset, _row, _root, target, prepared, _saved, current, _actual = _target(file, index, image)
        _check_writable(preset)
        if revision != current:
            raise MaskConflict("图片、蒙版或数据集配置已变化，请重新加载后编辑")
        with Image.open(BytesIO(png)) as uploaded:
            check_size(uploaded.size)
            if uploaded.format != "PNG" or uploaded.size != prepared.size:
                raise ValueError("蒙版必须是与编辑画布尺寸一致的 PNG")
            if uploaded.mode not in {"1", "L", "RGB", "RGBA"}:
                raise ValueError("不支持的蒙版像素格式")
            mask = uploaded.convert("L")
            if uploaded.mode == "RGBA" and uploaded.getchannel("A").getextrema() != (255, 255):
                raise ValueError("请提交不透明的黑白蒙版")
            _atomic_png(target, _png_data(mask))
        return {"ok": True, "message": "蒙版已保存", "revision": _target(file, index, image)[6]}


def apply_masks(file: str, index: int, revision: str) -> dict:
    with _write_lock(file):
        preset, _row, _source, _training, root = _context(file, index)
        _check_writable(preset)
        if revision != _digest(preset["content"].encode()):
            raise MaskConflict("数据集配置已变化，请刷新后再应用")
        doc = tomlkit.parse(preset["content"])
        subsets = [subset for dataset in doc.get("datasets", []) for subset in dataset.get("subsets", [])]
        if index >= len(subsets):
            raise ValueError("此数据集结构不支持直接应用蒙版")
        subsets[index]["mask_mode"] = "external"
        subsets[index]["mask_dir"] = presets._display_path(root)
        subsets[index]["alpha_mask"] = True
        ok, message, _warnings = presets.save_raw_file(file, tomlkit.dumps(doc), overwrite=True)
        if not ok:
            raise ValueError(message)
        return {"ok": True, "message": "已应用到此子集，后续启动的训练生效"}
