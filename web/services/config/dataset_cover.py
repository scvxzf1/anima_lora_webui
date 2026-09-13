"""Small, bounded, read-only dataset library covers."""

import base64
import os
import time
from collections import OrderedDict
from threading import Lock

from library.datasets.subsets import filter_paths_by_glob
from web.services.config.dataset_preview_thumbnail import render_dataset_preview_thumbnail

MAX_ENTRIES = 2000
MAX_DECODE_ATTEMPTS = 8
_cache = OrderedDict()
_lock = Lock()


def dataset_cover(file: str) -> dict:
    from web.services import config_service as service
    from web.services.config.dataset_presets_api import _resolve_project_path

    preset = service.load_dataset_preset(file)
    rows = preset.get("datasets", [])
    key = (str(service.ROOT), str(service.CONFIGS_DIR), file, repr(rows))
    # Serialize cold scans so a visible library cannot decode many originals at once.
    with _lock:
        cached = _cache.get(key)
        if cached and time.monotonic() - cached[0] < 60:
            _cache.move_to_end(key)
            return cached[1]
        result = _detect(rows, _resolve_project_path)
        _cache[key] = (time.monotonic(), result)
        _cache.move_to_end(key)
        while len(_cache) > 128:
            _cache.popitem(last=False)
        return result


def _detect(rows, resolve) -> dict:
    from web.services.config.dataset_media import DATASET_IMAGE_EXTS

    remaining = MAX_ENTRIES
    attempts = 0
    reason = "未配置图片目录"
    seen = set()
    for row in rows:
        for field in ("source_dir", "image_dir"):
            raw = row.get(field)
            if not raw:
                continue
            root = resolve(str(raw))
            identity = (root, str(row.get("path_pattern", "*")), str(row.get("recursive", True)))
            if identity in seen:
                continue
            seen.add(identity)
            if not root.is_dir():
                reason = "图片目录不存在或无法访问"
                continue
            pending = [root]
            while pending:
                directory = pending.pop()
                try:
                    with os.scandir(directory) as entries:
                        for entry in entries:
                            remaining -= 1
                            if remaining < 0:
                                return _empty("检测达到扫描上限，未找到可用图像")
                            if entry.is_symlink():
                                continue
                            if entry.is_dir(follow_symlinks=False):
                                if row.get("recursive", True) not in (False, "false", "False", 0):
                                    pending.append(resolve(entry.path))
                                continue
                            path = resolve(entry.path)
                            if path.suffix.lower() not in DATASET_IMAGE_EXTS or not path.is_relative_to(root):
                                continue
                            pattern = str(row.get("path_pattern") or "*")
                            if not filter_paths_by_glob([str(path)], str(root), pattern)[0]:
                                continue
                            attempts += 1
                            try:
                                thumb = render_dataset_preview_thumbnail(path, size=(96, 96))
                                encoded = base64.b64encode(thumb.content).decode("ascii")
                                return {"ok": True, "image": f"data:{thumb.content_type};base64,{encoded}", "reason": ""}
                            except (OSError, ValueError):
                                reason = "图片读取失败"
                            if attempts >= MAX_DECODE_ATTEMPTS:
                                return _empty("候选图片无法读取，已停止检测")
                except OSError:
                    reason = "图片目录读取失败"
            if reason == "未配置图片目录":
                reason = "未找到符合筛选条件的图像"
    return _empty(reason)


def _empty(reason: str) -> dict:
    return {"ok": True, "image": None, "reason": reason}
