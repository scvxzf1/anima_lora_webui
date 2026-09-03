"""WD14 EVA02 ONNX 本地打标 provider。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ..model_assets import asset_file_path, get_model_asset, inspect_asset
from .onnx_base import OnnxTaggerBase, onnxruntime_status


def _tag_key(value: Any) -> str:
    return str(value or "").replace("_", " ").strip().lower()


def _number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if number != number else max(0.0, min(1.0, number))


class WD14Tagger(OnnxTaggerBase):
    """使用固定 manifest 资产，不负责下载模型。"""

    name = "wd14"

    def __init__(
        self,
        settings: dict[str, Any] | None = None,
        *,
        overrides: dict[str, Any] | None = None,
    ):
        super().__init__(settings, overrides=overrides)
        self._tags: list[str] = []
        self._categories: list[int] = []
        self._input_size = 448
        self._input_layout = "nhwc"
        self._asset = None

    def _cfg(self) -> dict[str, Any]:
        values = dict(self.settings)
        return {
            "asset_id": str(
                values.get("asset_id")
                or values.get("model_id")
                or "wd14-eva02-large-v3"
            ).strip(),
            "batch_size": values.get("batch_size", 8),
            "general_threshold": _number(
                values.get("general_threshold", values.get("threshold_general", 0.35)),
                0.35,
            ),
            "character_threshold": _number(
                values.get(
                    "character_threshold", values.get("threshold_character", 0.85)
                ),
                0.85,
            ),
            "blacklist": values.get("blacklist", values.get("blacklist_tags", [])),
        }

    def _resolve_asset(self):
        asset = get_model_asset(self._cfg()["asset_id"])
        if asset.provider != self.name:
            raise ValueError(f"模型资产 {asset.id} 不属于 WD14 provider")
        return asset

    def _asset_status(self) -> tuple[Any, dict[str, Any]]:
        asset = self._resolve_asset()
        return asset, inspect_asset(asset, verify_hash=True)

    def is_available(self) -> tuple[bool, str]:
        runtime_ok, runtime_message = onnxruntime_status()
        if not runtime_ok:
            return False, runtime_message
        try:
            asset, status = self._asset_status()
        except (KeyError, OSError, ValueError) as exc:
            return False, str(exc)
        if status.get("state") != "installed":
            return False, f"模型未安装或校验失败：{asset.id}（请先在接入预设中下载）"
        return True, f"{asset.label}；{runtime_message}"

    def prepare(self) -> None:
        if self._session is not None:
            return
        asset, status = self._asset_status()
        if status.get("state") != "installed":
            raise FileNotFoundError(f"模型未安装或校验失败：{asset.id}；请先下载模型")
        model_path = asset_file_path(asset, asset.model_path)
        metadata_path = asset_file_path(asset, asset.metadata_paths[0])
        try:
            self._create_session(model_path)
            inputs = list(self._session.get_inputs())
            shape = list(getattr(inputs[0], "shape", [])) if inputs else []
            self._input_layout = _infer_layout(shape)
            self._input_size = _infer_size(shape, self._input_layout)
            self._load_tags(metadata_path)
        except Exception:
            self._clear_session()
            self._asset = None
            raise
        self._asset = asset

    def _load_tags(self, path: Path) -> None:
        tags: list[str] = []
        categories: list[int] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "name" not in reader.fieldnames:
                raise ValueError("selected_tags.csv 缺少 name 列")
            for row in reader:
                name = str(row.get("name") or "").strip()
                if not name:
                    continue
                try:
                    category = int(str(row.get("category") or "0").strip())
                except ValueError:
                    category = 0
                tags.append(name.replace("_", " "))
                categories.append(category)
        if not tags:
            raise ValueError("selected_tags.csv 没有可用标签")
        self._tags = tags
        self._categories = categories

    def known_tags(self) -> list[str]:
        return list(self._tags)

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        image = ImageOps.exif_transpose(image) or image
        if image.mode == "RGBA":
            canvas = Image.new("RGB", image.size, (255, 255, 255))
            canvas.paste(image, mask=image.getchannel("A"))
            image = canvas
        elif image.mode != "RGB":
            image = image.convert("RGB")
        image.thumbnail((self._input_size, self._input_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (self._input_size, self._input_size), (255, 255, 255))
        canvas.paste(
            image,
            (
                (self._input_size - image.width) // 2,
                (self._input_size - image.height) // 2,
            ),
        )
        array = np.asarray(canvas, dtype=np.float32)[..., ::-1]
        if self._input_layout == "nchw":
            array = array.transpose(2, 0, 1)
        return array

    def _postprocess_one(
        self, scores: np.ndarray
    ) -> tuple[list[str], dict[str, float]]:
        cfg = self._cfg()
        raw_blacklist = cfg["blacklist"] if isinstance(cfg["blacklist"], list) else []
        blacklist = {_tag_key(item) for item in raw_blacklist}
        result: list[tuple[str, float]] = []
        for index, value in enumerate(np.asarray(scores).reshape(-1)):
            if index >= len(self._tags):
                break
            category = self._categories[index] if index < len(self._categories) else 0
            tag = self._tags[index]
            if category == 9 or _tag_key(tag) in blacklist:
                continue
            threshold = (
                cfg["character_threshold"]
                if category == 4
                else cfg["general_threshold"]
            )
            score = float(value)
            if np.isfinite(score) and score >= threshold:
                result.append((tag, score))
        result.sort(key=lambda item: item[1], reverse=True)
        return [tag for tag, _score in result], {tag: score for tag, score in result}


def _infer_layout(shape: list[Any]) -> str:
    if len(shape) >= 4:
        if shape[1] == 3:
            return "nchw"
        if shape[-1] == 3:
            return "nhwc"
    return "nhwc"


def _infer_size(shape: list[Any], layout: str) -> int:
    indices = (2, 3) if layout == "nchw" else (1, 2)
    for index in indices:
        if index < len(shape):
            value = _positive_dimension(shape[index])
            if value is not None:
                return value
    return 448


def _positive_dimension(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


__all__ = ["WD14Tagger"]
