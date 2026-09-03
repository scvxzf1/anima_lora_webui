"""CLTagger ONNX 本地打标 provider。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from ..model_assets import asset_file_path, get_model_asset, inspect_asset
from .onnx_base import OnnxTaggerBase, onnxruntime_status


_CATEGORY_ALIASES = {
    "0": "General",
    "general": "General",
    "tag": "General",
    "tags": "General",
    "4": "Character",
    "character": "Character",
    "characters": "Character",
    "9": "Rating",
    "rating": "Rating",
    "copyright": "Copyright",
    "artist": "Artist",
    "meta": "Meta",
    "model": "Model",
    "quality": "Quality",
}


def _category(value: Any) -> str:
    text = str(value or "").strip()
    return _CATEGORY_ALIASES.get(text.lower(), text or "General")


def _tag_key(value: Any) -> str:
    return str(value or "").replace("_", " ").strip().lower()


def _number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if number != number else max(0.0, min(1.0, number))


@dataclass
class LabelData:
    names: list[str | None]
    categories: list[str]


class CLTagger(OnnxTaggerBase):
    name = "cltagger"

    def __init__(
        self,
        settings: dict[str, Any] | None = None,
        *,
        overrides: dict[str, Any] | None = None,
    ):
        super().__init__(settings, overrides=overrides)
        self._labels: LabelData | None = None
        self._input_size = 448
        self._input_layout = "nchw"
        self._is_v2 = False
        self._asset = None

    def _cfg(self) -> dict[str, Any]:
        values = dict(self.settings)
        return {
            "asset_id": str(
                values.get("asset_id") or values.get("model_id") or "cltagger-v1-02"
            ).strip(),
            "batch_size": values.get("batch_size", 8),
            "general_threshold": _number(
                values.get("general_threshold", values.get("threshold_general", 0.35)),
                0.35,
            ),
            "character_threshold": _number(
                values.get(
                    "character_threshold", values.get("threshold_character", 0.6)
                ),
                0.6,
            ),
            "blacklist": values.get("blacklist", values.get("blacklist_tags", [])),
            "add_copyright_tag": _bool(values.get("add_copyright_tag", True), True),
            "add_artist_tag": _bool(values.get("add_artist_tag", False), False),
            "add_meta_tag": _bool(values.get("add_meta_tag", False), False),
            "add_model_tag": _bool(values.get("add_model_tag", False), False),
            "add_rating_tag": _bool(values.get("add_rating_tag", False), False),
            "add_quality_tag": _bool(values.get("add_quality_tag", False), False),
        }

    def _resolve_asset(self):
        asset = get_model_asset(self._cfg()["asset_id"])
        if asset.provider != self.name:
            raise ValueError(f"模型资产 {asset.id} 不属于 CLTagger provider")
        return asset

    def is_available(self) -> tuple[bool, str]:
        runtime_ok, runtime_message = onnxruntime_status()
        if not runtime_ok:
            return False, runtime_message
        try:
            asset = self._resolve_asset()
            status = inspect_asset(asset, verify_hash=True)
        except (KeyError, OSError, ValueError) as exc:
            return False, str(exc)
        if status.get("state") != "installed":
            return False, f"模型未安装或校验失败：{asset.id}（请先在接入预设中下载）"
        return True, f"{asset.label}；{runtime_message}"

    def prepare(self) -> None:
        if self._session is not None:
            return
        asset = self._resolve_asset()
        status = inspect_asset(asset, verify_hash=True)
        if status.get("state") != "installed":
            raise FileNotFoundError(f"模型未安装或校验失败：{asset.id}；请先下载模型")
        # Version behavior follows the manifest entry, not an arbitrary profile
        # id.  This keeps renamed/imported profiles from selecting the wrong
        # preprocessing contract.
        self._is_v2 = (
            asset.repo_id == "cella110n/cl_tagger_v2"
            or Path(asset.metadata_paths[0]).name == "model_vocabulary.json"
        )
        model_path = asset_file_path(asset, asset.model_path)
        mapping_path = asset_file_path(asset, asset.metadata_paths[0])
        try:
            self._create_session(model_path)
            inputs = list(self._session.get_inputs())
            input_meta = inputs[0] if inputs else None
            if self._is_v2:
                input_meta = next(
                    (
                        item
                        for item in inputs
                        if getattr(item, "name", "") == "pixel_values"
                    ),
                    input_meta,
                )
            if input_meta is None:
                raise ValueError("CLTagger ONNX 模型没有输入节点")
            self._input_name = str(getattr(input_meta, "name", self._input_name or "input"))
            shape = list(getattr(input_meta, "shape", []))
            self._input_layout = _infer_layout(shape)
            self._input_size = _infer_size(shape, self._input_layout)
            if self._is_v2:
                logits = [
                    str(getattr(item, "name", ""))
                    for item in self._session.get_outputs()
                    if getattr(item, "name", "") == "logits"
                ]
                if logits:
                    self._output_names = logits
            self._labels = self._load_tag_mapping(mapping_path)
        except Exception:
            self._clear_session()
            self._asset = None
            self._labels = None
            raise
        self._asset = asset

    @staticmethod
    def _load_tag_mapping(path: Path) -> LabelData:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("tag_mapping.json 格式不受支持")
        category_ids, category_tags = _parse_categories(raw)
        if "idx_to_tag" in raw:
            index_to_tag = _parse_idx_to_tag(raw["idx_to_tag"])
        elif "tag_to_idx" in raw:
            index_to_tag = _parse_tag_to_idx(raw["tag_to_idx"])
        else:
            index_to_tag = {}
            for key, value in raw.items():
                if isinstance(value, dict) and "tag" in value:
                    index_to_tag[int(key)] = str(value["tag"])
                elif isinstance(value, str) and str(key).lstrip("-").isdigit():
                    index_to_tag[int(key)] = value
            if not index_to_tag:
                raise ValueError("tag_mapping.json 缺少标签索引")
        tag_categories = _parse_tag_categories(raw, category_ids)
        size = max(index_to_tag, default=-1) + 1
        names: list[str | None] = [None] * size
        categories = ["General"] * size
        for index, tag in index_to_tag.items():
            if index < 0:
                continue
            if index >= len(names):
                names.extend([None] * (index + 1 - len(names)))
                categories.extend(["General"] * (index + 1 - len(categories)))
            names[index] = tag
            inline = raw.get(str(index))
            inline_category = (
                inline.get("category") if isinstance(inline, dict) else None
            )
            categories[index] = tag_categories.get(
                tag, category_tags.get(tag, _category(inline_category))
            )
        return LabelData(names=names, categories=categories)

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        image = ImageOps.exif_transpose(image) or image
        if image.mode not in {"RGB", "RGBA"}:
            # Palette/LA images can carry transparency in ``info`` even when
            # their mode is not RGBA. Preserve that alpha before compositing.
            image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
        if image.mode == "RGBA":
            canvas = Image.new("RGB", image.size, (255, 255, 255))
            canvas.paste(image, mask=image.getchannel("A"))
            image = canvas
        if image.width != image.height:
            side = max(image.width, image.height)
            canvas = Image.new("RGB", (side, side), (255, 255, 255))
            canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
            image = canvas
        image = image.resize(
            (self._input_size, self._input_size), Image.Resampling.BICUBIC
        )
        if self._is_v2:
            array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
            return array.transpose(2, 0, 1) if self._input_layout == "nchw" else array
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = array[..., ::-1]
        if self._input_layout == "nchw":
            array = array.transpose(2, 0, 1)
            mean = np.full((3, 1, 1), 0.5, dtype=np.float32)
            std = np.full((3, 1, 1), 0.5, dtype=np.float32)
        else:
            mean = np.full((1, 1, 3), 0.5, dtype=np.float32)
            std = np.full((1, 1, 3), 0.5, dtype=np.float32)
        return (array - mean) / std

    def _postprocess_one(
        self, logits: np.ndarray
    ) -> tuple[list[str], dict[str, float]]:
        if self._labels is None:
            raise ValueError("标签映射尚未加载")
        cfg = self._cfg()
        blacklist = {
            _tag_key(value)
            for value in (
                cfg["blacklist"] if isinstance(cfg["blacklist"], list) else []
            )
        }
        gates = {
            "Copyright": cfg["add_copyright_tag"],
            "Artist": cfg["add_artist_tag"],
            "Meta": cfg["add_meta_tag"],
            "Model": cfg["add_model_tag"],
            "Rating": cfg["add_rating_tag"],
            "Quality": cfg["add_quality_tag"],
        }
        scores = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits).reshape(-1), -30, 30)))
        result: list[tuple[str, float]] = []
        for index, value in enumerate(scores):
            if index >= len(self._labels.names):
                break
            tag = self._labels.names[index]
            if not tag or _tag_key(tag) in blacklist:
                continue
            category = self._labels.categories[index]
            if category in gates and not gates[category]:
                continue
            threshold = (
                cfg["character_threshold"]
                if category == "Character"
                else cfg["general_threshold"]
            )
            score = float(value)
            if score >= threshold:
                result.append((tag.replace("_", " "), score))
        result.sort(key=lambda item: item[1], reverse=True)
        return [tag for tag, _score in result], {tag: score for tag, score in result}


def _parse_idx_to_tag(value: Any) -> dict[int, str]:
    if isinstance(value, dict):
        return {int(key): str(tag) for key, tag in value.items()}
    if isinstance(value, list):
        return {index: str(tag) for index, tag in enumerate(value) if str(tag).strip()}
    raise ValueError("idx_to_tag 格式不受支持")


def _parse_tag_to_idx(value: Any) -> dict[int, str]:
    if not isinstance(value, dict):
        raise ValueError("tag_to_idx 格式不受支持")
    return {int(index): str(tag) for tag, index in value.items()}


def _parse_categories(raw: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    value = raw.get("categories", {})
    if not isinstance(value, dict):
        return {}, {}
    ids: dict[str, str] = {}
    tags: dict[str, str] = {}
    for key, item in value.items():
        if isinstance(item, (str, int, float, bool)):
            ids[str(key).strip()] = _category(item)
        elif isinstance(item, list):
            category = _category(key)
            for tag in item:
                if str(tag).strip():
                    tags[str(tag)] = category
    return ids, tags


def _parse_tag_categories(
    raw: dict[str, Any], category_ids: dict[str, str]
) -> dict[str, str]:
    value = raw.get("tag_to_category", {})
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for tag, category in value.items():
        text = str(category).strip()
        result[str(tag)] = category_ids.get(text, _category(text))
    return result


def _infer_layout(shape: list[Any]) -> str:
    if len(shape) >= 4:
        if shape[1] == 3:
            return "nchw"
        if shape[-1] == 3:
            return "nhwc"
    return "nchw"


def _infer_size(shape: list[Any], layout: str) -> int:
    indices = (2, 3) if layout == "nchw" else (1, 2)
    for index in indices:
        if index < len(shape):
            value = _positive_dimension(shape[index])
            if value is not None:
                return value
    return 448


def _positive_dimension(value: Any) -> int | None:
    """Return a concrete ONNX dimension, including numeric string dims."""

    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


__all__ = ["CLTagger", "LabelData"]
