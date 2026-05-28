"""Caption sidecar helpers for text-encoder preprocessing.

The default path stays plain text sidecars. When explicitly enabled, structured
JSON sidecars are rendered with stable fixed fields and category-local shuffle.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Any, Callable


WarnFn = Callable[[str], None]


@dataclass
class StructuredCaption:
    fixed: list[str]
    appearance: list[str]
    tags: list[str]
    environment: list[str]
    nl: str = ""

    def render(
        self,
        *,
        appearance: list[str] | None = None,
        tags: list[str] | None = None,
        environment: list[str] | None = None,
    ) -> str:
        body = [
            *self.fixed,
            *(self.appearance if appearance is None else appearance),
            *(self.tags if tags is None else tags),
            *(self.environment if environment is None else environment),
        ]
        text = ", ".join(_clean_tags(body))
        nl = self.nl.strip()
        if not nl:
            return text
        return f"{text}. {nl}" if text else nl

    def generate_variants(self, num_variants: int, tag_dropout_rate: float) -> list[str]:
        variants = [self.render()]
        for _ in range(max(0, num_variants - 1)):
            appearance = _shuffle_and_drop(self.appearance, tag_dropout_rate)
            tags = _shuffle_and_drop(self.tags, tag_dropout_rate)
            environment = _shuffle_and_drop(self.environment, tag_dropout_rate)
            variants.append(
                self.render(
                    appearance=appearance,
                    tags=tags,
                    environment=environment,
                )
            )
        return variants


@dataclass
class CaptionSource:
    text: str = ""
    structured: StructuredCaption | None = None
    path: Path | None = None

    @property
    def from_json(self) -> bool:
        return self.structured is not None

    def render(self) -> str:
        if self.structured is not None:
            return self.structured.render()
        return self.text


def _clean_tags(values: list[str]) -> list[str]:
    return [value.strip() for value in values if value and value.strip()]


def _coerce_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return _clean_tags(value.split(","))
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(_coerce_tags(item))
        return out
    return _clean_tags([str(value)])


def _first_tags(*values: Any) -> list[str]:
    for value in values:
        tags = _coerce_tags(value)
        if tags:
            return tags
    return []


def _character_tags(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [
            *_coerce_tags(value.get("name")),
            *_coerce_tags(value.get("variant")),
        ]
    return _coerce_tags(value)


def _nested(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def structured_caption_from_json(data: dict[str, Any]) -> StructuredCaption:
    fixed = _nested(data, "fixed")
    character = data.get("character")
    from_path = _nested(data, "from_path")
    ai_output = _nested(data, "ai_output")

    fixed_tags = [
        *_first_tags(data.get("quality"), fixed.get("quality")),
        *_first_tags(data.get("count"), fixed.get("count"), ai_output.get("count")),
        *_character_tags(character),
        *_first_tags(data.get("series"), fixed.get("series")),
        *_first_tags(data.get("artist"), fixed.get("artist")),
    ]

    appearance = [
        *_coerce_tags(from_path.get("appearance")),
        *_coerce_tags(ai_output.get("appearance")),
        *_coerce_tags(data.get("appearance")),
    ]
    tags = [
        *_coerce_tags(ai_output.get("tags")),
        *_coerce_tags(data.get("tags")),
    ]
    environment = [
        *_coerce_tags(ai_output.get("environment")),
        *_coerce_tags(data.get("environment")),
    ]
    nl = str(data.get("nl") or ai_output.get("nl") or "").strip()

    return StructuredCaption(
        fixed=_clean_tags(fixed_tags),
        appearance=_clean_tags(appearance),
        tags=_clean_tags(tags),
        environment=_clean_tags(environment),
        nl=nl,
    )


def load_json_caption(path: Path) -> StructuredCaption:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON caption must be an object")
    return structured_caption_from_json(data)


def _text_caption_path(image_path: Path, caption_extension: str) -> Path:
    extension = caption_extension if caption_extension.startswith(".") else f".{caption_extension}"
    return image_path.with_suffix(extension)


def _read_text_caption(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[0].strip() if lines else ""


def read_caption_source(
    image_path: Path,
    *,
    prefer_json_caption: bool = False,
    caption_extension: str = ".txt",
    warn: WarnFn | None = None,
) -> CaptionSource:
    if prefer_json_caption:
        json_path = image_path.with_suffix(".json")
        if json_path.exists():
            try:
                return CaptionSource(
                    structured=load_json_caption(json_path),
                    path=json_path,
                )
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
                if warn is not None:
                    warn(f"could not parse JSON caption {json_path.name}: {e}; falling back to text")

    text_path = _text_caption_path(image_path, caption_extension)
    if text_path.exists():
        return CaptionSource(text=_read_text_caption(text_path), path=text_path)
    return CaptionSource(text="")


def _shuffle_and_drop(tokens: list[str], tag_dropout_rate: float) -> list[str]:
    shuffled = list(tokens)
    random.shuffle(shuffled)
    if tag_dropout_rate <= 0.0:
        return shuffled
    return [tag for tag in shuffled if random.random() >= tag_dropout_rate]
