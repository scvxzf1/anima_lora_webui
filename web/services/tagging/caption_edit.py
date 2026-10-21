"""Caption sidecar read/write semantics for the tagging workbench.

Phase 2 of the clean-room port (`docs/findings/tagging-port-spec.md` §3): one
place decides which sidecar is authoritative, what counts as an already captioned
image, and how edited tags are merged back into structured or plain captions.

This module only implements the contract for this repository's dataset-preset
architecture; no third-party source, prompt text or fixture is reproduced.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from library.preprocess.captions import structured_caption_from_json
from web.services.atomic_io import atomic_write_text

JSON_SUFFIX = ".json"
TEXT_SUFFIXES = (".txt", ".caption")
DEFAULT_TEXT_SUFFIX = ".txt"
POLICIES = ("overwrite", "skip", "append")
POLICY_OVERWRITE = "overwrite"
POLICY_SKIP = "skip"
POLICY_APPEND = "append"

# Keys that carry editable tags in a structured caption.  A reviewed caption
# replaces them wholesale, because the edited text is authoritative.
ABSORBED_JSON_KEYS = frozenset(
    {
        "tags",
        "appearance",
        "environment",
        "quality",
        "count",
        "character",
        "series",
        "artist",
        "fixed",
        "from_path",
        "ai_output",
        "nl",
    }
)


class CaptionSidecarError(ValueError):
    """A caption sidecar exists but cannot be edited safely."""


class CorruptCaptionError(CaptionSidecarError):
    """A structured caption sidecar exists but is not a usable JSON object."""


class AggregateCaptionWriteError(CaptionSidecarError):
    """The dataset row keeps captions in captions.json instead of sidecars.

    Writing <stem>.txt next to an aggregate caption would create two sources of
    truth, and the dataset reader prefers the aggregate, so the reviewed text
    would silently never reach training.
    """


def normalize_policy(value: Any) -> str:
    """Return a supported write policy, defaulting to overwrite."""

    policy = str(value or "").strip().lower()
    return policy if policy in POLICIES else POLICY_OVERWRITE


def split_tags(value: Any) -> list[str]:
    """Split caption text into tags, rejecting blanks and repeated entries."""

    if isinstance(value, (list, tuple)):
        raw_parts: list[str] = []
        for item in value:
            raw_parts.extend(split_tags(item))
        return raw_parts
    parts = re.split(r"[,，\n]", str(value or ""))
    return unique_tags(parts)


def unique_tags(values: Any) -> list[str]:
    """Keep the first spelling of every tag, comparing case-insensitively."""

    seen: set[str] = set()
    result: list[str] = []
    for item in values or []:
        tag = str(item or "").strip()
        if not tag:
            continue
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(tag)
    return result


def merge_tags(existing: Any, incoming: Any) -> list[str]:
    """Append new tags after the existing ones, preserving their order."""

    return unique_tags([*unique_tags(existing), *unique_tags(incoming)])


def apply_trigger(trigger: str, tags: list[str]) -> list[str]:
    """Prepend the trigger once, honouring separators and duplicate spellings.

    A trigger may carry several comma separated tags ("masterpiece, best
    quality").  Splitting it keeps the prepend idempotent: the next write back
    reads those tags from the sidecar and must not duplicate them.
    """

    parts = split_tags(trigger)
    if not parts:
        return unique_tags(tags)
    return unique_tags([*parts, *unique_tags(tags)])


def sidecar_for(image: Path, *, text_suffix: str = DEFAULT_TEXT_SUFFIX) -> tuple[Path, str]:
    """Return the authoritative sidecar path and its kind (``json``/``text``)."""

    image = Path(image)
    json_path = image.with_suffix(JSON_SUFFIX)
    if json_path.is_file():
        return json_path, "json"
    ordered = unique_tags([text_suffix, *TEXT_SUFFIXES]).__iter__()
    for suffix in ordered:
        candidate = image.with_suffix(suffix if suffix.startswith(".") else f".{suffix}")
        if candidate.is_file():
            return candidate, "text"
    fallback = image.with_suffix(text_suffix or DEFAULT_TEXT_SUFFIX)
    return fallback, "text"


def is_structured_caption(image: Path, *, text_suffix: str = DEFAULT_TEXT_SUFFIX) -> bool:
    """Report whether the authoritative sidecar is a structured JSON caption."""

    path, kind = sidecar_for(image, text_suffix=text_suffix)
    return kind == "json" and path.is_file()


def has_effective_caption(image: Path, *, text_suffix: str = DEFAULT_TEXT_SUFFIX) -> bool:
    """Report whether the image already carries usable caption content.

    text_suffix must be the dataset row caption extension: the same value
    decides the write target, so detection and write-back always inspect one
    sidecar instead of silently disagreeing.
    """

    path, kind = sidecar_for(image, text_suffix=text_suffix)
    if not path.is_file():
        return False
    if kind == "json":
        data = _load_json(path, required=False)
        if not isinstance(data, dict):
            return False
        meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
        rendered = structured_caption_from_json(data).render().strip()
        return bool(rendered or str(meta.get("trigger") or "").strip())
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return bool(split_tags(text))


def read_caption(image: Path, *, text_suffix: str = DEFAULT_TEXT_SUFFIX) -> dict[str, Any]:
    """Return the editable tags and rendered text of the authoritative sidecar."""

    path, kind = sidecar_for(image, text_suffix=text_suffix)
    if not path.is_file():
        return {"path": None, "kind": kind, "tags": [], "text": "", "exists": False}
    if kind == "json":
        data = _load_json(path, required=False)
        if not isinstance(data, dict):
            return {"path": path, "kind": kind, "tags": [], "text": "", "exists": True}
        structured = structured_caption_from_json(data)
        tags = unique_tags(
            [*structured.fixed, *structured.appearance, *structured.tags, *structured.environment]
        )
        return {"path": path, "kind": kind, "tags": tags, "text": structured.render(), "exists": True}
    text = path.read_text(encoding="utf-8", errors="replace")
    tags = split_tags(text)
    return {"path": path, "kind": kind, "tags": tags, "text": ", ".join(tags), "exists": True}


def read_tags(image: Path, *, text_suffix: str = DEFAULT_TEXT_SUFFIX) -> list[str]:
    """Return only the editable tags of the authoritative sidecar."""

    return list(read_caption(image, text_suffix=text_suffix)["tags"])


def write_tags(
    image: Path,
    tags: Any,
    *,
    text_suffix: str = DEFAULT_TEXT_SUFFIX,
    trigger: str = "",
) -> dict[str, Any]:
    """Write the edited tags back to the authoritative sidecar."""

    image = Path(image)
    path, kind = sidecar_for(image, text_suffix=text_suffix)
    edited = apply_trigger(trigger, unique_tags(tags))
    if kind == "json" and path.is_file():
        data = _load_json(path, required=True)
        payload = _structured_payload(data, edited, trigger)
        text = retag_json_text(payload, edited)
        atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        return {"path": path, "kind": "json", "tags": edited, "text": text}
    text = ", ".join(edited)
    atomic_write_text(path, f"{text}\n" if text else "")
    return {"path": path, "kind": "text", "tags": edited, "text": text}


def retag_json_text(payload: dict[str, Any], tags: list[str]) -> str:
    """Render the caption text a structured payload now produces."""

    rendered = ", ".join(tags)
    nl = str(payload.get("nl") or "").strip()
    if not nl:
        return rendered
    return f"{rendered}. {nl}" if rendered else nl


def _structured_payload(data: dict[str, Any], tags: list[str], trigger: str) -> dict[str, Any]:
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    nl = structured_caption_from_json(data).nl.strip()
    payload = {key: value for key, value in data.items() if key not in ABSORBED_JSON_KEYS}
    if trigger:
        meta = {**meta, "trigger": trigger}
    if meta:
        payload["meta"] = meta
    payload["tags"] = list(tags)
    if nl and _nl_still_missing(nl, tags):
        payload["nl"] = nl
    return payload


def _nl_still_missing(nl: str, tags: list[str]) -> bool:
    needle = re.sub(r"\s+", " ", nl).strip().casefold()
    if not needle:
        return False
    haystack = re.sub(r"\s+", " ", " ".join(tags)).casefold()
    return needle not in haystack


def _load_json(path: Path, *, required: bool) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        if required:
            raise CorruptCaptionError(f"结构化 caption 无法解析，已取消写入：{path.name}") from exc
        return None
    if not isinstance(data, dict):
        if required:
            raise CorruptCaptionError(f"结构化 caption 不是对象，已取消写入：{path.name}")
        return None
    return data
