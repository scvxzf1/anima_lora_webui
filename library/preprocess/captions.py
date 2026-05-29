"""Caption sidecar helpers for text-encoder preprocessing.

The default path stays plain text sidecars. When explicitly enabled, structured
JSON sidecars are rendered with stable fixed fields and category-local shuffle.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import random
import shutil
from pathlib import Path
from typing import Any, Callable, Sequence

from library.preprocess._dataset import walk_images


WarnFn = Callable[[str], None]
DEFAULT_CAPTION_BACKUP_EXTENSIONS = (".txt", ".caption", ".json")
CAPTIONS_JSON_FILE = "captions.json"
CAPTION_SOURCE_AUTO = "auto"
CAPTION_SOURCE_TXT = "txt"
CAPTION_SOURCE_JSON = "json"
CAPTION_SOURCE_CAPTIONS_JSON = "captions_json"
CAPTION_SOURCE_MODES = frozenset(
    {
        CAPTION_SOURCE_AUTO,
        CAPTION_SOURCE_TXT,
        CAPTION_SOURCE_JSON,
        CAPTION_SOURCE_CAPTIONS_JSON,
    }
)
_CAPTION_SOURCE_ALIASES = {
    "text": CAPTION_SOURCE_TXT,
    ".txt": CAPTION_SOURCE_TXT,
    "sd_scripts": CAPTION_SOURCE_TXT,
    "sd-scripts": CAPTION_SOURCE_TXT,
    "same_stem_json": CAPTION_SOURCE_JSON,
    ".json": CAPTION_SOURCE_JSON,
    "anima_lora_toolkit": CAPTION_SOURCE_JSON,
    "animaloratoolkit": CAPTION_SOURCE_JSON,
    "captions.json": CAPTION_SOURCE_CAPTIONS_JSON,
    "captions-json": CAPTION_SOURCE_CAPTIONS_JSON,
    "captionsjson": CAPTION_SOURCE_CAPTIONS_JSON,
    "diffpipeforge": CAPTION_SOURCE_CAPTIONS_JSON,
}


@dataclass
class CaptionBackupStats:
    images_seen: int = 0
    copied: int = 0
    missing: int = 0
    failed: int = 0


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
    captions: list[str] | None = None
    path: Path | None = None
    source_mode: str = CAPTION_SOURCE_AUTO
    detected_mode: str = ""

    @property
    def from_json(self) -> bool:
        return self.structured is not None

    @property
    def from_captions_json(self) -> bool:
        return self.detected_mode == CAPTION_SOURCE_CAPTIONS_JSON

    def caption_texts(self) -> list[str]:
        if self.captions is not None:
            return list(self.captions)
        return [self.render()]

    def render(self) -> str:
        if self.captions is not None:
            return self.captions[0] if self.captions else ""
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


def normalize_caption_source_mode(
    value: Any = None,
    prefer_json_caption: bool = False,
) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    if raw in CAPTION_SOURCE_MODES:
        return raw
    alias = _CAPTION_SOURCE_ALIASES.get(raw)
    if alias:
        return alias
    return CAPTION_SOURCE_JSON if prefer_json_caption else CAPTION_SOURCE_AUTO


def _coerce_caption_texts(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    out.append(text)
            elif item is not None:
                text = str(item).strip()
                if text:
                    out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def load_captions_json(path: Path) -> dict[str, list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("captions.json must be an object")
    out: dict[str, list[str]] = {}
    for key, value in data.items():
        if not isinstance(key, str) or not key.strip():
            continue
        captions = _coerce_caption_texts(value)
        if captions:
            out[key.strip().replace("\\", "/")] = captions
    return out


@lru_cache(maxsize=128)
def _load_captions_json_cached(path: str, mtime_ns: int, size: int) -> dict[str, tuple[str, ...]]:
    del mtime_ns, size
    return {key: tuple(value) for key, value in load_captions_json(Path(path)).items()}


def _load_captions_json_for_lookup(path: Path) -> dict[str, tuple[str, ...]]:
    stat = path.stat()
    return _load_captions_json_cached(str(path), stat.st_mtime_ns, stat.st_size)


def captions_json_texts_for_image(image_path: Path, captions_json_path: Path) -> list[str]:
    captions = _load_captions_json_for_lookup(captions_json_path)
    base_dir = captions_json_path.parent
    keys: list[str] = []
    try:
        keys.append(image_path.resolve().relative_to(base_dir.resolve()).as_posix())
    except ValueError:
        pass
    keys.extend([image_path.name, image_path.stem])
    for key in keys:
        value = captions.get(key)
        if value:
            return list(value)
    return []


def _text_caption_path(image_path: Path, caption_extension: str) -> Path:
    extension = caption_extension if caption_extension.startswith(".") else f".{caption_extension}"
    return image_path.with_suffix(extension)


def _read_text_caption(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return lines[0].strip() if lines else ""


def _unique_directories(directories: Sequence[Path]) -> list[Path]:
    out: list[Path] = []
    for directory in directories:
        try:
            resolved = Path(directory).resolve()
        except OSError:
            resolved = Path(directory)
        if resolved not in out:
            out.append(resolved)
    return out


def _probe_path_for_directory(image_path: Path, directory: Path) -> Path:
    try:
        image_path.resolve().relative_to(directory.resolve())
    except ValueError:
        return directory / image_path.name
    return image_path


def _captions_json_candidates(image_path: Path, root: Path | None = None) -> list[Path]:
    directories: list[Path] = []
    current = image_path.parent.resolve()
    stop = root.resolve() if root is not None else current
    while True:
        directories.append(current)
        if current == stop:
            break
        if root is None or stop not in current.parents:
            break
        parent = current.parent
        if parent == current:
            break
        current = parent
    return [directory / CAPTIONS_JSON_FILE for directory in directories]


def _read_captions_json_source(
    image_path: Path,
    *,
    root: Path | None,
    source_mode: str,
    warn: WarnFn | None,
) -> CaptionSource | None:
    for captions_path in _captions_json_candidates(image_path, root):
        if not captions_path.is_file():
            continue
        try:
            captions = captions_json_texts_for_image(image_path, captions_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
            if warn is not None:
                warn(f"could not parse captions.json {captions_path}: {e}")
            continue
        if captions:
            return CaptionSource(
                captions=captions,
                path=captions_path,
                source_mode=source_mode,
                detected_mode=CAPTION_SOURCE_CAPTIONS_JSON,
            )
    return None


def _read_json_sidecar_source(
    image_path: Path,
    *,
    source_mode: str,
    warn: WarnFn | None,
) -> CaptionSource | None:
    json_path = image_path.with_suffix(".json")
    if not json_path.exists():
        return None
    try:
        return CaptionSource(
            structured=load_json_caption(json_path),
            path=json_path,
            source_mode=source_mode,
            detected_mode=CAPTION_SOURCE_JSON,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
        if warn is not None:
            warn(f"could not parse JSON caption {json_path.name}: {e}; falling back to text")
    return None


def _read_text_sidecar_source(
    image_path: Path,
    *,
    caption_extension: str,
    source_mode: str,
) -> CaptionSource | None:
    text_path = _text_caption_path(image_path, caption_extension)
    if not text_path.exists():
        return None
    return CaptionSource(
        text=_read_text_caption(text_path),
        path=text_path,
        source_mode=source_mode,
        detected_mode=CAPTION_SOURCE_TXT,
    )


def read_caption_source(
    image_path: Path,
    *,
    prefer_json_caption: bool = False,
    caption_source_mode: str | None = None,
    caption_extension: str = ".txt",
    captions_root: Path | None = None,
    warn: WarnFn | None = None,
) -> CaptionSource:
    return read_caption_source_from_dirs(
        image_path,
        [image_path.parent],
        prefer_json_caption=prefer_json_caption,
        caption_source_mode=caption_source_mode,
        caption_extension=caption_extension,
        captions_root=captions_root,
        warn=warn,
    )


def read_caption_source_from_dirs(
    image_path: Path,
    directories: Sequence[Path],
    *,
    prefer_json_caption: bool = False,
    caption_source_mode: str | None = None,
    caption_extension: str = ".txt",
    captions_root: Path | None = None,
    warn: WarnFn | None = None,
) -> CaptionSource:
    source_mode = normalize_caption_source_mode(caption_source_mode, prefer_json_caption)
    explicit_source_mode = bool(str(caption_source_mode or "").strip())
    clean_dirs = _unique_directories([Path(d) for d in directories if d])
    if not clean_dirs:
        clean_dirs = [image_path.parent]
    probes = [
        (_probe_path_for_directory(image_path, directory), captions_root or directory)
        for directory in clean_dirs
    ]

    if source_mode == CAPTION_SOURCE_AUTO:
        order = (CAPTION_SOURCE_CAPTIONS_JSON, CAPTION_SOURCE_JSON, CAPTION_SOURCE_TXT)
    elif source_mode == CAPTION_SOURCE_JSON and prefer_json_caption and not explicit_source_mode:
        # Preserve the legacy prefer_json_caption behavior: same-stem JSON is
        # preferred, but a missing or invalid JSON sidecar can still fall back
        # to text.
        order = (CAPTION_SOURCE_JSON, CAPTION_SOURCE_TXT)
    else:
        order = (source_mode,)

    for candidate_mode in order:
        for probe, root in probes:
            if candidate_mode == CAPTION_SOURCE_CAPTIONS_JSON:
                source = _read_captions_json_source(
                    probe,
                    root=root,
                    source_mode=source_mode,
                    warn=warn,
                )
            elif candidate_mode == CAPTION_SOURCE_JSON:
                source = _read_json_sidecar_source(
                    probe,
                    source_mode=source_mode,
                    warn=warn,
                )
            else:
                source = _read_text_sidecar_source(
                    probe,
                    caption_extension=caption_extension,
                    source_mode=source_mode,
                )
            if source is not None:
                return source
    return CaptionSource(text="", source_mode=source_mode)


def caption_backup_extensions(caption_extension: str | None = ".txt") -> tuple[str, ...]:
    extensions: list[str] = []
    for ext in (*DEFAULT_CAPTION_BACKUP_EXTENSIONS, caption_extension or ""):
        clean = str(ext or "").strip()
        if not clean:
            continue
        if not clean.startswith("."):
            clean = f".{clean}"
        if clean not in extensions:
            extensions.append(clean)
    return tuple(extensions)


def backup_caption_sidecars(
    source_dir: Path,
    backup_dir: Path,
    *,
    recursive: bool = False,
    caption_extension: str | None = ".txt",
    warn: WarnFn | None = None,
) -> CaptionBackupStats:
    """Copy same-stem caption sidecars into a non-training backup directory.

    The backup mirrors the source dataset's relative layout and only copies
    sidecars for discovered images, so orphaned tag files stay out of the run
    archive. Copy failures are reported through ``warn`` and counted, but never
    raised: this is an audit trail, not training input.
    """
    stats = CaptionBackupStats()
    if not source_dir.is_dir():
        if warn is not None:
            warn(f"caption backup skipped: source directory does not exist: {source_dir}")
        return stats

    try:
        image_paths = walk_images(source_dir, recursive=recursive)
    except Exception as e:  # noqa: BLE001
        if warn is not None:
            warn(f"caption backup skipped: could not scan {source_dir}: {e}")
        stats.failed += 1
        return stats

    stats.images_seen = len(image_paths)
    extensions = caption_backup_extensions(caption_extension)
    for image_path in image_paths:
        try:
            rel_image = image_path.relative_to(source_dir)
        except ValueError:
            rel_image = Path(image_path.name)
        found_for_image = False
        for ext in extensions:
            sidecar = image_path.with_suffix(ext)
            if not sidecar.is_file():
                continue
            found_for_image = True
            target = backup_dir / rel_image.with_suffix(ext)
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sidecar, target)
                stats.copied += 1
            except OSError as e:
                stats.failed += 1
                if warn is not None:
                    warn(f"caption backup warning: could not copy {sidecar}: {e}")
        if not found_for_image:
            stats.missing += 1
    return stats


def _shuffle_and_drop(tokens: list[str], tag_dropout_rate: float) -> list[str]:
    shuffled = list(tokens)
    random.shuffle(shuffled)
    if tag_dropout_rate <= 0.0:
        return shuffled
    return [tag for tag in shuffled if random.random() >= tag_dropout_rate]
