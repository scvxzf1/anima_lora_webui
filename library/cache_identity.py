"""Cache-identity stamps: rebuild preprocess sidecars when their inputs change.

Why this exists
---------------
The preprocess sidecars (`{stem}_{WxH}_anima.npz` VAE latents,
`{stem}_anima_te.safetensors` text-encoder outputs) stay reusable only while
the frozen weights that produced them are unchanged. Before this module the
checks were:

* text caches: a `model_family` / `cache_schema` stamp plus the caption
  sidecar's mtime;
* latent caches: per-resolution key presence only.

Both miss a weight swap that keeps the configured path (captions keep their
mtime), and both miss a tokenizer change entirely. This module adds one
compact, versioned identity record per sidecar that names the weights actually
used, so swapping the text encoder, the VAE, the DiT backing the LLM adapter or
the tokenizer marks the affected sidecars stale and forces a rebuild.

Policy
------
=========================  ==================================================
recorded stamp             behaviour
=========================  ==================================================
absent (legacy sidecar)    reuse; log one warning per kind. With
                           `require_identity` it counts as stale.
present and matching       reuse.
present and different      stale -> re-encode.
=========================  ==================================================

`compare` is a *subset* check: the producer records everything it knows and a
consumer asserts only the keys it can derive from its own config. That keeps a
preprocess subprocess (which sees `--caption_shuffle_variants N`) and the
training strategy (which sees `use_shuffled_caption_variants`) from
disagreeing over keys neither side can compute identically.

Deliberately stdlib-only at import time so preprocess subprocesses and the
training stack share one implementation without importing torch.

Detection limits
----------------
Identity is path + size + mtime for files and a size/mtime inventory for
directories, mirroring `library.cache_pool.fingerprint`'s "light" mode. An
in-place weight swap that preserves both size and mtime is not detected; pass
`--overwrite` or use a fresh cache directory when a checkpoint's provenance is
uncertain.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1"

#: safetensors metadata keys.
METADATA_KEY = "cache_identity"
METADATA_SCHEMA_KEY = "cache_identity_schema"

#: `.npz` array key (0-d unicode array holding the canonical JSON).
NPZ_KEY = "cache_identity"

#: Sentinels that keep "unset config" comparable on both sides of a cache.
DEFAULT_KEY = "<default>"
UNUSED_KEY = "<unused>"

_MAX_DIR_ENTRIES = 4096

#: Config strings and torch dtypes spell the same precision differently.
_DTYPE_ALIASES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp16": "float16",
    "half": "float16",
    "float16": "float16",
    "fp32": "float32",
    "float": "float32",
    "float32": "float32",
}

_warned: set[str] = set()


# --- primitives -----------------------------------------------------------


def _raw(value: Any) -> str:
    text = "" if value is None else str(value).strip()
    return text or DEFAULT_KEY


def normalize_dtype(value: Any) -> str:
    """Collapse `torch.bfloat16` / `"bfloat16"` / `"BF16"` to one spelling."""
    if value is None:
        return ""
    name = getattr(value, "__name__", None)
    text = str(name if name else value).strip()
    if text.startswith("torch."):
        text = text[len("torch.") :]
    text = text.lower()
    return _DTYPE_ALIASES.get(text, text)


def _dir_entries(path: Path, max_entries: int) -> tuple[list[list[Any]], bool]:
    """Bounded, sorted `(relpath, size, mtime_ns)` inventory of files under `path`."""
    entries: list[list[Any]] = []
    truncated = False
    for root, dirs, files in os.walk(path):
        dirs.sort()
        base = Path(root)
        for name in sorted(files):
            child = base / name
            try:
                stat = child.stat()
            except OSError:
                continue
            entries.append(
                [
                    child.relative_to(path).as_posix(),
                    int(stat.st_size),
                    int(stat.st_mtime_ns),
                ]
            )
            if len(entries) >= max_entries:
                truncated = True
                return entries, truncated
    return entries, truncated


def path_identity(value: Any, *, max_entries: int = _MAX_DIR_ENTRIES) -> dict[str, Any]:
    """Identity of one weight path (single file or Diffusers-style directory).

    A missing path is recorded (`exists: False`) instead of raising: consumers
    compare dicts, and a vanished path is a mismatch, which is the rebuild signal
    we want. Sizes/mtimes are stat-only - no file contents are hashed.
    """
    text = "" if value is None else str(value).strip()
    if not text:
        return {"path": "", "exists": False}

    path = Path(os.path.abspath(os.path.expanduser(text)))
    record: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not record["exists"]:
        return record

    try:
        if path.is_file():
            stat = path.stat()
            record["size"] = int(stat.st_size)
            record["mtime_ns"] = int(stat.st_mtime_ns)
            return record
        entries, truncated = _dir_entries(path, max_entries)
        record["files"] = entries
        if truncated:
            record["truncated"] = True
    except OSError as exc:  # unreadable path is a mismatch, not a crash
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def vocab_pack_identity(value: Any, *, max_entries: int = _MAX_DIR_ENTRIES) -> dict[str, Any]:
    """Content-address one configured vocab pack without trusting its mtime.

    A pack may be a manifest file or a tokenizer-resource directory. Its digest
    covers sorted relative paths and file bytes, so an in-place edit that keeps
    size and mtime cannot silently reuse an Anima text cache.
    """
    text = "" if value is None else str(value).strip()
    if not text:
        return {"path": "", "exists": False}

    path = Path(os.path.abspath(os.path.expanduser(text)))
    record: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not record["exists"]:
        return record

    try:
        files = [path] if path.is_file() else sorted(
            child for child in path.rglob("*") if child.is_file()
        )
        if len(files) > max_entries:
            record["error"] = f"too many files (limit {max_entries})"
            return record
        digest = hashlib.sha256()
        for child in files:
            relative = child.name if path.is_file() else child.relative_to(path).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            with child.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        record.update(files=len(files), sha256=digest.hexdigest())
    except OSError as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def canonical(identity: Mapping[str, Any]) -> str:
    """Deterministic JSON for an identity dict (used for stamps and npz keys)."""
    return json.dumps(
        dict(identity), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


# --- identity builders ----------------------------------------------------


def text_cache_identity(
    *,
    family: str,
    text_encoder: Any = None,
    tokenizer: Any = None,
    t5_tokenizer: Any = None,
    vocab_pack: Any = None,
    dit: Any = None,
    llm_adapter_outputs: bool = False,
    diff_output_preservation_trigger: Any = None,
    diff_output_preservation_class: Any = None,
    compute_dtype: Any = None,
    caption_shuffle_variants: Any = None,
    tag_dropout_rate: Any = None,
) -> dict[str, Any]:
    """Identity inputs for a text-encoder output cache.

    `dit` is only asserted when LLM-adapter outputs are being cached: the
    adapter lives inside the DiT checkpoint, so a DiT swap only invalidates a
    text cache that carries `crossattn_emb`. Otherwise the DiT is recorded as
    :data:`UNUSED_KEY` so plain TE caches survive a base-model swap.

    `caption_shuffle_variants` / `tag_dropout_rate` / `compute_dtype` are
    producer-side provenance only; consumers that cannot derive them identically
    simply omit those keys (`compare` checks a subset).
    """
    identity: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "stage": "text-encoder",
        "family": str(family),
        "text_encoder": path_identity(text_encoder),
        "tokenizer": path_identity(tokenizer if tokenizer else text_encoder),
        "t5_tokenizer": _raw(t5_tokenizer),
        "vocab_pack": vocab_pack_identity(vocab_pack) if vocab_pack else UNUSED_KEY,
        "diff_output_preservation_trigger": ""
        if not diff_output_preservation_trigger
        else str(diff_output_preservation_trigger),
        "diff_output_preservation_class": ""
        if not diff_output_preservation_class
        else str(diff_output_preservation_class),
        "dit": path_identity(dit) if llm_adapter_outputs else UNUSED_KEY,
    }
    if compute_dtype is not None:
        identity["compute_dtype"] = normalize_dtype(compute_dtype)
    if caption_shuffle_variants is not None:
        identity["caption_shuffle_variants"] = int(caption_shuffle_variants)
    if tag_dropout_rate is not None:
        identity["tag_dropout_rate"] = f"{float(tag_dropout_rate):.6g}"
    return identity


def latent_cache_identity(*, family: str, vae: Any = None) -> dict[str, Any]:
    """Identity inputs for a VAE latent cache.

    Family doubles as the latent-space discriminator (the registry maps family to
    latent space 1:1), so an Anima latent is never compared against a Krea-2 one.
    """
    return {
        "schema": SCHEMA_VERSION,
        "stage": "vae-latents",
        "family": str(family),
        "vae": path_identity(vae),
    }


def dit_cache_identity(*, family: str, dit: Any = None) -> dict[str, Any]:
    """Identity inputs for artifacts that depend on the DiT weights themselves."""
    return {
        "schema": SCHEMA_VERSION,
        "stage": "dit",
        "family": str(family),
        "dit": path_identity(dit),
    }


# --- stamps and comparison ------------------------------------------------


def stamp(
    metadata: Mapping[str, Any] | None, identity: Mapping[str, Any] | None
) -> dict[str, str]:
    """Merge an identity record into a safetensors metadata dict."""
    out = {str(key): str(value) for key, value in dict(metadata or {}).items()}
    if identity is None:
        return out
    out[METADATA_KEY] = canonical(identity)
    out[METADATA_SCHEMA_KEY] = SCHEMA_VERSION
    return out


def read_stamp(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Parse the identity record out of a safetensors metadata dict."""
    raw = (metadata or {}).get(METADATA_KEY)
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def encode_npz(identity: Mapping[str, Any]):
    """`numpy` 0-d array holding the canonical identity JSON."""
    import numpy as np

    return np.array(canonical(identity))


def decode_npz(value: Any) -> dict[str, Any] | None:
    """Parse an `.npz` identity array (tolerates bytes/str/scalar arrays)."""
    text = value
    item = getattr(text, "item", None)
    if callable(item):
        try:
            text = item()
        except ValueError:  # non-scalar array: not our stamp
            return None
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    try:
        parsed = json.loads(str(text))
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def read_npz_stamp(npz: Any) -> dict[str, Any] | None:
    """Parse the identity record out of an open `NpzFile`."""
    keys = getattr(npz, "files", None)
    if keys is None or NPZ_KEY not in keys:
        return None
    try:
        return decode_npz(npz[NPZ_KEY])
    except (KeyError, OSError, ValueError):
        return None


def compare(
    recorded: Mapping[str, Any] | None, expected: Mapping[str, Any] | None
) -> str:
    """`"absent"` / `"match"` / `"mismatch"` for a recorded vs expected pair.

    Subset semantics: every key in `expected` must be present and equal in
    `recorded`; extra recorded keys are provenance and are ignored.
    """
    if not recorded:
        return "absent"
    for key, value in dict(expected or {}).items():
        if recorded.get(key) != value:
            return "mismatch"
    return "match"


def evaluate(
    recorded: Mapping[str, Any] | None,
    expected: Mapping[str, Any] | None,
    *,
    require_identity: bool = False,
) -> str:
    """Collapse :func:`compare` plus the legacy policy to `"reuse"`/`"stale"`."""
    state = compare(recorded, expected)
    if state == "match":
        return "reuse"
    if state == "absent":
        return "stale" if require_identity else "reuse"
    return "stale"


def warn_legacy(kind: str, path: Any) -> None:
    """Warn once per cache kind about an unstamped (legacy) sidecar."""
    label = str(kind)
    if label in _warned:
        return
    _warned.add(label)
    logger.warning(
        "Reusing an unstamped %s cache (e.g. %s): its text-encoder / VAE / "
        "tokenizer provenance is unknown. Re-run preprocessing (or pass "
        "--require_cache_identity, or --overwrite) to stamp and rebuild.",
        label,
        path,
    )


def reset_warnings() -> None:
    """Clear the warn-once latch (tests)."""
    _warned.clear()
