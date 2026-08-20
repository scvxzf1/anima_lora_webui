"""Header-only Anima checkpoint layout and identity inspection."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from library.io.safetensors_io import (
    MemoryEfficientSafeOpen,
    get_split_weight_filenames,
)

_KEY_PREFIXES = ("net.", "model.diffusion_model.", "")
_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.(.+)$")
_SUPPORTED_BLOCK_COUNTS = (28, 40)


@dataclass(frozen=True)
class AnimaCheckpointLayout:
    arch: str
    variant: str
    num_blocks: int
    model_channels: int
    num_heads: int
    key_prefix: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_checkpoint_files(
    path_or_files: str | os.PathLike | Sequence[str | os.PathLike],
) -> tuple[Path, ...]:
    raw = (
        [path_or_files]
        if isinstance(path_or_files, (str, os.PathLike))
        else list(path_or_files)
    )
    if not raw:
        raise ValueError("Anima checkpoint path list is empty")

    files: list[Path] = []
    seen: set[Path] = set()
    for item in raw:
        path = Path(item).expanduser().resolve()
        split = get_split_weight_filenames(str(path))
        for candidate in map(Path, split or [path]):
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            if not candidate.is_file():
                raise FileNotFoundError(f"Anima checkpoint file not found: {candidate}")
            if candidate.suffix.lower() != ".safetensors":
                raise ValueError(f"Anima checkpoint must use safetensors: {candidate}")
            files.append(candidate)
            seen.add(candidate)
    return tuple(files)


def _read_key_shapes(files: tuple[Path, ...]) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for path in files:
        with MemoryEfficientSafeOpen(str(path)) as handle:
            for key in handle.keys():
                if key in shapes:
                    raise ValueError(
                        f"Duplicate tensor key across Anima checkpoint shards: {key}"
                    )
                raw_shape = (handle.header.get(key) or {}).get("shape")
                if not isinstance(raw_shape, list) or not all(
                    isinstance(dim, int) and dim >= 0 for dim in raw_shape
                ):
                    raise ValueError(f"Invalid safetensors shape for {key!r} in {path}")
                shapes[key] = tuple(raw_shape)
    return shapes


def _resolve_block_prefix(keys: Sequence[str]) -> str:
    prefixes = {
        prefix
        for key in keys
        for prefix in _KEY_PREFIXES
        if key.startswith(prefix) and _BLOCK_RE.match(key[len(prefix) :])
    }
    if not prefixes:
        raise ValueError("Checkpoint has no Anima blocks.N.* tensors")
    if len(prefixes) != 1:
        display = [prefix or "<none>" for prefix in sorted(prefixes)]
        raise ValueError(f"Checkpoint mixes Anima key prefixes: {display}")
    return next(iter(prefixes))


def _block_indices(shapes: dict[str, tuple[int, ...]], prefix: str) -> list[int]:
    indices = sorted(
        {
            int(match.group(1))
            for key in shapes
            if key.startswith(prefix)
            and (match := _BLOCK_RE.match(key[len(prefix) :])) is not None
        }
    )
    supported = [list(range(count)) for count in _SUPPORTED_BLOCK_COUNTS]
    if indices not in supported:
        found = f"{indices[:4]}...{indices[-4:]}" if indices else "[]"
        raise ValueError(
            "Unsupported or incomplete Anima block layout: expected exactly "
            f"0..27 or 0..39, found {found} ({len(indices)} blocks)"
        )
    return indices


def _derive_model_geometry(
    shapes: dict[str, tuple[int, ...]], prefix: str, indices: Sequence[int]
) -> tuple[int, int]:
    q_suffix = "self_attn.q_proj.weight"
    norm_suffix = "self_attn.q_norm.weight"
    for index in indices:
        q_key = f"{prefix}blocks.{index}.{q_suffix}"
        norm_key = f"{prefix}blocks.{index}.{norm_suffix}"
        if q_key not in shapes or norm_key not in shapes:
            raise ValueError(f"Anima block {index} is missing q_proj/q_norm tensors")
        q_shape = shapes[q_key]
        norm_shape = shapes[norm_key]
        if q_shape != (2048, 2048) or norm_shape != (128,):
            raise ValueError(
                f"Unsupported Anima block {index} geometry: "
                f"q_proj={q_shape}, q_norm={norm_shape}"
            )
    return 2048, 16


def inspect_anima_checkpoint(
    path_or_files: str | os.PathLike | Sequence[str | os.PathLike],
) -> AnimaCheckpointLayout:
    files = resolve_checkpoint_files(path_or_files)
    shapes = _read_key_shapes(files)
    prefix = _resolve_block_prefix(tuple(shapes))
    indices = _block_indices(shapes, prefix)
    model_channels, num_heads = _derive_model_geometry(shapes, prefix, indices)
    num_blocks = len(indices)
    return AnimaCheckpointLayout(
        arch=f"anima-{model_channels}-{num_blocks}",
        variant=("anima-2.9b-preview-v1" if num_blocks == 40 else "anima-28-block"),
        num_blocks=num_blocks,
        model_channels=model_channels,
        num_heads=num_heads,
        key_prefix=prefix,
    )


def anima_checkpoint_sha256(
    path_or_files: str | os.PathLike | Sequence[str | os.PathLike],
) -> str:
    files = resolve_checkpoint_files(path_or_files)
    digests: list[bytes] = []
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(16 * 1024 * 1024):
                digest.update(chunk)
        digests.append(digest.digest())
    if len(digests) == 1:
        return digests[0].hex()
    combined = hashlib.sha256()
    for digest in digests:
        combined.update(digest)
    return combined.hexdigest()


def apply_layout_to_args(
    args, layout: AnimaCheckpointLayout, base_sha256: str | None = None
) -> None:
    args.anima_arch = layout.arch
    args.anima_variant = layout.variant
    args.anima_num_blocks = layout.num_blocks
    args.anima_model_channels = layout.model_channels
    args.anima_num_heads = layout.num_heads
    args._anima_checkpoint_layout = layout
    if base_sha256:
        args.anima_base_sha256 = base_sha256
