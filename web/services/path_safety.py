"""Shared path display/resolve and safetensors header helpers.

These helpers keep preview / weight analysis / continue-LoRA on one security
surface so path boundaries and safetensors reads do not silently diverge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import unquote, urlparse


def normalize_user_path_value(value: str) -> str:
    clean = str(value or "").strip().strip('"').strip("'").strip()
    if clean.startswith("file://"):
        parsed = urlparse(clean)
        clean = unquote(parsed.path or "")
    else:
        clean = unquote(clean)
    return clean.replace("\\", "/").strip()


def resolve_display_path(value: str, *, root: Path) -> Path | None:
    raw = normalize_user_path_value(value)
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (Path(root) / path).resolve()


def display_path(path: Path, *, root: Path) -> str:
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def display_project_path(value: str | Path, *, root: Path) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    if not raw:
        return ""
    path = Path(raw)
    if not path.is_absolute():
        return path.as_posix().strip("/")
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return raw


def path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def unique_resolved_dirs(paths: Iterable[Path | None]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if path is None:
            continue
        try:
            resolved = path.resolve()
        except OSError:
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def allowed_weight_dirs(
    *,
    root: Path,
    output_root: Path | None = None,
    task: Mapping[str, Any] | None = None,
    training_dirs: Iterable[str] = (),
) -> list[Path]:
    dirs: list[Path | None] = []
    if output_root is not None:
        dirs.append(Path(output_root))
    output_dir = str((task or {}).get("output_dir") or "").strip()
    if output_dir:
        dirs.append(resolve_display_path(output_dir, root=root))
    for training_dir in training_dirs:
        resolved = resolve_display_path(str(training_dir or ""), root=root)
        if resolved is None:
            continue
        dirs.append(resolved.parent if resolved.name == "sample" else resolved)
    return unique_resolved_dirs(dirs)


def is_under_allowed_dirs(path: Path, allowed_dirs: Iterable[Path]) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for allowed in allowed_dirs:
        try:
            resolved.relative_to(Path(allowed).resolve())
            return True
        except ValueError:
            continue
    return False


def read_safetensors_header_bytes(data: bytes) -> tuple[dict[str, str], list[str]]:
    """Parse safetensors header from in-memory bytes without loading tensors."""
    import json

    try:
        if len(data) < 8:
            raise ValueError("文件太小，缺少 safetensors header")
        header_len = int.from_bytes(data[:8], byteorder="little", signed=False)
        if header_len <= 0 or header_len > len(data) - 8:
            raise ValueError("safetensors header 长度不合法")
        raw_header = json.loads(data[8:8 + header_len].decode("utf-8"))
        if not isinstance(raw_header, dict):
            raise ValueError("safetensors header 不是 JSON object")
        metadata = raw_header.get("__metadata__")
        safe_metadata = (
            {str(k): str(v) for k, v in metadata.items()}
            if isinstance(metadata, dict)
            else {}
        )
        keys = [str(key) for key in raw_header.keys() if key != "__metadata__"]
        return safe_metadata, keys
    except Exception as exc:
        raise ValueError(f"读取 safetensors 上传文件失败: {exc}") from exc


def training_dirs_from_preview_settings(settings: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Collect configured/effective training dirs used by weight allowlists."""
    data = settings or {}
    return (
        str(data.get("training_dir") or ""),
        str(data.get("effective_training_dir") or ""),
    )


def read_safetensors_header(path: Path) -> tuple[dict[str, str], list[str]]:
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = {str(key): str(value) for key, value in (handle.metadata() or {}).items()}
            keys = list(handle.keys())
        return metadata, keys
    except Exception as exc:
        raise ValueError(f"读取 safetensors 权重失败: {exc}") from exc


def read_safetensors_metadata(path: Path) -> dict[str, str]:
    try:
        metadata, _keys = read_safetensors_header(path)
        return metadata
    except Exception:
        return {}


def contains_parent_ref(value: str | Path) -> bool:
    """Return True if any path part is ``..`` after normalization."""
    raw = normalize_user_path_value(str(value or ""))
    if not raw:
        return False
    return ".." in Path(raw).parts


def resolve_allowed_file(
    value: str,
    *,
    root: Path,
    allowed_dirs: Iterable[Path],
    require_suffix: str | None = ".safetensors",
) -> Path:
    """Resolve a user path and require it to live under ``allowed_dirs``.

    Relative paths are rooted at ``root`` and may not contain ``..``.
    Absolute paths must still fall under the allowlist.
    """
    raw = normalize_user_path_value(value)
    if not raw:
        raise ValueError("路径为空")
    if contains_parent_ref(raw):
        raise ValueError("路径不允许包含 ..")
    path = Path(raw)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (Path(root) / path).resolve()
        try:
            resolved.relative_to(Path(root).resolve())
        except ValueError as exc:
            raise ValueError("路径超出允许范围") from exc
    if require_suffix and resolved.suffix.lower() != require_suffix.lower():
        raise ValueError(f"只支持 {require_suffix} 文件")
    if not is_under_allowed_dirs(resolved, allowed_dirs):
        raise ValueError("路径超出允许范围")
    return resolved

