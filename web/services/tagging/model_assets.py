"""Fixed manifest and safe local paths for optional captioner assets.

The manifest is intentionally code-owned.  Browser requests may select an
``asset_id`` from this list, but they can never provide a repository, URL, or
destination path.  Model files are user data and are therefore kept below the
runtime ``models/captioners`` directory rather than in Git.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import urlsplit

from library.env import anima_home

MANIFEST_VERSION = 1
CAPTIONER_MODELS_ROOT_ENV = "ANIMA_CAPTIONER_MODELS_ROOT"
HUGGINGFACE_TOKEN_ENV_NAMES = (
    "ANIMA_HUGGINGFACE_TOKEN",
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGINGFACE_TOKEN",
)
MAX_ASSET_FILE_BYTES = 8 * 1024 * 1024 * 1024

# The initial request is always made against Hugging Face.  Its resolver may
# redirect to one of the CDN/Xet hosts below; all of them remain HTTPS-only and
# are checked again by the downloader before every request.
ALLOWED_DOWNLOAD_HOSTS = frozenset(
    {
        "huggingface.co",
        "hf.co",
    }
)
ALLOWED_DOWNLOAD_HOST_SUFFIXES = (".hf.co",)
ALLOWED_REPOSITORIES = frozenset(
    {
        "SmilingWolf/wd-eva02-large-tagger-v3",
        "SmilingWolf/wd-vit-tagger-v3",
        "SmilingWolf/wd-vit-large-tagger-v3",
        "SmilingWolf/wd-v1-4-convnext-tagger-v2",
        "cella110n/cl_tagger",
        "cella110n/cl_tagger_v2",
    }
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _normalize_relative_path(value: Any, *, label: str) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{label} must be a relative path")
    normalized = path.as_posix()
    if normalized != raw or normalized.startswith("/"):
        raise ValueError(f"{label} is not normalized")
    return normalized


@dataclass(frozen=True, slots=True)
class ModelAssetFile:
    """One immutable file declaration in the model manifest."""

    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        normalized = _normalize_relative_path(self.path, label="manifest file")
        if normalized != self.path:
            object.__setattr__(self, "path", normalized)
        if not isinstance(self.size, int) or self.size <= 0 or self.size > MAX_ASSET_FILE_BYTES:
            raise ValueError(f"manifest file size invalid: {self.path}")
        digest = str(self.sha256 or "").strip().lower()
        if not _SHA256.fullmatch(digest):
            raise ValueError(f"manifest sha256 invalid: {self.path}")
        object.__setattr__(self, "sha256", digest)


@dataclass(frozen=True, slots=True)
class ModelAsset:
    """A provider-specific, version-pinned model bundle."""

    id: str
    provider: str
    label: str
    repo_id: str
    revision: str
    files: tuple[ModelAssetFile, ...]
    model_path: str
    metadata_paths: tuple[str, ...] = ()
    license: str = ""
    description: str = ""
    requires_auth: bool = False
    auth_hint: str = ""

    def __post_init__(self) -> None:
        if not _SAFE_ID.fullmatch(self.id):
            raise ValueError(f"manifest asset id invalid: {self.id}")
        if self.provider not in {"wd14", "cltagger"}:
            raise ValueError(f"manifest provider invalid: {self.provider}")
        if self.repo_id not in ALLOWED_REPOSITORIES:
            raise ValueError(f"manifest repository is not allowlisted: {self.repo_id}")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError(f"manifest revision must be a commit hash: {self.id}")
        if not self.files:
            raise ValueError(f"manifest asset has no files: {self.id}")
        paths = {item.path for item in self.files}
        for path in (self.model_path, *self.metadata_paths):
            if _normalize_relative_path(path, label="manifest entrypoint") not in paths:
                raise ValueError(f"manifest entrypoint is not declared: {self.id}/{path}")

    @property
    def total_size(self) -> int:
        return sum(item.size for item in self.files)

    def url_for(self, file: ModelAssetFile) -> str:
        """Build a URL from immutable repository/revision fields."""

        path = _normalize_relative_path(file.path, label="manifest file")
        return f"https://huggingface.co/{self.repo_id}/resolve/{self.revision}/{path}"


# Hashes for the LFS files are the repository's SHA-256 object IDs.  The
# manifest contains metadata needed by the providers, but never the model
# weights themselves.  CLTagger v2 is a gated repository; users must have
# accepted its terms and configured Hugging Face credentials before clicking
# its download action.
MODEL_ASSETS: tuple[ModelAsset, ...] = (
    ModelAsset(
        id="wd14-eva02-large-v3",
        provider="wd14",
        label="WD14 EVA02 Large Tagger v3",
        repo_id="SmilingWolf/wd-eva02-large-tagger-v3",
        revision="b25b82a03f7282e41aa2f257a52c7583b710bd1c",
        files=(
            ModelAssetFile(
                path="model.onnx",
                size=1_260_435_999,
                sha256="9e768793060c7939b277ccb382783e8670e8a042d29d77aa736be0c8cc898bfc",
            ),
            ModelAssetFile(
                path="selected_tags.csv",
                size=308_468,
                sha256="298633d94d0031d2081c0893f29c82eab7f0df00b08483ba8f29d1e979441217",
            ),
        ),
        model_path="model.onnx",
        metadata_paths=("selected_tags.csv",),
        license="Apache-2.0",
        description="SmilingWolf WD14 标签模型，输出通用与角色 tags。",
    ),
    ModelAsset(
        id="wd14-vit-v3",
        provider="wd14",
        label="WD14 ViT Tagger v3",
        repo_id="SmilingWolf/wd-vit-tagger-v3",
        revision="7f6b584d0bd3f55c4531f14ba3d4761b2bccdc0f",
        files=(
            ModelAssetFile(
                path="model.onnx",
                size=378_536_310,
                sha256="35f23693620b668f4d53fd3c62bf65e40af739bc52c7eb0fbc49258b58d065b6",
            ),
            ModelAssetFile(
                path="selected_tags.csv",
                size=308_468,
                sha256="298633d94d0031d2081c0893f29c82eab7f0df00b08483ba8f29d1e979441217",
            ),
        ),
        model_path="model.onnx",
        metadata_paths=("selected_tags.csv",),
        license="Apache-2.0",
        description="SmilingWolf WD14 ViT 标签模型 v3。",
    ),
    ModelAsset(
        id="wd14-vit-large-v3",
        provider="wd14",
        label="WD14 ViT Large Tagger v3",
        repo_id="SmilingWolf/wd-vit-large-tagger-v3",
        revision="ae469aa2e4706a3af08d3673cf73a11d1add314c",
        files=(
            ModelAssetFile(
                path="model.onnx",
                size=1_260_645_673,
                sha256="e4c8001b000a6c98f2db10794f7c406daa79873d071d6ca924330fa053fa1845",
            ),
            ModelAssetFile(
                path="selected_tags.csv",
                size=308_468,
                sha256="298633d94d0031d2081c0893f29c82eab7f0df00b08483ba8f29d1e979441217",
            ),
        ),
        model_path="model.onnx",
        metadata_paths=("selected_tags.csv",),
        license="Apache-2.0",
        description="SmilingWolf WD14 ViT Large 标签模型 v3，体积较大。",
    ),
    ModelAsset(
        id="wd14-convnext-v2",
        provider="wd14",
        label="WD14 ConvNeXt Tagger v2",
        repo_id="SmilingWolf/wd-v1-4-convnext-tagger-v2",
        revision="4b34d1b07bdd8e95494072648960b8a6adcbc0ff",
        files=(
            ModelAssetFile(
                path="model.onnx",
                size=387_820_405,
                sha256="71f06ecb7b9df81d8f271da4d43997ea2ed363cdac29aa64fcb256c9631e656a",
            ),
            ModelAssetFile(
                path="selected_tags.csv",
                size=253_906,
                sha256="8c8750600db36233a1b274ac88bd46289e588b338218c2e4c62bbc9f2b516368",
            ),
        ),
        model_path="model.onnx",
        metadata_paths=("selected_tags.csv",),
        license="Apache-2.0",
        description="SmilingWolf WD v1.4 ConvNeXt 标签模型 v2。",
    ),
    ModelAsset(
        id="cltagger-v1-02",
        provider="cltagger",
        label="CLTagger v1.02",
        repo_id="cella110n/cl_tagger",
        revision="0b6e9b4e145b1423bfd1715119074a24a301b471",
        files=(
            ModelAssetFile(
                path="cl_tagger_1_02/model.onnx",
                size=1_425_860_192,
                sha256="1459e946ffc083159015e0eb26239016349ddd36c3daecfeb70bfa4ccf38b944",
            ),
            ModelAssetFile(
                path="cl_tagger_1_02/tag_mapping.json",
                size=4_088_861,
                sha256="9611988482f1bf9a174622c0067d0baa88fda69c0a855cc500411426e77832ec",
            ),
        ),
        model_path="cl_tagger_1_02/model.onnx",
        metadata_paths=("cl_tagger_1_02/tag_mapping.json",),
        license="Apache-2.0",
        description="CLTagger v1 ONNX 模型与标签映射。",
    ),
    ModelAsset(
        id="cltagger-v2-01a",
        provider="cltagger",
        label="CLTagger v2.01a（SigLIP2）",
        repo_id="cella110n/cl_tagger_v2",
        revision="b57909b8e9c63f71e208a26473e7aabdf45ed6b6",
        files=(
            ModelAssetFile(
                path="v2_01a/model.onnx",
                size=791_773,
                sha256="12581711ccf803f914129b9e87932a4cf93c80c3382b7c63305e67afdcc2a02f",
            ),
            ModelAssetFile(
                path="v2_01a/model.onnx.data",
                size=2_211_645_300,
                sha256="d9f162b7c8127790879f17fb87bc643a1c803bc17bcce44ab021fca65b2dafff",
            ),
            ModelAssetFile(
                path="v2_01a/model_vocabulary.json",
                size=14_594_140,
                sha256="4966d2779825a8a4c4e46644fa8e2741824622929bdb21dbe4f9c18df2ebcf95",
            ),
        ),
        model_path="v2_01a/model.onnx",
        metadata_paths=("v2_01a/model_vocabulary.json",),
        license="CL Tagger v2 Model License v1.0",
        description="CLTagger v2.01a SigLIP2 ONNX；需 Hugging Face gated 仓库授权。",
        requires_auth=True,
        auth_hint="请先在 Hugging Face 接受模型条款并登录（hf auth login 或设置 HF_TOKEN）。",
    ),
)

_ASSETS_BY_ID = {asset.id: asset for asset in MODEL_ASSETS}
_ASSET_ALIASES = {
    # Target-project repository IDs and variant keys are accepted for profile
    # migration, then normalized to the immutable local asset IDs.
    "SmilingWolf/wd-eva02-large-tagger-v3": "wd14-eva02-large-v3",
    "SmilingWolf/wd-vit-tagger-v3": "wd14-vit-v3",
    "SmilingWolf/wd-vit-large-tagger-v3": "wd14-vit-large-v3",
    "SmilingWolf/wd-v1-4-convnext-tagger-v2": "wd14-convnext-v2",
    "cella110n/cl_tagger": "cltagger-v1-02",
    "cella110n/cl_tagger_v2": "cltagger-v2-01a",
    "cl_tagger_1_02": "cltagger-v1-02",
    "cl_tagger_v2_v2_01a": "cltagger-v2-01a",
    "cl_tagger_1_02/model.onnx": "cltagger-v1-02",
    "v2_01a/model.onnx": "cltagger-v2-01a",
}


def iter_model_assets() -> tuple[ModelAsset, ...]:
    """Return the immutable manifest entries in display order."""

    return MODEL_ASSETS


def get_model_asset(asset_id: str) -> ModelAsset:
    key = str(asset_id or "").strip()
    asset = _ASSETS_BY_ID.get(key) or _ASSETS_BY_ID.get(_ASSET_ALIASES.get(key, ""))
    if asset is None:
        raise KeyError(f"模型资产不存在：{key or '空值'}")
    return asset


def canonical_asset_id(asset_id: str) -> str:
    """Normalize known legacy/repository identifiers without accepting unknown IDs."""

    key = str(asset_id or "").strip()
    return _ASSET_ALIASES.get(key, key)


def captioner_models_root() -> Path:
    """Return the controlled root for optional model files.

    ``ANIMA_HOME`` remains the normal relocation mechanism.  The explicit
    override is useful for installations that keep large user data on another
    volume, but it is still normalized and cannot contain ``..`` segments.
    """

    raw = os.environ.get(CAPTIONER_MODELS_ROOT_ENV, "").strip()
    if raw:
        path = Path(raw).expanduser()
        if ".." in path.parts:
            raise ValueError(f"{CAPTIONER_MODELS_ROOT_ENV} cannot contain '..'")
        if not path.is_absolute():
            path = anima_home() / path
        return path.resolve()
    return (anima_home() / "models" / "captioners").resolve()


def asset_directory(asset: ModelAsset | str) -> Path:
    """Return the controlled install directory for one manifest asset."""

    entry = get_model_asset(asset) if isinstance(asset, str) else asset
    root = captioner_models_root()
    raw_path = root / entry.provider / entry.id
    _assert_no_symlink_chain(raw_path, root)
    path = raw_path.resolve()
    _assert_within(path, root)
    return path


def asset_file_path(asset: ModelAsset | str, file: ModelAssetFile | str) -> Path:
    """Resolve a manifest file and reject traversal/symlink escapes."""

    entry = get_model_asset(asset) if isinstance(asset, str) else asset
    declared = next((item for item in entry.files if item.path == (file.path if isinstance(file, ModelAssetFile) else str(file))), None)
    if declared is None:
        raise ValueError(f"文件不属于模型资产：{entry.id}")
    directory = asset_directory(entry)
    raw_path = directory / declared.path
    _assert_no_symlink_chain(raw_path, directory)
    path = raw_path.resolve()
    _assert_within(path, directory)
    return path


def public_asset(asset: ModelAsset, status: dict[str, Any] | None = None) -> dict[str, Any]:
    """Project a manifest entry without exposing absolute local paths."""

    files = status.get("files", []) if isinstance(status, dict) else []
    file_by_path = {str(item.get("path")): item for item in files if isinstance(item, dict)}
    return {
        "id": asset.id,
        "provider": asset.provider,
        "label": asset.label,
        "description": asset.description,
        "repo": asset.repo_id,
        # ``repo_id`` is the manifest-facing name.  Keep ``repo`` as a
        # compatibility alias for the existing Dragon page and older API
        # clients.
        "repo_id": asset.repo_id,
        "revision": asset.revision,
        "license": asset.license,
        "requires_auth": asset.requires_auth,
        "auth_configured": bool(_huggingface_token()) if asset.requires_auth else True,
        "auth_hint": asset.auth_hint,
        "directory": f"models/captioners/{asset.provider}/{asset.id}",
        "model_path": asset.model_path,
        "metadata_paths": list(asset.metadata_paths),
        "total_size": asset.total_size,
        "files": [_public_file(item, file_by_path.get(item.path, {})) for item in asset.files],
        "state": str((status or {}).get("state") or "missing"),
        "installed": bool((status or {}).get("installed")),
        "available": bool((status or {}).get("installed")),
        "bytes_present": int((status or {}).get("bytes_present", 0) or 0),
        **(
            {
                "download_id": str((status or {}).get("download_id") or ""),
                "download": (status or {}).get("download"),
            }
            if isinstance((status or {}).get("download"), dict)
            else {}
        ),
    }


def _public_file(declared: ModelAssetFile, status: dict[str, Any]) -> dict[str, Any]:
    """Merge runtime status without allowing it to override manifest fields."""

    return {
        "path": declared.path,
        "size": declared.size,
        "sha256": declared.sha256,
        "bytes": int(status.get("bytes", 0) or 0),
        "present": bool(status.get("present")),
        "size_ok": bool(status.get("size_ok")),
        "verified": bool(status.get("verified")),
        "valid": bool(status.get("valid")),
        "error": str(status.get("error") or "")[:200],
    }


def inspect_asset(asset: ModelAsset | str, *, verify_hash: bool = True) -> dict[str, Any]:
    """Inspect a local asset without creating directories or downloading files."""

    entry = get_model_asset(asset) if isinstance(asset, str) else asset
    records: list[dict[str, Any]] = []
    bytes_present = 0
    any_present = False
    any_invalid = False
    all_valid = True
    for declared in entry.files:
        try:
            path = asset_file_path(entry, declared)
            is_file = path.is_file() and not path.is_symlink()
            size = path.stat().st_size if is_file else 0
        except (OSError, ValueError):
            is_file = False
            size = 0
        present = bool(is_file)
        any_present = any_present or present
        size_ok = present and size == declared.size
        digest = ""
        hash_ok = False
        if size_ok and verify_hash:
            try:
                digest = sha256_file(path)
                hash_ok = digest == declared.sha256
            except OSError:
                hash_ok = False
        elif size_ok:
            # A lightweight status check can avoid hashing a multi-gigabyte
            # file; the explicit download/start path performs full verification.
            hash_ok = True
        valid = bool(size_ok and hash_ok)
        if present:
            bytes_present += min(size, declared.size)
        if present and not valid:
            any_invalid = True
        all_valid = all_valid and valid
        records.append(
            {
                "path": declared.path,
                "size": declared.size,
                "bytes": size,
                "present": present,
                "size_ok": size_ok,
                "verified": bool(verify_hash and hash_ok),
                "valid": valid,
                "error": "文件大小或 SHA256 不匹配" if present and not valid else "",
            }
        )
    if all_valid:
        state = "installed"
    elif any_invalid:
        state = "corrupt"
    elif any_present:
        state = "partial"
    else:
        state = "missing"
    return {
        "state": state,
        "installed": state == "installed",
        "files": records,
        "bytes_present": bytes_present,
        "total_size": entry.total_size,
    }


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file in bounded chunks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _assert_within(path: Path, root: Path) -> None:
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("模型路径越出受控目录") from exc


def _assert_no_symlink_chain(path: Path, root: Path) -> None:
    """Reject symlinked components before ``Path.resolve`` follows them."""

    root = root.resolve()
    candidate = Path(path)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("模型路径越出受控目录") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("模型目录包含不安全的符号链接")


def validate_download_url(url: str, *, allow_hosts: Iterable[str] | None = None) -> str:
    """Validate an HTTPS URL against the downloader host allowlist."""

    parsed = urlsplit(str(url or "").strip())
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError("模型下载只允许 HTTPS URL")
    if parsed.username or parsed.password or parsed.fragment:
        raise ValueError("模型下载 URL 不得包含凭据或 fragment")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("模型下载 URL 端口无效") from exc
    if port not in (None, 443):
        raise ValueError("模型下载 URL 只允许 HTTPS 默认端口")
    host = parsed.hostname.lower().rstrip(".")
    allowed = set(allow_hosts or ALLOWED_DOWNLOAD_HOSTS)
    if host not in allowed and not any(host.endswith(suffix) for suffix in ALLOWED_DOWNLOAD_HOST_SUFFIXES):
        raise ValueError("模型下载主机不在 allowlist 中")
    return parsed.geturl()


def _huggingface_token() -> str:
    """Resolve an optional HF read token without persisting or exposing it."""

    for name in HUGGINGFACE_TOKEN_ENV_NAMES:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    # ``hf auth login`` stores a token outside this project.  Import lazily so
    # the default WebUI and public assets do not require huggingface_hub.
    try:
        from huggingface_hub import get_token  # type: ignore[import-not-found]

        value = get_token()
    except (ImportError, OSError, RuntimeError):
        return ""
    return str(value or "").strip()


__all__ = [
    "ALLOWED_DOWNLOAD_HOSTS",
    "ALLOWED_REPOSITORIES",
    "CAPTIONER_MODELS_ROOT_ENV",
    "HUGGINGFACE_TOKEN_ENV_NAMES",
    "MANIFEST_VERSION",
    "MODEL_ASSETS",
    "ModelAsset",
    "ModelAssetFile",
    "asset_directory",
    "asset_file_path",
    "captioner_models_root",
    "canonical_asset_id",
    "get_model_asset",
    "inspect_asset",
    "iter_model_assets",
    "public_asset",
    "sha256_file",
    "validate_download_url",
]
