"""Content identities computed once when attaching a standard cached dataset."""
import hashlib
import json
from pathlib import Path
import torch


def hash_file(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected regular identity file: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, k) != getattr(after, k) for k in fields):
        raise ValueError(f"Identity file changed while hashing: {path}")
    return {"size": after.st_size, "sha256": digest.hexdigest()}


def hash_data_file(path):
    """Bind a data alias to its resolved file; snapshot files remain symlink-free."""
    source = Path(path).absolute()
    before = source.lstat()
    resolved = source.resolve(strict=True)
    identity = hash_file(resolved)
    after = source.lstat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if (source.resolve(strict=True) != resolved
            or any(getattr(before, k) != getattr(after, k) for k in fields)):
        raise ValueError(f"Data alias changed while hashing: {source}")
    return {"resolved_path": str(resolved), **identity}


def _value(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _value(v) for k, v in value.items()}
    if isinstance(value, torch.Tensor):
        raise ValueError("Recovery requires disk cached data, not in-memory tensors")
    raise ValueError(f"Unsupported data identity value: {type(value).__name__}")


def build_data_identity(group, *, config=None):
    from library.datasets.group import DatasetGroup
    from library.datasets.dreambooth import DreamBoothDataset
    from library.datasets.subsets import ImageInfo
    if type(group) is not DatasetGroup or not group.datasets:
        raise ValueError("Recovery supports standard DatasetGroup only")
    result = {"schema": "recovery_data_identity_v1", "config": _value(config or {}), "datasets": []}
    for dataset in group.datasets:
        if type(dataset) is not DreamBoothDataset or dataset.caching_mode is not None:
            raise ValueError("Recovery rejects custom and stage datasets")
        if dataset.bucket_manager is None:
            raise ValueError("Recovery requires initialized buckets")
        settings = {k: _value(v) for k, v in vars(dataset).items()
                    if v is None or isinstance(v, (str, bool, int, float))}
        for k in ("current_epoch", "current_step", "_largest_bucket_index"):
            settings.pop(k, None)
        for name in ("replacements", "token_strings", "XTI_layers"):
            settings[name] = _value(getattr(dataset, name, None))
        entry = {"settings": settings, "subsets": [], "images": []}
        for subset in dataset.subsets:
            if type(subset).__module__ != "library.datasets.subsets":
                raise ValueError("Recovery rejects custom subsets")
            entry["subsets"].append({k: _value(v) for k, v in vars(subset).items()})
        for key, info in sorted(dataset.image_data.items()):
            if type(info) is not ImageInfo or not info.latents_npz or not info.text_encoder_outputs_npz:
                raise ValueError("Recovery requires standard disk latent and text caches")
            record = {"key": str(key), "fields": {}, "files": {}}
            for name, value in vars(info).items():
                record["fields"][name] = _value(value)
                if value and isinstance(value, str) and (name.endswith("_path") or name.endswith("_npz")):
                    record["files"][str(Path(value).absolute())] = hash_data_file(value)
            source = Path(info.absolute_path)
            for caption in (source.with_suffix(".txt"), source.with_suffix(".caption")):
                if caption.exists():
                    record["files"][str(caption.absolute())] = hash_data_file(caption)
            entry["images"].append(record)
        result["datasets"].append(entry)
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"schema": result["schema"], "sha256": hashlib.sha256(encoded).hexdigest()}


def attach_data_identity(loader, *, config=None):
    identity = build_data_identity(loader.dataset.group, config=config)
    loader.dataset.data_identity = identity
    return identity
