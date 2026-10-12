"""Committed, content-verified recovery snapshots (single process only)."""
import json
import os
from pathlib import Path
import torch
from .resume_data import capture_loader, restore_loader
from .resume_identity import hash_file
from .resume_rng import capture_rng, validate_rng, restore_rng


SCHEMA = "training_recovery_snapshot_v1"


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding="utf-8")


def _sync_snapshot(directory):
    for file in directory.iterdir():
        with file.open("rb") as stream:
            os.fsync(stream.fileno())
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _owned(path, owner_root):
    raw, owner = Path(path), Path(owner_root)
    if ".." in raw.parts or ".." in owner.parts:
        raise ValueError("Recovery path contains parent traversal")
    for value in (raw.absolute(), owner.absolute()):
        for candidate in (value, *value.parents):
            if candidate.is_symlink():
                raise ValueError("Recovery path contains symlink")
    try:
        path, root = raw.resolve(strict=True), owner.resolve(strict=True)
    except OSError as exc:
        raise ValueError("Recovery owner or path is missing") from exc
    if path == root or not path.is_relative_to(root) or path.name.startswith("."):
        raise ValueError("Recovery path is external or partial")
    if not path.is_dir() or not root.is_dir():
        raise ValueError("Recovery path is not a directory")
    return path


def _counters(counters, step):
    if not isinstance(counters, dict) or not counters:
        raise ValueError("Missing recovery counters")
    if any(type(v) is not int or v < 0 for v in counters.values()):
        raise ValueError("Invalid recovery counters")
    if counters.get("global_step") != step or counters.get("completed_steps") != step:
        raise ValueError("Recovery counter step differs")
    names = {"attempted_steps", "effective_steps", "skipped_steps"}
    if names.intersection(counters) and (not names.issubset(counters) or
            counters["attempted_steps"] != counters["effective_steps"] + counters["skipped_steps"] or
            counters["attempted_steps"] != step):
        raise ValueError("Recovery native counters differ")


def _precision(value, contract, manifest, counters):
    required = {"mode", "base", "execution"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Invalid recovery precision metadata")
    if value["mode"] not in {"native_fp16", "fp16_fp32"}:
        raise ValueError("Unsupported recovery precision mode")
    if value["mode"] == "native_fp16" and not {
            "attempted_steps", "effective_steps", "skipped_steps"}.issubset(counters):
        raise ValueError("Missing native recovery counters")
    if value["mode"] == "native_fp16":
        from .native_manifest import SCHEMA as native_schema, fingerprint
        base, execution = value["base"], value["execution"]
        if (not isinstance(base, dict) or not base or not isinstance(execution, dict)
                or execution.get("native_amp") is not True
                or not isinstance(execution.get("scaler_policy"), dict)
                or not isinstance(execution.get("trainables"), dict)
                or not isinstance(execution.get("lora_fp32_compute"), list)
                or fingerprint({"schema": native_schema, "base": base}) != manifest):
            raise ValueError("Invalid native recovery precision guards or manifest")
    elif value["base"] is not None or value["execution"] is not None:
        raise ValueError("Unexpected generic recovery precision guards")
    return {**value, "schema": "recovery_precision_v1",
            "precision_contract_id": contract, "precision_manifest_id": manifest}



def save_recovery_state(state, directory, *, precision_contract_id, precision_manifest_id,
                        counters, precision_metadata=None):
    accelerator = state.accelerator
    if accelerator.num_processes != 1 or accelerator.gradient_accumulation_steps != 1:
        raise ValueError("Recovery supports single process accumulation=1 only")
    loader = getattr(state, "train_dataloader", getattr(state, "dataloader", None))
    if loader is None or getattr(loader.dataset, "data_identity", None) is None:
        raise ValueError("Recovery loader requires attached data identity")
    step = state.global_step
    _counters(counters, step)
    directory = Path(directory).absolute()
    directory.mkdir(parents=True, exist_ok=True)
    target, temporary = directory / f"recovery-{step:08d}", directory / f".recovery-{step:08d}.partial"
    if target.exists() or temporary.exists():
        raise FileExistsError("Recovery snapshot already exists")
    rng = capture_rng()
    cursor = capture_loader(loader)
    temporary.mkdir()
    try:
        accelerator.save_state(str(temporary), safe_serialization=True)
        torch.save(cursor, temporary / "cursor.pt")
        torch.save(rng, temporary / "strict_rng.pt")
        _write_json(temporary / "precision.json", _precision(precision_metadata, precision_contract_id,
                    precision_manifest_id, counters))
        required = ["model.safetensors", "optimizer.bin", "scheduler.bin", "train_state.json",
                    "cursor.pt", "strict_rng.pt", "precision.json"]
        if accelerator.scaler is not None:
            required.append("scaler.pt")
        progress = json.loads((temporary / "train_state.json").read_text())
        if progress.get("current_step") != step:
            raise ValueError("Training hook step differs from recovery step")
        files = {}
        for file in temporary.iterdir():
            if file.is_symlink() or not file.is_file():
                raise ValueError("Unexpected recovery entry")
            files[file.name] = hash_file(file)
        if any(name not in files or not files[name]["size"] for name in required):
            raise ValueError("Incomplete recovery snapshot")
        metadata = {"schema": SCHEMA, "committed": True, "global_step": step,
                    "precision_contract_id": precision_contract_id, "precision_manifest_id": precision_manifest_id,
                    "counters": counters, "data_identity": cursor["data_identity"],
                    "scaler_required": accelerator.scaler is not None, "required": required, "files": files}
        _write_json(temporary / "snapshot.json", metadata)
        _sync_snapshot(temporary)
        temporary.rename(target)
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        restore_rng(rng)
    return target


def validate_recovery_state(path, *, owner_root, precision_contract_id,
                            precision_manifest_id=None, data_identity=None):
    path = _owned(path, owner_root)
    try:
        manifest = path / "snapshot.json"
        if manifest.is_symlink() or not manifest.is_file():
            raise ValueError("Missing committed recovery manifest")
        metadata = json.loads(manifest.read_text())
        if metadata.get("schema") != SCHEMA or metadata.get("committed") is not True:
            raise ValueError("Invalid recovery manifest")
        step = metadata["global_step"]
        if type(step) is not int or step < 0:
            raise ValueError("Invalid recovery step")
        _counters(metadata["counters"], step)
        if metadata["precision_contract_id"] != precision_contract_id:
            raise ValueError("Precision contract changed")
        if precision_manifest_id is not None and metadata["precision_manifest_id"] != precision_manifest_id:
            raise ValueError("Precision manifest changed")
        if data_identity is not None and metadata["data_identity"] != data_identity:
            raise ValueError("Data identity changed")
        required = {"model.safetensors", "optimizer.bin", "scheduler.bin", "train_state.json",
                    "cursor.pt", "strict_rng.pt", "precision.json"}
        if type(metadata.get("scaler_required")) is not bool:
            raise ValueError("Invalid scaler requirement")
        if metadata["scaler_required"]:
            required.add("scaler.pt")
        if not required.issubset(metadata["required"]) or not set(metadata["required"]).issubset(metadata["files"]):
            raise ValueError("Incomplete recovery inventory")
        if {p.name for p in path.iterdir()} != set(metadata["files"]) | {"snapshot.json"}:
            raise ValueError("Recovery file inventory changed")
        for name, identity in metadata["files"].items():
            if Path(name).name != name or hash_file(path / name) != identity or identity["size"] <= 0:
                raise ValueError(f"Recovery file hash differs: {name}")
        if json.loads((path / "train_state.json").read_text()).get("current_step") != step:
            raise ValueError("Recovery train step differs")
        precision = json.loads((path / "precision.json").read_text())
        binding = {k: precision[k] for k in ("mode", "base", "execution")}
        if precision != _precision(binding, metadata["precision_contract_id"],
                                  metadata["precision_manifest_id"], metadata["counters"]):
            raise ValueError("Recovery precision binding differs")
        cursor = torch.load(path / "cursor.pt", weights_only=False, map_location="cpu")
        validate_cursor(cursor, metadata["data_identity"])
        if cursor["loader"]["dataset_state"]["fetched"] != step:
            raise ValueError("Recovery cursor step differs")
        validate_rng(torch.load(path / "strict_rng.pt", weights_only=False, map_location="cpu"))
        return metadata
    except (KeyError, TypeError, OSError, RuntimeError, EOFError, AttributeError, IndexError) as exc:
        raise ValueError("Invalid recovery snapshot") from exc


def validate_cursor(cursor, identity):
    from .resume_data import validate_loader_snapshot
    validate_loader_snapshot(cursor)
    if cursor.get("data_identity") != identity or identity is None:
        raise ValueError("Cursor data identity differs")


def restore_recovery_auxiliary(accelerator, loader, path, *, owner_root,
                               precision_contract_id, precision_manifest_id=None, data_identity=None):
    if accelerator.num_processes != 1 or accelerator.gradient_accumulation_steps != 1:
        raise ValueError("Recovery requires single process accumulation=1")
    identity = data_identity or getattr(loader.dataset, "data_identity", None)
    metadata = validate_recovery_state(path, owner_root=owner_root, precision_contract_id=precision_contract_id,
                                       precision_manifest_id=precision_manifest_id, data_identity=identity)
    restore_loader(loader, torch.load(Path(path) / "cursor.pt", weights_only=False, map_location="cpu"))
    restore_rng(torch.load(Path(path) / "strict_rng.pt", weights_only=False, map_location="cpu"))
    return metadata
