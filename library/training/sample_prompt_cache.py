"""Persistent text-encoder outputs for training sample prompts."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

from library import train_util
from library.env import resolve_model_family, resolve_under_home
from library.models.family_registry import dispatch_model_family

logger = logging.getLogger(__name__)

_CACHE_SCHEMA = "1"
_DEFAULT_CACHE_DIR = "post_image_dataset/sample-prompt-te"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _path_fingerprint(value: object) -> dict[str, Any]:
    raw = str(value or "").strip()
    if not raw:
        return {"path": "", "exists": False}
    path = resolve_under_home(raw).resolve()
    result: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    if path.is_file():
        stat = path.stat()
        result.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        return result

    files = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        stat = child.stat()
        files.append(
            (str(child.relative_to(path)), int(stat.st_size), int(stat.st_mtime_ns))
        )
    result["files"] = files
    return result


def _krea2_encoding_profile(args) -> dict[str, Any]:
    from library.models.krea2_raw.strategy import (
        KREA2_MAX_LENGTH,
        KREA2_PAD_LENGTH,
        KREA2_PREFIX_IDX,
        KREA2_SELECT_LAYERS,
        KREA2_TE_BUNDLED_CONFIG_DIR,
    )

    return dict(
        max_length=KREA2_MAX_LENGTH,
        pad_length=KREA2_PAD_LENGTH,
        prefix_idx=KREA2_PREFIX_IDX,
        select_layers=list(KREA2_SELECT_LAYERS),
        tokenizer=_path_fingerprint(KREA2_TE_BUNDLED_CONFIG_DIR),
    )


def _anima_encoding_profile(args) -> dict[str, Any]:
    return dict(
        qwen3_max_token_length=getattr(args, "qwen3_max_token_length", None),
        t5_max_token_length=getattr(args, "t5_max_token_length", None),
        t5_tokenizer=_path_fingerprint(getattr(args, "t5_tokenizer_path", None)),
    )


def _z_image_encoding_profile(args) -> dict[str, Any]:
    from library.models.z_image.strategy import Z_IMAGE_MAX_LENGTH
    from library.models.z_image.weights import resolve_z_image_tokenizer_path

    return dict(
        max_length=Z_IMAGE_MAX_LENGTH,
        hidden_layer=-2,
        tokenizer=_path_fingerprint(
            resolve_z_image_tokenizer_path(getattr(args, "qwen3", ""))
        ),
    )


def _encoding_profile(args, family: str) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "family": family,
        "mixed_precision": str(getattr(args, "mixed_precision", "") or ""),
        "qwen3": _path_fingerprint(getattr(args, "qwen3", None)),
    }
    handler = dispatch_model_family(
        family,
        operation="sample prompt text-cache profile",
        handlers={
            "anima": _anima_encoding_profile,
            "krea2_raw": _krea2_encoding_profile,
            "z_image": _z_image_encoding_profile,
        },
    )
    profile.update(handler(args))
    return profile


def _prompt_values(prompts: Iterable[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        for value in (prompt.get("prompt", ""), prompt.get("negative_prompt", "")):
            text = str(value or "")
            if text not in seen:
                seen.add(text)
                values.append(text)
    return values


def _cache_context(args, prompts: list[dict[str, Any]], cache_root=None):
    family = resolve_model_family(args)
    source = resolve_under_home(str(args.sample_prompts)).resolve()
    profile = _encoding_profile(args, family)
    values = _prompt_values(prompts)
    signature = hashlib.sha256(
        _json(
            {
                "schema": _CACHE_SCHEMA,
                "profile": profile,
                "prompt_values": values,
            }
        ).encode("utf-8")
    ).hexdigest()
    cache_key = hashlib.sha256(
        _json(
            {
                "family": family,
                "source": str(source),
                "qwen3": profile["qwen3"].get("path", ""),
            }
        ).encode("utf-8")
    ).hexdigest()[:24]
    root = (
        Path(cache_root)
        if cache_root is not None
        else resolve_under_home(_DEFAULT_CACHE_DIR)
    )
    return root / f"{cache_key}.safetensors", family, signature, values


def load_sample_prompt_cache(args, *, cache_root=None):
    """Return ``(prompt snapshot, encoded outputs)`` on a valid cache hit."""
    prompts = [dict(prompt) for prompt in train_util.load_prompts(args.sample_prompts)]
    path, family, signature, values = _cache_context(args, prompts, cache_root)
    if not path.is_file():
        return None
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if (
                metadata.get("schema") != _CACHE_SCHEMA
                or metadata.get("family") != family
                or metadata.get("signature") != signature
            ):
                logger.info("Sample-prompt TE cache invalidated: %s", path)
                return None
            arities = json.loads(metadata.get("arities", "[]"))
            if len(arities) != len(values):
                return None
            outputs = {}
            for index, (prompt, arity) in enumerate(zip(values, arities)):
                outputs[prompt] = tuple(
                    handle.get_tensor(f"p{index}_t{tensor_index}")
                    for tensor_index in range(int(arity))
                )
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        SafetensorError,
    ) as exc:
        logger.warning("Ignoring unreadable sample-prompt TE cache %s: %s", path, exc)
        return None
    logger.info("Sample-prompt TE cache hit: %s", path)
    return prompts, outputs


def restore_sample_prompt_cache(trainer, args, *, cache_root=None) -> bool:
    cached = load_sample_prompt_cache(args, cache_root=cache_root)
    if cached is None:
        return False
    trainer.sample_prompts_snapshot, trainer.sample_prompts_te_outputs = cached
    return True


def save_sample_prompt_cache(
    args,
    prompts: list[dict[str, Any]],
    outputs: dict[str, Iterable[torch.Tensor]],
    *,
    cache_root=None,
) -> Path:
    """Atomically persist prompt encodings and return the cache path."""
    path, family, signature, values = _cache_context(args, prompts, cache_root)
    tensors: dict[str, torch.Tensor] = {}
    arities: list[int] = []
    for index, prompt in enumerate(values):
        encoded = tuple(outputs[prompt])
        arities.append(len(encoded))
        for tensor_index, value in enumerate(encoded):
            tensor = torch.as_tensor(value).detach().to("cpu").contiguous().clone()
            tensors[f"p{index}_t{tensor_index}"] = tensor

    metadata = {
        "schema": _CACHE_SCHEMA,
        "family": family,
        "signature": signature,
        "arities": _json(arities),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        save_file(tensors, str(temp), metadata=metadata)
        os.replace(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
    logger.info("Saved sample-prompt TE cache: %s", path)
    return path
