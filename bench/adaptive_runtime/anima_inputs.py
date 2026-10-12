"""Immutable real-cache Anima inputs for disposable precision probes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from library.models.family_registry import get_model_family_spec


SCHEMA = "adaptive_anima_inputs_v1"
MAX_BYTES = 128 * 1024**2
LATENT_KEY = re.compile(r"^latents_(\d+)x(\d+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_tensors(values):
    if set(values) != {"latents", "noise", "prompt"}:
        raise ValueError("Anima input keys must be latents/noise/prompt")
    latent, noise, prompt = (values[k] for k in ("latents", "noise", "prompt"))
    if (latent.ndim != 5 or latent.shape[:3] != (1, 16, 1)
            or latent.shape[3] < 4 or latent.shape[4] < 4
            or noise.shape != latent.shape or prompt.shape != (1, 512, 1024)):
        raise ValueError("Anima input shape mismatch; full 512-token prompt required")
    if sum(t.numel() * t.element_size() for t in values.values()) > MAX_BYTES:
        raise ValueError("Anima input byte budget exceeded")
    if not all(t.dtype == torch.float32 and bool(torch.isfinite(t).all()) for t in values.values()):
        raise ValueError("Anima input tensors must be finite FP32")


def read_source_caches(latent_cache: Path, text_cache: Path, latent_key: str | None = None):
    with np.load(latent_cache, allow_pickle=False) as source:
        candidates = [key for key in source.files if LATENT_KEY.fullmatch(key)]
        if latent_key is None:
            if len(candidates) != 1:
                raise ValueError("Expected exactly one latent key; specify --latent-key")
            latent_key = candidates[0]
        if latent_key not in candidates:
            raise ValueError("Selected latent key missing or malformed")
        match = LATENT_KEY.fullmatch(latent_key)
        raw = source[latent_key]
        if raw.shape != (16, int(match[1]), int(match[2])):
            raise ValueError("Latent key and tensor shape disagree")
        if raw.dtype.kind not in "f" or not np.isfinite(raw).all():
            raise ValueError("Latent cache must contain finite floating values")
        latent = torch.from_numpy(raw.astype(np.float32, copy=True))[None, :, None]
    spec = get_model_family_spec("anima")
    with safe_open(str(text_cache), framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
        if metadata.get("model_family") != spec.name or metadata.get("cache_schema") != spec.text_cache.schema:
            raise ValueError("Text cache family/schema mismatch")
        if "crossattn_emb" not in source.keys():
            raise ValueError("Anima text cache requires crossattn_emb")
        prompt = source.get_tensor("crossattn_emb")
    if prompt.shape == (512, 1024):
        prompt = prompt.unsqueeze(0)
    return latent, prompt.float().contiguous(), latent_key


def prepare_anima_inputs(latent_cache: Path, text_cache: Path, output: Path, *, seed=20260930,
                         latent_key=None):
    output = Path(output)
    manifest_path = output.with_suffix(".json")
    if output.exists() or manifest_path.exists():
        raise FileExistsError("Input package or manifest already exists")
    latent, prompt, key = read_source_caches(Path(latent_cache), Path(text_cache), latent_key)
    noise = torch.randn(latent.shape, generator=torch.Generator(device="cpu").manual_seed(seed))
    values = {"latents": latent.contiguous(), "noise": noise.contiguous(), "prompt": prompt}
    _validate_tensors(values)
    source = {"latent_cache": str(Path(latent_cache).resolve()),
              "latent_sha256": sha256_file(Path(latent_cache)),
              "text_cache": str(Path(text_cache).resolve()),
              "text_sha256": sha256_file(Path(text_cache)), "latent_key": key}
    metadata = {"schema": SCHEMA, "model_family": "anima", "seed": str(seed),
                **{k: v for k, v in source.items() if k.endswith("sha256") or k == "latent_key"}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        temporary = Path(directory) / "inputs.safetensors"
        save_file(values, str(temporary), metadata)
        os.link(temporary, output)
    report = {**metadata, **source, "status": "ok", "inputs_sha256": sha256_file(output),
              "shapes": {key: list(value.shape) for key, value in values.items()},
              "dtypes": {key: str(value.dtype) for key, value in values.items()}}
    try:
        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
    except Exception:
        output.unlink()
        raise
    return report


def load_anima_inputs(path, *, device, dtype):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES + 1024**2:
        raise ValueError("Anima input package byte budget exceeded")
    digest = sha256_file(path)
    with safe_open(str(path), framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
    if metadata.get("schema") != SCHEMA or metadata.get("model_family") != "anima":
        raise ValueError("Unknown Anima input package schema/family")
    report = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if (report.get("inputs_sha256") != digest or report.get("schema") != SCHEMA
            or any(report.get(k) != metadata.get(k) for k in
                   ("latent_sha256", "text_sha256", "latent_key", "seed"))):
        raise ValueError("Anima input manifest/hash mismatch")
    values = load_file(str(path), device="cpu")
    _validate_tensors(values)
    if report.get("shapes") != {k: list(v.shape) for k, v in values.items()}:
        raise ValueError("Anima input manifest shape mismatch")
    return {k: value.to(device=device, dtype=dtype) for k, value in values.items()}, digest
