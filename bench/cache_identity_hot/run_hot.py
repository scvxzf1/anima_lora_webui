#!/usr/bin/env python
"""Hot test: model-identity driven latent cache rebuild (real VAE, real GPU).

Runs the real preprocess CLI (`scripts.preprocess.cache_latents`) against the
on-disk Qwen-Image VAE and a throwaway 2-image dataset, then asserts that a
changed VAE identity forces a re-encode while an unchanged one keeps skipping.

Stages (each is a real subprocess run; counts come from the CLI summary line):

1. cold_encode              write both latents
2. rerun_same_vae           skip both (identity match)
3. vae_path_changed         re-encode (configured VAE path is part of identity)
4. vae_mtime_changed        re-encode (in-place weight swap signal)
5. legacy_stamp_default     skip unstamped sidecars, warn once
6. legacy_stamp_strict      re-encode with --require_cache_identity
7. stale_resolution_dropped other-resolution keys from the old VAE are dropped

Usage::

    .venv/bin/python -m bench.cache_identity_hot.run_hot --gpu 0

Results land in `output/bench/cache_identity_hot/<timestamp>/`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from bench._common import REPO_ROOT, make_run_dir, write_result

PY = str(REPO_ROOT / ".venv" / "bin" / "python")
MODULE = "scripts.preprocess.cache_latents"
DEFAULT_VAE = "models/vae/qwen_image_vae.safetensors"
IMAGE_SIZES = ((512, 512), (640, 384))
SUMMARY_RE = re.compile(r"complete: (\d+) cached, (\d+) skipped")


def _make_dataset(root: Path) -> list[tuple[int, int]]:
    root.mkdir(parents=True, exist_ok=True)
    sizes = []
    for index, (w, h) in enumerate(IMAGE_SIZES):
        Image.new("RGB", (w, h), color=(30 * index + 40, 60, 90)).save(root / f"img{index}.png")
        sizes.append((w, h))
    return sizes


def _run_stage(name: str, *, vae: str, data: Path, cache: Path, gpu: int,
               dtype: str, strict: bool, run_dir: Path) -> dict:
    cmd = [
        PY, "-m", MODULE,
        "--dir", str(data), "--cache_dir", str(cache),
        "--vae", vae, "--model_family", "anima",
        "--batch_size", "1", "--dtype", dtype,
    ]
    if strict:
        cmd.append("--require_cache_identity")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    (run_dir / f"stage-{name}.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    match = SUMMARY_RE.search(proc.stdout)
    counts = tuple(int(g) for g in match.groups()) if match else (None, None)
    return {
        "stage": name,
        "returncode": proc.returncode,
        "vae": vae,
        "written": counts[0],
        "skipped": counts[1],
        "strict": strict,
    }


def _stamp(path: Path) -> dict | None:
    with np.load(path) as npz:
        if "cache_identity" not in npz.files:
            return None
        return json.loads(str(npz["cache_identity"].item()))


def _rewrite(path: Path, drop_stamp: bool = False, extra: dict | None = None) -> None:
    with np.load(path) as npz:
        payload = {key: npz[key] for key in npz.files}
    if drop_stamp:
        payload.pop("cache_identity", None)
    payload.update(extra or {})
    np.savez(path, **payload)


def _consumer_checks(cache: Path, sizes, vae_now: str, vae_other: str) -> dict:
    from library import cache_identity
    from library.anima.strategy import AnimaLatentsCachingStrategy

    def strategy(vae: str):
        return AnimaLatentsCachingStrategy(
            True, 1, False,
            cache_identity=cache_identity.latent_cache_identity(family="anima", vae=vae),
        )

    npz = next(iter(sorted(cache.glob("*_anima.npz"))))
    w, h = sizes[0]
    return {
        "matching_vae_accepted": bool(
            strategy(vae_now).is_disk_cached_latents_expected((w, h), str(npz), False, False)
        ),
        "other_vae_rejected": not strategy(vae_other).is_disk_cached_latents_expected(
            (w, h), str(npz), False, False
        ),
    }


def _expect(results: list[dict], checks: list[tuple[str, str, int]]) -> list[str]:
    by_name = {row["stage"]: row for row in results}
    failures = []
    for stage, field, expected in checks:
        actual = by_name[stage].get(field)
        if actual != expected:
            failures.append(f"{stage}.{field}: expected {expected}, got {actual}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--vae", default=DEFAULT_VAE)
    parser.add_argument("--root", default="output/bench/cache_identity_hot")
    args = parser.parse_args()

    vae = str(REPO_ROOT / args.vae)
    if not Path(vae).is_file():
        print(f"missing VAE: {vae}", file=sys.stderr)
        return 2

    run_dir = make_run_dir("cache_identity_hot", root=args.root)
    workspace = run_dir / "workspace"
    data, cache = workspace / "images", workspace / "cache"
    sizes = _make_dataset(data)

    vae_copy = workspace / "qwen_image_vae_copy.safetensors"
    shutil.copy2(vae, vae_copy)

    results: list[dict] = []
    def stage(name, *, vae_path, strict=False):
        row = _run_stage(name, vae=vae_path, data=data, cache=cache, gpu=args.gpu,
                         dtype=args.dtype, strict=strict, run_dir=run_dir)
        results.append(row)
        print(f"{name:26s} rc={row['returncode']} written={row['written']} skipped={row['skipped']}")
        return row

    stage("cold_encode", vae_path=vae)
    stage("rerun_same_vae", vae_path=vae)
    stage("vae_path_changed", vae_path=str(vae_copy))

    os.utime(vae_copy, None)  # in-place swap signal: same path, new mtime
    stage("vae_mtime_changed", vae_path=str(vae_copy))

    for npz in sorted(cache.glob("*_anima.npz")):
        _rewrite(npz, drop_stamp=True)
    stage("legacy_stamp_default", vae_path=str(vae_copy))
    stage("legacy_stamp_strict", vae_path=str(vae_copy), strict=True)

    runtime_npz = next(iter(sorted(cache.glob("*_anima.npz"))))
    _rewrite(runtime_npz, extra={"latents_99x99": np.zeros((1, 16, 99, 99), dtype=np.float32)})
    os.utime(vae_copy, None)
    stage("stale_resolution_dropped", vae_path=str(vae_copy))
    with np.load(runtime_npz) as npz:
        stale_key_dropped = "latents_99x99" not in npz.files

    consumers = _consumer_checks(cache, sizes, str(vae_copy), vae)
    failures = _expect(results, [
        ("cold_encode", "written", 2),
        ("rerun_same_vae", "skipped", 2),
        ("vae_path_changed", "written", 2),
        ("vae_mtime_changed", "written", 2),
        ("legacy_stamp_default", "skipped", 2),
        ("legacy_stamp_strict", "written", 2),
        ("stale_resolution_dropped", "written", 2),
    ])
    if any(row["returncode"] != 0 for row in results) or not stale_key_dropped \
            or not all(consumers.values()):
        failures.append(f"stale_key_dropped={stale_key_dropped} consumers={consumers}")

    metrics = {"stages": results, "stale_resolution_key_dropped": stale_key_dropped,
               "consumer_checks": consumers, "passed": not failures,
               "failures": failures, "image_sizes": [list(s) for s in sizes]}
    (run_dir / "stages.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_result(run_dir, script=__file__, args=args, metrics=metrics,
                 label="latent-vae-identity", artifacts=["stages.json"],
                 device=f"cuda:{args.gpu}")
    print(f"\n{'PASS' if not failures else 'FAIL'} -> {run_dir}")
    for item in failures:
        print(f"  {item}")
    print(json.dumps(consumers))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
