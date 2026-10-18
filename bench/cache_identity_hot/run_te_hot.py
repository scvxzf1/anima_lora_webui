#!/usr/bin/env python
"""Hot test: model-identity driven text-encoder cache rebuild (real TE, real GPU).

Runs `scripts.preprocess.cache_text_embeddings` against the on-disk Anima
Qwen3 text encoder and a throwaway 2-image dataset, then asserts that a changed
text-encoder identity forces a re-encode while an unchanged one keeps skipping.

The encoder is loaded without `--dit`, so no LLM adapter is stacked and the
caches carry hiddens + mask + t5 ids only (that is the shape the identity check
targets; the DiT is only asserted when adapter outputs are cached).

Stages:

1. cold_encode            write both caps
2. rerun_same_te          skip both (identity match)
3. te_path_changed        re-encode (configured encoder path is part of identity)
4. te_mtime_changed       re-encode (in-place weight swap signal, on a copy)
5. legacy_stamp_default   skip unstamped sidecars, warn once
6. legacy_stamp_strict    re-encode with --require_cache_identity

Usage::

    .venv/bin/python -m bench.cache_identity_hot.run_te_hot --gpu 0

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

from PIL import Image
from safetensors.torch import load_file, save_file

from bench._common import REPO_ROOT, make_run_dir, write_result

PY = str(REPO_ROOT / ".venv" / "bin" / "python")
MODULE = "scripts.preprocess.cache_text_embeddings"
DEFAULT_TE = "/home/scv/nvme0n1p1/ComfyUI/models/text_encoders/anima/anima_qwen_3_06b_base.safetensors"
IMAGE_SIZES = ((512, 512), (640, 384))
SUMMARY_RE = re.compile(r"complete: (\d+) cached, (\d+) skipped")


def _make_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for index, (w, h) in enumerate(IMAGE_SIZES):
        Image.new("RGB", (w, h), color=(70, 30 * index + 40, 90)).save(root / f"img{index}.png")
        (root / f"img{index}.txt").write_text(
            f"1girl, solo, test image {index}", encoding="utf-8"
        )


def _run_stage(name: str, *, te: str, data: Path, cache: Path, gpu: int,
               dtype: str, strict: bool, run_dir: Path) -> dict:
    cmd = [
        PY, "-m", MODULE,
        "--dir", str(data), "--cache_dir", str(cache), "--qwen3", te,
        "--batch_size", "1", "--dtype", dtype, "--min_pixels", "0",
        "--caption_shuffle_variants", "0",
    ]
    if strict:
        cmd.append("--require_cache_identity")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    (run_dir / f"te-stage-{name}.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    match = SUMMARY_RE.search(proc.stdout)
    counts = tuple(int(g) for g in match.groups()) if match else (None, None)
    return {
        "stage": name,
        "returncode": proc.returncode,
        "te": te,
        "written": counts[0],
        "skipped": counts[1],
        "strict": strict,
    }


def _strip_identity(path: Path) -> None:
    """Rewrite a cached sidecar as a legacy (pre-identity) safetensors file."""
    tensors = load_file(str(path))
    save_file(tensors, str(path), metadata={"model_family": "anima", "cache_schema": "anima_te_v1"})


def _consumer_checks(cache: Path, te_now: str, te_other: str) -> dict:
    from library.anima.strategy import AnimaTextEncoderOutputsCachingStrategy

    path = next(iter(sorted(cache.glob("*_anima_te.safetensors"))))
    def strategy(te: str):
        return AnimaTextEncoderOutputsCachingStrategy(
            True, 1, False, identity_inputs={"text_encoder": te}
        )
    return {
        "matching_te_accepted": bool(strategy(te_now).is_disk_cached_outputs_expected(str(path))),
        "other_te_rejected": not strategy(te_other).is_disk_cached_outputs_expected(str(path)),
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
    parser.add_argument("--te", default=DEFAULT_TE)
    parser.add_argument("--root", default="output/bench/cache_identity_hot")
    args = parser.parse_args()

    te = str(Path(args.te).expanduser())
    if not Path(te).is_file():
        print(f"missing text encoder: {te}", file=sys.stderr)
        return 2

    run_dir = make_run_dir("cache_identity_hot", root=args.root)
    workspace = run_dir / "te-workspace"
    data, cache = workspace / "images", workspace / "cache"
    _make_dataset(data)

    te_link = workspace / "anima_te_symlink.safetensors"
    os.symlink(te, te_link)                   # same bytes, different path, 0 extra disk
    te_copy = workspace / "anima_te_copy.safetensors"
    shutil.copy2(te, te_copy)                 # touchable copy for the in-place stage

    results: list[dict] = []
    def stage(name, *, te_path, strict=False):
        row = _run_stage(name, te=te_path, data=data, cache=cache, gpu=args.gpu,
                         dtype=args.dtype, strict=strict, run_dir=run_dir)
        results.append(row)
        print(f"{name:24s} rc={row['returncode']} written={row['written']} skipped={row['skipped']}")
        return row

    stage("cold_encode", te_path=te)
    stage("rerun_same_te", te_path=te)
    stage("te_path_changed", te_path=str(te_link))

    os.utime(te_copy, None)
    stage("te_mtime_changed", te_path=str(te_copy))

    for path in sorted(cache.glob("*_anima_te.safetensors")):
        _strip_identity(path)
    stage("legacy_stamp_default", te_path=str(te_copy))
    stage("legacy_stamp_strict", te_path=str(te_copy), strict=True)

    consumers = _consumer_checks(cache, str(te_copy), te)
    failures = _expect(results, [
        ("cold_encode", "written", 2),
        ("rerun_same_te", "skipped", 2),
        ("te_path_changed", "written", 2),
        ("te_mtime_changed", "written", 2),
        ("legacy_stamp_default", "skipped", 2),
        ("legacy_stamp_strict", "written", 2),
    ])
    if any(row["returncode"] != 0 for row in results) or not all(consumers.values()):
        failures.append(f"consumers={consumers}")

    metrics = {"stages": results, "consumer_checks": consumers, "passed": not failures,
               "failures": failures, "text_encoder": te,
               "image_sizes": [list(s) for s in IMAGE_SIZES]}
    (run_dir / "te-stages.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_result(run_dir, script=__file__, args=args, metrics=metrics,
                 label="anima-te-identity", artifacts=["te-stages.json"],
                 device=f"cuda:{args.gpu}")
    print(f"\n{'PASS' if not failures else 'FAIL'} -> {run_dir}")
    for item in failures:
        print(f"  {item}")
    print(json.dumps(consumers))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
