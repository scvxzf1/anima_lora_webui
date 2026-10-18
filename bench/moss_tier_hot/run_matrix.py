#!/usr/bin/env python
"""Run the MOSS-Image token-budget tier hot-test matrix (protocol section 7).

Examples:
  python -m bench.moss_tier_hot.run_matrix plan
  python -m bench.moss_tier_hot.run_matrix prepare --rows 1024-short
  python -m bench.moss_tier_hot.run_matrix all --rows 1024-short --steps 10

Every artefact stays under --root (default output/bench/moss_tier_hot/<stamp>).
"""
from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

from bench._common import REPO_ROOT
from bench.moss_tier_hot import report, runner
from bench.moss_tier_hot.matrix import ROWS, select_rows, write_row_assets

DEFAULT_ROOT_PARENT = "output/bench/moss_tier_hot"
DEFAULT_GPU_UUID = "GPU-fe9cb2a3-4a2b-d6cd-f4ca-219596a8dc44"  # CMP 50HX, physical GPU 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "prepare", "cache", "run", "report", "all"))
    parser.add_argument("--root", default=None)
    parser.add_argument("--rows", default=None, help="comma separated row names")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--gpu-uuid", default=DEFAULT_GPU_UUID)
    parser.add_argument("--python", default=runner.DEFAULT_PYTHON)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--no-overwrite-cache", action="store_true")
    return parser.parse_args(argv)


def _resolve_root(args) -> Path:
    if args.root:
        return Path(args.root)
    if args.action in ("plan", "report"):
        return REPO_ROOT / DEFAULT_ROOT_PARENT
    stamp = __import__("time").strftime("%Y%m%d-%H%M%S")
    return REPO_ROOT / DEFAULT_ROOT_PARENT / f"matrix-{stamp}"


def _bucket_check(repo_root: Path, row) -> dict:
    from library.datasets.buckets import BucketManager

    data = tomllib.loads((repo_root / "configs/methods" / f"{row.method}.toml").read_text("utf-8"))
    dataset = data["datasets"][0]
    manager = BucketManager(max_reso=(dataset["resolution"], dataset["resolution"]),
                            min_size=dataset["min_bucket_reso"],
                            max_size=dataset["max_bucket_reso"],
                            reso_steps=dataset["bucket_reso_steps"])
    manager.make_buckets(constant_token_buckets=True)
    selected = manager.select_bucket(*row.src_size)
    bucket = selected[0] if isinstance(selected[0], tuple) else selected
    width, height = bucket[0], bucket[1]
    return {"bucket": f"{width}x{height}", "tokens": (width // 16) * (height // 16),
            "buckets": len(manager.predefined_resos)}


def _print_plan(repo_root: Path, rows) -> None:
    print(f"{len(rows)} rows; protocol order = {list(ROWS)}")
    print("| row | method | res | src | bucket | image tokens | cap |")
    print("| --- | --- | ---: | --- | --- | ---: | ---: |")
    for row in rows:
        check = _bucket_check(repo_root, row)
        print(f"| {row.name} | {row.method} | {row.resolution} | {row.src_size[0]}x{row.src_size[1]} "
              f"| {check['bucket']} | {check['tokens']} (expected {row.tokens}) "
              f"| {row.max_token_length} |")


def main(argv=None) -> int:
    args = parse_args(argv)
    root = _resolve_root(args)
    rows = select_rows(args.rows)
    print(f"root: {root}")
    print(f"gpu:  {args.gpu_uuid}")
    if args.action == "plan":
        _print_plan(REPO_ROOT, rows)
        return 0
    root.mkdir(parents=True, exist_ok=True)
    assets = [write_row_assets(REPO_ROOT, root, row, args.steps) for row in rows]
    if args.action in ("prepare",):
        print(f"prepared {len(assets)} rows")
        return 0
    cache_info: dict[str, dict] = {}
    if args.action in ("cache", "all"):
        for item in assets:
            result = runner.build_cache(REPO_ROOT, item, args.gpu_uuid, args.python,
                                        overwrite=not args.no_overwrite_cache)
            cache_info[result["row"]] = result
            print(f"cache {result['row']}: exit={result['cache_exit_code']} "
                  f"{result['cache_seconds']}s resized={result.get('resized_image')} "
                  f"keys={result.get('latent_keys')}")
            if result["cache_exit_code"] != 0:
                return result["cache_exit_code"]
    if args.action in ("run", "all"):
        for item in assets:
            result = runner.run_training(REPO_ROOT, item, args.gpu_uuid, args.python,
                                         timeout_sec=args.timeout,
                                         cache_info=cache_info.get(item["row"].name))
            print(f"run {result['row']}: exit={result['train_exit_code']} "
                  f"tail_mean={result.get('tail_mean_seconds')}s")
    if args.action in ("report", "all"):
        summary = report.summarize(root, rows)
        print(f"wrote {root / 'summary.md'} ({len(summary['rows'])} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
