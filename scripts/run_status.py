"""One-liner status for a training run, read from its ``progress.jsonl``.

Usage:
    python scripts/run_status.py                  # newest run under output/logs
    python scripts/run_status.py anima_turbo      # by output_name
    python scripts/run_status.py path/to/x.progress.jsonl
    python scripts/run_status.py --list           # every known run, newest first
    python scripts/run_status.py --json           # machine-readable digest

Answers "where is this run at?" without exporting the TensorBoard events file:
step N/total, rate, ETA, last losses, last checkpoint, and whether the run is
still alive (``running``), finished (``ok``), or died (``error`` / ``stopped`` /
``dead``). Works for any run that emits the sink — ``train.py`` methods and the
bespoke ``make turbo`` loop alike.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from library.env import resolve_under_home  # noqa: E402
from library.training.progress import read_status  # noqa: E402

_SUFFIX = ".progress.jsonl"


def _find_streams(logs_dir: Path) -> list[Path]:
    """Every progress stream under ``logs_dir``, newest-modified first."""
    if not logs_dir.is_dir():
        return []
    streams = [p for p in logs_dir.rglob(f"*{_SUFFIX}") if p.is_file()]
    return sorted(streams, key=lambda p: p.stat().st_mtime, reverse=True)


def _resolve(target: str | None, logs_dir: Path) -> Path:
    """Resolve a path / run name / nothing-at-all to one stream file."""
    if target:
        direct = resolve_under_home(target)
        if direct.is_file():
            return direct
        named = logs_dir / f"{target}{_SUFFIX}"
        if named.is_file():
            return named
        # a run name that doesn't sit flat in logs_dir (per-job daemon dirs)
        matches = [p for p in _find_streams(logs_dir) if p.name == f"{target}{_SUFFIX}"]
        if matches:
            return matches[0]
        raise SystemExit(f"no progress stream for {target!r} (looked under {logs_dir})")
    streams = _find_streams(logs_dir)
    if not streams:
        raise SystemExit(
            f"no *{_SUFFIX} under {logs_dir} — no run has started, or the sink is "
            "disabled (--no_log / --progress_jsonl off)."
        )
    return streams[0]


def _fmt_dur(seconds: float | None) -> str:
    if not seconds or seconds < 0:
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _fmt_metrics(metrics: dict, limit: int) -> str:
    out = []
    for key, val in list(metrics.items())[:limit]:
        out.append(f"{key}={val:.4g}" if isinstance(val, float) else f"{key}={val}")
    return " ".join(out)


def format_status(st: dict, *, metric_limit: int = 6) -> str:
    step = st["global_step"] if st["global_step"] is not None else "?"
    total = st["total_steps"] or "?"
    pct = f" ({st['pct']:.1f}%)" if st["pct"] is not None else ""
    rate = f"{st['rate']:.2f} it/s" if st["rate"] else "? it/s"
    head = (
        f"{st['run'] or st['path']} [{st['method'] or '?'}] {st['status'].upper()}  "
        f"step {step}/{total}{pct}  {rate}  "
        f"elapsed {_fmt_dur(st['elapsed'])}  ETA {_fmt_dur(st['eta'])}"
    )
    lines = [head]
    if st["metrics"]:
        lines.append(f"  last:  {_fmt_metrics(st['metrics'], metric_limit)}")
    if st["val"]:
        lines.append(f"  val:   {_fmt_metrics(_strip_ev(st['val']), metric_limit)}")
    if st["ckpt"]:
        lines.append(
            f"  ckpt:  {st['ckpt']['path']} (step {st['ckpt']['global_step']})"
        )
    if st["error"]:
        lines.append(f"  error: {st['error']}")
    if st["warnings"]:
        lines.append(f"  {st['warnings']} warning/error log event(s) in the stream")
    return "\n".join(lines)


def _strip_ev(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k not in ("ev", "ts")}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "target",
        nargs="?",
        help="run name (output_name), a .progress.jsonl path, or omit for the "
        "most recently updated run.",
    )
    p.add_argument(
        "--logs-dir",
        default="output/logs",
        help="where to look up run names (default: output/logs).",
    )
    p.add_argument(
        "--list", action="store_true", help="report every run found, newest first."
    )
    p.add_argument("--json", action="store_true", help="emit the status dict as JSON.")
    args = p.parse_args(argv)

    logs_dir = resolve_under_home(args.logs_dir)
    if args.list:
        streams = _find_streams(logs_dir)
        if not streams:
            raise SystemExit(f"no *{_SUFFIX} under {logs_dir}")
    else:
        streams = [_resolve(args.target, logs_dir)]

    statuses = []
    for path in streams:
        try:
            statuses.append(read_status(str(path)))
        except (OSError, ValueError) as exc:
            if not args.list:
                raise SystemExit(f"{path}: {exc}")
            print(f"{path}: {exc}", file=sys.stderr)

    if args.json:
        json.dump(statuses if args.list else statuses[0], sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print("\n\n".join(format_status(st) for st in statuses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
