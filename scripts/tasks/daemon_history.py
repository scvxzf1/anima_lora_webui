"""Read persisted daemon jobs and stdout without requiring a running daemon."""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Iterator, TextIO

from scripts.daemon import config


_JOB_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{6}\Z")
_MAX_RECORD_BYTES = 1024 * 1024
_MAX_TAIL_LINES = 10000


def _job_dir(job_id: str) -> Path:
    if not _JOB_ID.fullmatch(job_id):
        raise ValueError(f"invalid daemon job id: {job_id!r}")
    return config.job_dir(job_id)


@contextmanager
def _open_job_file(job_id: str, filename: str) -> Iterator[TextIO]:
    job_dir = _job_dir(job_id)
    dir_fd = os.open(job_dir, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        # Keep the opened directory fixed and reject symlinks/FIFOs before reading.
        file_fd = os.open(
            filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=dir_fd
        )
        try:
            if not stat.S_ISREG(os.fstat(file_fd).st_mode):
                raise ValueError(f"unsafe daemon job file: {job_id}/{filename}")
            with os.fdopen(file_fd, encoding="utf-8", errors="replace") as stream:
                file_fd = -1
                yield stream
        finally:
            if file_fd >= 0:
                os.close(file_fd)
    finally:
        os.close(dir_fd)


def _read_record(job_id: str) -> dict | None:
    try:
        with _open_job_file(job_id, "job.json") as stream:
            if os.fstat(stream.fileno()).st_size > _MAX_RECORD_BYTES:
                return None
            contents = stream.read(_MAX_RECORD_BYTES + 1)
            if len(contents.encode("utf-8")) > _MAX_RECORD_BYTES:
                return None
            record = json.loads(contents)
    except (OSError, UnicodeError, ValueError):
        return None
    if not isinstance(record, dict):
        return None
    record["id"] = job_id
    return record


def _sort_key(record: dict) -> tuple[float, str]:
    try:
        submitted = float(record.get("submitted_at"))
    except (TypeError, ValueError, OverflowError):
        submitted = 0.0
    if not math.isfinite(submitted):
        submitted = 0.0
    return submitted, record["id"]


def _jobs_from_disk() -> list[dict]:
    if not config.JOBS_DIR.is_dir():
        return []
    records = []
    for path in config.JOBS_DIR.iterdir():
        if not path.is_dir() or path.is_symlink() or not _JOB_ID.fullmatch(path.name):
            continue
        record = _read_record(path.name)
        if record is not None:
            records.append(record)
    return sorted(records, key=_sort_key)


def _job_line(record: dict) -> str:
    def display(key: str) -> str:
        value = str(record.get(key) or "-")
        return "".join(c if c.isprintable() else " " for c in value)[:80]

    return f"{record['id']}  {display('state'):8}  {display('method')}  {display('preset')}"


def cmd_daemon_jobs(extra):
    """List persisted daemon jobs, even when the daemon is stopped.

    Options: --limit N (default 15), --all, --state queued|running|done|error|stopped.
    """
    parser = argparse.ArgumentParser(prog="daemon-jobs")
    parser.add_argument("--limit", type=int, default=15)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--state", choices=("queued", "running", "done", "error", "stopped"))
    args = parser.parse_args(extra or [])
    if args.limit < 1:
        parser.error("--limit must be positive")
    records = _jobs_from_disk()
    if args.state:
        records = [record for record in records if record.get("state") == args.state]
    shown = records if args.all else records[-args.limit :]
    for record in shown:
        print(_job_line(record))
    print(f"{len(shown)} of {len(records)} jobs")


def cmd_daemon_log(extra):
    """Read a job's stdout from disk; JOB=<id> or the most recent job.

    Use -n N for the last N lines (default 100); -n 0 streams the whole file.
    """
    parser = argparse.ArgumentParser(prog="daemon-log")
    parser.add_argument("job_id", nargs="?")
    parser.add_argument("-n", "--lines", type=int, default=100)
    args = parser.parse_args(extra or [])
    if not 0 <= args.lines <= _MAX_TAIL_LINES:
        parser.error(f"-n must be between 0 and {_MAX_TAIL_LINES}")
    records = _jobs_from_disk()
    env_job_id = os.environ.get("JOB")
    if env_job_id and args.job_id:
        parser.error("specify either JOB or job_id, not both")
    job_id = env_job_id or args.job_id
    if not job_id:
        if not records:
            raise SystemExit("no daemon jobs on record")
        job_id = records[-1]["id"]
    try:
        with _open_job_file(job_id, "stdout.log") as stream:
            lines = stream if args.lines == 0 else deque(stream, maxlen=args.lines)
            print(f"# {job_id}  {_job_dir(job_id) / 'stdout.log'}")
            for line in lines:
                print(line, end="" if line.endswith("\n") else "\n")
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
