"""Offline daemon history commands do not require a running daemon."""

from __future__ import annotations

import json
import os

import pytest

from scripts.daemon import config
from scripts.tasks.daemon_history import cmd_daemon_jobs, cmd_daemon_log


@pytest.fixture
def jobs_dir(tmp_path, monkeypatch):
    path = tmp_path / "jobs"
    path.mkdir()
    monkeypatch.setattr(config, "JOBS_DIR", path)
    monkeypatch.delenv("JOB", raising=False)
    return path


def _job(jobs_dir, job_id, *, submitted_at, state="done", stdout="", **record_fields):
    path = jobs_dir / job_id
    path.mkdir()
    record = {
        "id": job_id,
        "method": "lora",
        "preset": "default",
        "state": state,
        "submitted_at": submitted_at,
        **record_fields,
    }
    (path / "job.json").write_text(json.dumps(record), encoding="utf-8")
    (path / "stdout.log").write_text(stdout, encoding="utf-8")
    return path


def test_daemon_jobs_reads_disk_in_oldest_first_order(jobs_dir, capsys):
    older = "20260925-010000-aaaaaa"
    newer = "20260925-020000-bbbbbb"
    _job(jobs_dir, newer, submitted_at=2, state="error")
    _job(jobs_dir, older, submitted_at=1)

    cmd_daemon_jobs(["--all"])
    output = capsys.readouterr().out
    assert output.index(older) < output.index(newer)
    assert "2 of 2 jobs" in output

    cmd_daemon_jobs(["--limit", "1", "--state", "error"])
    output = capsys.readouterr().out
    assert newer in output
    assert older not in output


def test_daemon_log_tails_recent_job_and_streams_full_log(jobs_dir, capsys, monkeypatch):
    older = "20260925-010000-aaaaaa"
    newer = "20260925-020000-bbbbbb"
    _job(jobs_dir, older, submitted_at=1, stdout="old\n")
    _job(jobs_dir, newer, submitted_at=2, stdout="first\nsecond\nthird\n")

    cmd_daemon_log(["-n", "2"])
    output = capsys.readouterr().out
    assert newer in output
    assert "first\n" not in output
    assert "second\nthird\n" in output

    monkeypatch.setenv("JOB", older)
    cmd_daemon_log(["-n", "0"])
    output = capsys.readouterr().out
    assert older in output
    assert "old\n" in output


def test_daemon_log_ignores_untrusted_stdout_path(jobs_dir, tmp_path, capsys):
    external = tmp_path / "secret.txt"
    external.write_text("private content", encoding="utf-8")
    job_id = "20260925-010000-aaaaaa"
    _job(jobs_dir, job_id, submitted_at=1, stdout="safe\n", stdout_path=str(external))

    cmd_daemon_log([job_id])
    output = capsys.readouterr().out
    assert "safe\n" in output
    assert "private content" not in output


@pytest.mark.parametrize("job_id", ["../../secret", "20260925-010000-aaaaaa/../x", "x"])
def test_daemon_log_rejects_invalid_job_id(jobs_dir, job_id):
    with pytest.raises(SystemExit, match="invalid daemon job id"):
        cmd_daemon_log([job_id])


def test_daemon_log_rejects_symlink_escape(jobs_dir, tmp_path):
    job_id = "20260925-010000-aaaaaa"
    path = _job(jobs_dir, job_id, submitted_at=1)
    (path / "stdout.log").unlink()
    outside = tmp_path / "outside.txt"
    outside.write_text("private content", encoding="utf-8")
    (path / "stdout.log").symlink_to(outside)

    with pytest.raises(SystemExit, match="Too many levels of symbolic links"):
        cmd_daemon_log([job_id])


def test_daemon_log_rejects_replaced_job_directory(jobs_dir, tmp_path, monkeypatch):
    job_id = "20260925-010000-aaaaaa"
    path = _job(jobs_dir, job_id, submitted_at=1, stdout="safe\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "stdout.log").write_text("private content\n", encoding="utf-8")
    original_open = os.open
    directory_opens = 0

    def replace_before_open(target, flags, *args, **kwargs):
        nonlocal directory_opens
        if target == path:
            directory_opens += 1
            if directory_opens == 2:
                path.rename(jobs_dir / "moved")
                path.symlink_to(outside, target_is_directory=True)
        return original_open(target, flags, *args, **kwargs)

    monkeypatch.setattr("scripts.tasks.daemon_history.os.open", replace_before_open)
    with pytest.raises(SystemExit, match="Not a directory|Too many levels of symbolic links"):
        cmd_daemon_log([job_id])


def test_daemon_log_rejects_conflicting_job_selectors(jobs_dir, monkeypatch, capsys):
    job_id = "20260925-010000-aaaaaa"
    _job(jobs_dir, job_id, submitted_at=1, stdout="safe\n")
    monkeypatch.setenv("JOB", job_id)

    with pytest.raises(SystemExit) as exc:
        cmd_daemon_log([job_id])
    assert exc.value.code == 2
    assert "specify either JOB or job_id" in capsys.readouterr().err


def test_daemon_log_tails_large_file_without_printing_earlier_lines(jobs_dir, capsys):
    job_id = "20260925-010000-aaaaaa"
    _job(jobs_dir, job_id, submitted_at=1, stdout="".join(f"line {i}\n" for i in range(20000)))

    cmd_daemon_log(["-n", "2", job_id])
    output = capsys.readouterr().out
    assert "line 19998\nline 19999\n" in output
    assert "line 19997\n" not in output


def test_daemon_jobs_skips_oversize_or_malformed_records(jobs_dir, capsys):
    oversized = jobs_dir / "20260925-010000-aaaaaa"
    oversized.mkdir()
    (oversized / "job.json").write_text(" " * (1024 * 1024 + 1), encoding="utf-8")
    malformed = jobs_dir / "20260925-020000-bbbbbb"
    malformed.mkdir()
    (malformed / "job.json").write_text("not json", encoding="utf-8")

    cmd_daemon_jobs([])
    assert capsys.readouterr().out == "0 of 0 jobs\n"


def test_daemon_jobs_skips_fifo_record_without_blocking(jobs_dir, capsys):
    job_id = "20260925-010000-aaaaaa"
    path = _job(jobs_dir, job_id, submitted_at=1)
    (path / "job.json").unlink()
    os.mkfifo(path / "job.json")

    cmd_daemon_jobs([])
    assert capsys.readouterr().out == "0 of 0 jobs\n"
