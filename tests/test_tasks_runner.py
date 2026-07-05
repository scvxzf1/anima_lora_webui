from __future__ import annotations

import os
import sys

import pytest

import tasks


def test_tasks_main_prints_global_help_without_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["tasks.py"])

    with pytest.raises(SystemExit) as exc:
        tasks.main()

    out = capsys.readouterr().out
    assert exc.value.code == 0
    assert "Usage: python tasks.py <command> [extra args...]" in out
    assert "Commands:" in out


def test_tasks_main_prints_global_help(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["tasks.py", "--help"])

    with pytest.raises(SystemExit) as exc:
        tasks.main()

    out = capsys.readouterr().out
    assert exc.value.code == 0
    assert "Usage: python tasks.py <command> [extra args...]" in out
    assert "type-check" in out


def test_tasks_main_rejects_unknown_command(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["tasks.py", "nope"])

    with pytest.raises(SystemExit) as exc:
        tasks.main()

    err = capsys.readouterr().err
    assert exc.value.code == 1
    assert "Unknown command: nope" in err


def test_tasks_main_prints_subcommand_help(monkeypatch, capsys) -> None:
    def fake_command(_extra: list[str]) -> None:
        """Detailed fake help."""

    monkeypatch.setattr(tasks, "COMMANDS", {"fake": (fake_command, "Fake command")})
    monkeypatch.setattr(sys, "argv", ["tasks.py", "fake", "--help"])

    with pytest.raises(SystemExit) as exc:
        tasks.main()

    out = capsys.readouterr().out
    assert exc.value.code == 0
    assert "python tasks.py fake -- Fake command" in out
    assert "Detailed fake help." in out


def test_tasks_main_prints_subcommand_help_without_docstring(monkeypatch, capsys) -> None:
    def fake_command(_extra: list[str]) -> None:
        return None

    monkeypatch.setattr(tasks, "COMMANDS", {"fake": (fake_command, "Fake command")})
    monkeypatch.setattr(sys, "argv", ["tasks.py", "fake", "--help"])

    with pytest.raises(SystemExit) as exc:
        tasks.main()

    out = capsys.readouterr().out
    assert exc.value.code == 0
    assert "python tasks.py fake -- Fake command" in out
    assert "(no detailed help available)" in out


def test_tasks_main_applies_inline_env_and_forwards_remaining_args(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_command(extra: list[str]) -> None:
        calls.append(extra)

    monkeypatch.delenv("ANIMA_TASKS_TEST_FLAG", raising=False)
    monkeypatch.setattr(tasks, "COMMANDS", {"fake": (fake_command, "Fake command")})
    monkeypatch.setattr(
        sys,
        "argv",
        ["tasks.py", "fake", "ANIMA_TASKS_TEST_FLAG=ok", "--flag"],
    )

    tasks.main()

    assert os.environ["ANIMA_TASKS_TEST_FLAG"] == "ok"
    assert calls == [["--flag"]]


def test_tasks_main_preserves_equals_inside_inline_env_value(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_command(extra: list[str]) -> None:
        calls.append(extra)

    monkeypatch.delenv("ANIMA_TASKS_TEST_TOKEN", raising=False)
    monkeypatch.setattr(tasks, "COMMANDS", {"fake": (fake_command, "Fake command")})
    monkeypatch.setattr(
        sys,
        "argv",
        ["tasks.py", "fake", "ANIMA_TASKS_TEST_TOKEN=a=b=c", "--flag"],
    )

    tasks.main()

    assert os.environ["ANIMA_TASKS_TEST_TOKEN"] == "a=b=c"
    assert calls == [["--flag"]]


def test_tasks_main_unknown_command_does_not_apply_inline_env(monkeypatch, capsys) -> None:
    monkeypatch.delenv("ANIMA_TASKS_UNKNOWN", raising=False)
    monkeypatch.setattr(sys, "argv", ["tasks.py", "nope", "ANIMA_TASKS_UNKNOWN=set"])

    with pytest.raises(SystemExit) as exc:
        tasks.main()

    err = capsys.readouterr().err
    assert exc.value.code == 1
    assert "Unknown command: nope" in err
    assert "ANIMA_TASKS_UNKNOWN" not in os.environ


def test_tasks_main_type_check_strips_separator_before_pyright(monkeypatch) -> None:
    commands: list[list[str]] = []

    monkeypatch.setattr(tasks.utilities.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(tasks.utilities, "run", lambda command: commands.append(command))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tasks.py",
            "type-check",
            "--",
            "--warnings",
            "web/services/config/common.py",
        ],
    )

    tasks.main()

    assert commands == [
        [
            tasks.utilities.PY,
            "-m",
            "pyright",
            "--warnings",
            "web/services/config/common.py",
        ]
    ]


def test_tasks_main_forwards_type_check_separator_and_targets(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_type_check(extra: list[str]) -> None:
        calls.append(extra)

    monkeypatch.setattr(
        tasks,
        "COMMANDS",
        {"type-check": (fake_type_check, "Run the configured pyright pilot gate.")},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tasks.py",
            "type-check",
            "--",
            "--warnings",
            "web/services/config/common.py",
        ],
    )

    tasks.main()

    assert calls == [["--", "--warnings", "web/services/config/common.py"]]


def test_tasks_main_forwards_invalid_inline_env_tokens(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_command(extra: list[str]) -> None:
        calls.append(extra)

    monkeypatch.delenv("BAD-NAME", raising=False)
    monkeypatch.setattr(tasks, "COMMANDS", {"fake": (fake_command, "Fake command")})
    monkeypatch.setattr(sys, "argv", ["tasks.py", "fake", "BAD-NAME=value"])

    tasks.main()

    assert "BAD-NAME" not in os.environ
    assert calls == [["BAD-NAME=value"]]
