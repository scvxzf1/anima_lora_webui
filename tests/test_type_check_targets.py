from __future__ import annotations

import pytest

from scripts.tasks import utilities
from scripts.tasks.utilities import TYPE_CHECK_TARGETS


def test_type_check_targets_stay_scoped_to_pilot_surface() -> None:
    assert set(TYPE_CHECK_TARGETS) == {
        "library/config",
        "scripts/config_compat.py",
        "scripts/config_explain.py",
        "scripts/tasks/_common.py",
        "scripts/tasks/utilities.py",
        "web/services/config/common.py",
        "web/services/config/estimation.py",
        "web/services/config/file_groups.py",
        "web/services/config/merge.py",
        "web/services/config/metadata.py",
        "web/services/config/output_runs.py",
        "web/services/config/paths.py",
        "web/services/config/preflight.py",
        "web/services/config/raw_files.py",
        "web/services/config/sample_prompts.py",
    }
    assert "." not in TYPE_CHECK_TARGETS
    assert "tests" not in TYPE_CHECK_TARGETS
    assert "web/services/config" not in TYPE_CHECK_TARGETS
    assert "web/services/config/datasets.py" not in TYPE_CHECK_TARGETS
    assert "web/services/config/_legacy.py" not in TYPE_CHECK_TARGETS


def test_cmd_type_check_uses_default_targets(monkeypatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(utilities.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(utilities, "run", lambda command: commands.append(command))

    utilities.cmd_type_check([])

    assert commands == [[utilities.PY, "-m", "pyright", *TYPE_CHECK_TARGETS]]


def test_cmd_type_check_strips_separator_for_explicit_targets(monkeypatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(utilities.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(utilities, "run", lambda command: commands.append(command))

    utilities.cmd_type_check(["--", "library/config"])

    assert commands == [[utilities.PY, "-m", "pyright", "library/config"]]


def test_cmd_type_check_empty_separator_uses_default_targets(monkeypatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(utilities.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(utilities, "run", lambda command: commands.append(command))

    utilities.cmd_type_check(["--"])

    assert commands == [[utilities.PY, "-m", "pyright", *TYPE_CHECK_TARGETS]]


def test_cmd_type_check_preserves_explicit_pyright_flags(monkeypatch) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(utilities.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(utilities, "run", lambda command: commands.append(command))

    utilities.cmd_type_check(["--", "--warnings", "web/services/config/common.py"])

    assert commands == [[utilities.PY, "-m", "pyright", "--warnings", "web/services/config/common.py"]]


def test_cmd_type_check_exits_when_pyright_is_missing(monkeypatch, capsys) -> None:
    monkeypatch.setattr(utilities.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(SystemExit) as exc:
        utilities.cmd_type_check([])

    assert exc.value.code == 2
    assert "pyright is not installed" in capsys.readouterr().err
