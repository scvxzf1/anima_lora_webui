from __future__ import annotations

from pathlib import Path

from tests.frontend_test_support import *  # noqa: F403


def test_frontend_state_tests_are_split_into_domain_modules() -> None:
    assert Path(__file__).with_name("test_training_frontend_modules.py").is_file()
    assert Path(__file__).with_name("test_training_frontend_config_ui.py").is_file()
