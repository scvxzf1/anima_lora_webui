from __future__ import annotations

import pytest

from library.runtime.launch import (
    ACCELERATE_NUM_PROCESSES_ENV,
    accelerate_training_command_prefix,
    resolve_accelerate_num_processes,
)


def test_accelerate_num_processes_defaults_to_single_process(monkeypatch):
    monkeypatch.delenv(ACCELERATE_NUM_PROCESSES_ENV, raising=False)
    cmd = accelerate_training_command_prefix("python", "train.py")
    assert cmd[cmd.index("--num_processes") + 1] == "1"


def test_accelerate_num_processes_env_override():
    env = {ACCELERATE_NUM_PROCESSES_ENV: "2"}
    assert resolve_accelerate_num_processes(env) == "2"


@pytest.mark.parametrize("value", ["0", "-1", "many"])
def test_accelerate_num_processes_rejects_invalid_values(value: str):
    with pytest.raises(ValueError):
        resolve_accelerate_num_processes({ACCELERATE_NUM_PROCESSES_ENV: value})
