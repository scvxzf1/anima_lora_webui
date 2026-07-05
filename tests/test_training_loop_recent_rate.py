from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from library.training.loop import (
    _log_step,
    _recent_step_seconds,
    _record_recent_step_seconds,
)


def _rate_state() -> SimpleNamespace:
    return SimpleNamespace(
        recent_step_rate_last=None,
        recent_step_seconds=deque(maxlen=9),
    )


def test_recent_step_seconds_starts_after_second_step() -> None:
    state = _rate_state()

    _record_recent_step_seconds(state, 1, now=100.0)

    assert _recent_step_seconds(state) is None

    _record_recent_step_seconds(state, 2, now=102.0)

    assert _recent_step_seconds(state) == pytest.approx(2.0)


def test_recent_step_seconds_uses_sliding_median() -> None:
    state = _rate_state()

    for step, now in ((1, 100.0), (2, 102.0), (3, 104.0), (4, 112.0)):
        _record_recent_step_seconds(state, step, now=now)

    assert list(state.recent_step_seconds) == [2.0, 2.0, 8.0]
    assert _recent_step_seconds(state) == pytest.approx(2.0)


def test_recent_step_seconds_resets_on_non_monotonic_sample() -> None:
    state = _rate_state()
    _record_recent_step_seconds(state, 1, now=100.0)
    _record_recent_step_seconds(state, 2, now=102.0)

    _record_recent_step_seconds(state, 1, now=110.0)

    assert list(state.recent_step_seconds) == []
    assert state.recent_step_rate_last == (110.0, 1)


class _LossRecorder:
    moving_average = 0.5

    def add(self, **_kwargs) -> None:
        return None


class _ProgressBar:
    def __init__(self) -> None:
        self.postfix = None

    def set_postfix(self, *, refresh, **kwargs) -> None:
        self.postfix = {"refresh": refresh, **kwargs}


def test_log_step_adds_recent_rate_postfix_without_step_unit() -> None:
    progress_bar = _ProgressBar()
    state = SimpleNamespace(
        args=SimpleNamespace(log_every_n_steps=1, max_train_steps=10),
        accelerator=SimpleNamespace(
            sync_gradients=True,
            device="cpu",
            unwrap_model=lambda network: network,
        ),
        global_step=2,
        loss_recorder=_LossRecorder(),
        progress_bar=progress_bar,
        network=SimpleNamespace(),
        is_tracking=False,
        recent_step_rate_last=(100.0, 1),
        recent_step_seconds=deque([1.92], maxlen=9),
    )
    trainer = SimpleNamespace(progress_sink=None)
    loss = SimpleNamespace(detach=lambda: SimpleNamespace(item=lambda: 0.5))

    _log_step(
        trainer,
        state,
        loss=loss,
        step=1,
        epoch=0,
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
        max_mean_logs={},
    )

    assert progress_bar.postfix["refresh"] is False
    assert progress_bar.postfix["recent_s_per_step"] == "1.92"
    assert "s/step" not in progress_bar.postfix["recent_s_per_step"]
