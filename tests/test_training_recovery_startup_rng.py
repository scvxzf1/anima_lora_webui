"""Real loop startup Rich logging must not perturb a committed RNG cursor."""
import inspect
import io
import logging
import random
from types import SimpleNamespace

import pytest
import numpy as np
import torch

from library.training import loop
from tests.test_training_recovery_cursor import make_loader
from library.training.adaptive_runtime.resume_data import capture_loader, restore_loader


def assert_cursor_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_cursor_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_cursor_equal(a, b)
    else:
        assert actual == expected


def build_startup(monkeypatch, *, resumed, scheduler_draw=False):
    monkeypatch.setattr(loop.train_util, "init_trackers", lambda *a: None)
    monkeypatch.setattr(loop, "clean_memory_on_device", lambda *a: None)
    args = SimpleNamespace(
        _adaptive_training_resume={"path": "committed"} if resumed else None,
        max_train_steps=18, max_validation_steps=None, validation_sigmas=None,
        t_min=None, t_max=None, timestep_sampling="sigmoid", sigmoid_scale=1.0,
        sigmoid_bias=0.0, sample_at_first=False,
    )
    trainer = SimpleNamespace(
        get_noise_scheduler=lambda *a: random.random() if scheduler_draw else object(),
        is_text_encoder_not_needed_for_training=lambda *a: True,
        sample_images=lambda *a, **k: None,
        _parse_profile_steps=lambda *a: None,
    )
    accelerator = SimpleNamespace(device=torch.device("cpu"), trackers=[],
                                  unwrap_model=lambda model: model, is_local_main_process=False)
    kwargs = {name: None for name in inspect.signature(loop.build_loop_state).parameters if name != "trainer"}
    kwargs.update(args=args, accelerator=accelerator, network=torch.nn.Linear(1, 1),
                  unet=SimpleNamespace(device="cpu"), text_encoders=[],
                  optimizer_train_fn=lambda: None, optimizer_eval_fn=lambda: None,
                  val_dataloader=[], unet_weight_dtype=torch.float16,
                  epoch_to_start=0, num_train_epochs=2, initial_step=0,
                  resume_global_step=1 if resumed else 0, metadata={})
    state = loop.build_loop_state(trainer, **kwargs)
    state.progress_bar.close()


@pytest.mark.parametrize("resumed", [False, True])
def test_actual_loop_startup_with_random_rich_hyperlinks(monkeypatch, resumed):
    from rich.console import Console
    from rich.logging import RichHandler
    handler = RichHandler(console=Console(file=io.StringIO(), width=80))
    logger = logging.Logger("recovery-startup", level=logging.INFO)
    logger.addHandler(handler)
    monkeypatch.setattr(loop, "logger", logger)
    random.seed(923)
    before = random.getstate()
    build_startup(monkeypatch, resumed=resumed)
    after = random.getstate()
    # Initial startup remains unchanged; only replayed startup logs are inert.
    assert (after == before) is resumed
    assert "text_encoder is not needed" in handler.console.file.getvalue()


def test_recovery_startup_keeps_nontelemetry_randomness(monkeypatch):
    random.seed(924)
    random.random()
    expected = random.getstate()
    random.seed(924)
    build_startup(monkeypatch, resumed=True, scheduler_draw=True)
    assert random.getstate() == expected


@pytest.mark.parametrize("cut", [1, 12])
def test_immediate_restored_wrapper_cursor_matches_committed_state(cut):
    loader = make_loader()
    iterator = iter(loader)
    for _ in range(cut):
        next(iterator)
    saved = capture_loader(loader)
    restored = make_loader()
    restore_loader(restored, saved)
    immediate = capture_loader(restored)
    # Exhausted cursors advance the epoch plan, but cached loader progress
    # must already expose the committed state before any new data is fetched.
    assert_cursor_equal(immediate["loader"], saved["loader"])
    assert immediate["iteration"] == saved["iteration"] + int(cut == len(loader))
