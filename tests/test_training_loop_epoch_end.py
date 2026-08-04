from __future__ import annotations

from types import SimpleNamespace

from library.training import loop as training_loop


class _FakeAccelerator:
    is_main_process = True
    device = "cpu"

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def print(self, *_args, **_kwargs) -> None:
        return None

    def unwrap_model(self, model):
        return model

    def wait_for_everyone(self) -> None:
        self.events.append("wait")


class _FakeNetwork:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def on_epoch_start(self, _text_encoder, _unet) -> None:
        self.events.append("epoch_start")


class _FakeSaver:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def maybe_save_epoch(self, _network, global_step, epoch, num_train_epochs) -> None:
        assert (global_step, epoch, num_train_epochs) == (6, 5, 6)
        self.events.append("save_epoch")

    def maybe_save_resumable(
        self, _network, global_step, epoch, num_train_epochs
    ) -> None:
        assert (global_step, epoch, num_train_epochs) == (6, 5, 6)
        self.events.append("save_resumable")


class _FakeTrainer:
    _adapters = []

    def __init__(self, events: list[str]) -> None:
        self.events = events

    def sample_images(self, *_args, **_kwargs) -> None:
        self.events.append("sample")


def test_final_step_runs_epoch_end_artifacts_before_loop_exit(monkeypatch) -> None:
    events: list[str] = []
    network = _FakeNetwork(events)
    state = SimpleNamespace(
        args=SimpleNamespace(max_train_steps=6),
        accelerator=_FakeAccelerator(events),
        current_epoch=SimpleNamespace(value=0),
        metadata={},
        epoch_to_start=5,
        num_train_epochs=6,
        text_encoder=None,
        unet=None,
        network=network,
        global_step=5,
        saver=_FakeSaver(events),
        optimizer_eval_fn=lambda: events.append("optimizer_eval"),
        optimizer_train_fn=lambda: events.append("optimizer_train"),
        vae=None,
        tokenizers=None,
        stage_index=-1,
    )

    monkeypatch.setattr(training_loop, "_maybe_apply_stage_schedule", lambda *_a, **_k: None)

    def run_steps(_trainer, loop_state, _epoch) -> None:
        loop_state.global_step = 6
        events.append("steps")

    monkeypatch.setattr(training_loop, "_run_epoch_steps", run_steps)
    monkeypatch.setattr(
        training_loop,
        "_run_epoch_validation",
        lambda *_args: events.append("validation"),
    )
    monkeypatch.setattr(
        training_loop, "_log_epoch_average", lambda *_args: events.append("epoch_log")
    )
    monkeypatch.setattr(
        training_loop,
        "_run_adapter_epoch_hooks",
        lambda *_args: events.append("adapter_hooks"),
    )

    training_loop.run_training_loop(_FakeTrainer(events), state)

    assert events == [
        "epoch_start",
        "steps",
        "validation",
        "epoch_log",
        "adapter_hooks",
        "wait",
        "optimizer_eval",
        "save_epoch",
        "save_resumable",
        "sample",
        "optimizer_train",
    ]
    assert state.current_epoch.value == 6
    assert "ss_training_finished_at" in state.metadata
