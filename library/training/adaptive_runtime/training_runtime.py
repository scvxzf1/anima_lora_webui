"""Opt-in real-loop progress boundaries for the isolated training worker."""

import logging
from pathlib import Path
import shutil

from library.training.auto_block_swap.process import write_result
from .training_state import save_training_state


class TrainingRuntime:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.stage = "setup"
        self.optimizer_started = False
        self.completed_steps = 0
        self.saved_state = None
        self.finished = False

    def phase(self, stage):
        self.stage = stage
        if stage == "optimizer":
            # Sticky before step(): an exception can follow a partial update.
            self.optimizer_started = True

    def after_optimizer(self, state):
        if state.accelerator.optimizer_step_was_skipped:
            raise RuntimeError("Adaptive worker stopped: loss scaler skipped optimizer update")

    def commit(self, state):
        if not self.optimizer_started or state.global_step != self.completed_steps + 1:
            raise RuntimeError("Adaptive worker observed an invalid optimizer boundary")
        self.stage = "checkpoint"
        saved = save_training_state(state, self.directory)
        previous = self.saved_state
        self.saved_state = str(saved)
        self.completed_steps = state.global_step
        write_result(self.directory / "latest-state.json", self.result("running"))
        if previous is not None:
            try:
                shutil.rmtree(previous)
            except OSError:
                logging.getLogger(__name__).warning("Could not remove previous owned adaptive snapshot")
        self.stage = "between_steps"

    def finish(self, state):
        if not (self.completed_steps > 0 and self.completed_steps == state.global_step
                and state.global_step == state.args.max_train_steps):
            raise RuntimeError("Adaptive worker returned without completing requested optimizer steps")
        self.finished = True
        self.stage = "finished"

    def result(self, status):
        return {"status": status, "stage": self.stage,
                "optimizer_started": self.optimizer_started,
                "completed_steps": self.completed_steps, "saved_state": self.saved_state,
                "committed_checkpoint": None, "data_cursor_resume_supported": False,
                "resume_limitation": "dataset_order_and_prefetch_cursor_not_restored"}


def notify(trainer, event, state=None):
    runtime = getattr(trainer, "_adaptive_training_runtime", None)
    if runtime is None:
        return
    if event in {"after_optimizer", "commit", "finish"}:
        getattr(runtime, event)(state)
    else:
        runtime.phase(event)


def validate_entry(trainer, args):
    if getattr(args, "adaptive_oom_retry", False):
        raise ValueError("adaptive_oom_retry must use the supervised train.py CLI entry")
    private = getattr(args, "_adaptive_training_worker", None)
    runtime = getattr(trainer, "_adaptive_training_runtime", None)
    if private or runtime is not None:
        if not private or not isinstance(runtime, TrainingRuntime):
            raise ValueError("Adaptive training worker requires an isolated runtime")
