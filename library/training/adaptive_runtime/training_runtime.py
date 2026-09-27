"""Opt-in real-loop progress boundaries for the isolated training worker."""

import logging
from pathlib import Path
import shutil

from library.training.auto_block_swap.process import write_result
from .training_state import save_training_state


class TrainingRuntime:
    def __init__(self, directory, *, precision_contract_id=None, args=None):
        self.directory = Path(directory)
        self.precision_contract_id = precision_contract_id
        self.args = args
        self.precision_manifest_id = None
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

    def validate_precision(self, model=None):
        if self.precision_contract_id is not None:
            from .training_precision import validate_local_precision_contract

            validate_local_precision_contract(self.args, self.precision_contract_id)
        if model is None:
            return
        names = getattr(model, "_adaptive_precision_manifest_names", None)
        expected = getattr(model, "_adaptive_precision_manifest_id", None)
        if names is None or expected is None:
            raise ValueError("precision_contract_mismatch: installed dtype manifest is missing")
        from .training_precision import realized_precision_manifest_id

        actual = realized_precision_manifest_id(model, names)
        if actual != expected:
            raise ValueError("precision_contract_mismatch: installed dtype manifest drifted")
        if self.precision_manifest_id is None:
            self.precision_manifest_id = actual
        elif self.precision_manifest_id != actual:
            raise ValueError("precision_contract_mismatch: realized dtype manifest drifted")

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
                "precision_contract_id": self.precision_contract_id,
                "precision_manifest_id": self.precision_manifest_id,
                "committed_checkpoint": None, "data_cursor_resume_supported": False,
                "resume_limitation": "dataset_order_and_prefetch_cursor_not_restored"}


def notify(trainer, event, state=None):
    runtime = getattr(trainer, "_adaptive_training_runtime", None)
    if runtime is None:
        return
    if event in {"setup", "commit"}:
        # The commit path also writes a progress result; validate before it
        # can publish an obsolete dtype identity.
        runtime.validate_precision(getattr(trainer, "_adaptive_precision_manifest_model", None))
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
