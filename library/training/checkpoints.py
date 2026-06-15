import argparse
from dataclasses import dataclass
import json
import logging
import math
import os
import shutil
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# checkpoint filename templates
EPOCH_STATE_NAME = "{}-{:06d}-state"
EPOCH_FILE_NAME = "{}-{:06d}"
EPOCH_DIFFUSERS_DIR_NAME = "{}-{:06d}"
LAST_STATE_NAME = "{}-state"
DEFAULT_EPOCH_NAME = "epoch"
DEFAULT_LAST_OUTPUT_NAME = "last"

DEFAULT_STEP_NAME = "at"
STEP_STATE_NAME = "{}-step{:08d}-state"
STEP_FILE_NAME = "{}-step{:08d}"
STEP_DIFFUSERS_DIR_NAME = "{}-step{:08d}"

CHECKPOINT_STATE_NAME = "{}-checkpoint-state"
CHECKPOINT_FILE_NAME = "{}-checkpoint"
CHECKPOINT_EPOCH_STATE_NAME = "{}-checkpoint-{:06d}-state"
CHECKPOINT_EPOCH_FILE_NAME = "{}-checkpoint-{:06d}"
CHECKPOINT_LATEST_STATE_FILE_NAME = "{}-checkpoint-latest.json"


@dataclass(frozen=True)
class ResumeStartPlan:
    initial_step: int
    epoch_to_start: int
    steps_from_state: Optional[int] = None


def default_if_none(value, default):
    return default if value is None else value


def _save_last_keep_count(value) -> Optional[int]:
    if value is None:
        return None
    try:
        keep_count = int(value)
    except (TypeError, ValueError):
        return None
    if keep_count < 0:
        return None
    if keep_count == 0:
        return 1
    return keep_count


def get_epoch_ckpt_name(args: argparse.Namespace, ext: str, epoch_no: int):
    model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)
    return EPOCH_FILE_NAME.format(model_name, epoch_no) + ext


def get_step_ckpt_name(args: argparse.Namespace, ext: str, step_no: int):
    model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)
    return STEP_FILE_NAME.format(model_name, step_no) + ext


def get_last_ckpt_name(args: argparse.Namespace, ext: str):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
    return model_name + ext


def get_remove_epoch_no(args: argparse.Namespace, epoch_no: int):
    keep_count = _save_last_keep_count(getattr(args, "save_last_n_epochs", None))
    if keep_count is None:
        return None

    save_every_n_epochs = getattr(args, "save_every_n_epochs", None)
    if not save_every_n_epochs:
        return None
    remove_epoch_no = epoch_no - save_every_n_epochs * keep_count
    if remove_epoch_no <= 0:
        return None
    return remove_epoch_no


def get_remove_step_no(args: argparse.Namespace, step_no: int):
    last_n_steps = getattr(args, "save_last_n_steps", None)
    if last_n_steps is None:
        return None
    try:
        last_n_steps = int(last_n_steps)
    except (TypeError, ValueError):
        return None
    if last_n_steps < 0:
        return None

    remove_step_no = step_no - last_n_steps - 1
    remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)
    if remove_step_no < 0:
        return None
    return remove_step_no


def save_sd_model_on_epoch_end_or_stepwise_common(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    sd_saver,
    diffusers_saver,
):
    if on_epoch_end:
        epoch_no = epoch + 1
        saving = (
            epoch_no % args.save_every_n_epochs == 0 and epoch_no < num_train_epochs
        )
        if not saving:
            return

        model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)
        remove_no = get_remove_epoch_no(args, epoch_no)
    else:
        model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)
        epoch_no = epoch
        remove_no = get_remove_step_no(args, global_step)

    os.makedirs(args.output_dir, exist_ok=True)
    if save_stable_diffusion_format:
        ext = ".safetensors" if use_safetensors else ".ckpt"

        if on_epoch_end:
            ckpt_name = get_epoch_ckpt_name(args, ext, epoch_no)
        else:
            ckpt_name = get_step_ckpt_name(args, ext, global_step)

        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        logger.info("")
        logger.info(f"saving checkpoint: {ckpt_file}")
        sd_saver(ckpt_file, epoch_no, global_step)

        # remove older checkpoints
        if remove_no is not None:
            if on_epoch_end:
                remove_ckpt_name = get_epoch_ckpt_name(args, ext, remove_no)
            else:
                remove_ckpt_name = get_step_ckpt_name(args, ext, remove_no)

            remove_ckpt_file = os.path.join(args.output_dir, remove_ckpt_name)
            if os.path.exists(remove_ckpt_file):
                logger.info(f"removing old checkpoint: {remove_ckpt_file}")
                os.remove(remove_ckpt_file)

    else:
        if on_epoch_end:
            out_dir = os.path.join(
                args.output_dir, EPOCH_DIFFUSERS_DIR_NAME.format(model_name, epoch_no)
            )
        else:
            out_dir = os.path.join(
                args.output_dir, STEP_DIFFUSERS_DIR_NAME.format(model_name, global_step)
            )

        logger.info("")
        logger.info(f"saving model: {out_dir}")
        diffusers_saver(out_dir)

        # remove older checkpoints
        if remove_no is not None:
            if on_epoch_end:
                remove_out_dir = os.path.join(
                    args.output_dir,
                    EPOCH_DIFFUSERS_DIR_NAME.format(model_name, remove_no),
                )
            else:
                remove_out_dir = os.path.join(
                    args.output_dir,
                    STEP_DIFFUSERS_DIR_NAME.format(model_name, remove_no),
                )

            if os.path.exists(remove_out_dir):
                logger.info(f"removing old model: {remove_out_dir}")
                shutil.rmtree(remove_out_dir)

    if args.save_state:
        if on_epoch_end:
            save_and_remove_state_on_epoch_end(args, accelerator, epoch_no)
        else:
            save_and_remove_state_stepwise(args, accelerator, global_step)


def save_and_remove_state_on_epoch_end(args: argparse.Namespace, accelerator, epoch_no):
    model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)

    logger.info("")
    logger.info(f"saving state at epoch {epoch_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(
        args.output_dir, EPOCH_STATE_NAME.format(model_name, epoch_no)
    )
    accelerator.save_state(state_dir)

    last_n_epochs = (
        args.save_last_n_epochs_state
        if args.save_last_n_epochs_state is not None
        else args.save_last_n_epochs
    )
    keep_count = _save_last_keep_count(last_n_epochs)
    if keep_count is not None:
        remove_epoch_no = epoch_no - args.save_every_n_epochs * keep_count
        if remove_epoch_no <= 0:
            return
        state_dir_old = os.path.join(
            args.output_dir, EPOCH_STATE_NAME.format(model_name, remove_epoch_no)
        )
        if os.path.exists(state_dir_old):
            logger.info(f"removing old state: {state_dir_old}")
            shutil.rmtree(state_dir_old)


def save_and_remove_state_stepwise(args: argparse.Namespace, accelerator, step_no):
    model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)

    logger.info("")
    logger.info(f"saving state at step {step_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(
        args.output_dir, STEP_STATE_NAME.format(model_name, step_no)
    )
    accelerator.save_state(state_dir)

    last_n_steps = (
        args.save_last_n_steps_state
        if args.save_last_n_steps_state
        else args.save_last_n_steps
    )
    if last_n_steps is not None:
        remove_step_no = step_no - last_n_steps - 1
        remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)

        if remove_step_no > 0:
            state_dir_old = os.path.join(
                args.output_dir, STEP_STATE_NAME.format(model_name, remove_step_no)
            )
            if os.path.exists(state_dir_old):
                logger.info(f"removing old state: {state_dir_old}")
                shutil.rmtree(state_dir_old)


def get_checkpoint_state_dir(args: argparse.Namespace, epoch_no: Optional[int] = None):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
    if epoch_no is not None:
        return os.path.join(
            args.output_dir, CHECKPOINT_EPOCH_STATE_NAME.format(model_name, epoch_no)
        )
    return os.path.join(args.output_dir, CHECKPOINT_STATE_NAME.format(model_name))


def get_checkpoint_ckpt_name(
    args: argparse.Namespace, ext: str, epoch_no: Optional[int] = None
):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
    if epoch_no is not None:
        return CHECKPOINT_EPOCH_FILE_NAME.format(model_name, epoch_no) + ext
    return CHECKPOINT_FILE_NAME.format(model_name) + ext


def get_checkpoint_latest_state_file(args: argparse.Namespace):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
    return os.path.join(
        args.output_dir, CHECKPOINT_LATEST_STATE_FILE_NAME.format(model_name)
    )


def _checkpoint_epoch_no_from_state_name(
    args: argparse.Namespace, name: str
) -> Optional[int]:
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
    prefix = f"{model_name}-checkpoint-"
    suffix = "-state"
    if not (name.startswith(prefix) and name.endswith(suffix)):
        return None
    raw = name[len(prefix) : -len(suffix)]
    if len(raw) != 6 or not raw.isdigit():
        return None
    return int(raw)


def _checkpointing_last_n_epochs(args: argparse.Namespace) -> int:
    value = getattr(args, "checkpointing_last_n_epochs", 1)
    try:
        keep_last = int(value)
    except (TypeError, ValueError):
        return 1
    if keep_last == -1:
        return -1
    return keep_last if keep_last > 0 else 1


def _checkpoint_epoch_state_entries(args: argparse.Namespace) -> list[tuple[int, str]]:
    output_dir = getattr(args, "output_dir", "")
    if not output_dir or not os.path.isdir(output_dir):
        return []
    entries: list[tuple[int, str]] = []
    for name in os.listdir(output_dir):
        epoch_no = _checkpoint_epoch_no_from_state_name(args, name)
        if epoch_no is None:
            continue
        path = os.path.join(output_dir, name)
        if os.path.isdir(path):
            entries.append((epoch_no, path))
    entries.sort(key=lambda item: (item[0], item[1]))
    return entries


def _read_checkpoint_train_state(state_dir: str) -> dict[str, Any]:
    train_state_file = os.path.join(state_dir, "train_state.json")
    if not os.path.exists(train_state_file):
        return {}
    try:
        with open(train_state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001 - bad state means this candidate is skipped
        logger.info(f"skip checkpoint state because train_state.json is unreadable: {e}")
        return {}
    return data if isinstance(data, dict) else {}


def _write_checkpoint_latest_state_marker(
    args: argparse.Namespace, state_dir: str, epoch_no: int
) -> None:
    marker_file = get_checkpoint_latest_state_file(args)
    tmp_file = marker_file + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "state_dir": os.path.abspath(state_dir),
                "epoch": epoch_no,
            },
            f,
        )
    os.replace(tmp_file, marker_file)


def _checkpoint_latest_marker_state_dir(args: argparse.Namespace) -> Optional[str]:
    marker_file = get_checkpoint_latest_state_file(args)
    if not os.path.exists(marker_file):
        return None
    try:
        with open(marker_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001 - stale marker should not block fresh runs
        logger.info(f"skip checkpoint marker because it is unreadable: {e}")
        return None
    state_dir = str(data.get("state_dir") or "").strip()
    if not state_dir:
        return None
    state_dir = os.path.abspath(state_dir)
    output_dir = os.path.abspath(args.output_dir)
    try:
        if os.path.commonpath([output_dir, state_dir]) != output_dir:
            return None
    except ValueError:
        return None
    return state_dir if os.path.isdir(state_dir) else None


def _checkpoint_state_candidates(args: argparse.Namespace) -> list[tuple[int, float, str]]:
    candidates: list[tuple[int, float, str]] = []
    marker_state_dir = _checkpoint_latest_marker_state_dir(args)
    legacy_state_dir = get_checkpoint_state_dir(args)
    state_dirs = [marker_state_dir] if marker_state_dir else []
    if os.path.isdir(legacy_state_dir):
        state_dirs.append(legacy_state_dir)

    seen: set[str] = set()
    for state_dir in state_dirs:
        state_dir = os.path.abspath(state_dir)
        if state_dir in seen:
            continue
        seen.add(state_dir)
        data = _read_checkpoint_train_state(state_dir)
        try:
            step = int(data.get("current_step"))
        except (TypeError, ValueError):
            continue
        try:
            mtime = os.path.getmtime(os.path.join(state_dir, "train_state.json"))
        except OSError:
            mtime = 0.0
        candidates.append((step, mtime, state_dir))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return candidates


def plan_resume_start(
    args: argparse.Namespace,
    *,
    steps_from_state: Optional[int],
    batches_per_epoch: int,
    num_processes: int,
) -> ResumeStartPlan:
    """Resolve resume/initial-step counters without mutating checkpoint state."""
    steps_from_state_out = steps_from_state
    initial_steps_per_epoch = math.ceil(
        batches_per_epoch / num_processes / args.gradient_accumulation_steps
    )
    skip_batches_per_epoch = math.ceil(
        batches_per_epoch / args.gradient_accumulation_steps
    )

    initial_step = 0
    if args.initial_epoch is not None or args.initial_step is not None:
        if steps_from_state is not None:
            logger.warning(
                "steps from the state is ignored because initial_step is specified"
            )
        if args.initial_step is not None:
            initial_step = args.initial_step
        else:
            initial_step = (args.initial_epoch - 1) * initial_steps_per_epoch
    else:
        if steps_from_state is not None:
            initial_step = steps_from_state
            steps_from_state_out = None

    if initial_step > 0 and args.max_train_steps <= initial_step:
        raise ValueError(
            f"恢复点已训练到 step {initial_step}，当前配置目标是 {args.max_train_steps}，"
            "继续训练不会产生新步数。请增加 max_train_steps / max_train_epochs 后再续训。"
        )

    epoch_to_start = 0
    if initial_step > 0:
        if args.skip_until_initial_step:
            if not args.resume:
                logger.info(
                    "initial_step is specified but not resuming. lr scheduler will be started from the beginning"
                )
            logger.info(f"skipping {initial_step} steps")
            initial_step *= args.gradient_accumulation_steps
            epoch_to_start = initial_step // skip_batches_per_epoch
        else:
            epoch_to_start = initial_step // skip_batches_per_epoch
            initial_step = 0

    return ResumeStartPlan(
        initial_step=initial_step,
        epoch_to_start=epoch_to_start,
        steps_from_state=steps_from_state_out,
    )


def _remove_path(path: str) -> None:
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def _recover_checkpoint_state_dirs(state_dir: str, tmp_dir: str, backup_dir: str) -> None:
    """Clean stale tmp/backup dirs before writing a new resumable state."""
    if os.path.exists(tmp_dir):
        _remove_path(tmp_dir)
    if os.path.exists(backup_dir):
        if not os.path.exists(state_dir):
            os.replace(backup_dir, state_dir)
        else:
            _remove_path(backup_dir)


def _checkpoint_network_state_compatible(state_dir: str, network: Any) -> bool:
    model_state_file = os.path.join(state_dir, "model.safetensors")
    if not os.path.exists(model_state_file):
        logger.info(
            f"skip auto-resume because checkpoint model state is missing: {model_state_file}"
        )
        return False
    try:
        from safetensors.torch import load_file

        checkpoint_keys = set(load_file(model_state_file, device="cpu").keys())
    except Exception as e:  # noqa: BLE001 - corrupted state should not block a fresh run
        logger.info(f"skip auto-resume because checkpoint model state is unreadable: {e}")
        return False

    network_keys = set(network.state_dict().keys())
    normalized_network_keys = {
        key.removeprefix("module.") for key in network_keys
    } | network_keys
    normalized_checkpoint_keys = {
        key.removeprefix("module.") for key in checkpoint_keys
    } | checkpoint_keys
    if not normalized_checkpoint_keys:
        logger.info("skip auto-resume because checkpoint model state is empty")
        return False
    if normalized_checkpoint_keys.issubset(normalized_network_keys):
        return True

    missing = sorted(normalized_checkpoint_keys - normalized_network_keys)[:5]
    logger.info(
        "skip auto-resume because checkpoint model state is incompatible; "
        f"unexpected keys: {missing}"
    )
    return False


def save_checkpoint_state(
    args: argparse.Namespace, accelerator, epoch_no: Optional[int] = None
):
    state_dir = get_checkpoint_state_dir(args, epoch_no)
    tmp_dir = state_dir + ".tmp"
    backup_dir = state_dir + ".backup"

    logger.info("")
    logger.info(f"saving checkpoint state to {state_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    _recover_checkpoint_state_dirs(state_dir, tmp_dir, backup_dir)
    try:
        accelerator.save_state(tmp_dir)
        if os.path.exists(state_dir):
            os.replace(state_dir, backup_dir)
        os.replace(tmp_dir, state_dir)
        if epoch_no is not None:
            _write_checkpoint_latest_state_marker(args, state_dir, epoch_no)
        if os.path.exists(backup_dir):
            _remove_path(backup_dir)
    except Exception:
        if os.path.exists(tmp_dir):
            _remove_path(tmp_dir)
        if os.path.exists(backup_dir):
            if os.path.exists(state_dir):
                _remove_path(state_dir)
            os.replace(backup_dir, state_dir)
        raise


def save_state_on_train_end(args: argparse.Namespace, accelerator):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)

    logger.info("")
    logger.info("saving last state.")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, LAST_STATE_NAME.format(model_name))
    accelerator.save_state(state_dir)


def save_sd_model_on_train_end_common(
    args: argparse.Namespace,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    epoch: int,
    global_step: int,
    sd_saver,
    diffusers_saver,
):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)

    if save_stable_diffusion_format:
        os.makedirs(args.output_dir, exist_ok=True)

        ckpt_name = model_name + (".safetensors" if use_safetensors else ".ckpt")
        ckpt_file = os.path.join(args.output_dir, ckpt_name)

        logger.info(f"save trained model as StableDiffusion checkpoint to {ckpt_file}")
        sd_saver(ckpt_file, epoch, global_step)
    else:
        out_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)

        logger.info(f"save trained model as Diffusers to {out_dir}")
        diffusers_saver(out_dir)


class CheckpointSaver:
    """Owns every save / remove operation across a training run.

    Replaces the cluster of save_model / remove_model / save_model_hook /
    load_model_hook closures and the inline save-tick blocks scattered through
    train(). State that used to live in closures (metadata refs, save_dtype,
    sai-spec callable, mp.Value handles, ``steps_from_state``) becomes
    instance attributes.

    ``metadata`` is a shared mutable dict — the trainer also writes
    ``ss_epoch`` between saves; the saver only writes during a save.
    """

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        accelerator,
        save_dtype,
        metadata: dict,
        minimum_metadata: dict,
        get_sai_model_spec_fn: Callable[[argparse.Namespace], dict],
        current_epoch,
        current_step,
        progress_sink=None,
    ):
        self.args = args
        self.accelerator = accelerator
        self.save_dtype = save_dtype
        self.metadata = metadata
        self.minimum_metadata = minimum_metadata
        self.get_sai_model_spec_fn = get_sai_model_spec_fn
        self.current_epoch = current_epoch
        self.current_step = current_step
        # Optional structured-progress sink (Phase 0). When set, every
        # checkpoint write emits a ``ckpt`` event.
        self.progress_sink = progress_sink
        # Set by the load_state pre-hook when resuming. Read by train() to
        # decide initial_step.
        self.steps_from_state: Optional[int] = None

    def register_hooks(self, network: Any) -> None:
        """Install accelerator save/load pre-hooks that persist epoch/step
        state to ``train_state.json`` and strip non-network models from the
        save list (we only want the adapter weights, not the frozen DiT)."""
        accelerator = self.accelerator
        unwrap_type = type(accelerator.unwrap_model(network))

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                remove_indices = []
                for i, model in enumerate(models):
                    if not isinstance(model, unwrap_type):
                        remove_indices.append(i)
                for i in reversed(remove_indices):
                    if len(weights) > i:
                        weights.pop(i)

            train_state_file = os.path.join(output_dir, "train_state.json")
            logger.info(
                f"save train state to {train_state_file} at epoch "
                f"{self.current_epoch.value} step {self.current_step.value + 1}"
            )
            with open(train_state_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "current_epoch": self.current_epoch.value,
                        "current_step": self.current_step.value + 1,
                    },
                    f,
                )

        def load_model_hook(models, input_dir):
            remove_indices = []
            for i, model in enumerate(models):
                if not isinstance(model, unwrap_type):
                    remove_indices.append(i)
            for i in reversed(remove_indices):
                models.pop(i)

            train_state_file = os.path.join(input_dir, "train_state.json")
            if os.path.exists(train_state_file):
                with open(train_state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.steps_from_state = data["current_step"]
                logger.info(f"load train state from {train_state_file}: {data}")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    def auto_resume(self, network: Any | None = None) -> None:
        """If ``checkpointing_epochs`` is enabled and a resumable checkpoint
        exists below ``max_train_steps``, point ``args.resume`` at it and
        force ``skip_until_initial_step``. No-op when ``args.resume`` is
        already set or no checkpoint exists.

        When ``network`` is provided, require the resumable checkpoint's
        ``model.safetensors`` keys to be compatible before resuming.
        """
        args = self.args
        if not getattr(args, "checkpointing_epochs", None) or args.resume:
            return
        candidates = _checkpoint_state_candidates(args)
        if not candidates:
            return
        ckpt_step, _, checkpoint_state_dir = candidates[0]
        if network is not None and not _checkpoint_network_state_compatible(
            checkpoint_state_dir, network
        ):
            return
        if ckpt_step < args.max_train_steps:
            args.resume = checkpoint_state_dir
            args.skip_until_initial_step = True
            logger.info(
                f"auto-resuming from checkpoint at step {ckpt_step}: {checkpoint_state_dir}"
            )
        else:
            logger.info(
                f"checkpoint already reached max_train_steps "
                f"({ckpt_step} >= {args.max_train_steps}), starting fresh"
            )

    def save(self, ckpt_name: str, network: Any, steps: int, epoch_no: int) -> None:
        """Write a network checkpoint with up-to-date training metadata."""
        args = self.args
        accelerator = self.accelerator
        unwrapped_nw = accelerator.unwrap_model(network)

        os.makedirs(args.output_dir, exist_ok=True)
        ckpt_file = os.path.join(args.output_dir, ckpt_name)

        accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
        self.metadata["ss_training_finished_at"] = str(time.time())
        self.metadata["ss_steps"] = str(steps)
        self.metadata["ss_epoch"] = str(epoch_no)

        metadata_to_save = self.minimum_metadata if args.no_metadata else self.metadata
        sai_metadata = self.get_sai_model_spec_fn(args)
        metadata_to_save.update(sai_metadata)

        unwrapped_nw.save_weights(ckpt_file, self.save_dtype, metadata_to_save)

        if self.progress_sink is not None:
            self.progress_sink.ckpt(global_step=steps, path=ckpt_file)

    def remove(self, old_ckpt_name: str) -> None:
        """Delete an old checkpoint plus its HydraLoRA ``_moe`` sibling if present."""
        args = self.args
        accelerator = self.accelerator
        old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
        if os.path.exists(old_ckpt_file):
            accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
            os.remove(old_ckpt_file)
        moe_file = os.path.splitext(old_ckpt_file)[0] + "_moe.safetensors"
        if os.path.exists(moe_file):
            accelerator.print(f"removing old checkpoint: {moe_file}")
            os.remove(moe_file)

    def maybe_save_step(self, network: Any, global_step: int, epoch: int) -> None:
        """Step-cadence save. ``global_step`` must already be incremented."""
        args = self.args
        accelerator = self.accelerator
        if (
            args.save_every_n_steps is None
            or global_step % args.save_every_n_steps != 0
        ):
            return
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return
        ckpt_name = get_step_ckpt_name(args, "." + args.save_model_as, global_step)
        self.save(ckpt_name, network, global_step, epoch)
        if args.save_state:
            save_and_remove_state_stepwise(args, accelerator, global_step)
        remove_step_no = get_remove_step_no(args, global_step)
        if remove_step_no is not None:
            remove_ckpt_name = get_step_ckpt_name(
                args, "." + args.save_model_as, remove_step_no
            )
            self.remove(remove_ckpt_name)

    def maybe_save_epoch(
        self, network: Any, global_step: int, epoch: int, num_train_epochs: int
    ) -> None:
        """Epoch-cadence save. ``epoch`` is 0-indexed; saver writes ``epoch+1``."""
        args = self.args
        accelerator = self.accelerator
        if args.save_every_n_epochs is None:
            return
        epoch_no = epoch + 1
        saving = (
            epoch_no % args.save_every_n_epochs == 0 and epoch_no < num_train_epochs
        )
        if not saving or not accelerator.is_main_process:
            return
        ckpt_name = get_epoch_ckpt_name(args, "." + args.save_model_as, epoch_no)
        self.save(ckpt_name, network, global_step, epoch_no)
        remove_epoch_no = get_remove_epoch_no(args, epoch_no)
        if remove_epoch_no is not None:
            remove_ckpt_name = get_epoch_ckpt_name(
                args, "." + args.save_model_as, remove_epoch_no
            )
            self.remove(remove_ckpt_name)
        if args.save_state:
            save_and_remove_state_on_epoch_end(args, accelerator, epoch_no)

    def maybe_save_resumable(
        self, network: Any, global_step: int, epoch: int, num_train_epochs: int
    ) -> None:
        """``checkpointing_epochs``-cadence resumable save. ``epoch`` is 0-indexed."""
        args = self.args
        accelerator = self.accelerator
        if not (
            args.checkpointing_epochs is not None and args.checkpointing_epochs > 0
        ):
            return
        epoch_no = epoch + 1
        if not (
            epoch_no % args.checkpointing_epochs == 0 and epoch_no < num_train_epochs
        ):
            return
        if accelerator.is_main_process:
            ckpt_name = get_checkpoint_ckpt_name(
                args, "." + args.save_model_as, epoch_no
            )
            self.save(ckpt_name, network, global_step, epoch_no)
        save_checkpoint_state(args, accelerator, epoch_no)
        if accelerator.is_main_process:
            self._cleanup_old_resumable_checkpoints()

    def _cleanup_old_resumable_checkpoints(self) -> None:
        """Keep only the configured number of numbered resumable checkpoints."""
        args = self.args
        keep_last = _checkpointing_last_n_epochs(args)
        if keep_last == -1:
            return
        entries = _checkpoint_epoch_state_entries(args)
        if len(entries) <= keep_last:
            return
        for epoch_no, state_dir in entries[: len(entries) - keep_last]:
            if os.path.exists(state_dir):
                logger.info(f"removing old checkpoint state: {state_dir}")
                shutil.rmtree(state_dir)
            ckpt_name = get_checkpoint_ckpt_name(
                args, "." + args.save_model_as, epoch_no
            )
            self.remove(ckpt_name)

    def cleanup_resumable(self) -> None:
        """At training end, remove the legacy single resumable checkpoint.

        Numbered checkpoint states are intentionally kept according to
        ``checkpointing_last_n_epochs`` so history can resume any retained point.
        """
        args = self.args
        if not getattr(args, "checkpointing_epochs", None):
            return
        if not self.accelerator.is_main_process:
            return
        checkpoint_state_dir = get_checkpoint_state_dir(args)
        explicit_resume = os.path.abspath(str(getattr(args, "resume", "") or ""))
        keep_explicit_resume_state = (
            explicit_resume
            and explicit_resume == os.path.abspath(checkpoint_state_dir)
        )
        if keep_explicit_resume_state:
            logger.info(
                f"training complete, keeping explicit resume checkpoint state: {checkpoint_state_dir}"
            )
        elif os.path.exists(checkpoint_state_dir):
            logger.info(
                f"training complete, removing checkpoint state: {checkpoint_state_dir}"
            )
            shutil.rmtree(checkpoint_state_dir)
        latest_marker = get_checkpoint_latest_state_file(args)
        if os.path.exists(latest_marker) and not keep_explicit_resume_state:
            logger.info(f"training complete, removing checkpoint marker: {latest_marker}")
            os.remove(latest_marker)
        checkpoint_ckpt = os.path.join(
            args.output_dir,
            get_checkpoint_ckpt_name(args, "." + args.save_model_as),
        )
        if os.path.exists(checkpoint_ckpt) and not keep_explicit_resume_state:
            logger.info(f"removing checkpoint weights: {checkpoint_ckpt}")
            os.remove(checkpoint_ckpt)

    def save_final(self, network: Any, global_step: int, num_train_epochs: int) -> None:
        """Write the final ``<output_name>.<ext>`` checkpoint. Main-process only."""
        if not self.accelerator.is_main_process:
            return
        args = self.args
        ckpt_name = get_last_ckpt_name(args, "." + args.save_model_as)
        self.save(ckpt_name, network, global_step, num_train_epochs)
        logger.info("model saved.")
