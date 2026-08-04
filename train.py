# Anima LoRA training script (merged standalone)

import argparse
import sys

from library.runtime.allocator import default_expandable_segments

if default_expandable_segments():
    print(
        "Anima: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        "(default; set ANIMA_EXPANDABLE_SEGMENTS=0 to disable)"
    )

import torch

from library.training.method_adapter import (
    MethodAdapter,
)
from library.training.bootstrap import TrainingBootstrap
from library.training.contexts import RuntimeState
from library.training.trainer_bootstrap_mixin import TrainerBootstrapMixin
from library.training.trainer_logging_mixin import TrainerLoggingMixin
from library.training.trainer_network_mixin import TrainerNetworkMixin
from library.log import setup_logging

setup_logging()
import logging  # noqa: E402

logger = logging.getLogger(__name__)

# Compatibility shims for tests/callers that still import private helpers from train.py.
import signal  # noqa: E402

from library.training.train_bootstrap import (  # noqa: E402
    collect_bucket_resolutions as _collect_bucket_resolutions,
    collect_compile_resolutions as _collect_compile_resolutions,
    decode_deferred_samples_safely as _decode_deferred_samples_safely,
    normalize_sample_args as _normalize_sample_args,
    resolve_block_swap_profile_jsonl as _resolve_block_swap_profile_jsonl,
    sample_preview_enabled as _sample_preview_enabled,
)
from library.training.probes import (  # noqa: E402
    attach_peak_probe_to_network as _attach_peak_probe_to_network,
    maybe_probe as _maybe_probe,
    maybe_probe_components as _maybe_probe_components,
)
from library.training.v100_flash import (  # noqa: E402, F401
    flash_attn_v100_doc as _flash_attn_v100_doc,
    resolve_v100_flash_stability as _resolve_v100_flash_stability,
)


def _install_stop_signal_handlers() -> None:
    """Make SIGTERM follow the same cleanup path as Ctrl-C.

    Kept in train.py so source-level tests can see the SIGTERM binding.
    """

    if not hasattr(signal, "SIGTERM"):
        return

    def _raise_keyboard_interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)


class AnimaTrainer(
    TrainerLoggingMixin,
    TrainerNetworkMixin,
    TrainerBootstrapMixin,
):
    def __init__(self, bootstrap: TrainingBootstrap | None = None):
        self.bootstrap = bootstrap or TrainingBootstrap()
        self.sample_prompts_te_outputs = None
        self.sample_prompts_snapshot = None
        self.memory_probe = None
        self.peak_probe = None
        self._padding_mask_cache = {}
        # Per-method extensions (EasyControl, IP-Adapter, …). Resolved
        # from args+network in train() right after _create_and_apply_network.
        self._adapters: list[MethodAdapter] = []
        # Feature-specific per-run state — see ``RuntimeState``.
        self._state = RuntimeState()

    # Source-contract methods that tests match against train.py text.

    def get_text_encoder_outputs_caching_strategy(self, args, weight_dtype: torch.dtype):
        # Source-contract tests expect cache_dtype=weight_dtype to remain visible here.
        from library.training.anima_strategies import (
            get_text_encoder_outputs_caching_strategy as _impl,
        )

        return _impl(args, weight_dtype)  # cache_dtype=weight_dtype

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        from library.training.anima_strategies import get_models_for_text_encoding as _impl

        return _impl(args, accelerator, text_encoders)

    def sample_images(
        self,
        accelerator,
        args,
        epoch,
        global_step,
        device,
        vae,
        tokenizer,
        text_encoder,
        unet,
        network=None,
    ):
        from library.training.sample_preview import sample_images as _sample_images

        # Source-contract: sample_prompts_snapshot=self.sample_prompts_snapshot
        return _sample_images(
            self,
            accelerator,
            args,
            epoch,
            global_step,
            device,
            vae,
            tokenizer,
            text_encoder,
            unet,
            network=network,
        )

    def cache_text_encoder_outputs_if_needed(
        self,
        args,
        accelerator,
        text_encoders,
        dataset,
    ):
        from library.training.text_encoder_cache import (
            cache_text_encoder_outputs_if_needed as _cache_text_encoder_outputs_if_needed,
        )

        return _cache_text_encoder_outputs_if_needed(
            self,
            args,
            accelerator,
            text_encoders,
            dataset,
        )


def setup_parser() -> argparse.ArgumentParser:
    from library.training.cli_args import setup_parser as _setup_parser

    return _setup_parser()


from library.training.cli_entry import build_network_extras  # noqa: E402


def _install_crash_reporter(argv: list[str]) -> None:
    from library.training.crash_reporter import _install_crash_reporter as _impl

    return _impl(argv)


if __name__ == "__main__":
    from library.training.cli_entry import run_training_cli

    run_training_cli(
        setup_parser=setup_parser,
        trainer_factory=AnimaTrainer,
        install_stop_signal_handlers=_install_stop_signal_handlers,
        install_crash_reporter=_install_crash_reporter,
    )
