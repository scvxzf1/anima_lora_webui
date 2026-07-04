"""Shared training compatibility checks for optimization flags.

The helpers here are intentionally pure: they inspect a mapping/object and
return errors, warnings, and suggested mutations. CLI and Web preflight can
consume the same rules without importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


VALID_SELECTIVE_CHECKPOINTS = {
    "off",
    "adapter_aware",
    "every_other",
    "mlp_only",
    "mlp_layer1_only",
    "peak_blocks_adapter_aware",
    "peak_blocks_mlp",
    "peak_blocks_mlp_layer1",
}


@dataclass(frozen=True)
class TrainingCompatIssue:
    code: str
    key: str
    message: str


@dataclass(frozen=True)
class TrainingCompatMutation:
    code: str
    key: str
    value: Any
    message: str


@dataclass(frozen=True)
class TrainingCompatResult:
    errors: tuple[TrainingCompatIssue, ...]
    warnings: tuple[TrainingCompatIssue, ...]
    mutations: tuple[TrainingCompatMutation, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


class _CompatBuilder:
    def __init__(self) -> None:
        self.errors: list[TrainingCompatIssue] = []
        self.warnings: list[TrainingCompatIssue] = []
        self.mutations: list[TrainingCompatMutation] = []

    def error(self, code: str, key: str, message: str) -> None:
        self.errors.append(TrainingCompatIssue(code, key, message))

    def warning(self, code: str, key: str, message: str) -> None:
        self.warnings.append(TrainingCompatIssue(code, key, message))

    def mutate(self, code: str, key: str, value: Any, message: str) -> None:
        self.mutations.append(TrainingCompatMutation(code, key, value, message))

    def build(self) -> TrainingCompatResult:
        return TrainingCompatResult(
            errors=tuple(self.errors),
            warnings=tuple(self.warnings),
            mutations=tuple(self.mutations),
        )


def _get(config: Mapping[str, Any] | object, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _bool_value(value: Any, fallback: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", "none", ""}:
        return False
    return fallback


def _int_value(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _float_value(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def check_training_compat(config: Mapping[str, Any] | object) -> TrainingCompatResult:
    """Validate optimization-flag combinations used by training and Web preflight."""

    out = _CompatBuilder()

    selective_checkpoint = str(_get(config, "selective_checkpoint", "off") or "off").strip().lower()
    gradient_checkpointing = _bool_value(_get(config, "gradient_checkpointing"), False)
    cpu_offload_checkpointing = _bool_value(_get(config, "cpu_offload_checkpointing"), False)
    unsloth_offload_checkpointing = _bool_value(_get(config, "unsloth_offload_checkpointing"), False)
    blocks_to_swap = _int_value(_get(config, "blocks_to_swap"), 0)
    torch_compile = _bool_value(_get(config, "torch_compile"), False)
    use_lokr = _bool_value(_get(config, "use_lokr"), False)
    network_module = str(_get(config, "network_module", "") or "")
    functional_loss_weight = _float_value(_get(config, "functional_loss_weight"), 0.0)
    dynamo_backend = str(_get(config, "dynamo_backend", "") or "")
    compile_inductor_mode = _get(config, "compile_inductor_mode", None)

    if selective_checkpoint not in VALID_SELECTIVE_CHECKPOINTS:
        out.error(
            "invalid_selective_checkpoint",
            "selective_checkpoint",
            "--selective_checkpoint must be one of: "
            + ", ".join(sorted(VALID_SELECTIVE_CHECKPOINTS)),
        )

    if blocks_to_swap < 0:
        out.error(
            "negative_blocks_to_swap",
            "blocks_to_swap",
            "blocks_to_swap must be greater than or equal to 0.",
        )

    block_swap_enabled = blocks_to_swap > 0
    selective_enabled = selective_checkpoint != "off"

    if selective_enabled and gradient_checkpointing:
        out.error(
            "selective_full_gradient_checkpointing",
            "gradient_checkpointing",
            "--selective_checkpoint is a selective DiT checkpoint mode; "
            "do not combine it with full --gradient_checkpointing.",
        )
    if selective_enabled and cpu_offload_checkpointing:
        out.error(
            "selective_cpu_offload",
            "cpu_offload_checkpointing",
            "--selective_checkpoint does not support CPU activation offload.",
        )
    if selective_enabled and unsloth_offload_checkpointing:
        out.error(
            "selective_unsloth_offload",
            "unsloth_offload_checkpointing",
            "--selective_checkpoint cannot be combined with "
            "--unsloth_offload_checkpointing.",
        )

    if block_swap_enabled and cpu_offload_checkpointing:
        out.error(
            "block_swap_cpu_offload",
            "cpu_offload_checkpointing",
            "blocks_to_swap supports standard gradient_checkpointing, but is "
            "not supported with cpu_offload_checkpointing",
        )

    if unsloth_offload_checkpointing:
        if not gradient_checkpointing and not selective_enabled:
            out.warning(
                "unsloth_enables_gradient_checkpointing",
                "gradient_checkpointing",
                "unsloth_offload_checkpointing is enabled, so gradient_checkpointing is also enabled",
            )
            out.mutate(
                "unsloth_enables_gradient_checkpointing",
                "gradient_checkpointing",
                True,
                "unsloth_offload_checkpointing is enabled, so gradient_checkpointing is also enabled",
            )
        if cpu_offload_checkpointing:
            out.error(
                "unsloth_cpu_offload",
                "cpu_offload_checkpointing",
                "Cannot use both --unsloth_offload_checkpointing and --cpu_offload_checkpointing",
            )
        if block_swap_enabled:
            out.error(
                "block_swap_unsloth_offload",
                "unsloth_offload_checkpointing",
                "blocks_to_swap supports standard gradient_checkpointing, but is "
                "not supported with unsloth_offload_checkpointing",
            )

    if cpu_offload_checkpointing and not gradient_checkpointing:
        out.warning(
            "cpu_offload_without_gradient_checkpointing",
            "cpu_offload_checkpointing",
            "cpu_offload_checkpointing only affects activation checkpointing; "
            "enable gradient_checkpointing or disable cpu_offload_checkpointing.",
        )

    if use_lokr and gradient_checkpointing and torch_compile:
        out.warning(
            "lokr_full_checkpoint_compile",
            "torch_compile",
            "LoKr with full gradient_checkpointing under torch_compile is "
            "experimental. Training keeps torch_compile enabled, pins larger "
            "Dynamo graph budgets, and traces the factorized LoKr forward "
            "in-graph; blocks_to_swap remains independent.",
        )

    if block_swap_enabled:
        if torch_compile and dynamo_backend == "cudagraphs":
            out.warning(
                "block_swap_cudagraphs_disable_compile",
                "torch_compile",
                "blocks_to_swap moves DiT block weights between CPU/GPU, "
                "so dynamo_backend='cudagraphs' is unsafe. Disabling torch_compile.",
            )
            out.mutate(
                "block_swap_cudagraphs_disable_compile",
                "torch_compile",
                False,
                "blocks_to_swap moves DiT block weights between CPU/GPU, "
                "so dynamo_backend='cudagraphs' is unsafe. Disabling torch_compile.",
            )
            torch_compile = False

        if torch_compile and compile_inductor_mode in {"reduce-overhead", "max-autotune"}:
            safe_mode = (
                "max-autotune-no-cudagraphs"
                if compile_inductor_mode == "max-autotune"
                else None
            )
            out.warning(
                "block_swap_compile_mode_cudagraphs",
                "compile_inductor_mode",
                "blocks_to_swap is incompatible with Inductor CUDAGraph modes "
                f"({compile_inductor_mode!r}); using {safe_mode or 'default'} mode instead.",
            )
            out.mutate(
                "block_swap_compile_mode_cudagraphs",
                "compile_inductor_mode",
                safe_mode,
                "blocks_to_swap is incompatible with Inductor CUDAGraph modes "
                f"({compile_inductor_mode!r}); using {safe_mode or 'default'} mode instead.",
            )

        if network_module == "networks.methods.soft_tokens":
            out.error(
                "block_swap_soft_tokens",
                "blocks_to_swap",
                "blocks_to_swap is not supported with soft_tokens. "
                "Keep blocks_to_swap=0 for this multi-forward method.",
            )
        if functional_loss_weight > 0.0:
            out.error(
                "block_swap_functional_loss",
                "blocks_to_swap",
                "blocks_to_swap is not supported with functional_loss_weight > 0. "
                "Disable block swap for postfix/functional multi-forward training.",
            )

    return out.build()


def apply_training_compat_mutations(
    target: Mapping[str, Any] | object,
    result: TrainingCompatResult,
) -> None:
    """Apply suggested mutations to a mutable mapping or argparse namespace."""

    for mutation in result.mutations:
        if isinstance(target, dict):
            target[mutation.key] = mutation.value
        else:
            setattr(target, mutation.key, mutation.value)
