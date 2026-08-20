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
KREA2_ATTN_MODES = {"", "torch", "flash", "sdpa"}
KREA2_SELECTIVE_CHECKPOINTS = {"off", "every_other"}
KREA2_UNSUPPORTED_ADAPTER_FLAGS = (
    "use_ip_adapter", "use_easycontrol", "use_byg", "use_lokr", "use_loha",
    "use_glora", "use_vera", "use_ortho", "use_chimera_hydra",
    "use_timestep_mask", "add_reft", "train_llm_adapter",
)


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
    model_family = str(_get(config, "model_family", "") or "").strip().lower()
    krea2_family = model_family == "krea2_raw"
    compile_dynamic_seq = _bool_value(_get(config, "compile_dynamic_seq"), False)
    v100_flash_stability = str(
        _get(config, "v100_flash_stability", "off") or "off"
    ).strip().lower()

    if selective_checkpoint not in VALID_SELECTIVE_CHECKPOINTS:
        out.error(
            "invalid_selective_checkpoint",
            "selective_checkpoint",
            "--selective_checkpoint must be one of: "
            + ", ".join(sorted(VALID_SELECTIVE_CHECKPOINTS)),
        )

    if krea2_family:
        adapter_conflicts = []
        if network_module not in {"", "networks.lora_anima"}:
            adapter_conflicts.append(f"network_module={network_module!r}")
        adapter_conflicts.extend(
            flag for flag in KREA2_UNSUPPORTED_ADAPTER_FLAGS
            if _bool_value(_get(config, flag), False)
        )
        moe_style = str(_get(config, "use_moe_style", "false") or "false").strip().lower()
        if moe_style not in {"", "0", "false", "none", "off"}:
            adapter_conflicts.append(f"use_moe_style={moe_style!r}")
        if _bool_value(_get(config, "route_per_layer"), False):
            adapter_conflicts.append("route_per_layer")
        router_source = str(_get(config, "router_source", "none") or "none").strip().lower()
        if router_source != "none":
            adapter_conflicts.append(f"router_source={router_source!r}")
        if _float_value(_get(config, "dora_wd"), 0.0) > 0:
            adapter_conflicts.append("dora_wd")
        if _int_value(_get(config, "step_expert_K"), 0) > 1:
            adapter_conflicts.append("step_expert_K")
        if functional_loss_weight > 0:
            adapter_conflicts.append("functional_loss_weight")
        if adapter_conflicts:
            out.error(
                "krea2_plain_lora_only",
                "network_module",
                "Krea-2 training currently supports only plain LoRA; unsupported: "
                + ", ".join(adapter_conflicts),
            )

        krea2_attn_mode = str(_get(config, "attn_mode", "") or "").strip().lower()
        if krea2_attn_mode not in KREA2_ATTN_MODES:
            out.error(
                "krea2_invalid_attn_mode",
                "attn_mode",
                "Krea-2 attn_mode supports only torch or flash (sdpa is a "
                f"torch alias); got {krea2_attn_mode!r}.",
            )
        if compile_dynamic_seq:
            message = (
                "Krea-2 compile uses two fixed padded token-family graphs; "
                "compile_dynamic_seq will be disabled."
            )
            out.warning(
                "krea2_compile_dynamic_seq",
                "compile_dynamic_seq",
                message,
            )
            out.mutate(
                "krea2_compile_dynamic_seq",
                "compile_dynamic_seq",
                False,
                message,
            )
        normalized_inductor_mode = str(compile_inductor_mode or "default").strip().lower()
        if normalized_inductor_mode != "default":
            out.error(
                "krea2_compile_inductor_mode",
                "compile_inductor_mode",
                "Krea-2 compile supports only compile_inductor_mode=default; "
                f"got {compile_inductor_mode!r}.",
            )
        if selective_checkpoint not in KREA2_SELECTIVE_CHECKPOINTS:
            out.error(
                "krea2_selective_checkpoint",
                "selective_checkpoint",
                "Krea-2 selective_checkpoint supports only off or every_other; "
                f"got {selective_checkpoint!r}.",
            )
        if v100_flash_stability != "off":
            out.error(
                "krea2_v100_flash_stability",
                "v100_flash_stability",
                "v100_flash_stability is Anima-only and must be off for Krea-2.",
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

    base_compute = str(_get(config, "base_compute", "bf16") or "bf16").strip().lower()
    block_swap_transfer_dtype = str(
        _get(config, "block_swap_transfer_dtype", "bf16") or "bf16"
    ).strip().lower()
    convrot_active = base_compute in {
        "w8a16_convrot",
        "w8a8_convrot",
        "w8a16",
        "w8a8",
    }
    nf4_active = base_compute == "nf4"
    if convrot_active and block_swap_transfer_dtype in {"int8", "int8_linear", "i8"}:
        out.error(
            "convrot_block_swap_int8_mutex",
            "base_compute",
            "base_compute ConvRot paths are mutually exclusive with "
            "block_swap_transfer_dtype=int8 (double dequant / confused semantics). "
            "Use block_swap_transfer_dtype=bf16 when enabling ConvRot.",
        )
    # NF4 (Krea-2 QLoRA) 与 ConvRot 互斥: ConvRot 是 anima cross-attn/AdaLN 专属
    # 量化路径, NF4 是 Krea-2 single-stream 通用 bnb 4-bit, 两者架构假设不同.
    if nf4_active and convrot_active:
        out.error(
            "nf4_convrot_mutex",
            "base_compute",
            "base_compute=nf4 is mutually exclusive with ConvRot paths "
            "(w8a16_convrot/w8a8_convrot). NF4 is the Krea-2 QLoRA path; "
            "ConvRot is anima-specific. Pick one.",
        )
    # NF4 × block_swap 已验证通过 (方向 A: deepcopy master + Params4bit.to() 整体
    # 搬运, offloading.py isinstance 分流不碰 bf16/int8/fp8 路径). 端到端探针
    # (probe_nf4_blockswap.py, PG199, 1024, swap=4, 30 步): host RAM 18.18GB
    # (bf16 路径 22.64GB master 单项超它曾在 62GB 机宕机), GPU 10.26GB, loss
    # 0.0084->0.0017 单调下降, LoRA grad 非零, DiT frozen 不变. 保留 warning:
    # NF4 block swap 主战场是 host RAM (master 5.66GB) 而非 GPU, 提醒用户关注
    # pinned master 内存预算.
    if nf4_active and block_swap_enabled:
        out.warning(
            "nf4_block_swap_host_ram",
            "base_compute",
            "base_compute=nf4 + blocks_to_swap verified (offloader NF4 path "
            "via Params4bit integral transport, end-to-end probe green). "
            "Main constraint is host RAM: pinned NF4 masters ~5.7GB (bf16 "
            "path 22.64GB), watch pinned-master memory budget on low-RAM hosts.",
        )
    # nf4_prequantized_path: 只在 base_compute=nf4 时生效. 给了路径但 base_compute
    # 不是 nf4 → 路径被静默忽略, 提醒用户 (常见于从 NF4 配置切回 bf16 忘删路径).
    nf4_path = _get(config, "nf4_prequantized_path", None)
    if nf4_path and not nf4_active:
        out.warning(
            "nf4_path_ignored",
            "nf4_prequantized_path",
            "--nf4_prequantized_path only takes effect with --base_compute nf4; "
            "ignored under current base_compute. Drop the path or switch to nf4.",
        )
    if base_compute not in {
        "bf16",
        "fp16",
        "none",
        "off",
        "w8a16_convrot",
        "w8a8_convrot",
        "w8a16",
        "w8a8",
        "nf4",
    }:
        out.error(
            "invalid_base_compute",
            "base_compute",
            f"unknown base_compute={base_compute!r}; expected bf16 | "
            "w8a16_convrot | w8a8_convrot | nf4",
        )

    # ConvRot on attention linears + flash + torch.compile can materialize
    # float32 Q/K/V into FlashAttention (FA only accepts fp16/bf16). The
    # dispatcher now hard-casts at the flash entry, so this is informational
    # rather than a hard block — still surface so operators know the combo is
    # on the known-risk path.
    attn_mode = str(_get(config, "attn_mode", "") or "").strip().lower()
    convrot_scope = str(_get(config, "convrot_scope", "mlp") or "mlp").strip().lower()
    convrot_touches_attn = False
    if convrot_active:
        scope_tokens = {item.strip() for item in convrot_scope.split(",") if item.strip()}
        attn_scope_tokens = {
            "all",
            "attention",
            "attn",
            "self",
            "self_attn",
            "self_attn_qkv",
            "self_qkv",
            "self_attn_out",
            "self_out",
            "cross",
            "cross_attn",
            "cross_attn_q",
            "cross_q",
            "cross_attn_kv",
            "cross_kv",
            "cross_attn_out",
            "cross_out",
            "attention_out",
            "attn_out",
        }
        convrot_touches_attn = bool(scope_tokens & attn_scope_tokens)
    if (
        convrot_active
        and convrot_touches_attn
        and attn_mode in {"flash", "flash4"}
        and torch_compile
    ):
        out.warning(
            "convrot_attn_flash_compile_dtype",
            "base_compute",
            "base_compute ConvRot with convrot_scope covering attention "
            f"({convrot_scope!r}) + attn_mode={attn_mode!r} + torch_compile can "
            "materialize float32 Q/K/V; FlashAttention only accepts fp16/bf16. "
            "The flash dispatcher hard-casts at entry as a safety net. Prefer "
            "convrot_scope=mlp to avoid the cast tax if attention quant is not required.",
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
