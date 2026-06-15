"""LoRA/LoHa/LoKr/GLoRA continue-training weight inspection.

This module is intentionally stateless: callers provide the already-loaded
training config and project root. That keeps config preflight and training
process management from importing each other just to validate a safetensors
header.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

CONTINUE_LORA_KINDS = {"LoRA", "DoRA", "LoHa", "LoKr", "GLoRA"}
CONTINUE_LORA_ACCEPTED_LORA_SPECS = {
    "",
    "lora",
    "dora",
    "standard",
    "ortho",
    "ortholora",
    "tlora",
    "t_lora",
}
CONTINUE_LORA_UNSUPPORTED_SPEC_TOKENS = (
    "hydra",
    "chimera",
    "stacked",
    "fera",
    "moe",
    "reft",
    "postfix",
    "ip_adapter",
    "easycontrol",
    "soft_tokens",
)
CONTINUE_LORA_UNSUPPORTED_KEY_FRAGMENTS = (
    ".lora_ups.",
    ".lora_downs.",
    ".lora_up_weight",
    ".lora_down_weight",
    ".lora_up_c_weight",
    ".lora_up_f_weight",
    ".lora_down_c.",
    ".lora_down_f.",
    ".router.",
    "freq_router.",
    "content_router.",
)


def inspect_continue_lora_weight(
    path: str,
    *,
    variant: str = "lora",
    preset: str = "default",
    methods_subdir: str = "gui-methods",
    cfg: Mapping[str, Any] | None = None,
    config_error: Exception | str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    del preset
    project_root = Path(root).resolve() if root is not None else Path.cwd().resolve()
    raw_path = str(path or "").strip()
    if not raw_path:
        raise ValueError("请填写 LoRA/LoHa/LoKr/GLoRA 权重路径")
    weight_path = _resolve_display_path(raw_path, project_root)
    if weight_path is None:
        raise ValueError("权重路径不合法")
    if not _path_exists(weight_path):
        raise FileNotFoundError("权重文件不存在")
    if not weight_path.is_file():
        raise ValueError("权重路径不是文件")
    if weight_path.suffix.lower() != ".safetensors":
        raise ValueError("只支持 .safetensors 权重文件")
    if not os.access(weight_path, os.R_OK):
        raise ValueError("权重文件不可读取")

    metadata, keys = _read_safetensors_header(weight_path)
    kind = _detect_continue_lora_kind(keys, metadata)
    if kind not in CONTINUE_LORA_KINDS:
        raise ValueError("这个 safetensors 未识别为 LoRA、DoRA、LoHa、LoKr 或 GLoRA 权重")

    compatible, message = _continue_lora_compatibility(
        kind,
        variant=variant,
        methods_subdir=methods_subdir,
        cfg=cfg,
        config_error=config_error,
    )
    return {
        "ok": True,
        "name": weight_path.name,
        "abs_path": str(weight_path),
        "path": _display_project_path(weight_path, project_root),
        "kind": kind,
        "metadata": _safe_continue_lora_metadata(metadata),
        "compatible": compatible,
        "message": message,
    }


def _read_safetensors_header(path: Path) -> tuple[dict[str, str], list[str]]:
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = {str(k): str(v) for k, v in (f.metadata() or {}).items()}
            keys = list(f.keys())
        return metadata, keys
    except Exception as exc:
        raise ValueError(f"读取 safetensors 权重失败: {exc}") from exc


def _detect_continue_lora_kind(keys: list[str], metadata: dict[str, str]) -> str:
    from networks.registry import continue_weight_kind_from_plugins

    meta_spec = str(metadata.get("ss_network_spec") or "").strip().lower()
    adapter_variant = str(metadata.get("ss_adapter_variant") or "").strip().lower()
    lowered_keys = [str(key).lower() for key in keys]
    plugin_kind = continue_weight_kind_from_plugins(keys, metadata)
    if plugin_kind:
        return plugin_kind
    if _continue_lora_has_unsupported_structure(lowered_keys, metadata, meta_spec):
        return ""
    has_plain_lora_keys = any(
        key.endswith(".lora_down.weight") or key.endswith(".lora_up.weight")
        for key in lowered_keys
    )
    has_dora_magnitude = any(
        key.endswith((".dora_scale", ".dora_magnitude", ".magnitude"))
        for key in lowered_keys
    )
    if has_plain_lora_keys and (
        meta_spec == "dora" or adapter_variant == "dora" or has_dora_magnitude
    ):
        return "DoRA"
    if has_plain_lora_keys and meta_spec in CONTINUE_LORA_ACCEPTED_LORA_SPECS:
        return "LoRA"
    return ""


def _continue_lora_has_unsupported_structure(
    lowered_keys: list[str],
    metadata: dict[str, str],
    meta_spec: str,
) -> bool:
    if any(token in meta_spec for token in CONTINUE_LORA_UNSUPPORTED_SPEC_TOKENS):
        return True
    use_moe_style = str(metadata.get("ss_use_moe_style") or "").strip().lower()
    if use_moe_style not in {"", "false", "none"}:
        return True
    router_source = str(metadata.get("ss_router_source") or "").strip().lower()
    if router_source not in {"", "false", "none"}:
        return True
    if _truthy(metadata.get("ss_use_chimera_hydra")):
        return True
    if any(
        key in metadata for key in ("ss_num_experts_content", "ss_num_experts_freq")
    ):
        return True
    for key in lowered_keys:
        if key.startswith("reft_"):
            return True
        if key.endswith(".s_p") or key.endswith(".s_q"):
            return True
        if any(fragment in key for fragment in CONTINUE_LORA_UNSUPPORTED_KEY_FRAGMENTS):
            return True
    return False


def _safe_continue_lora_metadata(metadata: dict[str, str]) -> dict[str, str]:
    allowed = (
        "ss_network_spec",
        "ss_output_name",
        "ss_epoch",
        "ss_steps",
        "ss_num_epochs",
        "ss_max_train_steps",
        "ss_learning_rate",
        "ss_network_dim",
        "ss_network_alpha",
        "ss_adapter_variant",
        "ss_dora_compatible_export",
        "modelspec.architecture",
        "modelspec.implementation",
    )
    return {key: str(metadata[key]) for key in allowed if key in metadata}


def _continue_lora_compatibility(
    kind: str,
    *,
    variant: str,
    methods_subdir: str,
    cfg: Mapping[str, Any] | None,
    config_error: Exception | str | None,
) -> tuple[bool, str]:
    if cfg is None and config_error is not None:
        return False, f"无法读取当前训练配置用于兼容性检查: {config_error}"
    current_kind = _continue_lora_config_kind(variant, methods_subdir, dict(cfg or {}))
    if current_kind == "LoHa":
        if kind == "LoHa":
            return True, "兼容：当前变体为 LoHa，会基于该 LoHa 权重热启动训练"
        return (
            False,
            f"{kind} 权重不能直接用于 LoHa 变体；请切换到匹配的训练变体",
        )
    if current_kind == "LoKr":
        if kind == "LoKr":
            return True, "兼容：当前变体为 LoKr，会基于该 LoKr 权重热启动训练"
        return (
            False,
            f"{kind} 权重不能直接用于 LoKr 变体；请切换到匹配的训练变体",
        )
    if current_kind == "GLoRA":
        if kind == "GLoRA":
            return True, "兼容：当前变体为 GLoRA，会基于该 GLoRA 权重热启动训练"
        return (
            False,
            f"{kind} 权重不能直接用于 GLoRA 变体；请切换到匹配的训练变体",
        )
    if current_kind == "DoRA":
        if kind == "DoRA":
            return (
                True,
                "兼容：当前配置已启用 DoRA，会基于该 DoRA 权重热启动训练",
            )
        return (
            False,
            f"{kind} 权重不能直接用于 DoRA 配置；请切换到匹配的训练变体",
        )
    if current_kind == "LoRA":
        if kind == "LoRA":
            return (
                True,
                "兼容：当前配置属于 LoRA 家族，会基于该 LoRA 权重热启动训练",
            )
        if kind == "DoRA":
            return (
                False,
                "DoRA 权重需要当前配置启用 DoRA，请先在 LoRA 结构中选择 DoRA",
            )
        if kind == "LoHa":
            return False, "LoHa 权重需要当前变体为 loha，请先切换到 LoHa 变体"
        if kind == "LoKr":
            return False, "LoKr 权重需要当前变体为 lokr，请先切换到 LoKr 变体"
        return False, "GLoRA 权重需要当前变体为 glora，请先切换到 GLoRA 变体"
    return False, "当前只支持 LoRA / DoRA / LoHa / LoKr / GLoRA 家族配置权重热启动"


def _continue_lora_config_kind(
    variant: str, methods_subdir: str, cfg: dict[str, Any]
) -> str:
    module_name = str(cfg.get("network_module") or "")
    variant_key = str(variant or "").strip().lower()
    if _truthy(cfg.get("use_glora")) or variant_key == "glora":
        return "GLoRA"
    if _truthy(cfg.get("use_loha")) or variant_key == "loha":
        return "LoHa"
    if _truthy(cfg.get("use_lokr")) or variant_key == "lokr":
        return "LoKr"
    if _truthy(cfg.get("dora_wd")) or variant_key == "dora":
        return "DoRA"
    if module_name and "lora_anima" not in module_name:
        return ""
    if str(methods_subdir or "") == "gui-methods":
        blocked = (
            "hydra",
            "fera",
            "reft",
            "ip_adapter",
            "easycontrol",
            "soft_tokens",
            "postfix",
            "chimera",
        )
        if any(token in variant_key for token in blocked):
            return ""
    if _truthy(cfg.get("use_chimera_hydra")):
        return ""
    if _truthy(cfg.get("add_reft")):
        return ""
    if str(cfg.get("use_moe_style") or "").strip().lower() not in {
        "",
        "false",
        "none",
    }:
        return ""
    return "LoRA"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_display_path(value: str, root: Path) -> Path | None:
    raw = str(value or "").replace("\\", "/").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def _display_project_path(value: str | Path, root: Path) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    if not raw:
        return ""
    path = Path(raw)
    if not path.is_absolute():
        return path.as_posix().strip("/")
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return raw


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False
