"""Adapter type detection and analysis payload bases."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from web.services.weight_analysis.constants import (
    MAX_METADATA_ITEMS,
    UNSUPPORTED_KEY_FRAGMENTS,
    UNSUPPORTED_SPEC_TOKENS,
)
from web.services.weight_analysis.context import call


def _detect_adapter_type(keys: list[str], metadata: Mapping[str, str]) -> tuple[str, dict[str, Any]]:
    lowered_keys = [str(key).lower() for key in keys]
    meta_spec = str(metadata.get("ss_network_spec") or metadata.get("network_spec") or "").strip().lower()
    unsupported_reason = _unsupported_reason(lowered_keys, metadata, meta_spec)
    if unsupported_reason:
        return _label_from_meta_or_keys(lowered_keys, meta_spec), {"unsupported": True, "reason": unsupported_reason}
    if any(key.endswith(".hada_w1_a") or key.endswith(".hada_w2_a") for key in lowered_keys) or meta_spec == "loha":
        return "LoHa", {"unsupported": False, "reason": ""}
    if any(key.endswith(".lokr_w1") or key.endswith(".lokr_w2") for key in lowered_keys) or meta_spec == "lokr":
        return "LoKr", {"unsupported": False, "reason": ""}
    if any(key.endswith(".lora_down.weight") or key.endswith(".lora_up.weight") for key in lowered_keys):
        return "LoRA", {"unsupported": False, "reason": ""}
    return "Unknown", {"unsupported": True, "reason": "未识别为第一版支持的 LoRA / LoHa / LoKr 权重结构。"}


def _unsupported_reason(lowered_keys: list[str], metadata: Mapping[str, str], meta_spec: str) -> str:
    adapter_variant = str(metadata.get("ss_adapter_variant") or "").strip().lower()
    if (
        meta_spec == "dora"
        or adapter_variant == "dora"
        or any(
            key.endswith((".dora_scale", ".dora_magnitude"))
            for key in lowered_keys
        )
    ):
        return (
            "DoRA 需要底模权重才能还原幅度归一后的真实 ΔW，"
            "静态分析页暂不做近似重建。"
        )
    if any(token in meta_spec for token in UNSUPPORTED_SPEC_TOKENS):
        return "该权重疑似 Hydra / Chimera / FeRA / ReFT / VeRA 等结构，第一版暂不重建完整 ΔW。"
    use_moe_style = str(metadata.get("ss_use_moe_style") or "").strip().lower()
    if use_moe_style not in {"", "false", "none", "0"}:
        return "该权重包含 MoE/Hydra 风格路由结构，第一版暂不支持完整 ΔW 解释。"
    router_source = str(metadata.get("ss_router_source") or "").strip().lower()
    if router_source not in {"", "false", "none", "0"}:
        return "该权重包含 router 路由参数，第一版暂不支持完整 ΔW 解释。"
    if _truthy(metadata.get("ss_use_chimera_hydra")):
        return "该权重为 Chimera/Hydra 结构，第一版暂不支持完整 ΔW 解释。"
    if any(key in metadata for key in ("ss_num_experts_content", "ss_num_experts_freq")):
        return "该权重包含 Chimera 专家元数据，第一版暂不支持完整 ΔW 解释。"
    if any(fragment in key for key in lowered_keys for fragment in UNSUPPORTED_KEY_FRAGMENTS):
        return "该权重包含第一版暂不支持的专家、路由、ReFT 或 VeRA 参数。"
    return ""


def _label_from_meta_or_keys(lowered_keys: list[str], meta_spec: str) -> str:
    if "dora" in meta_spec or any(
        key.endswith((".dora_scale", ".dora_magnitude")) for key in lowered_keys
    ):
        return "DoRA"
    if "vera" in meta_spec or any("vera_lambda_" in key for key in lowered_keys):
        return "VeRA"
    if any(token in meta_spec for token in ("chimera", "hydra", "fera", "moe")):
        return "Hydra/Chimera"
    if "reft" in meta_spec or any(key.startswith("reft_") for key in lowered_keys):
        return "ReFT"
    return meta_spec.upper() if meta_spec else "Unsupported"


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _base_payload(
    weight_path: Path,
    metadata: Mapping[str, str],
    adapter_type: str,
    unsupported: Mapping[str, Any],
) -> dict[str, Any]:
    stat = weight_path.stat()
    return {
        "ok": True,
        "file": {
            "path": call("_display_path", weight_path),
            "abs_path": str(weight_path.resolve()),
            "name": weight_path.name,
            "size_bytes": stat.st_size,
            "mtime": stat.st_mtime,
            "mtime_text": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metadata": _safe_metadata(metadata),
        "adapter_type": adapter_type,
        "unsupported": dict(unsupported),
        "analysis_kind": "static_delta_weight_norm",
        "disclaimer": "这里分析的是 safetensors 内的静态 ΔW 范数，不是 prompt 激活图；不会启动跑图或占用 GPU。",
    }


def _uploaded_base_payload(
    filename: str,
    size_bytes: int,
    metadata: Mapping[str, str],
    adapter_type: str,
    unsupported: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "ok": True,
        "file": {
            "path": f"uploaded://{filename}",
            "abs_path": "",
            "name": filename,
            "size_bytes": int(size_bytes),
            "mtime": None,
            "mtime_text": "拖拽上传临时分析",
            "source": "upload",
        },
        "metadata": _safe_metadata(metadata),
        "adapter_type": adapter_type,
        "unsupported": dict(unsupported),
        "analysis_kind": "static_delta_weight_norm",
        "disclaimer": "这里分析的是 safetensors 内的静态 ΔW 范数，不是 prompt 激活图；拖入文件只做临时读取，不写入权重目录、不启动跑图或占用 GPU。",
    }


def _safe_metadata(metadata: Mapping[str, str]) -> dict[str, str]:
    preferred = (
        "ss_network_spec",
        "ss_output_name",
        "ss_epoch",
        "ss_steps",
        "ss_num_epochs",
        "ss_max_train_steps",
        "ss_learning_rate",
        "ss_network_dim",
        "ss_network_alpha",
        "modelspec.architecture",
        "modelspec.implementation",
        "ss_base_model_version",
        "ss_training_started_at",
    )
    safe: dict[str, str] = {}
    for key in preferred:
        if key in metadata:
            safe[key] = str(metadata[key])
    for key in sorted(metadata):
        if key in safe:
            continue
        if len(safe) >= MAX_METADATA_ITEMS:
            break
        if key.startswith(("ss_", "modelspec.")):
            safe[key] = str(metadata[key])
    return safe



def _int_or_none(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None
