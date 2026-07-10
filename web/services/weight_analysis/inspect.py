"""Public inspect entrypoints for file and upload analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from web.services.weight_analysis.constants import MAX_UPLOAD_WEIGHT_BYTES, WEIGHT_EXTS
from web.services.weight_analysis.context import call


def inspect_weight(path: str, *, task: dict[str, Any] | None = None) -> dict[str, Any]:
    """Inspect one safetensors file and summarize static ΔW energy."""

    weight_path = call("resolve_analysis_weight", path, task=task)
    metadata, keys = call("_read_safetensors_header", weight_path)
    adapter_type, unsupported = call("_detect_adapter_type", keys, metadata)
    base_payload = call("_base_payload", weight_path, metadata, adapter_type, unsupported)
    return call(
        "_inspect_loaded_weight",
        file_name=weight_path.name,
        metadata=metadata,
        adapter_type=adapter_type,
        unsupported=unsupported,
        base_payload=base_payload,
        load_state=lambda: call("_load_safetensors_tensors", weight_path),
    )


def inspect_weight_bytes(data: bytes, *, filename: str = "uploaded.safetensors") -> dict[str, Any]:
    """Inspect an uploaded safetensors file without persisting it to outputs."""

    clean_name = Path(str(filename or "uploaded.safetensors").replace("\\", "/")).name
    if Path(clean_name).suffix.lower() not in WEIGHT_EXTS:
        raise ValueError("只支持 .safetensors 权重文件")
    if len(data) > MAX_UPLOAD_WEIGHT_BYTES:
        raise ValueError(f"拖入文件过大，最大支持 {MAX_UPLOAD_WEIGHT_BYTES // (1024 * 1024)} MiB")
    metadata, keys = call("_read_safetensors_header_bytes", data)
    adapter_type, unsupported = call("_detect_adapter_type", keys, metadata)
    base_payload = call("_uploaded_base_payload", clean_name, len(data), metadata, adapter_type, unsupported)
    return call(
        "_inspect_loaded_weight",
        file_name=clean_name,
        metadata=metadata,
        adapter_type=adapter_type,
        unsupported=unsupported,
        base_payload=base_payload,
        load_state=lambda: call("_load_safetensors_tensors_bytes", data),
    )


def _inspect_loaded_weight(
    *,
    file_name: str,
    metadata: Mapping[str, str],
    adapter_type: str,
    unsupported: Mapping[str, Any],
    base_payload: dict[str, Any],
    load_state: Any,
) -> dict[str, Any]:
    if unsupported.get("unsupported"):
        return {
            **base_payload,
            "summary": call("_empty_summary"),
            "layers": [],
            "component_summary": [],
            "block_summary": [],
            "style_top20": [],
            "character_top20": [],
            "heatmap": call("_empty_heatmap"),
        }

    state = load_state()
    try:
        layer_rows = call("_compute_layers", state, adapter_type, metadata)
    finally:
        state.clear()
    if not layer_rows:
        return {
            **base_payload,
            "unsupported": {
                "unsupported": True,
                "reason": "没有找到可重建 ΔW 的 LoRA / LoHa / LoKr 层。",
            },
            "summary": call("_empty_summary"),
            "layers": [],
            "component_summary": [],
            "block_summary": [],
            "style_top20": [],
            "character_top20": [],
            "heatmap": call("_empty_heatmap"),
        }

    layers = call("_finalize_layer_contributions", layer_rows)
    return {
        **base_payload,
        "summary": call("_summary", file_name, adapter_type, metadata, layers),
        "layers": layers,
        "component_summary": call("_component_summary", layers),
        "block_summary": call("_block_summary", layers),
        "style_top20": call("_top_candidates", layers, kind="style"),
        "character_top20": call("_top_candidates", layers, kind="character"),
        "heatmap": call("_heatmap", layers),
    }
