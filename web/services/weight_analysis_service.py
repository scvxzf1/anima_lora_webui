"""Static safetensors ΔW analysis for LoRA / LoHa / LoKr weights.

The service deliberately stays CPU-only and model-free: it reads tensors from a
single ``.safetensors`` file, reconstructs equivalent adapter deltas where the
layout is supported, and summarizes static weight energy.  It does not load the
DiT, run prompt inference, or write back to user checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping
from urllib.parse import unquote, urlparse

import torch

from web.services import preview_service, settings_service

ROOT = Path(__file__).resolve().parents[2]
WEIGHT_EXTS = {".safetensors"}
MAX_ANALYSIS_WEIGHT_LIMIT = 500
MAX_METADATA_ITEMS = 80
MAX_UPLOAD_WEIGHT_BYTES = 512 * 1024 * 1024
TOP_LAYER_LIMIT = 20

STYLE_PRIORITY = {
    "mlp_layer1": 1.38,
    "cross_attn_k_proj": 1.34,
    "cross_attn_v_proj": 1.34,
    "self_attn_output_proj": 1.24,
    "self_attn_q_proj": 1.12,
    "self_attn_v_proj": 1.12,
    "cross_attn_output_proj": 1.08,
    "mlp_layer2": 0.86,
}
CHARACTER_PRIORITY = {
    "self_attn_q_proj": 1.28,
    "self_attn_v_proj": 1.26,
    "cross_attn_q_proj": 1.20,
    "cross_attn_k_proj": 1.14,
    "cross_attn_v_proj": 1.14,
    "self_attn_k_proj": 1.10,
    "self_attn_output_proj": 1.02,
    "cross_attn_output_proj": 0.98,
    "mlp_layer1": 0.86,
    "mlp_layer2": 0.82,
}
UNSUPPORTED_SPEC_TOKENS = (
    "hydra",
    "chimera",
    "fera",
    "moe",
    "reft",
    "vera",
    "postfix",
    "ip_adapter",
    "easycontrol",
    "soft_tokens",
)
UNSUPPORTED_KEY_FRAGMENTS = (
    ".lora_ups.",
    ".lora_downs.",
    ".lora_up_weight",
    ".lora_down_weight",
    ".router.",
    "freq_router.",
    "content_router.",
    ".s_p",
    ".s_q",
    "vera_lambda_",
)
BLOCK_RE = re.compile(r"(?:^|_)blocks_(?P<block>\d+)_(?P<component>.+)$")
LORA_SUFFIX = ".lora_down.weight"
LOHA_SUFFIX = ".hada_w1_a"
LOKR_SUFFIX = ".lokr_w1"


@dataclass(frozen=True)
class WeightListingContext:
    task: dict[str, Any] | None = None


def list_analysis_weights(
    *,
    task: dict[str, Any] | None = None,
    allow_latest_fallback: bool = True,
    training_service: Any | None = None,
    include_archived: bool = False,
    limit: int = MAX_ANALYSIS_WEIGHT_LIMIT,
) -> dict[str, Any]:
    """Return training weight candidates usable by the analysis page."""

    limit = max(1, min(int(limit or MAX_ANALYSIS_WEIGHT_LIMIT), MAX_ANALYSIS_WEIGHT_LIMIT))
    if task:
        payload = preview_service.list_training_weights(
            task,
            allow_latest_fallback=allow_latest_fallback,
        )
        weights = [
            _analysis_weight_meta(item)
            for item in payload.get("weights", [])[:limit]
            if isinstance(item, dict)
        ]
        return {
            "ok": True,
            "directory": payload.get("directory", ""),
            "directory_exists": bool(payload.get("directory_exists")),
            "count": len(weights),
            "total": payload.get("total", len(weights)),
            "task_count": payload.get("task_count", 0),
            "weights": weights,
            "message": payload.get("message", ""),
            "analysis_note": "只读取 .safetensors 静态权重，不加载模型、不跑图、不占用 GPU。",
        }

    weights_by_path: dict[str, dict[str, Any]] = {}
    directories: list[str] = []
    errors: list[str] = []

    def add_listing(payload: Mapping[str, Any], *, source_task: dict[str, Any] | None = None) -> None:
        directory = str(payload.get("directory") or "")
        if directory and directory not in directories:
            directories.append(directory)
        source_meta = _source_task_meta(source_task) if source_task else {}
        for item in payload.get("weights") or []:
            if not isinstance(item, dict):
                continue
            meta = _analysis_weight_meta(item)
            if source_meta:
                meta["source_task"] = source_meta
                meta["scope_label"] = meta.get("scope_label") or source_meta.get("label", "")
            key = str(meta.get("abs_path") or meta.get("file") or meta.get("name") or "")
            if not key:
                continue
            prev = weights_by_path.get(key)
            if prev is None or float(meta.get("mtime") or 0) > float(prev.get("mtime") or 0):
                weights_by_path[key] = meta

    try:
        add_listing(preview_service.list_training_weights(None, allow_latest_fallback=allow_latest_fallback))
    except Exception as exc:  # 仅影响下拉候选，不影响手填路径分析。
        errors.append(str(exc))

    for source_task in _analysis_source_tasks(training_service, include_archived=include_archived):
        try:
            add_listing(
                preview_service.list_training_weights(source_task, allow_latest_fallback=False),
                source_task=source_task,
            )
        except Exception as exc:
            label = _source_task_meta(source_task).get("label") or source_task.get("id") or "历史任务"
            errors.append(f"{label}: {exc}")

    weights = sorted(
        weights_by_path.values(),
        key=lambda item: (float(item.get("mtime") or 0), str(item.get("name") or "")),
        reverse=True,
    )[:limit]
    return {
        "ok": True,
        "directory": " · ".join(directories[:2]) + (" · ..." if len(directories) > 2 else ""),
        "directories": directories,
        "directory_exists": bool(directories),
        "count": len(weights),
        "total": len(weights_by_path),
        "task_count": sum(1 for item in weights if item.get("source_task")),
        "weights": weights,
        "message": "" if weights else "未找到可分析权重文件",
        "errors": errors[:8],
        "analysis_note": "只读取 .safetensors 静态权重，不加载模型、不跑图、不占用 GPU。",
    }


def inspect_weight(path: str, *, task: dict[str, Any] | None = None) -> dict[str, Any]:
    """Inspect one safetensors file and summarize static ΔW energy."""

    weight_path = resolve_analysis_weight(path, task=task)
    metadata, keys = _read_safetensors_header(weight_path)
    adapter_type, unsupported = _detect_adapter_type(keys, metadata)
    base_payload = _base_payload(weight_path, metadata, adapter_type, unsupported)
    return _inspect_loaded_weight(
        file_name=weight_path.name,
        metadata=metadata,
        adapter_type=adapter_type,
        unsupported=unsupported,
        base_payload=base_payload,
        load_state=lambda: _load_safetensors_tensors(weight_path),
    )


def inspect_weight_bytes(data: bytes, *, filename: str = "uploaded.safetensors") -> dict[str, Any]:
    """Inspect an uploaded safetensors file without persisting it to outputs."""

    clean_name = Path(str(filename or "uploaded.safetensors").replace("\\", "/")).name
    if Path(clean_name).suffix.lower() not in WEIGHT_EXTS:
        raise ValueError("只支持 .safetensors 权重文件")
    if len(data) > MAX_UPLOAD_WEIGHT_BYTES:
        raise ValueError(f"拖入文件过大，最大支持 {MAX_UPLOAD_WEIGHT_BYTES // (1024 * 1024)} MiB")
    metadata, keys = _read_safetensors_header_bytes(data)
    adapter_type, unsupported = _detect_adapter_type(keys, metadata)
    base_payload = _uploaded_base_payload(clean_name, len(data), metadata, adapter_type, unsupported)
    return _inspect_loaded_weight(
        file_name=clean_name,
        metadata=metadata,
        adapter_type=adapter_type,
        unsupported=unsupported,
        base_payload=base_payload,
        load_state=lambda: _load_safetensors_tensors_bytes(data),
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
            "summary": _empty_summary(),
            "layers": [],
            "component_summary": [],
            "block_summary": [],
            "style_top20": [],
            "character_top20": [],
            "heatmap": _empty_heatmap(),
        }

    state = load_state()
    try:
        layer_rows = _compute_layers(state, adapter_type, metadata)
    finally:
        state.clear()
    if not layer_rows:
        return {
            **base_payload,
            "unsupported": {
                "unsupported": True,
                "reason": "没有找到可重建 ΔW 的 LoRA / LoHa / LoKr 层。",
            },
            "summary": _empty_summary(),
            "layers": [],
            "component_summary": [],
            "block_summary": [],
            "style_top20": [],
            "character_top20": [],
            "heatmap": _empty_heatmap(),
        }

    layers = _finalize_layer_contributions(layer_rows)
    return {
        **base_payload,
        "summary": _summary(file_name, adapter_type, metadata, layers),
        "layers": layers,
        "component_summary": _component_summary(layers),
        "block_summary": _block_summary(layers),
        "style_top20": _top_candidates(layers, kind="style"),
        "character_top20": _top_candidates(layers, kind="character"),
        "heatmap": _heatmap(layers),
    }


def resolve_analysis_weight(value: str, *, task: dict[str, Any] | None = None) -> Path:
    clean = _normalize_user_path_value(value)
    if not clean:
        raise ValueError("请填写权重路径")
    path = Path(clean)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        if ".." in path.parts:
            raise ValueError("权重路径不能包含 ..")
        resolved = (ROOT / clean.lstrip("/")).resolve()
        try:
            resolved.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError("权重路径必须在项目目录内") from exc

    if resolved.suffix.lower() not in WEIGHT_EXTS:
        raise ValueError("只支持 .safetensors 权重文件")
    if not _is_under_allowed_weight_dir(resolved, task=task):
        raise ValueError("权重文件只允许从训练输出目录或全局输出目录读取")
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError("权重文件不存在")
    if not os.access(resolved, os.R_OK):
        raise ValueError("权重文件不可读取")
    return resolved


def _normalize_user_path_value(value: str) -> str:
    clean = str(value or "").strip().strip('"').strip("'").strip()
    if clean.startswith("file://"):
        parsed = urlparse(clean)
        clean = unquote(parsed.path or "")
    else:
        clean = unquote(clean)
    return clean.replace("\\", "/").strip()


def _analysis_source_tasks(training_service: Any | None, *, include_archived: bool) -> list[dict[str, Any]]:
    if training_service is None:
        return []
    tasks: list[dict[str, Any]] = []
    current_output_dir = str(getattr(training_service, "current_output_dir", "") or "").strip()
    if current_output_dir:
        tasks.append(
            {
                "id": str(getattr(training_service, "current_task_id", "") or "current"),
                "job": "training",
                "name": "当前训练",
                "output_dir": current_output_dir,
                "variant": str(getattr(training_service, "current_variant", "") or ""),
                "state": str(getattr(training_service, "status", "") or ""),
            }
        )
    try:
        history = training_service.list_history_tasks(include_archived=include_archived, limit=160)
    except Exception:
        history = []
    for item in history or []:
        if not isinstance(item, dict) or item.get("job") != "training":
            continue
        if not str(item.get("output_dir") or "").strip():
            continue
        tasks.append(item)
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in tasks:
        key = str(item.get("id") or item.get("output_dir") or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _source_task_meta(task: dict[str, Any] | None) -> dict[str, Any]:
    task = task or {}
    return {
        "id": str(task.get("id") or ""),
        "label": str(task.get("name") or task.get("history_run_label") or task.get("variant") or task.get("id") or "训练任务"),
        "state": str(task.get("state") or ""),
        "started_at": task.get("started_at"),
        "started_at_text": str(task.get("started_at_text") or ""),
        "output_dir": str(task.get("output_dir") or ""),
    }


def _analysis_weight_meta(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "file": item.get("file", ""),
        "abs_path": item.get("abs_path", ""),
        "name": item.get("name", ""),
        "kind": item.get("kind", ""),
        "scope": item.get("scope", ""),
        "scope_label": item.get("scope_label", ""),
        "epoch": item.get("epoch"),
        "steps": item.get("steps"),
        "mtime": item.get("mtime"),
        "mtime_text": item.get("mtime_text", ""),
        "size_bytes": item.get("size_bytes", 0),
        "output_name": item.get("output_name", ""),
    }


def _allowed_weight_dirs(task: dict[str, Any] | None = None) -> list[Path]:
    dirs = [settings_service.resolve_output_root()]
    output_dir = str((task or {}).get("output_dir") or "").strip()
    if output_dir:
        dirs.append(_resolve_display_path(output_dir))
    try:
        settings = preview_service.get_preview_settings()
    except Exception:
        settings = {}
    for key in ("training_dir", "effective_training_dir"):
        training_dir = str(settings.get(key) or "").strip()
        if not training_dir:
            continue
        resolved = _resolve_display_path(training_dir)
        dirs.append(resolved.parent if resolved.name == "sample" else resolved)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in dirs:
        resolved = path.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def _is_under_allowed_weight_dir(path: Path, *, task: dict[str, Any] | None = None) -> bool:
    resolved = path.resolve()
    for allowed in _allowed_weight_dirs(task):
        try:
            resolved.relative_to(allowed)
            return True
        except ValueError:
            continue
    return False


def _resolve_display_path(value: str) -> Path:
    path = Path(str(value or "").replace("\\", "/").strip())
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _read_safetensors_header(path: Path) -> tuple[dict[str, str], list[str]]:
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = {str(k): str(v) for k, v in (f.metadata() or {}).items()}
            keys = list(f.keys())
        return metadata, keys
    except Exception as exc:
        raise ValueError(f"读取 safetensors 权重失败: {exc}") from exc


def _load_safetensors_tensors(path: Path) -> dict[str, torch.Tensor]:
    try:
        from safetensors import safe_open

        tensors: dict[str, torch.Tensor] = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[str(key)] = f.get_tensor(key).detach().cpu()
        return tensors
    except Exception as exc:
        raise ValueError(f"读取 safetensors 张量失败: {exc}") from exc


def _read_safetensors_header_bytes(data: bytes) -> tuple[dict[str, str], list[str]]:
    try:
        if len(data) < 8:
            raise ValueError("文件太小，缺少 safetensors header")
        header_len = int.from_bytes(data[:8], byteorder="little", signed=False)
        if header_len <= 0 or header_len > len(data) - 8:
            raise ValueError("safetensors header 长度不合法")
        raw_header = json.loads(data[8:8 + header_len].decode("utf-8"))
        if not isinstance(raw_header, dict):
            raise ValueError("safetensors header 不是 JSON object")
        metadata = raw_header.get("__metadata__")
        safe_metadata = {str(k): str(v) for k, v in metadata.items()} if isinstance(metadata, dict) else {}
        keys = [str(key) for key in raw_header.keys() if key != "__metadata__"]
        return safe_metadata, keys
    except Exception as exc:
        raise ValueError(f"读取 safetensors 上传文件失败: {exc}") from exc


def _load_safetensors_tensors_bytes(data: bytes) -> dict[str, torch.Tensor]:
    try:
        from safetensors.torch import load

        return {str(key): tensor.detach().cpu() for key, tensor in load(data).items()}
    except Exception as exc:
        raise ValueError(f"读取 safetensors 上传张量失败: {exc}") from exc


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
            "path": _display_path(weight_path),
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


def _compute_layers(
    state: Mapping[str, torch.Tensor],
    adapter_type: str,
    metadata: Mapping[str, str],
) -> list[dict[str, Any]]:
    if adapter_type == "LoRA":
        return _compute_lora_layers(state, metadata)
    if adapter_type == "LoHa":
        return _compute_loha_layers(state, metadata)
    if adapter_type == "LoKr":
        return _compute_lokr_layers(state, metadata)
    return []


def _compute_lora_layers(state: Mapping[str, torch.Tensor], metadata: Mapping[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(state):
        if not key.endswith(LORA_SUFFIX):
            continue
        prefix = key.removesuffix(LORA_SUFFIX)
        down = state.get(key)
        up = state.get(f"{prefix}.lora_up.weight")
        if down is None or up is None:
            continue
        alpha = _alpha_value(state, prefix, default=_rank_from_down(down, metadata))
        rank = max(1, _rank_from_down(down, metadata))
        try:
            delta = _lora_delta(up, down, alpha, rank)
        except ValueError as exc:
            rows.append(_layer_error(prefix, "LoRA", str(exc), [down, up]))
            continue
        rows.append(_layer_stats(prefix, "LoRA", delta, [down, up], alpha=alpha, rank=rank))
        del delta
    return rows


def _compute_loha_layers(state: Mapping[str, torch.Tensor], metadata: Mapping[str, str]) -> list[dict[str, Any]]:
    del metadata
    rows: list[dict[str, Any]] = []
    for key in sorted(state):
        if not key.endswith(LOHA_SUFFIX):
            continue
        prefix = key.removesuffix(LOHA_SUFFIX)
        w1_a = state.get(f"{prefix}.hada_w1_a")
        w1_b = state.get(f"{prefix}.hada_w1_b")
        w2_a = state.get(f"{prefix}.hada_w2_a")
        w2_b = state.get(f"{prefix}.hada_w2_b")
        tensors = [tensor for tensor in (w1_a, w1_b, w2_a, w2_b) if tensor is not None]
        if len(tensors) != 4:
            continue
        alpha = _alpha_value(state, prefix, default=_rank_from_loha(w1_a, w1_b))
        rank = max(1, _rank_from_loha(w1_a, w1_b))
        try:
            delta = ((w1_a.float() @ w1_b.float()) * (w2_a.float() @ w2_b.float())) * (alpha / rank)
        except RuntimeError as exc:
            rows.append(_layer_error(prefix, "LoHa", str(exc), tensors))
            continue
        rows.append(_layer_stats(prefix, "LoHa", delta, tensors, alpha=alpha, rank=rank))
        del delta
    return rows


def _compute_lokr_layers(state: Mapping[str, torch.Tensor], metadata: Mapping[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in sorted(state):
        if not key.endswith(LOKR_SUFFIX):
            continue
        prefix = key.removesuffix(LOKR_SUFFIX)
        w1 = state.get(f"{prefix}.lokr_w1")
        w2 = state.get(f"{prefix}.lokr_w2")
        if w1 is None or w2 is None:
            continue
        network_dim = _metadata_network_dim(metadata)
        rank_guess = max(1, int(w1.shape[0]) if w1.dim() >= 1 else 1)
        alpha = _alpha_value(state, prefix, default=network_dim or rank_guess)
        # LoKr 的实际 Kronecker factor 不是 LoRA rank。训练加载器在没有
        # ss_network_dim 时会退回用 alpha 作为 modules_dim，因此这里同样
        # 优先使用元数据 rank，其次使用 alpha，最后才退回 factor 大小。
        rank = max(1, int(network_dim or round(alpha) or rank_guess))
        try:
            delta = torch.kron(w1.float(), w2.float()) * (alpha / rank)
        except RuntimeError as exc:
            rows.append(_layer_error(prefix, "LoKr", str(exc), [w1, w2]))
            continue
        rows.append(_layer_stats(prefix, "LoKr", delta, [w1, w2], alpha=alpha, rank=rank))
        del delta
    return rows


def _lora_delta(up: torch.Tensor, down: torch.Tensor, alpha: float, rank: int) -> torch.Tensor:
    up_f = up.float()
    down_f = down.float()
    if down_f.dim() == 2 and up_f.dim() == 2:
        return (up_f @ down_f) * (alpha / max(rank, 1))
    if down_f.dim() == 4 and up_f.dim() == 4:
        if tuple(down_f.shape[2:4]) == (1, 1):
            return (
                up_f.squeeze(3).squeeze(2) @ down_f.squeeze(3).squeeze(2)
            ).unsqueeze(2).unsqueeze(3) * (alpha / max(rank, 1))
        return torch.nn.functional.conv2d(down_f.permute(1, 0, 2, 3), up_f).permute(1, 0, 2, 3) * (
            alpha / max(rank, 1)
        )
    raise ValueError(f"暂不支持的 LoRA 张量维度: up={tuple(up.shape)}, down={tuple(down.shape)}")


def _alpha_value(state: Mapping[str, torch.Tensor], prefix: str, *, default: int | float) -> float:
    tensor = state.get(f"{prefix}.alpha")
    if tensor is None:
        return float(default or 1)
    try:
        return float(tensor.detach().float().cpu().reshape(-1)[0].item())
    except Exception:
        return float(default or 1)


def _rank_from_down(down: torch.Tensor, metadata: Mapping[str, str]) -> int:
    if down.dim() >= 1:
        return int(down.shape[0])
    try:
        return int(float(metadata.get("ss_network_dim") or 1))
    except (TypeError, ValueError):
        return 1


def _rank_from_loha(w1_a: torch.Tensor, w1_b: torch.Tensor) -> int:
    if w1_a.dim() == 2:
        return int(w1_a.shape[1])
    if w1_b.dim() == 2:
        return int(w1_b.shape[0])
    return 1


def _metadata_network_dim(metadata: Mapping[str, str]) -> int | None:
    try:
        value = int(float(metadata.get("ss_network_dim") or ""))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _layer_stats(
    name: str,
    adapter_type: str,
    delta: torch.Tensor,
    source_tensors: Iterable[torch.Tensor],
    *,
    alpha: float,
    rank: int,
) -> dict[str, Any]:
    delta_f = delta.detach().float()
    abs_delta = delta_f.abs()
    fro_sq = float(delta_f.pow(2).sum().item())
    fro_norm = float(fro_sq ** 0.5)
    mean_abs = float(abs_delta.mean().item()) if delta_f.numel() else 0.0
    max_abs = float(abs_delta.max().item()) if delta_f.numel() else 0.0
    parsed = parse_layer_name(name)
    return {
        "name": name,
        "adapter_type": adapter_type,
        "block": parsed["block"],
        "block_label": parsed["block_label"],
        "component": parsed["component"],
        "component_label": parsed["component_label"],
        "fro_norm": fro_norm,
        "fro_norm_sq": fro_sq,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "param_count": int(sum(int(t.numel()) for t in source_tensors)),
        "delta_param_count": int(delta_f.numel()),
        "shape": list(delta_f.shape),
        "alpha": float(alpha),
        "rank": int(rank),
        "contribution": 0.0,
        "energy_contribution": 0.0,
        "style_score": 0.0,
        "character_score": 0.0,
        "notes": [],
    }


def _layer_error(name: str, adapter_type: str, error: str, source_tensors: Iterable[torch.Tensor]) -> dict[str, Any]:
    parsed = parse_layer_name(name)
    return {
        "name": name,
        "adapter_type": adapter_type,
        "block": parsed["block"],
        "block_label": parsed["block_label"],
        "component": parsed["component"],
        "component_label": parsed["component_label"],
        "fro_norm": 0.0,
        "fro_norm_sq": 0.0,
        "mean_abs": 0.0,
        "max_abs": 0.0,
        "param_count": int(sum(int(t.numel()) for t in source_tensors)),
        "delta_param_count": 0,
        "shape": [],
        "alpha": 0.0,
        "rank": 0,
        "contribution": 0.0,
        "energy_contribution": 0.0,
        "style_score": 0.0,
        "character_score": 0.0,
        "notes": [f"该层 ΔW 重建失败: {error}"],
    }


def parse_layer_name(name: str) -> dict[str, Any]:
    match = BLOCK_RE.search(name)
    if match:
        component = _normalize_component(match.group("component"))
        block = int(match.group("block"))
        return {
            "block": block,
            "block_label": str(block),
            "component": component,
            "component_label": component,
        }
    return {
        "block": None,
        "block_label": "其他",
        "component": _normalize_component(name),
        "component_label": _normalize_component(name),
    }


def _normalize_component(component: str) -> str:
    value = str(component or "").strip("_")
    for prefix in ("lora_unet_", "unet_"):
        if value.startswith(prefix):
            value = value.removeprefix(prefix)
    return value or "unknown"


def _finalize_layer_contributions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total_fro = sum(float(row.get("fro_norm") or 0.0) for row in rows) or 0.0
    total_energy = sum(float(row.get("fro_norm_sq") or 0.0) for row in rows) or 0.0
    for row in rows:
        fro = float(row.get("fro_norm") or 0.0)
        energy = float(row.get("fro_norm_sq") or 0.0)
        row["contribution"] = fro / total_fro if total_fro > 0 else 0.0
        row["energy_contribution"] = energy / total_energy if total_energy > 0 else 0.0
        row["style_score"] = _candidate_score(row, kind="style")
        row["character_score"] = _candidate_score(row, kind="character")
        row["notes"] = _layer_notes(row)
        row.pop("fro_norm_sq", None)
    return sorted(rows, key=lambda item: float(item.get("fro_norm") or 0.0), reverse=True)


def _candidate_score(row: Mapping[str, Any], *, kind: str) -> float:
    component = str(row.get("component") or "")
    block = row.get("block")
    block_int = int(block) if isinstance(block, int) else None
    fro = float(row.get("fro_norm") or 0.0)
    if kind == "style":
        weight = STYLE_PRIORITY.get(component, 0.72)
        if block_int is not None:
            if 13 <= block_int <= 18:
                weight *= 1.24
            elif 25 <= block_int <= 26:
                weight *= 1.20
            elif 9 <= block_int <= 24:
                weight *= 1.08
            elif 0 <= block_int <= 8:
                weight *= 0.84
        return fro * weight
    weight = CHARACTER_PRIORITY.get(component, 0.76)
    if block_int is not None:
        if 0 <= block_int <= 8:
            weight *= 1.24
        elif 9 <= block_int <= 18:
            weight *= 1.12
        elif block_int >= 25:
            weight *= 0.86
    return fro * weight


def _layer_notes(row: Mapping[str, Any]) -> list[str]:
    notes: list[str] = []
    component = str(row.get("component") or "")
    block = row.get("block")
    if component in {"mlp_layer1", "cross_attn_k_proj", "cross_attn_v_proj", "self_attn_output_proj"}:
        notes.append("风格候选高权重层类型")
    if component in {"self_attn_q_proj", "self_attn_v_proj", "cross_attn_q_proj", "cross_attn_k_proj", "cross_attn_v_proj"}:
        notes.append("角色/结构候选注意力层")
    if isinstance(block, int) and (13 <= block <= 18 or 25 <= block <= 26):
        notes.append("中后段重点 block")
    if isinstance(block, int) and 0 <= block <= 8:
        notes.append("早期 block，通常更偏结构基础")
    return notes


def _summary(
    file_name: str,
    adapter_type: str,
    metadata: Mapping[str, str],
    layers: list[Mapping[str, Any]],
) -> dict[str, Any]:
    total_fro = sum(float(layer.get("fro_norm") or 0.0) for layer in layers)
    total_energy = sum(float(layer.get("fro_norm") or 0.0) ** 2 for layer in layers)
    total_params = sum(int(layer.get("param_count") or 0) for layer in layers)
    blocks = sorted({int(layer["block"]) for layer in layers if isinstance(layer.get("block"), int)})
    components = sorted({str(layer.get("component") or "") for layer in layers if layer.get("component")})
    early = _average_fro_for_blocks(layers, range(0, 9))
    mid_late = _average_fro_for_blocks(layers, range(13, 27))
    ratio = (mid_late / early) if early > 0 else None
    top_layer = layers[0] if layers else {}
    return {
        "file_name": file_name,
        "adapter_type": adapter_type,
        "output_name": str(metadata.get("ss_output_name") or ""),
        "layer_count": len(layers),
        "component_count": len(components),
        "block_count": len(blocks),
        "blocks": blocks,
        "components": components,
        "total_fro_norm": total_fro,
        "total_energy": total_energy,
        "total_param_count": total_params,
        "top_layer": top_layer.get("name", ""),
        "top_component": top_layer.get("component", ""),
        "top_block": top_layer.get("block"),
        "early_avg_fro_norm": early,
        "mid_late_avg_fro_norm": mid_late,
        "mid_late_vs_early_ratio": ratio,
        "conclusion": [
            "这是基于权重能量的静态推断，不是实际 prompt 激活。",
            "中后段 block 与高能层更可能承载风格信息，但仍需 prompt probe / heatmap / ablation 验证。",
        ],
    }


def _average_fro_for_blocks(layers: Iterable[Mapping[str, Any]], block_range: range) -> float:
    values = [float(layer.get("fro_norm") or 0.0) for layer in layers if layer.get("block") in block_range]
    return sum(values) / len(values) if values else 0.0


def _component_summary(layers: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for layer in layers:
        groups.setdefault(str(layer.get("component") or "unknown"), []).append(layer)
    rows = [_aggregate_group(component, items, key="component") for component, items in groups.items()]
    return sorted(rows, key=lambda item: float(item.get("fro_norm") or 0.0), reverse=True)


def _block_summary(layers: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for layer in layers:
        block = layer.get("block")
        label = str(block) if isinstance(block, int) else "其他"
        groups.setdefault(label, []).append(layer)
    rows = [_aggregate_group(label, items, key="block") for label, items in groups.items()]
    return sorted(rows, key=lambda item: (item.get("block") is None, -(float(item.get("fro_norm") or 0.0))))


def _aggregate_group(label: str, items: list[Mapping[str, Any]], *, key: str) -> dict[str, Any]:
    fro = sum(float(item.get("fro_norm") or 0.0) for item in items)
    energy = sum(float(item.get("fro_norm") or 0.0) ** 2 for item in items)
    params = sum(int(item.get("param_count") or 0) for item in items)
    max_item = max(items, key=lambda item: float(item.get("fro_norm") or 0.0))
    row = {
        key: None if key == "block" and label == "其他" else (_int_or_none(label) if key == "block" else label),
        "label": label,
        "layer_count": len(items),
        "fro_norm": fro,
        "energy": energy,
        "mean_abs": sum(float(item.get("mean_abs") or 0.0) for item in items) / len(items),
        "max_abs": max(float(item.get("max_abs") or 0.0) for item in items),
        "param_count": params,
        "top_layer": max_item.get("name", ""),
        "top_component": max_item.get("component", ""),
        "contribution": sum(float(item.get("contribution") or 0.0) for item in items),
        "energy_contribution": sum(float(item.get("energy_contribution") or 0.0) for item in items),
    }
    return row


def _top_candidates(layers: list[Mapping[str, Any]], *, kind: str) -> list[dict[str, Any]]:
    score_key = "style_score" if kind == "style" else "character_score"
    sorted_layers = sorted(layers, key=lambda item: float(item.get(score_key) or 0.0), reverse=True)
    return [
        {
            "rank": index + 1,
            "name": layer.get("name", ""),
            "block": layer.get("block"),
            "component": layer.get("component", ""),
            "fro_norm": layer.get("fro_norm", 0.0),
            "mean_abs": layer.get("mean_abs", 0.0),
            "max_abs": layer.get("max_abs", 0.0),
            "contribution": layer.get("contribution", 0.0),
            "score": layer.get(score_key, 0.0),
            "reason": _candidate_reason(layer, kind=kind),
        }
        for index, layer in enumerate(sorted_layers[:TOP_LAYER_LIMIT])
    ]


def _candidate_reason(layer: Mapping[str, Any], *, kind: str) -> str:
    component = str(layer.get("component") or "")
    block = layer.get("block")
    parts: list[str] = []
    if kind == "style":
        if component == "mlp_layer1":
            parts.append("mlp_layer1 更像风格染色器")
        elif component in {"cross_attn_k_proj", "cross_attn_v_proj"}:
            parts.append("cross-attn k/v 更像提示词到风格的绑定")
        elif component == "self_attn_output_proj":
            parts.append("self-attn output 影响整体气质和一致性")
        else:
            parts.append("按 ΔW 范数与层类型权重排序")
        if isinstance(block, int) and (13 <= block <= 18 or 25 <= block <= 26):
            parts.append("处于中后段重点 block")
    else:
        if component in {"self_attn_q_proj", "self_attn_v_proj"}:
            parts.append("self-attn q/v 更偏结构塑形候选")
        elif component in {"cross_attn_q_proj", "cross_attn_k_proj", "cross_attn_v_proj"}:
            parts.append("cross-attn q/k/v 可能影响角色提示绑定")
        else:
            parts.append("按早中段与注意力层启发式排序")
        if isinstance(block, int) and 0 <= block <= 8:
            parts.append("低 block 高能，偏角色/结构候选")
    return "；".join(parts)


def _heatmap(layers: list[Mapping[str, Any]]) -> dict[str, Any]:
    block_values = sorted({int(layer["block"]) for layer in layers if isinstance(layer.get("block"), int)})
    component_values = _ordered_components(str(layer.get("component") or "unknown") for layer in layers)
    matrix: list[list[float]] = []
    cells: list[dict[str, Any]] = []
    max_value = 0.0
    lookup: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for layer in layers:
        block = layer.get("block")
        component = str(layer.get("component") or "unknown")
        if not isinstance(block, int):
            continue
        lookup.setdefault((block, component), []).append(layer)
    for block in block_values:
        row: list[float] = []
        for component in component_values:
            items = lookup.get((block, component), [])
            value = sum(float(item.get("fro_norm") or 0.0) for item in items)
            row.append(value)
            max_value = max(max_value, value)
            if value > 0:
                cells.append(
                    {
                        "block": block,
                        "component": component,
                        "fro_norm": value,
                        "layer_count": len(items),
                        "top_layer": max(items, key=lambda item: float(item.get("fro_norm") or 0.0)).get("name", ""),
                    }
                )
        matrix.append(row)
    for cell in cells:
        cell["intensity"] = float(cell["fro_norm"] or 0.0) / max_value if max_value > 0 else 0.0
    return {
        "blocks": block_values,
        "components": component_values,
        "matrix": matrix,
        "max_value": max_value,
        "cells": cells,
    }


def _ordered_components(components: Iterable[str]) -> list[str]:
    preferred = [
        "self_attn_q_proj",
        "self_attn_k_proj",
        "self_attn_v_proj",
        "self_attn_output_proj",
        "cross_attn_q_proj",
        "cross_attn_k_proj",
        "cross_attn_v_proj",
        "cross_attn_output_proj",
        "mlp_layer1",
        "mlp_layer2",
    ]
    unique = sorted(set(components))
    return [item for item in preferred if item in unique] + [item for item in unique if item not in preferred]


def _empty_summary() -> dict[str, Any]:
    return {
        "file_name": "",
        "adapter_type": "",
        "output_name": "",
        "layer_count": 0,
        "component_count": 0,
        "block_count": 0,
        "blocks": [],
        "components": [],
        "total_fro_norm": 0.0,
        "total_energy": 0.0,
        "total_param_count": 0,
        "top_layer": "",
        "top_component": "",
        "top_block": None,
        "early_avg_fro_norm": 0.0,
        "mid_late_avg_fro_norm": 0.0,
        "mid_late_vs_early_ratio": None,
        "conclusion": [],
    }


def _empty_heatmap() -> dict[str, Any]:
    return {"blocks": [], "components": [], "matrix": [], "max_value": 0.0, "cells": []}


def _int_or_none(value: Any) -> int | None:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None
