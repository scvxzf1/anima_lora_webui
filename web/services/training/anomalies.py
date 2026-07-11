"""Training anomaly classification and user-facing hint formatting."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Any

CUDA_OOM_RE = re.compile(
    r"(?:"
    r"cuda\s+out\s+of\s+memory"
    r"|torch\.outofmemoryerror"
    r"|outofmemoryerror:\s*cuda"
    r"|cublas_status_alloc_failed"
    r"|cudnn_status_alloc_failed"
    r")",
    re.IGNORECASE,
)

OOM_HINT = "大概率爆显存"


def classify_training_error(text: str) -> str:
    """Return a short user-facing hint for known high-signal training failures."""
    if text and CUDA_OOM_RE.search(text):
        return OOM_HINT
    return ""


def format_training_anomaly(status_data: dict[str, Any]) -> str | None:
    """检测训练异常状态并生成可读的错误提示。"""
    latest_metric = status_data.get("latest_metric", {}) if isinstance(status_data, dict) else {}
    if not isinstance(latest_metric, dict) or not latest_metric:
        return None

    loss = latest_metric.get("loss")
    lr = latest_metric.get("lr")
    step = latest_metric.get("step")
    rate = str(latest_metric.get("rate") or "").strip() or "未知"

    anomaly_kind = _loss_anomaly_kind(loss)
    if anomaly_kind is None:
        return None

    title = {
        "nan": "损失值变为 NaN",
        "inf": "损失值变为无穷大",
    }.get(anomaly_kind, "损失值异常")

    lines = [
        f"⚠️ 训练异常：{title}",
        f"  • 发生步数：第 {step} 步" if step is not None else "  • 发生步数：未知",
        f"  • 当前学习率：{_format_anomaly_value(lr)}",
        f"  • 训练速度：{rate}",
    ]

    latest_system = status_data.get("latest_system", {})
    if isinstance(latest_system, dict) and latest_system:
        vram_used = _float_or_none(latest_system.get("vram_used_gb"))
        vram_total = _float_or_none(latest_system.get("vram_total_gb"))
        if (
            vram_used is not None
            and vram_total is not None
            and math.isfinite(vram_used)
            and math.isfinite(vram_total)
        ):
            vram_pct = (vram_used / vram_total * 100) if vram_total > 0 else 0
            lines.append(f"  • 显存占用：{vram_used:.2f}GB / {vram_total:.2f}GB ({vram_pct:.1f}%)")

    lines.extend([
        "",
        "常见原因（按可能性排序）：",
        "  1. 学习率过高",
        "     → 建议降至 5e-5 或更低，并添加 warmup_steps = 50",
        "  2. 混合精度数值溢出",
        "     → 尝试改用 bf16 或临时关闭混合精度 (mixed_precision = \"no\")",
        "  3. 缓存文件损坏",
        "     → 删除 *_anima*.npz 和 *_anima_te.safetensors 后重新运行预处理",
        "  4. 图片或 caption 异常",
        "     → 检查是否有全黑/全白图片或空 caption 文件",
    ])

    config_file = str(status_data.get("history_source_config_file") or "").strip()
    if config_file:
        config_name = Path(config_file).name
        lines.extend([
            "",
            f"配置文件：{config_name}",
            f"完整路径：{config_file}",
        ])

    lines.extend([
        "",
        "详细排查步骤请参考项目文档或查看训练日志。",
    ])
    return "\n".join(lines)


def _loss_anomaly_kind(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"nan", "+nan", "-nan"}:
            return "nan"
        if text in {"inf", "+inf", "infinity", "+infinity"}:
            return "inf"
        if text in {"-inf", "-infinity"}:
            return "inf"
    number = _float_or_none(value)
    if number is None:
        return None
    if math.isnan(number):
        return "nan"
    if math.isinf(number):
        return "inf"
    return None


def _format_anomaly_value(value: Any) -> str:
    if value is None or value == "":
        return "未知"
    number = _float_or_none(value)
    if number is None:
        return str(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "Infinity" if number > 0 else "-Infinity"
    return str(value)


def _message_with_error_hint(message: str, hint: str) -> str:
    if not hint or not message:
        return message
    if hint in message:
        return message
    return f"{message}：{hint}"


def _float_or_none(value: Any) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def classify_training_failure(
    *,
    reason: str = "",
    message: str = "",
    returncode: int | None = None,
    stop_requested: bool = False,
) -> str:
    """Classify a queue/process failure for auto-retry decisions.

    Returns one of:
    - user_stop
    - checkpoint_missing
    - oom
    - process_exit
    - launch_failure
    - unknown
    """
    if stop_requested or str(reason or "").strip().lower() in {"user_stop", "canceled", "cancelled"}:
        return "user_stop"
    text = f"{reason}\n{message}".lower()
    if (
        "train_state.json" in text
        or "检查点" in f"{reason}\n{message}"
        or ("checkpoint" in text and "missing" in text)
    ):
        return "checkpoint_missing"
    if "续训检查点状态已不存在" in f"{reason}\n{message}":
        return "checkpoint_missing"
    if "缺少" in f"{reason}\n{message}" and (".bin" in text or "scheduler" in text or "optimizer" in text):
        return "checkpoint_missing"
    if CUDA_OOM_RE.search(f"{reason}\n{message}") or "out of memory" in text or "oom" == str(reason).lower():
        return "oom"
    if str(reason or "").strip().lower() == "launch_failure":
        return "launch_failure"
    if str(reason or "").strip().lower() == "process_exit" or returncode not in (None, 0):
        return "process_exit"
    return "unknown"


def should_auto_retry_failure(kind: str) -> bool:
    """Return whether a classified failure is eligible for auto_retry."""
    return str(kind or "").strip().lower() in {
        "oom",
        "process_exit",
        "launch_failure",
        "unknown",
    }

