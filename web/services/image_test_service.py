"""Lightweight image-test subprocess management for WebUI."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime
from pathlib import Path
import time
from typing import Any

from aiohttp import web

from library.inference.precision import (
    RUNTIME_DTYPE_CHOICES,
    TEXT_ENCODER_DTYPE_CHOICES,
    normalize_runtime_dtype,
    normalize_text_encoder_dtype,
)
from library.inference.request import GenerationRequest
from web.services.preview_service import DEFAULT_INFERENCE_DIR
from web.services.project_python import resolve_web_python_executable
from web.services.settings_service import display_path
from web.services.weight_analysis_service import resolve_analysis_weight

ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ALLOWED_SAMPLERS = {"euler", "er_sde", "lcm"}
ALLOWED_ATTN_MODES = {"flash", "torch", "sageattn", "flex", "xformers", "sdpa"}
ALLOWED_RUNTIME_DTYPES = set(RUNTIME_DTYPE_CHOICES)
ALLOWED_TEXT_ENCODER_DTYPES = set(TEXT_ENCODER_DTYPE_CHOICES)
MAX_LOG_LINES = 80


class ImageTestService:
    def __init__(self, app: web.Application):
        self.app = app
        self.process: asyncio.subprocess.Process | None = None
        self.status: str = "idle"
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.exit_code: int | None = None
        self.error: str = ""
        self.output_dir: Path = _resolve_save_dir(DEFAULT_INFERENCE_DIR)
        self.command: list[str] = []
        self.last_request: dict[str, Any] = {}
        self._logs: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self._monitor_task: asyncio.Task | None = None
        self._stop_requested = False

    def get_status_snapshot(self) -> dict[str, Any]:
        output_files = _list_output_images(self.output_dir)
        return {
            "ok": self.status != "error",
            "status": self.status,
            "running": self.status == "running",
            "started_at": self.started_at,
            "started_at_text": _format_ts(self.started_at),
            "finished_at": self.finished_at,
            "finished_at_text": _format_ts(self.finished_at),
            "exit_code": self.exit_code,
            "error": self.error,
            "output_dir": display_path(self.output_dir),
            "output_count": len(output_files),
            "output_files": output_files[:12],
            "command": self.command,
            "last_request": self.last_request,
            "logs": list(self._logs),
        }

    async def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.status == "running":
            raise RuntimeError("已有生图测试任务正在运行")
        training_svc = self.app.get("training_service")
        if training_svc and str(getattr(training_svc, "status", "") or "") == "running":
            raise RuntimeError("当前有训练或预处理任务正在运行，请先停止后再进行生图测试")

        normalized = _normalize_image_test_request(payload)
        cmd = _build_generation_command(normalized)
        output_dir = _resolve_save_dir(normalized["save_path"])
        output_dir.mkdir(parents=True, exist_ok=True)

        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(ROOT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        self.status = "running"
        self.started_at = time.time()
        self.finished_at = None
        self.exit_code = None
        self.error = ""
        self.output_dir = output_dir
        self.command = cmd
        self.last_request = _request_summary(normalized)
        self._logs.clear()
        self._logs.append("生图测试已启动")
        self._stop_requested = False
        self._monitor_task = asyncio.create_task(self._monitor_process())
        return self.get_status_snapshot()

    async def stop(self) -> dict[str, Any]:
        if self.status != "running" or self.process is None:
            raise RuntimeError("当前没有运行中的生图测试任务")
        self._stop_requested = True
        self._logs.append("正在停止生图测试任务…")
        self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=5)
        except asyncio.TimeoutError:
            self.process.kill()
            await self.process.wait()
        return self.get_status_snapshot()

    async def shutdown(self) -> None:
        if self.status == "running" and self.process is not None:
            self._stop_requested = True
            self.process.kill()
            await self.process.wait()
        if self._monitor_task is not None:
            await self._monitor_task

    async def _monitor_process(self) -> None:
        assert self.process is not None
        stdout = self.process.stdout
        if stdout is not None:
            while True:
                line = await stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    self._logs.append(text)
        code = await self.process.wait()
        self.exit_code = code
        self.finished_at = time.time()
        if self._stop_requested:
            self.status = "canceled"
            self.error = ""
        elif code == 0:
            self.status = "done"
            self.error = ""
            self._logs.append("生图测试已完成")
        else:
            self.status = "error"
            self.error = f"生图测试异常退出 (code={code})"
            self._logs.append(self.error)
        self.process = None
        self._monitor_task = None
        self._stop_requested = False


def _normalize_image_test_request(payload: dict[str, Any]) -> dict[str, Any]:
    raw_cfg = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    cfg = _apply_global_model_path_defaults(dict(raw_cfg or {}))

    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        raise ValueError("请输入正向提示词")

    sampler = _normalize_choice(payload.get("sampler") or cfg.get("sample_sampler") or "euler", ALLOWED_SAMPLERS, "采样器")
    attn_mode = _normalize_choice(payload.get("attn_mode") or cfg.get("attn_mode") or "flash", ALLOWED_ATTN_MODES, "注意力后端")
    seed = _normalize_optional_int(payload.get("seed"))
    width = _normalize_positive_int(payload.get("width"), default=1024, label="宽度")
    height = _normalize_positive_int(payload.get("height"), default=1024, label="高度")
    infer_steps = _normalize_positive_int(payload.get("infer_steps"), default=28, label="采样步数")
    guidance_scale = _normalize_float(payload.get("guidance_scale"), default=4.0, label="CFG")
    flow_shift = _normalize_float(payload.get("flow_shift"), default=1.0, label="Flow Shift")
    lora_multiplier = _normalize_float(payload.get("lora_multiplier"), default=1.0, label="LoRA 强度")
    runtime_dtype = _normalize_runtime_dtype(payload.get("runtime_dtype"), cfg)
    text_encoder_dtype = _normalize_text_encoder_dtype(payload.get("text_encoder_dtype"))
    weight_path_raw = str(payload.get("weight_path") or "").strip()
    weight_path = str(resolve_analysis_weight(weight_path_raw)) if weight_path_raw else ""

    dit = str(cfg.get("pretrained_model_name_or_path") or "").strip()
    text_encoder = str(cfg.get("qwen3") or "").strip()
    vae = str(cfg.get("vae") or "").strip()
    if not dit:
        raise ValueError("当前配置缺少基础模型路径")
    if not text_encoder:
        raise ValueError("当前配置缺少 Qwen3 文本编码器路径")
    if not vae:
        raise ValueError("当前配置缺少 VAE 路径")

    return {
        "prompt": prompt,
        "negative_prompt": str(payload.get("negative_prompt") or ""),
        "width": width,
        "height": height,
        "infer_steps": infer_steps,
        "guidance_scale": guidance_scale,
        "flow_shift": flow_shift,
        "sampler": sampler,
        "attn_mode": attn_mode,
        "runtime_dtype": runtime_dtype,
        "text_encoder_dtype": text_encoder_dtype,
        "seed": seed,
        "weight_path": weight_path,
        "lora_multiplier": lora_multiplier,
        "save_path": DEFAULT_INFERENCE_DIR,
        "config": cfg,
    }


def _build_generation_command(request: dict[str, Any]) -> list[str]:
    cfg = request["config"]
    generation_request = GenerationRequest(
        prompt=request["prompt"],
        negative_prompt=request["negative_prompt"],
        image_size=(request["height"], request["width"]),
        infer_steps=request["infer_steps"],
        guidance_scale=request["guidance_scale"],
        flow_shift=request["flow_shift"],
        sampler=request["sampler"],
        seed=request["seed"],
        lora_weight=[request["weight_path"]] if request["weight_path"] else None,
        lora_multiplier=request["lora_multiplier"],
        dit=str(cfg["pretrained_model_name_or_path"]),
        vae=str(cfg["vae"]),
        text_encoder=str(cfg["qwen3"]),
        attn_mode=request["attn_mode"],
        save_path=request["save_path"],
        extra_argv=[
            "--runtime_dtype",
            request["runtime_dtype"],
            "--text_encoder_dtype",
            request["text_encoder_dtype"],
        ],
    )
    return [
        resolve_web_python_executable(),
        "inference.py",
        *generation_request.to_argv(),
    ]


def _request_summary(request: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": request["prompt"],
        "negative_prompt": request["negative_prompt"],
        "width": request["width"],
        "height": request["height"],
        "infer_steps": request["infer_steps"],
        "guidance_scale": request["guidance_scale"],
        "flow_shift": request["flow_shift"],
        "sampler": request["sampler"],
        "attn_mode": request["attn_mode"],
        "runtime_dtype": request["runtime_dtype"],
        "text_encoder_dtype": request["text_encoder_dtype"],
        "seed": request["seed"],
        "weight_path": request["weight_path"],
        "lora_multiplier": request["lora_multiplier"],
    }


def _normalize_runtime_dtype(value: Any, cfg: dict[str, Any]) -> str:
    raw = str(value or "").strip()
    if raw:
        return _normalize_choice(raw, ALLOWED_RUNTIME_DTYPES, "推理精度")
    cfg_value = str(cfg.get("precision_preference") or "").strip().lower()
    if cfg_value in ALLOWED_RUNTIME_DTYPES:
        return cfg_value
    return normalize_runtime_dtype("bf16")


def _normalize_text_encoder_dtype(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return normalize_text_encoder_dtype("same")
    return _normalize_choice(raw, ALLOWED_TEXT_ENCODER_DTYPES, "文本编码器精度")


def _normalize_choice(value: Any, allowed: set[str], label: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in allowed:
        raise ValueError(f"{label} 不支持: {value}")
    return normalized


def _normalize_positive_int(value: Any, *, default: int, label: str) -> int:
    raw = "" if value is None else str(value).strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise ValueError(f"{label} 必须是整数") from exc
    if parsed <= 0:
        raise ValueError(f"{label} 必须大于 0")
    return parsed


def _normalize_optional_int(value: Any) -> int | None:
    raw = "" if value is None else str(value).strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError("随机种子必须是整数") from exc


def _normalize_float(value: Any, *, default: float, label: str) -> float:
    raw = "" if value is None else str(value).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{label} 必须是数字") from exc


def _resolve_save_dir(value: str) -> Path:
    clean = str(value or DEFAULT_INFERENCE_DIR).replace("\\", "/").strip().lstrip("/")
    if ".." in Path(clean).parts:
        raise ValueError("生图输出目录不能包含 ..")
    return (ROOT / clean).resolve()


def _list_output_images(directory: Path) -> list[dict[str, Any]]:
    if not directory.exists() or not directory.is_dir():
        return []
    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    files.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return [
        {
            "name": path.name,
            "file": display_path(path),
            "mtime": path.stat().st_mtime,
            "mtime_text": _format_ts(path.stat().st_mtime),
        }
        for path in files
    ]


def _format_ts(value: float | None) -> str:
    if not value:
        return ""
    return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _apply_global_model_path_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    from web.services.config.preflight import apply_global_model_path_defaults

    return apply_global_model_path_defaults(cfg)
