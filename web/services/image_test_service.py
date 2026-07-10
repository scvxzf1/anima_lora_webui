"""Lightweight image-test subprocess management for WebUI."""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import unquote, urlparse

from aiohttp import web

from library.inference.precision import (
    RUNTIME_DTYPE_CHOICES,
    TEXT_ENCODER_DTYPE_CHOICES,
    normalize_runtime_dtype,
    normalize_text_encoder_dtype,
)
from library.inference.selective_lora import (
    enabled_blocks_from_anima_selective_strengths,
    normalize_anima_selective_block_strengths,
    normalize_anima_selective_blocks,
    normalize_anima_selective_preset,
)
from library.inference.request import GenerationRequest
from web.services import config_service, settings_service
from web.services import path_safety
from web.services.preview_service import DEFAULT_INFERENCE_DIR
from web.services.project_python import resolve_web_python_executable
from web.services.settings_service import display_path

ROOT = Path(__file__).resolve().parents[2]
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
ALLOWED_SAMPLERS = {"euler", "er_sde", "lcm"}
ALLOWED_ATTN_MODES = {"flash", "torch", "sageattn", "flex", "xformers", "sdpa"}
ALLOWED_RUNTIME_DTYPES = set(RUNTIME_DTYPE_CHOICES)
ALLOWED_TEXT_ENCODER_DTYPES = set(TEXT_ENCODER_DTYPE_CHOICES)
MAX_LOG_LINES = 80
IMAGE_TEST_MODEL_PATH_LABELS = {
    "pretrained_model_name_or_path": "基础 DiT 模型",
    "qwen3": "Qwen3 文本编码器",
    "vae": "VAE 路径",
}


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

    def resolve_weight_path(self, value: str) -> dict[str, Any]:
        resolved = _resolve_image_test_weight_path(value, app=self.app)
        return {
            "ok": True,
            "weight_path": str(resolved),
            "name": resolved.name,
            "display_path": display_path(resolved),
        }

    async def start(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.status == "running":
            raise RuntimeError("已有生图测试任务正在运行")
        training_svc = self.app.get("training_service")
        if training_svc and str(getattr(training_svc, "status", "") or "") == "running":
            raise RuntimeError("当前有训练或预处理任务正在运行，请先停止后再进行生图测试")
        try:
            available_gpus = await training_svc.list_gpus() if training_svc is not None else []
        except Exception:
            available_gpus = []

        normalized = _normalize_image_test_request(
            payload,
            app=self.app,
            available_gpus=available_gpus,
        )
        cmd = _build_generation_command(normalized)
        env = _build_generation_env(normalized)
        output_dir = _resolve_save_dir(normalized["save_path"])
        output_dir.mkdir(parents=True, exist_ok=True)

        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(ROOT),
            env=env,
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


def _normalize_image_test_request(
    payload: dict[str, Any],
    *,
    app: web.Application | None = None,
    available_gpus: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    raw_cfg = payload.get("config") if isinstance(payload.get("config"), dict) else {}
    cfg = _apply_global_model_path_defaults(dict(raw_cfg or {}))
    cfg = _resolve_image_test_model_paths(cfg)

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
    device, gpu_index, gpu_label = _normalize_image_test_gpu_selection(
        payload.get("gpu_index"),
        available_gpus=available_gpus,
    )
    weight_path_raw = str(payload.get("weight_path") or "").strip()
    weight_path = str(_resolve_image_test_weight_path(weight_path_raw, app=app)) if weight_path_raw else ""
    anima_selective_lora = _normalize_bool(payload.get("anima_selective_lora"))
    anima_selective_preset = normalize_anima_selective_preset(
        payload.get("anima_selective_preset")
    )
    anima_selective_strength = _normalize_float(
        payload.get("anima_selective_strength"),
        default=1.0,
        label="分层倍率",
    )
    anima_selective_block_strengths = (
        normalize_anima_selective_block_strengths(
            payload.get("anima_selective_block_strengths"),
            preset=anima_selective_preset,
        )
        if anima_selective_lora
        else {}
    )
    anima_selective_blocks = (
        enabled_blocks_from_anima_selective_strengths(
            anima_selective_block_strengths,
            preset=anima_selective_preset,
        )
        if anima_selective_lora
        else []
    )
    if anima_selective_lora and not anima_selective_blocks:
        anima_selective_blocks = normalize_anima_selective_blocks(
            payload.get("anima_selective_blocks"),
            preset=anima_selective_preset,
        )
    if anima_selective_lora and not weight_path:
        raise ValueError("启用 LoRA 分层加载时，需要先选择一个 LoRA 权重")

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
        "device": device,
        "gpu_index": gpu_index,
        "gpu_label": gpu_label,
        "seed": seed,
        "weight_path": weight_path,
        "lora_multiplier": lora_multiplier,
        "anima_selective_lora": anima_selective_lora,
        "anima_selective_preset": anima_selective_preset,
        "anima_selective_strength": anima_selective_strength,
        "anima_selective_blocks": anima_selective_blocks,
        "anima_selective_block_strengths": anima_selective_block_strengths,
        "save_path": DEFAULT_INFERENCE_DIR,
        "config": cfg,
    }


def _resolve_image_test_model_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    next_cfg = dict(cfg)
    global_settings = _image_test_global_model_settings()
    default_settings = (
        global_settings.get("defaults")
        if isinstance(global_settings.get("defaults"), dict)
        else {}
    )
    for key, label in IMAGE_TEST_MODEL_PATH_LABELS.items():
        resolved = _resolve_image_test_required_model_path(
            key,
            current_value=next_cfg.get(key),
            global_value=global_settings.get(key),
            default_value=default_settings.get(key),
            label=label,
        )
        next_cfg[key] = str(resolved)
    return next_cfg


def _resolve_image_test_required_model_path(
    key: str,
    *,
    current_value: Any,
    global_value: Any,
    default_value: Any,
    label: str,
) -> Path:
    current_raw = str(current_value or "").strip()
    global_raw = str(global_value or "").strip()
    default_raw = str(default_value or "").strip()
    current_path = _resolve_image_test_model_config_path(current_raw)
    global_path = _resolve_image_test_model_config_path(global_raw)
    if (
        global_path is not None
        and global_path.exists()
        and _is_image_test_default_model_path(
            current_raw=current_raw,
            default_raw=default_raw,
        )
    ):
        return global_path.resolve()
    if current_path is not None and current_path.exists():
        return current_path.resolve()
    if global_path is not None and global_path.exists():
        return global_path.resolve()
    if current_raw:
        raise ValueError(
            f"{label} 不存在: {current_raw}。请到“全局设置”或当前配置里填写真实路径。"
        )
    if global_raw:
        raise ValueError(
            f"{label} 不存在: {global_raw}。请到“全局设置”里检查该模型路径。"
        )
    raise ValueError(f"当前配置缺少{label}")


def _resolve_image_test_model_config_path(value: str) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return config_service._resolve_project_path(raw)
    except Exception:
        path = Path(os.path.expandvars(raw)).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (ROOT / raw.lstrip("/")).resolve()


def _is_image_test_default_model_path(*, current_raw: str, default_raw: str) -> bool:
    if not current_raw or not default_raw:
        return False
    if current_raw.replace("\\", "/").strip() == default_raw.replace("\\", "/").strip():
        return True
    current_path = _resolve_image_test_model_config_path(current_raw)
    default_path = _resolve_image_test_model_config_path(default_raw)
    if current_path is None or default_path is None:
        return False
    return current_path.resolve() == default_path.resolve()


def _image_test_global_model_settings() -> dict[str, Any]:
    try:
        payload = settings_service.get_global_settings()
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_image_test_gpu_selection(
    value: Any,
    *,
    available_gpus: list[dict[str, Any]] | None = None,
) -> tuple[str | None, int | None, str]:
    raw = str(value or "").strip().lower()
    if not raw or raw in {"auto", "default"}:
        return None, None, "自动"
    try:
        index = int(raw)
    except ValueError as exc:
        raise ValueError("GPU 选择不合法") from exc
    if index < 0:
        raise ValueError("GPU 选择不合法")
    matched = None
    if available_gpus:
        matched = next(
            (
                item
                for item in available_gpus
                if int(item.get("index", -1)) == index
            ),
            None,
        )
        if matched is None:
            raise ValueError(f"当前未检测到 GPU {index}")
    label = str((matched or {}).get("label") or f"GPU {index}")
    return "cuda", index, label


def _resolve_image_test_weight_path(
    value: str,
    *,
    app: web.Application | None = None,
) -> Path:
    clean = _normalize_image_test_weight_value(value)
    if not clean:
        raise ValueError("请填写 LoRA 权重路径")
    if not clean.lower().endswith(".safetensors"):
        raise ValueError("只支持 .safetensors 权重文件")
    if ".." in Path(clean).parts:
        raise ValueError("LoRA 权重路径不允许包含 ..")
    path = Path(clean)
    preferred_dirs = _preferred_image_test_weight_dirs(app)
    search_dirs = _search_image_test_weight_dirs(preferred_dirs)
    allowlist = _image_test_weight_allowlist(preferred_dirs=preferred_dirs, search_dirs=search_dirs)
    if _is_image_test_weight_file_name_only(path):
        resolved = _resolve_image_test_weight_by_name(
            path.name,
            search_dirs=search_dirs,
            preferred_dirs=preferred_dirs,
            enable_fallback_search=app is not None,
        )
    else:
        if path.is_absolute():
            resolved = path.resolve()
        else:
            # Relative paths must stay under repo root and may not escape via ..
            resolved = (ROOT / path).resolve()
            try:
                resolved.relative_to(ROOT.resolve())
            except ValueError as exc:
                raise ValueError("LoRA 权重路径超出允许范围") from exc
        if not path_safety.is_under_allowed_dirs(resolved, allowlist):
            raise ValueError("LoRA 权重路径超出允许范围")
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError("LoRA 权重文件不存在")
    if not os.access(resolved, os.R_OK):
        raise ValueError("LoRA 权重文件不可读取")
    return resolved


def _image_test_weight_allowlist(
    *,
    preferred_dirs: list[Path],
    search_dirs: list[Path],
) -> list[Path]:
    """Dirs that may host explicit weight paths for image_test."""
    dirs: list[Path] = []
    dirs.extend(preferred_dirs)
    dirs.extend(search_dirs)
    dirs.append(ROOT.resolve())
    # Unique preserve order
    out: list[Path] = []
    seen: set[str] = set()
    for d in dirs:
        try:
            key = str(Path(d).resolve())
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(Path(key))
    return out


def _normalize_image_test_weight_value(value: str) -> str:
    clean = str(value or "").strip().strip('"').strip("'").strip()
    if clean.startswith("file://"):
        parsed = urlparse(clean)
        clean = unquote(parsed.path or "")
        if clean.startswith("/") and len(clean) >= 4 and clean[2] == ":":
            clean = clean[1:]
    else:
        clean = unquote(clean)
    return clean.replace("\\", "/").strip()


def _is_image_test_weight_file_name_only(path: Path) -> bool:
    return not path.is_absolute() and len(path.parts) == 1


def _resolve_image_test_weight_by_name(
    name: str,
    *,
    search_dirs: list[Path],
    preferred_dirs: list[Path],
    enable_fallback_search: bool,
) -> Path:
    matches = _collect_image_test_weight_matches(name, search_dirs)
    if not matches and enable_fallback_search:
        matches = _collect_image_test_weight_matches(
            name,
            _fallback_image_test_weight_dirs(search_dirs),
        )
    if not matches:
        raise FileNotFoundError("未找到对应的 LoRA / LokR 权重文件。建议直接填写完整路径。")
    return max(matches, key=lambda path: _image_test_weight_match_sort_key(path, preferred_dirs))


def _collect_image_test_weight_matches(name: str, search_dirs: list[Path]) -> list[Path]:
    matches: list[Path] = []
    seen: set[str] = set()
    for search_dir in search_dirs:
        if not search_dir.exists() or not search_dir.is_dir():
            continue
        for candidate in search_dir.rglob(name):
            if not candidate.is_file() or candidate.suffix.lower() != ".safetensors":
                continue
            resolved = candidate.resolve()
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            matches.append(resolved)
    return matches


def _image_test_weight_match_sort_key(path: Path, preferred_dirs: list[Path]) -> tuple[int, float, int, str]:
    resolved = path.resolve()
    preferred_rank = _image_test_weight_dir_rank(resolved, preferred_dirs)
    try:
        mtime = resolved.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (preferred_rank, float(mtime), -len(resolved.parts), str(resolved))


def _image_test_weight_dir_rank(path: Path, allowed_dirs: list[Path]) -> int:
    for index, allowed in enumerate(allowed_dirs):
        try:
            path.relative_to(allowed)
            return max(len(allowed_dirs) - index, 0)
        except ValueError:
            continue
    return 0


def _preferred_image_test_weight_dirs(app: web.Application | None = None) -> list[Path]:
    dirs: list[Path] = []
    training_service = app.get("training_service") if app is not None else None
    current_output_dir = str(getattr(training_service, "current_output_dir", "") or "").strip()
    if current_output_dir:
        dirs.append(_resolve_image_test_display_path(current_output_dir))
    current_sample_dir = str(getattr(training_service, "current_sample_dir", "") or "").strip()
    if current_sample_dir:
        sample_dir = _resolve_image_test_display_path(current_sample_dir)
        dirs.append(sample_dir.parent if sample_dir.name == "sample" else sample_dir)
    dirs.append(settings_service.resolve_output_root().resolve())

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


def _search_image_test_weight_dirs(preferred_dirs: list[Path]) -> list[Path]:
    roots: list[Path] = []
    model_roots = _image_test_model_search_dirs()
    # Keep search bounded to preferred output dirs + configured model roots.
    # Do not walk the broader workspace tree by default.
    for path in [*preferred_dirs, *model_roots]:
        resolved = path.resolve()
        if any(_is_same_or_child_path(resolved, existing) for existing in roots):
            continue
        roots = [existing for existing in roots if not _is_same_or_child_path(existing, resolved)]
        roots.append(resolved)
    return roots


def _image_test_allow_home_search() -> bool:
    settings = _image_test_global_model_settings()
    raw = settings.get("image_test_allow_home_search", False)
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _fallback_image_test_weight_dirs(existing_roots: list[Path]) -> list[Path]:
    # Home rglob is opt-in only; default remains closed for safety/perf.
    if not _image_test_allow_home_search():
        return []
    home_dir = Path.home().resolve()
    if any(_is_same_or_child_path(home_dir, existing) for existing in existing_roots):
        return []
    return [home_dir]


def _is_same_or_child_path(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _image_test_model_search_dirs() -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    global_settings = _image_test_global_model_settings()
    for key in IMAGE_TEST_MODEL_PATH_LABELS:
        raw = str(global_settings.get(key) or "").strip()
        path = _resolve_image_test_model_config_path(raw)
        if path is None:
            continue
        resolved = path.resolve()
        model_root = _nearest_named_parent(resolved, "models")
        candidate = model_root or resolved.parent
        key_text = str(candidate)
        if key_text in seen:
            continue
        seen.add(key_text)
        roots.append(candidate)
    return roots


def _nearest_named_parent(path: Path, name: str) -> Path | None:
    for candidate in [path.parent, *path.parents]:
        if candidate.name == name:
            return candidate
    return None


def _resolve_image_test_display_path(value: str) -> Path:
    path = Path(str(value or "").replace("\\", "/").strip())
    if path.is_absolute():
        return path.resolve()
    return (ROOT / path).resolve()


def _build_generation_env(request: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    gpu_index = request.get("gpu_index")
    if gpu_index is None:
        return env
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_index))
    return env


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
        device=request["device"],
        attn_mode=request["attn_mode"],
        save_path=request["save_path"],
        extra_argv=[
            "--runtime_dtype",
            request["runtime_dtype"],
            "--text_encoder_dtype",
            request["text_encoder_dtype"],
            *_build_selective_lora_argv(request),
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
        "device": request["device"],
        "gpu_index": request["gpu_index"],
        "gpu_label": request["gpu_label"],
        "seed": request["seed"],
        "weight_path": request["weight_path"],
        "lora_multiplier": request["lora_multiplier"],
        "anima_selective_lora": request["anima_selective_lora"],
        "anima_selective_preset": request["anima_selective_preset"],
        "anima_selective_strength": request["anima_selective_strength"],
        "anima_selective_blocks": request["anima_selective_blocks"],
        "anima_selective_block_count": len(request["anima_selective_blocks"]),
        "anima_selective_block_strengths": {
            block: request["anima_selective_block_strengths"].get(block, 0.0)
            for block in request["anima_selective_blocks"]
        },
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


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _build_selective_lora_argv(request: dict[str, Any]) -> list[str]:
    if not request.get("anima_selective_lora"):
        return []
    argv = [
        "--anima_selective_lora",
        "--anima_selective_preset",
        str(request["anima_selective_preset"]),
    ]
    if float(request.get("anima_selective_strength", 1.0) or 1.0) != 1.0:
        argv.extend([
            "--anima_selective_strength",
            str(request["anima_selective_strength"]),
        ])
    argv.append("--anima_selective_block_strengths")
    argv.extend(
        f"{block}={request['anima_selective_block_strengths'].get(block, 0.0):.2f}"
        for block in request.get("anima_selective_block_strengths", {})
    )
    return argv


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
