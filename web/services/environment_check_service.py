"""Environment checks for WebUI."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Callable

from web.services.project_python import resolve_web_python_executable, venv_bin_dir, venv_python_path
from web.services.config.metadata import PREPROCESS_ENV_REQUIRED_FILES
from web.services import config_service as _config_service

ROOT = _config_service.ROOT

PREPROCESS_ENV_CHECK_KEY = "preprocess_environment"
PROJECT_FILE_CHECKS = tuple(dict.fromkeys(("pyproject.toml", "uv.lock", *PREPROCESS_ENV_REQUIRED_FILES)))
CORE_IMPORT_MODULES = ("torch", "aiohttp", "toml", "accelerate", "safetensors", "transformers", "diffusers", "PIL")
MODEL_PATH_CHECKS = (
    ("pretrained_model_name_or_path", "基础 DiT 模型", (".safetensors", ".pt", ".pth", ".ckpt")),
    ("qwen3", "Qwen3 文本编码器", (".safetensors", ".pt", ".pth", ".bin")),
    ("vae", "VAE 模型", (".safetensors", ".pt", ".pth", ".ckpt")),
)


def run_environment_check() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(level, key, message, *, path=None, group="runtime", hint="", detail=""):
        item = {"level": level, "key": key, "message": message, "group": group}
        if path is not None:
            item["path"] = str(path)
        if hint:
            item["hint"] = hint
        if detail:
            item["detail"] = detail
        checks.append(item)

    info = {
        "system": platform.system(),
        "platform": sys.platform,
        "python_version": platform.python_version(),
        "web_executable": sys.executable,
        "project_python": resolve_web_python_executable(),
        "venv_python": str(venv_python_path() or ""),
        "project_root": str(ROOT),
        "cuda_track": "unknown",
    }
    pv = info["python_version"].split(".")[:2]
    if tuple(int(x) for x in pv) != (3, 13):
        add("error", "python_version", f"需要 Python 3.13，当前 {info['python_version']}", hint="uv sync 后用项目 venv 启动 WebUI")
    else:
        add("ok", "python_version", f"Python {info['python_version']}")
    if not venv_python_path():
        add("error", "project_venv", "未找到 .venv 解释器", path=ROOT / ".venv", hint="在项目根目录执行 uv sync")
    else:
        add("ok", "project_venv", f"venv: {info['venv_python']}", path=info["venv_python"])
    if venv_python_path() and Path(sys.executable).resolve() != venv_python_path().resolve():
        add("warning", "web_interpreter", "Web 未用项目 venv 启动", detail=f"Web={sys.executable}")
    missing = [r for r in PROJECT_FILE_CHECKS if not (ROOT / r).is_file()]
    if missing:
        add("error", "project_files", "缺少: " + ", ".join(missing), path=ROOT / missing[0], group="project_files")
    else:
        add("ok", "project_files", "项目关键文件完整", path=ROOT, group="project_files")
    for key, cmd, label in (
        ("tool_git", ["git", "--version"], "git"),
        ("tool_uv", ["uv", "--version"], "uv"),
        ("tool_hf", ["hf", "--version"], "hf"),
        ("tool_nvidia_smi", ["nvidia-smi", "--version"], "nvidia-smi"),
    ):
        ok, detail = _run_cmd(cmd)
        if ok:
            add("ok", key, f"{label} 可用", group="platform_tools", detail=detail)
        elif key in {"tool_hf", "tool_nvidia_smi"}:
            add("warning", key, f"{label} 不可用", group="platform_tools", detail=detail)
        else:
            add("error", key, f"{label} 不可用", group="platform_tools", detail=detail)
    _check_model_paths(add)
    probe = _probe_imports(info["project_python"])
    if probe.get("error"):
        add("error", "python_import_probe", probe["error"], group="python_packages")
    else:
        for name in CORE_IMPORT_MODULES:
            mod = probe.get("modules", {}).get(name, {})
            if mod.get("ok"):
                add("ok", f"pkg_{name}", f"{name} {mod.get('version','')}", group="python_packages")
            else:
                add("error", f"pkg_{name}", f"{name}: {mod.get('error','')}", group="python_packages", hint="uv sync")
        tc = probe.get("torch_cuda", {})
        info["cuda_track"] = _cuda_track_from_probe(tc)
        if tc.get("available"):
            add("ok", "cuda_available", "PyTorch CUDA 可用", group="gpu_stack", detail=str(tc.get("devices")))
        else:
            add("warning", "cuda_available", "PyTorch 未检测到 CUDA", group="gpu_stack")
    try:
        from web.services.settings_service import resolve_output_root
        root = resolve_output_root()
        root.mkdir(parents=True, exist_ok=True)
        t = root / ".env_probe"
        t.write_text("ok", encoding="utf-8")
        t.unlink(missing_ok=True)
        add("ok", "output_root_writable", f"输出根目录可写: {root}", path=root, group="web_runtime")
    except Exception as e:
        add("warning", "web_runtime", str(e), group="web_runtime")
    errors = [c for c in checks if c["level"] == "error"]
    warnings = [c for c in checks if c["level"] == "warning"]
    titles = {
        "runtime": "运行环境",
        "project_files": "项目文件",
        "platform_tools": "系统工具",
        "model_paths": "模型路径",
        "python_packages": "Python 依赖",
        "gpu_stack": "GPU / CUDA",
        "web_runtime": "Web 运行",
    }
    grouped = {}
    for c in checks:
        grouped.setdefault(c["group"], []).append(c)
    groups = [{"key": k, "title": titles.get(k, k), "checks": grouped[k]} for k in titles if grouped.get(k)]
    return {"ok": not errors, "platform": info, "summary": {"errors": len(errors), "warnings": len(warnings), "checks": len(checks)}, "checks": checks, "errors": errors, "warnings": warnings, "groups": groups}


def check_preprocess_environment_for_preflight(add: Callable[..., None]) -> None:
    python_exe = Path(resolve_web_python_executable())
    if not python_exe.is_file():
        add("error", PREPROCESS_ENV_CHECK_KEY, f"预处理启动环境异常: Python 不存在 {python_exe}", python_exe)
        return
    missing = [rel for rel in PREPROCESS_ENV_REQUIRED_FILES if not (ROOT / rel).is_file()]
    if missing:
        add("error", PREPROCESS_ENV_CHECK_KEY, f"预处理启动环境异常: 缺少 {', '.join(missing)}", ROOT / missing[0])
        return
    add("ok", PREPROCESS_ENV_CHECK_KEY, "预处理启动环境文件检查通过", ROOT)


def _run_cmd(cmd, timeout=5.0):
    env = os.environ.copy()
    b = venv_bin_dir()
    if b:
        env["PATH"] = os.pathsep.join([str(b), env.get("PATH", "")])
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        return False, f"not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    out = (proc.stdout or proc.stderr or "").strip()
    return (proc.returncode == 0, out.splitlines()[0] if out else "ok")


def _check_model_paths(add: Callable[..., None]) -> None:
    settings = _load_model_path_settings()
    for key, label, suffixes in MODEL_PATH_CHECKS:
        raw = str(settings.get(key) or "").strip()
        if not raw:
            add("error", f"model_{key}", f"{label} 路径为空", group="model_paths", hint="在全局设置填写模型路径，或执行 python tasks.py download-models")
            continue
        path = _resolve_project_path(raw)
        if not path.exists():
            add(
                "error",
                f"model_{key}",
                f"{label} 不存在",
                path=path,
                group="model_paths",
                hint="检查全局设置路径，或执行 python tasks.py download-models",
            )
            continue
        if not path.is_file():
            add("error", f"model_{key}", f"{label} 不是文件", path=path, group="model_paths")
            continue
        if path.suffix.lower() not in suffixes:
            add(
                "warning",
                f"model_{key}",
                f"{label} 后缀可能不匹配: {path.suffix or '无后缀'}",
                path=path,
                group="model_paths",
                hint="确认这是可加载的模型权重文件",
            )
            continue
        add("ok", f"model_{key}", f"{label} 已找到", path=path, group="model_paths")


def _load_model_path_settings() -> dict[str, str]:
    settings: dict[str, str] = {}
    try:
        from web.services.settings_service import get_global_settings

        global_settings = get_global_settings()
    except Exception:
        global_settings = {}
    defaults = global_settings.get("defaults") if isinstance(global_settings.get("defaults"), dict) else {}
    for key, _, _ in MODEL_PATH_CHECKS:
        settings[key] = str(defaults.get(key) or "").strip()
    for key, _, _ in MODEL_PATH_CHECKS:
        if key in global_settings:
            value = str(global_settings.get(key) or "").strip()
            if value:
                settings[key] = value
    for root in _model_path_config_roots():
        root_settings = _load_model_path_settings_from_config_root(root)
        for key, _, _ in MODEL_PATH_CHECKS:
            if not settings.get(key) and root_settings.get(key):
                settings[key] = root_settings[key]
    return settings


def _load_model_path_settings_from_config_root(configs_root: Path) -> dict[str, str]:
    settings: dict[str, str] = {}
    base = _read_toml(configs_root / "base.toml")
    for key, _, _ in MODEL_PATH_CHECKS:
        settings[key] = str(base.get(key) or "").strip()
    web_settings = _read_toml(configs_root / "web-ui-settings.toml")
    section = web_settings.get("global") if isinstance(web_settings.get("global"), dict) else {}
    for key, _, _ in MODEL_PATH_CHECKS:
        if key in section:
            value = str(section.get(key) or "").strip()
            if value:
                settings[key] = value
    return settings


def _model_path_config_roots() -> list[Path]:
    roots: list[Path] = []
    try:
        from library.env import get_configs_root

        roots.append(get_configs_root())
    except Exception:
        pass
    try:
        from web.services import settings_service

        roots.append(settings_service.SETTINGS_FILE.parent)
    except Exception:
        pass
    roots.append(ROOT / "configs")
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            marker = str(root.resolve())
        except OSError:
            marker = str(root)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(root)
    return unique


def _read_toml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _resolve_project_path(value: str) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if expanded.is_absolute():
        return expanded.resolve()
    return (ROOT / expanded).resolve()


def _cuda_track_from_probe(torch_cuda: dict[str, Any]) -> str:
    version = str(torch_cuda.get("runtime_version") or "").strip()
    if not version:
        return "unknown"
    parts = version.split(".")
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        return "unknown"
    return f"cu{int(parts[0])}{int(parts[1])}"



def _probe_imports(project_python: str) -> dict[str, Any]:
    script = f"""
import importlib
import json

out = {{'modules': {{}}, 'torch_cuda': {{}}, 'error': ''}}
names = {json.dumps(CORE_IMPORT_MODULES)}
for name in names:
    item = {{'ok': False, 'version': '', 'error': ''}}
    try:
        m = importlib.import_module(name)
        item['ok'] = True
        item['version'] = getattr(m, '__version__', '') or ''
    except Exception as e:
        item['error'] = str(e)
    out['modules'][name] = item
try:
    import torch
    out['torch_cuda'] = {{
        'available': bool(torch.cuda.is_available()),
        'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        'runtime_version': torch.version.cuda or '',
    }}
except Exception as e:
    out['torch_cuda'] = {{'error': str(e), 'available': False, 'devices': [], 'runtime_version': ''}}
print(json.dumps(out))
"""
    try:
        proc = subprocess.run(
            [project_python, "-c", script],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except Exception as e:
        return {"error": str(e)}
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout or "").strip()}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as e:
        return {"error": str(e)}
