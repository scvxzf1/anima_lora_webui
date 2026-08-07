"""Minimal ``.env`` loader — no external dependency.

Used by scripts that need user-specific paths and credentials (HF token,
ComfyUI registry token, external corpus directories) without hardcoding
them in the repo.

Format: standard ``KEY=VALUE`` lines, ``#`` for comments, optional surrounding
single or double quotes around the value. No shell interpolation; values are
taken literally. Existing process env wins over file values (so a CLI
``CAPTION_CORPUS_DIR=… make foo`` overrides the file).

Looks for ``.env`` at the project root by default — the directory two levels
up from this file (``anima_lora/``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def anima_home() -> Path:
    """Repo home used to anchor every repo-relative path.

    Defaults to :func:`project_root` (the ``anima_lora/`` checkout that holds
    ``configs/``, ``models/``, ``output/`` …). Set ``ANIMA_HOME`` to override —
    this is what lets ``import anima_lora`` and the CLI run from *any* working
    directory instead of requiring a ``cd`` into the repo first.
    """
    override = os.environ.get("ANIMA_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return project_root()


def resolve_under_home(path) -> Path:
    """Resolve a possibly-relative path against :func:`anima_home`.

    Absolute and ``~``-prefixed paths pass through untouched; bare relative
    paths are interpreted relative to the repo home rather than the current
    working directory. Idempotent (absolute in → same path out), so it is safe
    to call at every layer of a call chain without double-anchoring.
    """
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return anima_home() / p


def load_dotenv(path: Optional[Path] = None) -> dict[str, str]:
    """Read a ``.env`` file into ``os.environ`` (without overriding existing keys).

    Returns the dict of values that were *added* (useful for logging /
    test introspection). A missing file is a no-op — callers shouldn't
    depend on .env being present.
    """
    if path is None:
        path = anima_home() / ".env"
    added: dict[str, str] = {}
    if not path.exists():
        return added
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            val = val[1:-1]
        if key and key not in os.environ:
            os.environ[key] = val
            added[key] = val
    return added


def expand_env_vars(value: str) -> str:
    """Expand env placeholders in user-facing config strings."""
    load_dotenv()
    return os.path.expanduser(os.path.expandvars(value))


def expand_env_vars_in_obj(value: Any) -> Any:
    """Recursively expand env placeholders in TOML/JSON-like trees."""
    if isinstance(value, str):
        if "$" not in value and not value.startswith("~"):
            return value
        return expand_env_vars(value)
    if isinstance(value, dict):
        return {k: expand_env_vars_in_obj(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env_vars_in_obj(v) for v in value]
    if isinstance(value, tuple):
        return tuple(expand_env_vars_in_obj(v) for v in value)
    return value


def _resolve_project_relative_override(value: str, *, label: str) -> Path:
    clean = expand_env_vars(value).strip()
    if not clean:
        raise ValueError(f"{label} cannot be empty")
    path = Path(clean)
    if ".." in path.parts:
        raise ValueError(f"{label} cannot contain '..'")
    if not path.is_absolute():
        path = project_root() / path
    return path.resolve()



def _webui_paths_section() -> dict[str, str]:
    """Read project-local ``.anima-webui-settings.toml [paths]`` overrides."""
    webui_paths_file = project_root() / ".anima-webui-settings.toml"
    if not webui_paths_file.exists():
        return {}
    try:
        import toml

        raw = toml.loads(webui_paths_file.read_text(encoding="utf-8"))
    except Exception:
        return {}
    section = raw.get("paths", {}) if isinstance(raw, dict) else {}
    if not isinstance(section, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in section.items():
        text_value = str(value or "").strip()
        if text_value:
            out[str(key)] = text_value
    return out


def get_configs_root() -> Path:
    """获取配置根目录，支持 WebUI 设置和环境变量覆盖。

    优先级：
    1. WebUI .claude/webui-settings.toml [paths].configs_root（项目根目录下的固定配置文件）
    2. ANIMA_CONFIGS_ROOT 环境变量
    3. 默认 project_root()/configs

    相对路径相对于项目根目录解析，绝对路径直接使用。
    """
    # 确保 .env 已加载
    load_dotenv()

    # 1. 优先读取项目根目录下的 WebUI 路径配置文件（不跟随 configs/ 移动）
    webui_paths_file = project_root() / ".anima-webui-settings.toml"
    if webui_paths_file.exists():
        try:
            import toml
            raw = toml.loads(webui_paths_file.read_text(encoding="utf-8"))
        except Exception:
            pass
        else:
            section = raw.get("paths", {})
            if isinstance(section, dict):
                webui_value = str(section.get("configs_root") or "").strip()
                if webui_value:
                    return _resolve_project_relative_override(
                        webui_value,
                        label="configs_root",
                    )

    # 2. 读取环境变量
    env_value = os.environ.get("ANIMA_CONFIGS_ROOT")
    if env_value:
        return _resolve_project_relative_override(
            env_value,
            label="ANIMA_CONFIGS_ROOT",
        )

    # 3. 默认值
    return project_root() / "configs"


def get_training_history_root() -> Path:
    """获取训练历史根目录。

    优先级：
    1. WebUI .anima-webui-settings.toml [paths].history_root
    2. ANIMA_TRAINING_HISTORY_ROOT 环境变量
    3. 默认 configs_root/web-training-history
    """
    # 确保 .env 已加载
    load_dotenv()

    webui_value = _webui_paths_section().get("history_root", "").strip()
    if webui_value:
        return _resolve_project_relative_override(
            webui_value,
            label="history_root",
        )

    env_value = os.environ.get("ANIMA_TRAINING_HISTORY_ROOT")
    if env_value:
        return _resolve_project_relative_override(
            env_value,
            label="ANIMA_TRAINING_HISTORY_ROOT",
        )
    return get_configs_root() / "web-training-history"


def get_training_queue_root() -> Path:
    """获取训练队列根目录。

    优先级：
    1. WebUI .anima-webui-settings.toml [paths].queue_root
    2. ANIMA_TRAINING_QUEUE_ROOT 环境变量
    3. 默认 configs_root/web-training-queue
    """
    # 确保 .env 已加载
    load_dotenv()

    webui_value = _webui_paths_section().get("queue_root", "").strip()
    if webui_value:
        return _resolve_project_relative_override(
            webui_value,
            label="queue_root",
        )

    env_value = os.environ.get("ANIMA_TRAINING_QUEUE_ROOT")
    if env_value:
        return _resolve_project_relative_override(
            env_value,
            label="ANIMA_TRAINING_QUEUE_ROOT",
        )
    return get_configs_root() / "web-training-queue"


# ── Model family (multi-model support, see docs/multi_model_support.md) ──
#
# `model_family` is the switch between Anima (default) and Krea-2-Raw
# (docs/proposal/krea2_raw_migration.md). It is not yet wired into base.toml
# (that lands in migration stage 6); for stage 0 the env var
# `ANIMA_MODEL_FAMILY` is the single source of truth, and anima-only scripts
# (distill_turbo / distill_mod / distill_spd / dcw / edit / merge_to_dit)
# read it via `resolve_model_family()` to refuse running under a non-anima
# family instead of silently assuming Anima cross-attn / AdaLN geometry.
DEFAULT_MODEL_FAMILY = "anima"
_ANIMA_ONLY_FAMILIES = ("anima",)


def resolve_model_family() -> str:
    """当前 model family（默认 "anima"，可被 ``ANIMA_MODEL_FAMILY`` 覆盖）。

    阶段 0 过渡实现：最终真相源是 ``configs/base.toml`` 的 ``model_family``
    键（阶段 6 落地），在那之前用环境变量作为单一来源，让 anima-only 脚本
    能统一读取而不各自硬编码。
    """
    load_dotenv()
    value = os.environ.get("ANIMA_MODEL_FAMILY")
    if value:
        return value.strip().lower()
    return DEFAULT_MODEL_FAMILY


def assert_anima_only(script: str, family: str | None = None) -> None:
    """Anim-only 脚本守卫：当 model family 不是 anima 时硬拒绝。

    用于依赖 Anima 专属 cross-attention / AdaLN / fusion-head / DirectEdit
    几何的脚本（蒸馏、DCW、edit、merge）。这些在 Krea-2-Raw single-stream
    MMDiT 上首日不支持（提案 §1 非目标），用 anima 假设硬跑会产生难定位的
    错误结果。

    ``family`` 通常来自脚本的 ``--model_family`` argparse flag；缺省时回退到
    :func:`resolve_model_family`（环境变量）。
    """
    family = (family or resolve_model_family()).strip().lower()
    if family in _ANIMA_ONLY_FAMILIES:
        return
    raise SystemExit(
        f"{script} 当前只支持 anima family（收到 model_family={family!r}）。"
        "Krea-2-Raw 上的对应能力首日未实现，见 "
        "docs/proposal/krea2_raw_migration.md §1 非目标。"
    )
