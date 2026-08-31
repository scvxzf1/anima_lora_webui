"""Configuration and secret handling for the external captioning bridge.

The tagging workbench deliberately keeps provider credentials outside the normal
WebUI settings response.  Non-secret options are stored in an ignored TOML
file, while the API key is stored in a separate ignored file (or supplied via
the environment for deployments).  Callers only receive the public projection
returned by :func:`get_public_settings`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import toml

from library.env import anima_home
from web.services.atomic_io import atomic_write_text

from .memory_log import DEFAULT_LOG_RETENTION_LINES, normalize_log_retention

ROOT = anima_home()
SETTINGS_FILE = ROOT / "configs" / "captioning" / "settings.toml"
SECRETS_FILE = ROOT / ".anima-captioning-secrets.toml"

DEFAULT_SETTINGS: dict[str, Any] = {
    "provider": "openai_compatible",
    "base_url": "https://api.openai.com/v1",
    "model": "",
    "system_prompt": (
        "Describe the image for a training caption. Return one concise caption "
        "as plain text, without markdown or a leading label."
    ),
    "timeout_seconds": 120,
    "retry_count": 2,
    "retry_interval_seconds": 1.5,
    "concurrency": 2,
    "allow_private_network": False,
    "log_retention_lines": DEFAULT_LOG_RETENTION_LINES,
}

_ENV_BASE_URL = ("ANIMA_CAPTIONING_BASE_URL", "ANIMA_TAGGING_BASE_URL", "TAGGING_BASE_URL")
_ENV_MODEL = ("ANIMA_CAPTIONING_MODEL", "ANIMA_TAGGING_MODEL", "TAGGING_MODEL")
_ENV_KEY = ("ANIMA_CAPTIONING_API_KEY", "ANIMA_TAGGING_API_KEY", "TAGGING_API_KEY")


def get_public_settings() -> dict[str, Any]:
    """Return normalized provider settings without exposing the API key."""

    settings = load_settings()
    key = get_api_key()
    return {
        "ok": True,
        **settings,
        "api_key_configured": bool(key),
        "api_key_hint": "已配置 API Key" if key else "未配置 API Key",
    }


def load_settings() -> dict[str, Any]:
    """Load and normalize the ignored provider settings file."""

    raw = _read_toml(SETTINGS_FILE)
    section = raw.get("tagging") if isinstance(raw.get("tagging"), dict) else raw
    values = {**DEFAULT_SETTINGS, **(section if isinstance(section, dict) else {})}
    env_base = _first_env(_ENV_BASE_URL)
    env_model = _first_env(_ENV_MODEL)
    if env_base:
        values["base_url"] = env_base
    if env_model:
        values["model"] = env_model
    return normalize_settings(values)


def save_settings(data: dict[str, Any] | None) -> dict[str, Any]:
    """Persist non-secret settings and optionally update the API key.

    An empty ``api_key`` keeps the existing key.  ``clear_api_key`` explicitly
    removes the file value; an environment-provided key remains active because
    deployment configuration cannot be mutated by the browser.
    """

    payload = data if isinstance(data, dict) else {}
    current = load_settings()
    next_values = {**current}
    for key in DEFAULT_SETTINGS:
        if key in payload:
            next_values[key] = payload[key]
    normalized = normalize_settings(next_values)
    _write_toml(SETTINGS_FILE, {"tagging": normalized})

    if _as_bool(payload.get("clear_api_key"), False):
        _write_secret("")
    else:
        candidate = str(payload.get("api_key") or "").strip()
        if candidate:
            _write_secret(candidate)
    return get_public_settings()


def get_api_key() -> str:
    """Return the deployment key, preferring environment over local secrets."""

    env_key = _first_env(_ENV_KEY)
    if env_key:
        return env_key
    raw = _read_toml(SECRETS_FILE)
    section = raw.get("secrets") if isinstance(raw.get("secrets"), dict) else raw
    return str(section.get("api_key") or "").strip() if isinstance(section, dict) else ""


def normalize_settings(values: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize user-controlled values to a small, predictable schema."""

    source = values if isinstance(values, dict) else {}
    base_url = _normalize_base_url(source.get("base_url", DEFAULT_SETTINGS["base_url"]))
    model = str(source.get("model") or "").strip()[:256]
    system_prompt = str(source.get("system_prompt") or DEFAULT_SETTINGS["system_prompt"]).strip()
    if not system_prompt:
        system_prompt = DEFAULT_SETTINGS["system_prompt"]
    return {
        "provider": "openai_compatible",
        "base_url": base_url,
        "model": model,
        "system_prompt": system_prompt[:10000],
        "timeout_seconds": _clamp_int(source.get("timeout_seconds"), 120, 5, 900),
        "retry_count": _clamp_int(source.get("retry_count"), 2, 0, 6),
        "retry_interval_seconds": _clamp_float(source.get("retry_interval_seconds"), 1.5, 0.0, 60.0),
        "concurrency": _clamp_int(source.get("concurrency"), 2, 1, 8),
        "allow_private_network": _as_bool(source.get("allow_private_network"), False),
        "log_retention_lines": normalize_log_retention(source.get("log_retention_lines")),
    }


def validate_endpoint(settings: dict[str, Any] | None = None) -> str:
    """Validate the configured endpoint and enforce the private-network gate."""

    values = normalize_settings(settings or load_settings())
    try:
        parsed = urlsplit(values["base_url"])
        parsed.port
    except ValueError as exc:
        raise ValueError("API URL 的主机或端口无效") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("API URL 必须是带主机名的 http(s) 地址")
    if parsed.username or parsed.password:
        raise ValueError("API URL 不允许嵌入用户名或密码")
    if not values["allow_private_network"] and _is_private_hostname(parsed.hostname):
        raise ValueError("API URL 指向本机或私有网络；如确认安全，请开启“允许私有网络 API”")
    return values["base_url"]


def _normalize_base_url(value: Any) -> str:
    raw = str(value or DEFAULT_SETTINGS["base_url"]).strip().rstrip("/")
    if len(raw) > 2048:
        raise ValueError("API URL 过长")
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("API URL 的主机或端口无效") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("API URL 必须是带主机名的 http(s) 地址")
    if parsed.username or parsed.password:
        raise ValueError("API URL 不允许嵌入用户名或密码")
    if port is not None and not (1 <= port <= 65535):
        raise ValueError("API URL 端口必须在 1 到 65535 之间")
    if parsed.query or parsed.fragment:
        raise ValueError("API URL 不允许包含 query 或 fragment")
    return raw


def _is_private_hostname(hostname: str) -> bool:
    import ipaddress

    host = str(hostname or "").strip().lower().rstrip(".")
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(address.is_private or address.is_loopback or address.is_link_local or address.is_reserved)


def _first_env(names: tuple[str, ...]) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        raw = toml.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else {}
    except (OSError, toml.TomlDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _write_toml(path: Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    atomic_write_text(target, toml.dumps(payload))
    _restrict_file_permissions(target)


def _write_secret(value: str) -> None:
    target = Path(SECRETS_FILE)
    if not value:
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise ValueError(f"清除 API Key 失败：{exc}") from exc
        return
    _write_toml(target, {"secrets": {"api_key": value}})


def _restrict_file_permissions(path: Path) -> None:
    try:
        path.chmod(0o600)
    except OSError:
        # Windows and some mounted filesystems do not expose POSIX modes.
        pass


def _clamp_int(value: Any, default: int, lower: int, upper: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(lower, min(upper, number))


def _clamp_float(value: Any, default: float, lower: float, upper: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if number != number:  # NaN
        number = default
    return max(lower, min(upper, number))


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
