"""Persistent tagging connection profiles.

Profiles are deliberately separate from the legacy flat ``settings.toml``
schema.  The compatibility layer in :mod:`settings` can continue serving old
clients while the Dragon UI uses this module for multiple connections.
"""

from __future__ import annotations

import re
import time
import uuid
from pathlib import Path
from typing import Any

import toml

from web.services.atomic_io import atomic_write_text

from . import settings as tagging_settings
from .provider_registry import get_provider_type, list_provider_types, normalize_provider

PROFILE_FILE_NAME = "provider-profiles.toml"
LEGACY_PROFILE_ID = "legacy-openai"
MAX_PROFILES = 100
MAX_PROFILE_NAME_LENGTH = 80
MAX_PROFILE_ID_LENGTH = 64
_PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_LOCAL_DEFAULT_ASSETS = {
    "wd14": "wd14-eva02-large-v3",
    # Keep the public v1 as the default.  The fixed manifest exposes four
    # WD14 choices and two CLTagger choices; v2 is an explicit gated choice
    # and is never downloaded implicitly.
    "cltagger": "cltagger-v1-02",
}
_LOCAL_DEVICES = {"auto", "cpu", "cuda"}


def profiles_file() -> Path:
    """Resolve the profile file beside the existing tagging settings file."""

    return Path(tagging_settings.SETTINGS_FILE).with_name(PROFILE_FILE_NAME)


def profile_store_exists() -> bool:
    return profiles_file().is_file()


def list_profiles() -> dict[str, Any]:
    profiles, active_id = _load_profiles()
    return {
        "ok": True,
        "active_profile_id": active_id,
        "profiles": [_public_profile(profile) for profile in profiles],
        "provider_types": list_provider_types(),
    }


def get_profile(profile_id: str | None = None) -> dict[str, Any]:
    profiles, active_id = _load_profiles()
    target_id = str(profile_id or active_id).strip() or active_id
    profile = next((item for item in profiles if item["id"] == target_id), None)
    if profile is None:
        raise KeyError("打标接入预设不存在")
    return _public_profile(profile)


def get_effective_settings(profile_id: str | None = None) -> dict[str, Any]:
    """Return an internal provider snapshot for a job or connectivity check."""

    profiles, active_id = _load_profiles()
    target_id = str(profile_id or active_id).strip() or active_id
    profile = next((item for item in profiles if item["id"] == target_id), None)
    if profile is None:
        raise KeyError("打标接入预设不存在")
    config = dict(profile.get("config") or {})
    if profile["provider"] == "openai_compatible":
        values = {**tagging_settings.DEFAULT_SETTINGS, **config, "provider": "openai_compatible"}
        effective = tagging_settings.normalize_settings(values)
    else:
        effective = {"provider": profile["provider"], **config}
    # These fields are deliberately private and are removed by public job
    # projection.  They let the provider and secret resolver remain profile-aware.
    effective["_profile_id"] = profile["id"]
    effective["_profile_name"] = profile["name"]
    return effective


def get_profile_api_key(profile_id: str | None = None) -> str:
    """Resolve a profile-specific key without exposing it to callers."""

    profiles, active_id = _load_profiles()
    target_id = str(profile_id or active_id).strip() or active_id
    if target_id == LEGACY_PROFILE_ID:
        return tagging_settings._get_legacy_api_key()
    raw = _read_secrets()
    profile_secrets = raw.get("profiles") if isinstance(raw.get("profiles"), dict) else {}
    value = profile_secrets.get(target_id) if isinstance(profile_secrets, dict) else None
    if isinstance(value, dict):
        return str(value.get("api_key") or "").strip()
    return ""


def create_profile(payload: dict[str, Any] | None) -> dict[str, Any]:
    source = payload if isinstance(payload, dict) else {}
    profiles, active_id = _load_profiles()
    if len(profiles) >= MAX_PROFILES:
        raise ValueError(f"最多保存 {MAX_PROFILES} 个接入预设")
    profile = _normalize_profile(source)
    if any(item["id"] == profile["id"] for item in profiles):
        raise ValueError("接入预设 ID 已存在")
    profiles.append(profile)
    # Creating a profile must not silently switch a running workspace.  The
    # user explicitly activates it after reviewing its status.
    if not active_id:
        active_id = profile["id"]
    _save_profiles(profiles, active_id)
    _update_profile_secret(profile["id"], source)
    return _response(profiles, active_id, profile["id"])


def update_profile(profile_id: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    profiles, active_id = _load_profiles()
    target_id = str(profile_id or "").strip()
    index = next((idx for idx, item in enumerate(profiles) if item["id"] == target_id), None)
    if index is None:
        raise KeyError("打标接入预设不存在")
    source = payload if isinstance(payload, dict) else {}
    current = profiles[index]
    if "provider" in source and normalize_provider(source.get("provider")) != current["provider"]:
        raise ValueError("暂不支持修改已有预设的接入类型，请新建预设")
    merged = {**current, **source}
    if isinstance(current.get("config"), dict):
        merged["config"] = {**current["config"], **(source.get("config") if isinstance(source.get("config"), dict) else {})}
    profile = _normalize_profile(merged, current=current)
    profile["id"] = target_id
    profiles[index] = profile
    _save_profiles(profiles, active_id)
    _update_profile_secret(target_id, source)
    return _response(profiles, active_id, target_id)


def delete_profile(profile_id: str) -> dict[str, Any]:
    profiles, active_id = _load_profiles()
    target_id = str(profile_id or "").strip()
    if not any(item["id"] == target_id for item in profiles):
        raise KeyError("打标接入预设不存在")
    if len(profiles) <= 1:
        raise ValueError("至少保留一个接入预设")
    remaining = [item for item in profiles if item["id"] != target_id]
    if active_id == target_id:
        active_id = remaining[0]["id"]
    _save_profiles(remaining, active_id)
    _update_profile_secret(target_id, {"clear_api_key": True})
    return _response(remaining, active_id, "")


def activate_profile(profile_id: str) -> dict[str, Any]:
    profiles, _active_id = _load_profiles()
    target_id = str(profile_id or "").strip()
    target = next((item for item in profiles if item["id"] == target_id), None)
    if target is None:
        raise KeyError("打标接入预设不存在")
    public = _public_profile(target)
    if not public["available"]:
        reason = public.get("runtime_message") or public.get("status") or "接入预设尚未就绪"
        raise ValueError(f"接入预设当前不可用：{reason}")
    if public["kind"] == "local":
        _assert_local_asset_verified(target)
    _save_profiles(profiles, target_id)
    return _response(profiles, target_id, target_id)


def save_active_compat(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Apply the old flat settings form to the active profile."""

    profiles, active_id = _load_profiles()
    target = next((item for item in profiles if item["id"] == active_id), None)
    if target is None:
        raise KeyError("打标接入预设不存在")
    if target["provider"] != "openai_compatible":
        raise ValueError("当前接入不是外部 API，不能使用旧版设置表单")
    config = {**target.get("config", {})}
    source = payload if isinstance(payload, dict) else {}
    for key in tagging_settings.DEFAULT_SETTINGS:
        if key in source:
            config[key] = source[key]
    updated = _normalize_profile({**target, "config": config}, current=target)
    updated["id"] = target["id"]
    profiles[profiles.index(target)] = updated
    _save_profiles(profiles, active_id)
    _update_profile_secret(active_id, source)
    return tagging_settings.get_public_settings()


def _load_profiles() -> tuple[list[dict[str, Any]], str]:
    if not profile_store_exists():
        legacy = _legacy_profile()
        return [legacy], legacy["id"]
    raw = _read_toml(profiles_file())
    values = raw.get("profiles") if isinstance(raw.get("profiles"), list) else []
    profiles: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in values[:MAX_PROFILES]:
        if not isinstance(item, dict):
            continue
        try:
            normalized = _normalize_profile(item)
        except ValueError:
            continue
        if normalized["id"] in seen:
            continue
        seen.add(normalized["id"])
        profiles.append(normalized)
    if not profiles:
        legacy = _legacy_profile()
        return [legacy], legacy["id"]
    active = str(raw.get("active_profile_id") or "").strip()
    if active not in seen:
        active = profiles[0]["id"]
    return profiles, active


def _legacy_profile() -> dict[str, Any]:
    settings = tagging_settings._load_legacy_settings()
    config = dict(settings)
    config.pop("provider", None)
    return {
        "id": LEGACY_PROFILE_ID,
        "name": "默认外部 API",
        "provider": "openai_compatible",
        "kind": "external",
        "config": config,
        "created_at": 0.0,
        "updated_at": 0.0,
    }


def _normalize_profile(payload: dict[str, Any], *, current: dict[str, Any] | None = None) -> dict[str, Any]:
    source = dict(current or {})
    source.update(payload)
    provider = normalize_provider(source.get("provider"), allow_empty=True)
    profile_id = str(source.get("id") or uuid.uuid4().hex[:12]).strip()
    if not _PROFILE_ID_RE.fullmatch(profile_id):
        raise ValueError("接入预设 ID 只能包含字母、数字、下划线和短横线")
    name = str(source.get("name") or "").strip()
    if not name:
        raise ValueError("请输入接入预设名称")
    if len(name) > MAX_PROFILE_NAME_LENGTH:
        raise ValueError(f"接入预设名称最多 {MAX_PROFILE_NAME_LENGTH} 个字符")
    raw_config = source.get("config") if isinstance(source.get("config"), dict) else {}
    flat_keys = set(tagging_settings.DEFAULT_SETTINGS) | {
        "asset_id",
        "model_id",
        "device",
        "gpu_index",
        "batch_size",
        "general_threshold",
        "character_threshold",
        "blacklist",
        "add_copyright_tag",
        "add_artist_tag",
        "add_meta_tag",
        "add_model_tag",
        "add_rating_tag",
        "add_quality_tag",
    }
    raw_config = {**raw_config, **{key: source[key] for key in flat_keys if key in source}}
    if provider == "openai_compatible":
        values = {**tagging_settings.DEFAULT_SETTINGS, **raw_config, "provider": provider}
        normalized_config = tagging_settings.normalize_settings(values)
        normalized_config.pop("provider", None)
    else:
        normalized_config = _normalize_local_config(provider, raw_config)
    now = time.time()
    return {
        "id": profile_id[:MAX_PROFILE_ID_LENGTH],
        "name": name,
        "provider": provider,
        "kind": get_provider_type(provider)["kind"],
        "config": normalized_config,
        "created_at": _timestamp(source.get("created_at"), now),
        "updated_at": now if current else _timestamp(source.get("updated_at"), now),
    }


def _normalize_local_config(provider: str, source: dict[str, Any]) -> dict[str, Any]:
    from .model_assets import canonical_asset_id

    asset_id = canonical_asset_id(
        str(source.get("asset_id") or source.get("model_id") or _LOCAL_DEFAULT_ASSETS[provider]).strip()
    )[:128]
    if not asset_id:
        raise ValueError("本地接入必须选择模型资产")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", asset_id):
        raise ValueError("模型资产 ID 只能包含字母、数字、点、下划线和短横线")
    device = str(source.get("device") or "auto").strip().lower()
    if device not in _LOCAL_DEVICES:
        raise ValueError("本地模型设备只能是 auto、cpu 或 cuda")
    gpu_index = _optional_gpu_index(source.get("gpu_index")) if device == "cuda" else None
    batch_size = _clamp_int(source.get("batch_size"), 8, 1, 64)
    general = _clamp_float(source.get("general_threshold"), 0.35, 0.0, 1.0)
    character = _clamp_float(source.get("character_threshold"), 0.6 if provider == "cltagger" else 0.85, 0.0, 1.0)
    raw_blacklist = source.get("blacklist")
    blacklist = raw_blacklist if isinstance(raw_blacklist, list) else []
    cleaned = [str(value).strip()[:128] for value in blacklist if str(value).strip()][:200]
    normalized = {
        "asset_id": asset_id,
        "device": device,
        "batch_size": batch_size,
        "general_threshold": general,
        "character_threshold": character,
        "blacklist": cleaned,
    }
    if gpu_index is not None:
        normalized["gpu_index"] = gpu_index
    if provider == "cltagger":
        # These gates mirror CLTagger's category vocabulary. They are kept in
        # the profile so a job is reproducible even when the active profile
        # changes later; the provider profile editor exposes all six switches.
        normalized.update(
            {
                "add_copyright_tag": tagging_settings._as_bool(source.get("add_copyright_tag"), True),
                "add_artist_tag": tagging_settings._as_bool(source.get("add_artist_tag"), False),
                "add_meta_tag": tagging_settings._as_bool(source.get("add_meta_tag"), False),
                "add_model_tag": tagging_settings._as_bool(source.get("add_model_tag"), False),
                "add_rating_tag": tagging_settings._as_bool(source.get("add_rating_tag"), False),
                "add_quality_tag": tagging_settings._as_bool(source.get("add_quality_tag"), False),
            }
        )
    return normalized


def _public_profile(profile: dict[str, Any]) -> dict[str, Any]:
    provider = get_provider_type(profile["provider"])
    config = dict(profile.get("config") or {})
    key_configured = bool(get_profile_api_key(profile["id"])) if provider["kind"] == "external" else False
    if provider["kind"] == "external":
        ready = bool(config.get("base_url") and config.get("model"))
        status = "ready" if ready else "needs_config"
        model = config.get("model", "")
    else:
        asset_state = "unknown"
        runtime_available = True
        runtime_message = ""
        try:
            from .model_assets import get_model_asset, inspect_asset
            from .providers.onnx_base import onnxruntime_status

            asset = get_model_asset(str(config.get("asset_id") or ""))
            asset_state = str(inspect_asset(asset, verify_hash=False).get("state") or "missing")
            runtime_available, runtime_message = onnxruntime_status()
        except (KeyError, OSError, ValueError):
            asset_state = "unknown"
            runtime_available = False
            runtime_message = "模型资产无效"
        status = {
            "installed": "ready",
            "missing": "not_installed",
            "partial": "needs_install",
            "corrupt": "needs_install",
        }.get(asset_state, "unknown_asset")
        if status == "ready" and not runtime_available:
            status = "needs_runtime"
        model = config.get("asset_id", "")
    return {
        "id": profile["id"],
        "name": profile["name"],
        "provider": profile["provider"],
        "provider_label": provider["label"],
        "kind": provider["kind"],
        "config": config,
        "model": model,
        "asset_id": config.get("asset_id", ""),
        "asset_state": asset_state if provider["kind"] == "local" else "",
        "status": status,
        "available": bool(provider["implemented"] and status == "ready"),
        "runtime_available": runtime_available if provider["kind"] == "local" else True,
        "runtime_message": runtime_message if provider["kind"] == "local" else "",
        "api_key_configured": key_configured,
        "api_key_hint": "已配置 API Key" if key_configured else "未配置 API Key",
        "created_at": profile.get("created_at", 0.0),
        "updated_at": profile.get("updated_at", 0.0),
    }


def _assert_local_asset_verified(profile: dict[str, Any]) -> None:
    """Recheck the complete asset hash before making a local profile active.

    Listing profiles intentionally uses a lightweight size-only inspection so
    opening the settings page does not hash gigabytes of user data. Activation
    is the state-changing boundary, so it performs the authoritative check.
    """

    config = profile.get("config") if isinstance(profile.get("config"), dict) else {}
    try:
        from .model_assets import get_model_asset, inspect_asset

        asset = get_model_asset(str(config.get("asset_id") or ""))
        if asset.provider != profile.get("provider"):
            raise ValueError("模型资产与 provider 不匹配")
        status = inspect_asset(asset, verify_hash=True)
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(f"接入预设当前不可用：{exc}") from exc
    if status.get("state") != "installed":
        raise ValueError("接入预设当前不可用：模型文件校验失败，请重新下载")


def _response(profiles: list[dict[str, Any]], active_id: str, selected_id: str) -> dict[str, Any]:
    selected = next((item for item in profiles if item["id"] == selected_id), None)
    return {
        "ok": True,
        "active_profile_id": active_id,
        "profile": _public_profile(selected) if selected else None,
        "profiles": [_public_profile(item) for item in profiles],
        "provider_types": list_provider_types(),
    }


def _save_profiles(profiles: list[dict[str, Any]], active_id: str) -> None:
    payload = {
        "active_profile_id": active_id,
        "profiles": profiles,
    }
    path = profiles_file()
    atomic_write_text(path, toml.dumps(payload))
    _restrict_file_permissions(path)


def _update_profile_secret(profile_id: str, payload: dict[str, Any]) -> None:
    if profile_id == LEGACY_PROFILE_ID:
        source = payload if isinstance(payload, dict) else {}
        if bool(source.get("clear_api_key")):
            tagging_settings._write_secret("")
        elif str(source.get("api_key") or "").strip():
            tagging_settings._write_secret(str(source["api_key"]).strip())
        return
    source = payload if isinstance(payload, dict) else {}
    clear = bool(source.get("clear_api_key"))
    candidate = str(source.get("api_key") or "").strip()
    if not clear and not candidate:
        return
    raw = _read_secrets()
    profiles = raw.get("profiles") if isinstance(raw.get("profiles"), dict) else {}
    profiles = dict(profiles)
    if clear:
        profiles.pop(profile_id, None)
    elif candidate:
        profiles[profile_id] = {"api_key": candidate}
    raw["profiles"] = profiles
    path = Path(tagging_settings.SECRETS_FILE)
    atomic_write_text(path, toml.dumps(raw))
    _restrict_file_permissions(path)


def _read_secrets() -> dict[str, Any]:
    raw = tagging_settings._read_toml(tagging_settings.SECRETS_FILE)
    return raw if isinstance(raw, dict) else {}


def _read_toml(path: Path) -> dict[str, Any]:
    try:
        raw = toml.loads(Path(path).read_text(encoding="utf-8")) if Path(path).is_file() else {}
    except (OSError, toml.TomlDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _restrict_file_permissions(path: Path) -> None:
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _timestamp(value: Any, default: float) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return default


def _clamp_int(value: Any, default: int, lower: int, upper: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(lower, min(upper, number))


def _optional_gpu_index(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("GPU 序号必须是非负整数")
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("GPU 序号必须是非负整数") from exc
    if index < 0 or index > 1024 or str(value).strip() != str(index):
        raise ValueError("GPU 序号必须是非负整数")
    return index


def _clamp_float(value: Any, default: float, lower: float, upper: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    if number != number:
        number = default
    return max(lower, min(upper, number))


__all__ = [
    "LEGACY_PROFILE_ID",
    "activate_profile",
    "create_profile",
    "delete_profile",
    "get_effective_settings",
    "get_profile",
    "get_profile_api_key",
    "list_profiles",
    "profile_store_exists",
    "profiles_file",
    "save_active_compat",
    "update_profile",
]
