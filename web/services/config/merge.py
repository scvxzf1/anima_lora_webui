"""Configuration merge and data-directory helpers.

This module is loaded by ``web.services.config_service`` as part of the
compatibility facade.  It snapshots legacy globals at import time and syncs
mutable path settings from the facade before exported calls so existing tests
and callers that monkeypatch ``config_service.ROOT`` continue to work.
"""

from __future__ import annotations

from functools import wraps

from web.services import config_service as _facade

for _name, _value in _facade.__dict__.items():
    if _name.startswith("__") and _name.endswith("__"):
        continue
    globals().setdefault(_name, _value)

_SYNC_NAMES = (
    "ROOT",
    "CONFIGS_DIR",
    "GUI_METHODS_DIR",
    "IMPORTED_CONFIGS_DIR",
    "PRESETS_FILE",
    "WEB_FILE_GROUPS_FILE",
    "WEB_USER_LOCKS_FILE",
    "DATASET_PRESETS_DIR",
    "resolve_output_root",
    "_display_settings_path",
    "save_raw_file",
    "load_raw_file",
    "delete_raw_file",
    "patch_raw_file_values",
    "preview_raw_file_patch",
    "get_config_file_meta",
    "list_config_file_groups",
    "move_config_file_to_group",
    "_inspect_network_weight",
    "LOGGER",
)


def _sync_from_facade() -> None:
    _exported_names = set(globals().get("__all__", ()))
    _legacy_module = getattr(_facade, "_legacy", None)
    for _name in _SYNC_NAMES:
        if not hasattr(_facade, _name):
            continue
        _value = getattr(_facade, _name)
        if _name not in _exported_names:
            globals()[_name] = _value
        if _legacy_module is not None:
            setattr(_legacy_module, _name, _value)


def _exported(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        _sync_from_facade()
        return fn(*args, **kwargs)

    return wrapper

__all__ = ['list_methods', 'list_variants', 'list_all_variants', 'list_presets', 'load_merged_config', 'suggest_data_dirs', 'suggest_dataset_dirs', 'apply_auto_data_dirs']

def list_methods() -> list[str]:
    return [
        "lora", "lokr", "ortholora", "tlora", "hydralora",
        "reft", "chimera", "soft_tokens", "ip_adapter", "easycontrol",
        "spd",
    ]


def list_variants(method: str) -> list[str]:
    if method == "spd":
        spd_config = CONFIGS_DIR / "methods" / "spd.toml"
        return ["spd"] if spd_config.exists() else []
    if not GUI_METHODS_DIR.exists():
        return []
    variants = [stem for _order, stem in _builtin_variants_by_family().get(method, [])]
    if not variants:
        variants = _legacy_exact_variant_for_method(method)
    variants.extend(_custom_gui_variants())
    return variants


def _builtin_variants_by_family() -> dict[str, list[tuple[int, str]]]:
    by_family: dict[str, list[tuple[int, str]]] = {}
    if not GUI_METHODS_DIR.exists():
        return by_family
    for path in GUI_METHODS_DIR.glob("*.toml"):
        if _display_path(path) in HIDDEN_CONFIG_FILES:
            continue
        meta = _read_variant_metadata(path)
        family = meta.get("family")
        if not isinstance(family, str) or not family:
            continue
        order = meta.get("order")
        order_int = order if isinstance(order, int) else 100
        by_family.setdefault(family, []).append((order_int, path.stem))
    for entries in by_family.values():
        entries.sort(key=lambda item: (item[0], item[1]))
    return by_family


def _read_variant_metadata(path: Path) -> dict[str, Any]:
    try:
        data = toml.loads(path.read_text(encoding="utf-8"))
    except (OSError, toml.TomlDecodeError):
        return {}
    meta = data.get("variant")
    return meta if isinstance(meta, dict) else {}


def _legacy_exact_variant_for_method(method: str) -> list[str]:
    path = GUI_METHODS_DIR / f"{method}.toml"
    if not path.is_file() or _display_path(path) in HIDDEN_CONFIG_FILES:
        return []
    return [method]


def _custom_gui_variants() -> list[str]:
    custom_dir = GUI_METHODS_DIR / "custom"
    if not custom_dir.exists():
        return []
    return [f"custom/{p.stem}" for p in sorted(custom_dir.glob("*.toml"))]


def list_all_variants() -> list[str]:
    if not GUI_METHODS_DIR.exists():
        return []
    return sorted(
        p.stem for p in GUI_METHODS_DIR.glob("*.toml")
        if _display_path(p) not in HIDDEN_CONFIG_FILES
    )


def list_presets() -> list[str]:
    if not PRESETS_FILE.exists():
        return []
    data = toml.loads(PRESETS_FILE.read_text(encoding="utf-8"))
    return sorted(k for k, v in data.items() if isinstance(v, dict))


def load_merged_config(variant: str, preset: str, methods_subdir: str = "gui-methods") -> dict[str, Any]:
    methods_dir = _safe_config_subdir(methods_subdir)
    if methods_dir is None:
        raise ValueError("配置目录不合法")
    base = _load(CONFIGS_DIR / "base.toml")
    presets_data = _load(PRESETS_FILE)
    pset = presets_data.get(preset, {}) if isinstance(presets_data.get(preset), dict) else {}
    meth = _load(methods_dir / f"{variant}.toml")

    merged: dict[str, Any] = {}
    for k, v in base.items():
        if k not in ("general", "datasets"):
            merged[k] = v
    for k, v in pset.items():
        merged[k] = v
    for k, v in meth.items():
        merged[k] = v
    merged.setdefault("max_train_steps", DEFAULT_MAX_TRAIN_STEPS)
    merged = expand_env_vars_in_obj(merged)
    return apply_auto_data_dirs(merged)


def suggest_data_dirs(source_image_dir: str) -> dict[str, Any]:
    source_path = _resolve_project_path(str(source_image_dir or ""))
    if not str(source_image_dir or "").strip():
        return {"ok": False, "error": "请先填写源图像目录 / source_image_dir"}
    return {
        "ok": True,
        "source_image_dir": _display_path(source_path),
        "resized_image_dir": _display_path(_derived_data_dir(source_path, "resized")),
        "lora_cache_dir": _display_path(_derived_data_dir(source_path, "lora_cache")),
    }


def suggest_dataset_dirs(source_dirs: list[str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(source_dirs):
        source = str(raw or "").strip()
        if not source:
            continue
        source_path = _resolve_project_path(source)
        rows.append({
            "index": idx,
            "source_dir": _display_path(source_path),
            "image_dir": _display_path(_derived_data_dir(source_path, "resized")),
            "cache_dir": _display_path(_derived_data_dir(source_path, "lora_cache")),
        })
    if not rows:
        return {"ok": False, "error": "请至少填写一个原始数据集路径"}
    return {"ok": True, "datasets": rows}


def apply_auto_data_dirs(cfg: dict[str, Any], *, create: bool = False) -> dict[str, Any]:
    next_cfg = dict(cfg)
    source_raw = str(next_cfg.get("source_image_dir") or "").strip()
    if not source_raw:
        return next_cfg
    source_path = _resolve_project_path(source_raw)
    resized_path = _auto_data_dir_for_key(next_cfg.get("resized_image_dir"), source_path, "resized")
    cache_path = _auto_data_dir_for_key(next_cfg.get("lora_cache_dir"), source_path, "lora_cache")
    next_cfg["resized_image_dir"] = _display_path(resized_path)
    next_cfg["lora_cache_dir"] = _display_path(cache_path)
    if create:
        resized_path.mkdir(parents=True, exist_ok=True)
        cache_path.mkdir(parents=True, exist_ok=True)
    return next_cfg




for _name in __all__:
    globals()[_name] = _exported(globals()[_name])
