"""Sample prompt file loading and per-config prompt forks.

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

__all__ = ['load_sample_prompts_file', 'save_sample_prompts_file', '_normalize_prompt_file_path', '_sample_prompts_path_for_config']

def load_sample_prompts_file(rel_path: str | None = None) -> dict[str, Any]:
    normalized = _normalize_prompt_file_path(rel_path or DEFAULT_SAMPLE_PROMPTS_FILE)
    path = (ROOT / normalized).resolve()
    if not path.exists():
        return {"ok": True, "file": normalized, "content": "", "prompts": []}
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines()
    prompts = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    return {
        "ok": True,
        "file": normalized,
        "content": content,
        "prompts": prompts,
    }


def save_sample_prompts_file(
    content: str,
    rel_path: str | None = None,
    *,
    train_config_file: str | None = None,
) -> dict[str, Any]:
    normalized = _normalize_prompt_file_path(rel_path or DEFAULT_SAMPLE_PROMPTS_FILE)
    if train_config_file:
        normalized = _sample_prompts_path_for_config(train_config_file)
    text = str(content or "")
    lines = text.splitlines()
    prompts = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    path = (ROOT / normalized).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {
        "ok": True,
        "file": normalized,
        "content": text,
        "prompts": prompts,
        "message": f"已保存 {len(prompts)} 条预览提示词",
    }


def _normalize_prompt_file_path(value: str) -> str:
    clean = str(value or "").replace("\\", "/").strip()
    if not clean:
        clean = DEFAULT_SAMPLE_PROMPTS_FILE
    path = Path(clean)
    if path.is_absolute():
        try:
            clean = path.resolve().relative_to(ROOT.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError("提示词文件必须在项目目录内") from exc
        path = Path(clean)
    if ".." in path.parts:
        raise ValueError("提示词文件路径不能包含 ..")
    if path.suffix.lower() != ".txt":
        raise ValueError("提示词文件必须是 .txt")
    if not path.as_posix().startswith("configs/"):
        raise ValueError("提示词文件必须保存在 configs/ 下")
    return path.as_posix().lstrip("/")


def _sample_prompts_path_for_config(train_config_file: str) -> str:
    normalized_config = _normalize_config_rel_path(train_config_file)
    config_path = _safe_resolve(normalized_config)
    if config_path is None or Path(normalized_config).suffix.lower() != ".toml":
        raise ValueError("训练配置文件路径不合法")
    try:
        rel_to_configs = Path(normalized_config).relative_to("configs")
    except ValueError as exc:
        raise ValueError("训练配置文件必须保存在 configs/ 下") from exc
    prompt_path = Path("configs") / "sample-prompts" / rel_to_configs.with_suffix(".txt")
    return _normalize_prompt_file_path(prompt_path.as_posix())




for _name in __all__:
    globals()[_name] = _exported(globals()[_name])
