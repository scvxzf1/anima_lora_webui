"""Config provenance tracing helpers.

This module is intentionally read-side only: it mirrors the existing config
merge order and records where each final value came from, without changing the
training loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import toml

from library.env import resolve_under_home

from .io import _display_path, _flatten_toml, _load_toml_with_base, _resolve_preset


@dataclass(frozen=True)
class ConfigLayer:
    kind: str
    source: str
    values: dict[str, Any]


def _config_path(path: str | Path) -> Path:
    p = Path(path)
    if p.suffix != ".toml":
        p = p.with_suffix(".toml")
    if not p.is_absolute():
        p = resolve_under_home(str(p))
    return p


def _read_flat(path: Path, *, strict: bool = False) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return _flatten_toml(toml.load(f), source=str(path), strict=strict)


def _merge_layers(layers: list[ConfigLayer]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for layer in layers:
        merged.update(layer.values)
    return merged


def _history(layers: list[ConfigLayer]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for layer in layers:
        for key, value in layer.values.items():
            entries = out.setdefault(key, [])
            entries.append(
                {
                    "kind": layer.kind,
                    "source": layer.source,
                    "value": value,
                    "overrides_previous": bool(entries),
                }
            )
    return out


def _current_sources(history: Mapping[str, list[dict[str, Any]]]) -> dict[str, str]:
    return {key: entries[-1]["source"] for key, entries in history.items() if entries}


def trace_method_config(
    method: str,
    preset: str = "default",
    *,
    configs_dir: str | Path = "configs",
    methods_subdir: str = "methods",
    runtime_config: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Return a serializable trace for the effective training config.

    Layer order:
    ``base.toml`` -> preset -> method TOML -> optional runtime config ->
    optional caller-provided overrides. The runtime layer is useful for WebUI
    frozen ``config.runtime.toml`` files; overrides model CLI or scripted
    last-mile values.
    """

    root = Path(resolve_under_home(str(configs_dir)))
    base_path = root / "base.toml"
    method_path = root / methods_subdir / f"{method}.toml"
    for path in (base_path, method_path):
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

    layers: list[ConfigLayer] = [
        ConfigLayer("base", _display_path(str(base_path)), _read_flat(base_path, strict=strict)),
    ]

    preset_section, preset_path, preset_tag = _resolve_preset(preset, str(root))
    layers.append(
        ConfigLayer(
            "preset",
            preset_tag,
            _flatten_toml({preset: preset_section}, source=preset_path, strict=strict),
        )
    )

    layers.append(
        ConfigLayer("method", _display_path(str(method_path)), _read_flat(method_path, strict=strict))
    )

    if runtime_config is not None:
        runtime_path = _config_path(runtime_config)
        if not runtime_path.exists():
            raise FileNotFoundError(f"Runtime config file not found: {runtime_path}")
        layers.append(
            ConfigLayer(
                "runtime",
                _display_path(str(runtime_path)),
                _load_toml_with_base(str(runtime_path), strict=strict),
            )
        )

    if overrides:
        layers.append(ConfigLayer("override", "CLI/override", dict(overrides)))

    hist = _history(layers)
    values = _merge_layers(layers)
    return {
        "method": method,
        "preset": preset,
        "methods_subdir": methods_subdir,
        "configs_dir": _display_path(str(root)),
        "values": values,
        "sources": _current_sources(hist),
        "history": hist,
        "layers": [
            {
                "kind": layer.kind,
                "source": layer.source,
                "keys": sorted(layer.values),
            }
            for layer in layers
        ],
    }


def explain_key(trace: Mapping[str, Any], key: str) -> dict[str, Any]:
    values = trace.get("values", {})
    history = trace.get("history", {})
    if key not in values:
        raise KeyError(key)
    return {
        "key": key,
        "value": values[key],
        "source": trace.get("sources", {}).get(key),
        "history": history.get(key, []),
    }
