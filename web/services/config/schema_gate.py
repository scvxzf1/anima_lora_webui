"""Schema gate helpers for WebUI raw/preflight config validation."""

from __future__ import annotations

from typing import Any

from library.config import schema as config_schema


def ensure_schema_populated() -> dict[str, config_schema.ConfigKey]:
    """Populate CONFIG_SCHEMA if empty; safe to call repeatedly."""
    schema = config_schema.get_schema()
    if schema:
        return schema
    try:
        import train as train_mod

        parser = train_mod.setup_parser()
        extras = (
            train_mod.build_network_extras()
            if hasattr(train_mod, "build_network_extras")
            else None
        )
        config_schema.populate_schema(parser, extras=extras)
    except Exception:
        # Keep soft: validation becomes no-op when schema cannot load.
        return config_schema.get_schema()
    return config_schema.get_schema()


def validate_patch_values(values: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for a top-level patch dict.

    Policy:
    - unknown key -> warning
    - choices mismatch / coerce failure -> error
    """
    ensure_schema_populated()
    schema = config_schema.get_schema()
    if not schema:
        return [], []

    errors: list[str] = []
    warnings: list[str] = []
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            continue
        resolved = config_schema.resolve_alias(key)
        if resolved not in schema:
            warnings.append(f"unknown key: {key}")
            continue
        spec = schema[resolved]
        try:
            coerced = config_schema._coerce_value(spec, value)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"{key}: 类型无法转换 ({exc})")
            continue
        if spec.choices and coerced not in spec.choices:
            if not (coerced is None and None in spec.choices):
                errors.append(
                    f"{key}={coerced!r} 不在允许取值 {list(spec.choices)} 中"
                )
    return errors, warnings


def validate_config_mapping(cfg: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Validate a merged/config dict with the same warning/error policy."""
    if not isinstance(cfg, dict):
        return [], []
    values = {
        key: value
        for key, value in cfg.items()
        if isinstance(key, str) and not isinstance(value, (dict, list))
    }
    return validate_patch_values(values)
