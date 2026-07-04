#!/usr/bin/env python
"""Explain effective config values and their source layers.

Examples:
  python scripts/config_explain.py --method lora --preset default --key network_dim
  python scripts/config_explain.py --method rokkotsu --methods-subdir imported --runtime-config output/runs/x/config.runtime.toml --json
  python tasks.py explain-config METHOD=lora PRESET=balanced_16g -- --key blocks_to_swap
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import toml

from library.config.provenance import explain_key, trace_method_config


def _parse_override(text: str) -> tuple[str, Any]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("override must be KEY=VALUE")
    key, raw = text.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("override key cannot be empty")
    try:
        parsed = toml.loads(f"value = {raw}\n")["value"]
    except Exception:
        parsed = raw
    return key, parsed


def _render_text(payload: dict[str, Any], *, keys: list[str] | None) -> str:
    lines: list[str] = []
    lines.append(
        "Config trace: "
        f"method={payload['method']} preset={payload['preset']} "
        f"methods_subdir={payload['methods_subdir']}"
    )
    lines.append("Layers:")
    for layer in payload["layers"]:
        lines.append(f"  - {layer['kind']}: {layer['source']} ({len(layer['keys'])} keys)")
    lines.append("")

    selected = keys or sorted(payload["values"])
    for key in selected:
        if key not in payload["values"]:
            lines.append(f"{key}: <missing>")
            continue
        item = explain_key(payload, key)
        lines.append(f"{key} = {item['value']!r}  # from {item['source']}")
        for entry in item["history"]:
            marker = " overrides" if entry["overrides_previous"] else ""
            lines.append(
                f"    {entry['kind']}:{marker} {entry['source']} -> {entry['value']!r}"
            )
    return "\n".join(lines) + "\n"


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    trace = trace_method_config(
        args.method,
        args.preset,
        configs_dir=args.configs_dir,
        methods_subdir=args.methods_subdir,
        runtime_config=args.runtime_config,
        overrides=dict(args.override),
        strict=args.strict,
    )
    if args.key:
        missing = [key for key in args.key if key not in trace["values"]]
        if missing:
            raise KeyError(", ".join(missing))
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--preset", default="default")
    parser.add_argument("--methods-subdir", default="methods")
    parser.add_argument("--configs-dir", default="configs")
    parser.add_argument("--runtime-config", default=None)
    parser.add_argument("--key", action="append", default=None, help="Limit output to one key; repeatable.")
    parser.add_argument("--override", action="append", type=_parse_override, default=[])
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    try:
        trace = build_payload(args)
    except KeyError as exc:
        raise SystemExit(f"unknown config key(s): {exc.args[0]}") from exc
    if args.key:
        if args.json:
            print(json.dumps({key: explain_key(trace, key) for key in args.key}, indent=2, ensure_ascii=False))
            return
    if args.json:
        print(json.dumps(trace, indent=2, ensure_ascii=False))
    else:
        print(_render_text(trace, keys=args.key), end="")


if __name__ == "__main__":
    main()
