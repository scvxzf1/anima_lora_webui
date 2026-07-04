#!/usr/bin/env python
"""Print training compatibility diagnostics for an effective config.

Examples:
  python scripts/config_compat.py --method lora --preset default
  python scripts/config_compat.py --method lora --override blocks_to_swap=8 --override dynamo_backend='"cudagraphs"'
  python scripts/config_compat.py --config-file output/runs/x/config.runtime.toml --json
  python tasks.py config-compat METHOD=lora PRESET=default -- --override blocks_to_swap=8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import toml

from library.config.io import _load_toml_with_base
from library.config.provenance import trace_method_config
from library.env import resolve_under_home
from library.training.compat_matrix import check_training_compat


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


def _config_file_path(path: str) -> Path:
    p = Path(path)
    if p.suffix != ".toml":
        p = p.with_suffix(".toml")
    if not p.is_absolute():
        p = resolve_under_home(str(p))
    return p


def _issue_dict(item) -> dict[str, Any]:
    return {"code": item.code, "key": item.key, "message": item.message}


def _mutation_dict(item) -> dict[str, Any]:
    return {
        "code": item.code,
        "key": item.key,
        "value": item.value,
        "message": item.message,
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    overrides = dict(args.override)
    if args.config_file:
        config_path = _config_file_path(args.config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        values = _load_toml_with_base(str(config_path), strict=args.strict)
        values.update(overrides)
        source = str(config_path)
        layers = [{"kind": "config_file", "source": source, "keys": sorted(values)}]
    else:
        trace = trace_method_config(
            args.method,
            args.preset,
            configs_dir=args.configs_dir,
            methods_subdir=args.methods_subdir,
            runtime_config=args.runtime_config,
            overrides=overrides,
            strict=args.strict,
        )
        values = dict(trace["values"])
        source = f"method={args.method} preset={args.preset} methods_subdir={args.methods_subdir}"
        layers = trace["layers"]

    result = check_training_compat(values)
    return {
        "ok": result.ok,
        "source": source,
        "layers": layers,
        "errors": [_issue_dict(item) for item in result.errors],
        "warnings": [_issue_dict(item) for item in result.warnings],
        "mutations": [_mutation_dict(item) for item in result.mutations],
        "effective": {
            "blocks_to_swap": values.get("blocks_to_swap"),
            "gradient_checkpointing": values.get("gradient_checkpointing"),
            "selective_checkpoint": values.get("selective_checkpoint"),
            "cpu_offload_checkpointing": values.get("cpu_offload_checkpointing"),
            "unsloth_offload_checkpointing": values.get("unsloth_offload_checkpointing"),
            "torch_compile": values.get("torch_compile"),
            "dynamo_backend": values.get("dynamo_backend"),
            "compile_inductor_mode": values.get("compile_inductor_mode"),
            "network_module": values.get("network_module"),
            "functional_loss_weight": values.get("functional_loss_weight"),
        },
    }


def _render_text(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    status = "OK" if payload["ok"] else "ERROR"
    lines.append(f"Training compat: {status}")
    lines.append(f"Source: {payload['source']}")
    lines.append("")
    lines.append("Effective:")
    for key, value in payload["effective"].items():
        if value is not None:
            lines.append(f"  {key}: {value!r}")

    for title, key in (("Errors", "errors"), ("Warnings", "warnings"), ("Mutations", "mutations")):
        items = payload[key]
        lines.append("")
        lines.append(f"{title}:")
        if not items:
            lines.append("  - none")
            continue
        for item in items:
            suffix = f" -> {item['value']!r}" if "value" in item else ""
            lines.append(f"  - [{item['code']}] {item['key']}{suffix}: {item['message']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", default=None, help="Inspect a direct TOML config file.")
    parser.add_argument("--method", default="lora")
    parser.add_argument("--preset", default="default")
    parser.add_argument("--methods-subdir", default="methods")
    parser.add_argument("--configs-dir", default="configs")
    parser.add_argument("--runtime-config", default=None)
    parser.add_argument("--override", action="append", type=_parse_override, default=[])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    payload = build_payload(args)
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(_render_text(payload), end="")


if __name__ == "__main__":
    main()
