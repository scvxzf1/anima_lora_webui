#!/usr/bin/env python
"""Export a native ConvRot prequant checkpoint from a live bf16 DiT.

Writes ``anima_lora_convrot_prequant_v1`` safetensors for
``--convrot_weight_source prequant_checkpoint``.

Example::

    .venv/bin/python scripts/experiments/convrot_export_prequant.py \\
        --dit-path models/diffusion_models/anima-preview3-base.safetensors \\
        --scope mlp --group-size 256 \\
        --out output/tests/convrot_prequant_mlp_g256.safetensors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from library.runtime.convrot.prequant import (
    build_prequant_layers_from_modules,
    save_prequant_checkpoint,
)
from library.runtime.convrot.scope import classify_convrot_linear_module
from scripts.experiments.int8_linear_equivalence_probe import DEFAULT_DIT_PATH


def _collect_scope_linears(anima: nn.Module, scope: str) -> dict[str, nn.Linear]:
    out: dict[str, nn.Linear] = {}
    for name, module in anima.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if classify_convrot_linear_module(name, scope=scope) is None:
            continue
        if module.bias is not None:
            continue
        module.weight.requires_grad_(False)
        out[name] = module
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dit-path", type=Path, default=DEFAULT_DIT_PATH)
    parser.add_argument("--scope", default="mlp")
    parser.add_argument("--group-size", type=int, default=256)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/tests/convrot_prequant_mlp_g256.safetensors"),
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from library.anima.weights import load_anima_model

    device = torch.device(args.device)
    anima = load_anima_model(
        device=device,
        dit_path=str(args.dit_path),
        attn_mode="torch",
        loading_device=device,
        dit_weight_dtype=torch.bfloat16,
    )
    anima.to(device=device)
    modules = _collect_scope_linears(anima, args.scope)
    if not modules:
        raise SystemExit(f"no Linear modules matched scope={args.scope!r}")
    layers = build_prequant_layers_from_modules(modules, group_size=args.group_size)
    path = save_prequant_checkpoint(
        args.out,
        layers,
        group_size=args.group_size,
        mode="w8a16",
        metadata_extra={"scope": args.scope, "dit_path": str(args.dit_path)},
    )
    print(f"wrote {path} layers={len(layers)} group={args.group_size}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
