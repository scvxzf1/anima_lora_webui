#!/usr/bin/env python3
"""Bake a LoRA adapter into the base DiT and save as a new safetensors file.

The merged output is a standalone DiT checkpoint (ComfyUI-compatible, `net.`
prefixed) that reproduces LoRA+base inference without needing the adapter at
load time.

Supported: plain LoRA, OrthoLoRA, T-LoRA, LoHa, GLoRA. (T-LoRA's timestep mask is
training-only — inference already runs full rank, so baking is bit-equivalent.
GLoRA is bakeable because the merge path has access to the base Linear weight.)

Not supported (refuse by default; --allow-partial to drop and proceed):
  - ReFT              (block-level hook, not a Linear weight delta)
  - HydraLoRA moe     (layer-local router can't be baked under static weights)
  - step-expert turbo (per-step heads can't be baked into one DiT weight)
  - postfix / prefix  (cross-attn KV splice, not a weight delta)
  - register tokens   (ride the self-attn sequence, not a weight delta)

Same merge path as train.py:1499's --base_weights warm-start.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

import torch


from library.anima import weights as anima_weights  # noqa: E402
from library.inference.models import _classify_adapter_capability  # noqa: E402
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


# Marker → human-readable kind. Substring match on safetensors keys.
_NON_BAKEABLE_MARKERS: dict[str, str] = {
    "reft_": "ReFT (block-level hook)",
    ".lora_up_weight": "HydraLoRA stacked (per-layer router)",
    ".lora_ups.": "HydraLoRA split (per-layer router) / step-expert turbo (per-step heads)",
    "postfix_": "postfix (cross-attn KV splice)",
    "prefix_": "prefix (cross-attn KV splice)",
    "register_tokens": "register tokens (ride the self-attn sequence, not a weight delta)",
}

_NON_BAKEABLE_METADATA_SPECS: dict[str, str] = {
    "reft": "ReFT (block-level hook)",
    "hydra": "HydraLoRA moe (per-layer router)",
    "ortho_hydra": "OrthoHydraLoRA moe (per-layer router)",
    "chimera_hydra": "ChimeraHydra (dual-pool router)",
    "stacked_experts_global_fei": "HydraLoRA stacked experts (global FEI router)",
    "step_expert": "step-expert turbo (per-step heads)",
    "ip_adapter": "IP-Adapter (side network, not a Linear delta)",
    "easycontrol": "EasyControl (side network, not a Linear delta)",
    "soft_tokens": "Soft Tokens (prompt-side state, not a Linear delta)",
    "register": "register tokens (ride the self-attn sequence, not a weight delta)",
}
_BAKEABLE_METADATA_SPECS = {"", "lora", "ortho", "dora", "loha", "lokr", "glora", "vera"}

_DTYPE_MAP: dict[str, torch.dtype] = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def pick_latest_adapter(adapter_dir: Path) -> Path:
    """Latest `*.safetensors` in adapter_dir that is bakeable.

    Skips ``*_moe.safetensors`` (HydraLoRA router-live), ``*.bak.*`` (backups),
    and any file whose name contains ``postfix`` / ``prefix`` (those are
    separate non-weight-delta adapters).
    """
    candidates = sorted(
        (
            f
            for f in adapter_dir.glob("*.safetensors")
            if not f.name.endswith("_moe.safetensors")
            and ".bak." not in f.name
            and "postfix" not in f.name.lower()
            and "prefix" not in f.name.lower()
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No bakeable *.safetensors found in {adapter_dir} "
            "(excludes *_moe, *postfix*, *prefix*, and *.bak.*)"
        )
    return candidates[0]


def scan_non_bakeable_keys(weights_sd: dict) -> dict[str, int]:
    """Return ``{kind: count}`` for any key that matches a non-bakeable marker."""
    found: dict[str, int] = {}
    for key in weights_sd.keys():
        for marker, kind in _NON_BAKEABLE_MARKERS.items():
            if marker in key:
                found[kind] = found.get(kind, 0) + 1
                break
    return found


def read_safetensors_metadata(path: Path) -> dict[str, str]:
    """Read safetensors metadata without loading tensors."""
    if path.suffix != ".safetensors":
        return {}
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as f:
        return dict(f.metadata() or {})


def scan_non_bakeable_metadata(metadata: dict[str, str]) -> dict[str, int]:
    """Return non-bakeable adapter kinds proven by metadata stamps."""
    base_compute = str(metadata.get("ss_base_compute") or "bf16").strip().lower()
    if base_compute in {"w8a16_convrot", "w8a8_convrot", "w8a16", "w8a8"}:
        return {
            "ConvRot base_compute (requires dequant/high-precision base before bake)": 1
        }
    mode = str(metadata.get("ss_convrot_mode") or "").strip().lower()
    if mode in {"w8a16", "w8a8"}:
        return {
            "ConvRot base_compute (requires dequant/high-precision base before bake)": 1
        }
    spec = str(metadata.get("ss_network_spec") or "").strip().lower()
    if spec in _NON_BAKEABLE_METADATA_SPECS:
        return {_NON_BAKEABLE_METADATA_SPECS[spec]: 1}
    if spec not in _BAKEABLE_METADATA_SPECS:
        module = str(metadata.get("ss_network_module") or "").strip()
        if module.startswith("networks.methods."):
            label = spec or module
            return {f"{label} (non-LoRA method adapter)": 1}
    return {}


def scan_non_bakeable_adapter(adapter: Path, weights_sd: dict) -> dict[str, int]:
    """Combine key, metadata, and capability checks for static bake refusal."""
    found = scan_non_bakeable_keys(weights_sd)
    metadata = read_safetensors_metadata(adapter)
    for kind, count in scan_non_bakeable_metadata(metadata).items():
        found[kind] = found.get(kind, 0) + count
    capability = _classify_adapter_capability(str(adapter))
    if not capability.supports_static_merge:
        kind = f"{capability.kind} (dynamic-only adapter)"
        found[kind] = found.get(kind, 0) + 1
    return found


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bake a LoRA adapter into the base DiT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--adapter_dir",
        type=Path,
        default=Path("output/ckpt"),
        help="Directory to pick the latest adapter from (ignored if --adapter is set).",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Explicit adapter .safetensors path (overrides --adapter_dir).",
    )
    parser.add_argument(
        "--dit",
        type=Path,
        default=Path("models/diffusion_models/anima-base-v1.0.safetensors"),
        help="Base DiT safetensors.",
    )
    parser.add_argument(
        "--multiplier",
        type=float,
        default=1.0,
        help="LoRA strength to bake in.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path. Defaults to <adapter-stem>_merged.safetensors next to the adapter.",
    )
    parser.add_argument("--dtype", choices=list(_DTYPE_MAP), default="bf16")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for the merge math.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Drop unsupported keys (ReFT / Hydra moe / postfix / prefix) and bake the rest. "
        "The merged DiT will not reproduce those components.",
    )
    parser.add_argument(
        "--network_module",
        default="networks.lora_anima",
        help="Network module providing create_network_from_weights.",
    )
    args = parser.parse_args()

    adapter = args.adapter or pick_latest_adapter(args.adapter_dir)
    logger.info(f"adapter: {adapter}")

    from safetensors.torch import load_file

    weights_sd = load_file(str(adapter))
    non_bakeable = scan_non_bakeable_adapter(adapter, weights_sd)
    if non_bakeable:
        parts = [f"{count} {kind}" for kind, count in non_bakeable.items()]
        msg = "Non-bakeable keys detected: " + ", ".join(parts) + "."
        if not args.allow_partial:
            logger.error(
                msg
                + " Re-run with --allow-partial to drop them and bake the LoRA portion, "
                "or retrain without these components. These cannot be folded into DiT Linear weights."
            )
            return 2
        logger.warning(
            msg
            + " --allow-partial set; these components will be absent from the merged DiT."
        )

    dtype = _DTYPE_MAP[args.dtype]

    logger.info(f"loading base DiT: {args.dit}")
    unet = anima_weights.load_anima_model(
        device=args.device,
        dit_path=str(args.dit),
        attn_mode="torch",  # merge never runs a forward pass
        loading_device=args.device,
        dit_weight_dtype=dtype,
    )

    logger.info(f"building adapter network from weights (multiplier={args.multiplier})")
    network_module = importlib.import_module(args.network_module)
    network, weights_sd = network_module.create_network_from_weights(
        args.multiplier, str(adapter), None, None, unet, for_inference=True
    )

    logger.info("merging adapter into DiT")
    network.merge_to(None, unet, weights_sd, dtype, args.device)

    out = args.out or adapter.with_name(adapter.stem + "_merged.safetensors")
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "ss_merged_from": adapter.name,
        "ss_merge_multiplier": str(args.multiplier),
        "ss_base_dit": args.dit.name,
    }
    logger.info(f"saving merged DiT: {out}")
    anima_weights.save_anima_model(str(out), unet.state_dict(), metadata, dtype=dtype)
    logger.info("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
