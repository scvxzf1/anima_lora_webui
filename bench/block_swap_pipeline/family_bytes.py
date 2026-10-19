"""Cross-family block payload bytes, read from safetensors headers only.

The auto block-swap tuner promotes swapped blocks as long as VRAM headroom
allows. How much that buys depends on the per-block *transfer* payload, which is
numel * bytes-per-element of the transfer dtype (bf16 => 2), not the on-disk
dtype. This module computes that payload straight from a safetensors header, so
every registered family can be compared without loading multi-GB weights.

Reference point: Anima-base measures 138.4 MB/block in the profiler
(see docs/findings/block_swap_restore_link_limit_50hx_20261019.md), and this
module reproduces that number exactly.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

BLOCK_TOKENS = ("blocks", "layers", "transformer_blocks", "block")

DTYPE_BYTES = {
    "F64": 8, "I64": 8, "F32": 4, "I32": 4,
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1, "BOOL": 1,
}

TRANSFER_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}

FAMILY_DITS = {
    "anima": "/home/scv/nvme0n1p1/ComfyUI/models/diffusion_models/anima/anima-base-v1.0-sha.safetensors",
    "z_image": "/home/scv/nvme0n1p1/ComfyUI/models/diffusion_models/z-image/z_image_bf16.safetensors",
    "krea2_raw": "models/diffusion_models/krea2_raw_bf16.safetensors",
    # The ComfyUI qwen/qwenimage2512-* and qwenedit-2511-* checkpoints are a 60-block
    # fp8 architecture that library/models/qwen_image_2_1/weights.py rejects at
    # _validate_header; this is the real 32-block bf16 transformer.
    "qwen_image_2_1": "/home/scv/nvme0n1p1/ComfyUI/local_models/diffusion_models/Qwen-Image-2.1/qwen_image_2.1_bf16.safetensors",
}

ANIMA_MEASURED_BLOCK_MIB = 132.0


def read_header(path: str | Path) -> dict:
    """Safetensors header as {tensor_name: {dtype, shape, data_offsets}}."""
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError("not a safetensors file: " + str(path))
        length = struct.unpack("<Q", raw)[0]
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header


def _block_key(name: str) -> tuple[str, int] | None:
    """(prefix, index) for a tensor inside a block list, else None.

    The first block token wins and the prefix is kept so the caller can tell the
    transformer block list from a nested one. Anima stores both net.blocks.N
    (the DiT, 28 blocks) and net.llm_adapter.blocks.N (6 adapter blocks); only
    the former takes part in block swap.
    """
    parts = name.split(".")
    for pos, token in enumerate(parts):
        if token in BLOCK_TOKENS and pos + 1 < len(parts) and parts[pos + 1].isdigit():
            return ".".join(parts[: pos + 1]), int(parts[pos + 1])
    return None


def _dominant_prefix(header: dict) -> str | None:
    """Block-list prefix with the most distinct indices (ties: most tensors)."""
    seen: dict[str, set[int]] = {}
    counts: dict[str, int] = {}
    for name in header:
        found = _block_key(name)
        if found is None:
            continue
        prefix, index = found
        seen.setdefault(prefix, set()).add(index)
        counts[prefix] = counts.get(prefix, 0) + 1
    if not seen:
        return None
    return max(seen, key=lambda prefix: (len(seen[prefix]), counts[prefix]))


def _numel(shape) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def block_payload_bytes(path: str | Path, *, transfer_dtype: str = "bf16") -> dict:
    """Per-block transfer payload, plus the non-block remainder."""
    header = read_header(path)
    per_element = TRANSFER_BYTES[str(transfer_dtype).lower()]
    prefix = _dominant_prefix(header)
    blocks: dict[int, int] = {}
    native_blocks: dict[int, int] = {}
    non_block = 0
    for name, spec in header.items():
        numel = _numel(spec["shape"])
        native = numel * DTYPE_BYTES.get(str(spec["dtype"]).upper(), 2)
        found = _block_key(name)
        if found is None or found[0] != prefix:
            non_block += numel * per_element
            continue
        index = found[1]
        blocks[index] = blocks.get(index, 0) + numel * per_element
        native_blocks[index] = native_blocks.get(index, 0) + native
    ordered = [blocks[i] for i in sorted(blocks)]
    return {
        "path": str(path),
        "block_prefix": prefix,
        "transfer_dtype": str(transfer_dtype).lower(),
        "num_blocks": len(ordered),
        "per_block_bytes": ordered,
        "native_per_block_bytes": [native_blocks[i] for i in sorted(native_blocks)],
        "total_block_bytes": sum(ordered),
        "non_block_bytes": non_block,
        "uniform": len(set(ordered)) == 1,
    }


def promotion_plan(
    info: dict, *, free_mib: float, margin_mib: float = 400, swap: int | None = None
) -> dict:
    """Replicate plan_promotion arithmetic for a hypothetical free-VRAM reading."""
    block = info["per_block_bytes"][-1] if info["per_block_bytes"] else 0
    current = int(swap if swap is not None else info["num_blocks"] - 2)
    headroom = (free_mib - margin_mib) * 1024 * 1024
    if block <= 0 or headroom <= 0:
        return {"current_swap": current, "promotable": 0, "new_swap": current}
    promotable = min(int(headroom // block), max(0, current - 2))
    return {
        "current_swap": current,
        "promotable": promotable,
        "new_swap": current - promotable,
        "headroom_mib": headroom / 1024 / 1024,
        "block_mib": block / 1024 / 1024,
    }


def summarize(info: dict, *, free_mib: float | None = None, margin_mib: float = 400) -> dict:
    """Compact metrics for one family, optionally with a promotion decision."""
    block = info["per_block_bytes"][-1] if info["per_block_bytes"] else 0
    out = {
        "num_blocks": info["num_blocks"],
        "swap_upper_bound": max(0, info["num_blocks"] - 2),
        "block_mib": round(block / 1024 / 1024, 1),
        "total_block_gib": round(info["total_block_bytes"] / 1024**3, 2),
        "non_block_mib": round(info["non_block_bytes"] / 1024 / 1024, 1),
        "uniform_blocks": info["uniform"],
    }
    if free_mib is not None:
        out["plan"] = promotion_plan(info, free_mib=free_mib, margin_mib=margin_mib)
    return out


HEADER = f"{'family':<16}{'prefix':<20}{'blocks':>7}{'cap':>5}{'MiB/block':>11}{'GiB/blocks':>12}{'uniform':>9}"


def _format_row(family: str, info: dict, summary: dict) -> str:
    return (
        f"{family:<16}{str(info.get('block_prefix') or '-'):<20}"
        f"{summary['num_blocks']:>7}{summary['swap_upper_bound']:>5}"
        f"{summary['block_mib']:>11}{summary['total_block_gib']:>12}"
        f"{str(summary['uniform_blocks']):>9}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default=None, help="anima|z_image|krea2_raw|qwen_image_2_1")
    parser.add_argument("--dit", default=None, help="override with an explicit safetensors path")
    parser.add_argument("--all", action="store_true", help="every built-in family")
    parser.add_argument("--transfer-dtype", default="bf16", choices=sorted(TRANSFER_BYTES))
    parser.add_argument("--free-mib", type=float, default=None, help="run plan_promotion arithmetic")
    parser.add_argument("--margin-mib", type=float, default=400.0)
    parser.add_argument("--swap", type=int, default=None, help="current blocks_to_swap for the plan")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.all:
        targets = list(FAMILY_DITS.items())
    elif args.dit:
        targets = [(args.family or "custom", args.dit)]
    elif args.family:
        if args.family not in FAMILY_DITS:
            parser.error(f"unknown family {args.family!r}; known: {sorted(FAMILY_DITS)}")
        targets = [(args.family, FAMILY_DITS[args.family])]
    else:
        parser.error("pass --family, --dit or --all")

    rows = []
    if not args.json:
        print(HEADER)
    for family, path in targets:
        info = block_payload_bytes(path, transfer_dtype=args.transfer_dtype)
        summary = summarize(info, free_mib=args.free_mib, margin_mib=args.margin_mib)
        if args.swap is not None:
            summary["plan"] = promotion_plan(
                info, free_mib=args.free_mib or 0.0, margin_mib=args.margin_mib, swap=args.swap
            )
        rows.append({"family": family, "path": path, **summary})
        if args.json:
            continue
        print(_format_row(family, info, summary))
        plan = summary.get("plan")
        if plan:
            print(
                f"{'':16}free={args.free_mib:.0f} MiB margin={args.margin_mib:.0f} MiB -> "
                f"promotable={plan['promotable']} new_swap={plan['new_swap']}"
            )
        if family == "anima" and args.free_mib is None:
            measured = ANIMA_MEASURED_BLOCK_MIB
            flag = "ok" if abs(summary["block_mib"] - measured) < 1.0 else "MISMATCH"
            print(f"{'':16}Anima check vs profiler {measured} MiB/block: {flag}")
    if args.json:
        print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
