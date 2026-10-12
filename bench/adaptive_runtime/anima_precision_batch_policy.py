"""Bounded, nonmonotonic MLP2 candidate generation for the Anima pilot batch."""

from itertools import combinations
import math

from safetensors import safe_open


QUARTERS = (
    tuple(range(0, 7)),
    tuple(range(7, 14)),
    tuple(range(14, 21)),
    tuple(range(21, 28)),
)
MAX_CANDIDATES = 12


def candidate(blocks, stage):
    blocks = tuple(sorted(set(blocks)))
    if not blocks or any(type(i) is not int or not 0 <= i < 28 for i in blocks):
        raise ValueError("Nonempty Anima block subset in 0..27 required")
    return {
        "name": "mlp2-" + "-".join(map(str, blocks)),
        "blocks": list(blocks),
        "stage": stage,
        "fp16_patterns": [f"blocks.{i}.mlp.layer2" for i in blocks],
    }


def score(item):
    metrics = item["comparison"]["cases"][0]
    values = [
        metrics[k].get("relative_l2")
        for k in ("prediction", "input_gradient", "adapter_gradient")
    ]
    return max(
        v / limit if isinstance(v, (int, float)) and math.isfinite(v) else math.inf
        for v, limit in zip(values, (0.01, 0.02, 0.02), strict=True)
    )


def next_stage(history):
    """Actual measurements decide proposals; every proposed combination is rechecked."""
    if not history:
        return [candidate(blocks, "initial") for blocks in QUARTERS]
    seen = {tuple(item["candidate"]["blocks"]) for item in history}
    if len(seen) >= MAX_CANDIDATES:
        return []
    ranked = sorted(history, key=score)
    passed = sorted(
        (
            item
            for item in history
            if item["comparison"]["within_experimental_tolerances"]
        ),
        key=lambda item: (-len(item["candidate"]["blocks"]), score(item)),
    )
    proposals = []
    stage = "refine" if len(history) < 8 else "confirm"
    if passed:
        pools = passed[:3] if len(passed) > 1 else [passed[0], *ranked[:3]]
        for a, b in combinations(pools, 2):
            proposals.append(
                set(a["candidate"]["blocks"]) | set(b["candidate"]["blocks"])
            )
        proposals.append(
            set().union(*(set(item["candidate"]["blocks"]) for item in pools))
        )
    for item in ranked:
        blocks = item["candidate"]["blocks"]
        if len(blocks) > 1:
            middle = len(blocks) // 2
            proposals.extend((blocks[:middle], blocks[middle:]))
    result = []
    for blocks in proposals:
        blocks = tuple(sorted(blocks))
        if blocks and blocks not in seen:
            result.append(candidate(blocks, stage))
            seen.add(blocks)
        if len(result) == min(4, MAX_CANDIDATES - len(history)):
            break
    return result


def weight_inventory(path):
    """Header-only storage estimate; not a GPU memory or FLOP measurement."""
    with safe_open(str(path), framework="pt", device="cpu") as source:
        sizes = {
            name.removeprefix("net."): math.prod(source.get_slice(name).get_shape())
            for name in source.keys()
        }
        if any(source.get_slice(name).get_dtype() != "BF16" for name in source.keys()):
            raise ValueError("Exclusively BF16 source checkpoint required")
    required = {f"blocks.{i}.mlp.layer2.weight" for i in range(28)}
    if not required <= set(sizes):
        raise ValueError("Expected all 28 Anima MLP2 weights")
    return sizes


def storage_estimate(blocks, inventory):
    total = sum(inventory.values())
    selected = sum(inventory[f"blocks.{i}.mlp.layer2.weight"] for i in blocks)
    return {
        "source_elements": total,
        "fp16_target_weight_elements": selected,
        "fp32_source_value_payload_bytes": total * 4,
        "mixed_source_value_payload_bytes": total * 4 - selected * 2,
        "weight_payload_reduction_fraction": selected / (2 * total),
        "scope": "source_tensor_representation_estimate_not_peak_memory_or_flops",
    }
