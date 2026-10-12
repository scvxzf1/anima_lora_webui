"""Analytic costs against dense BF16, not measured timings."""

from dataclasses import asdict, dataclass

from .config import NMConfig


@dataclass(frozen=True)
class LinearCost:
    tokens: int
    in_features: int
    out_features: int
    bf16_weight_bytes: int
    int4_value_bytes: int
    index_bytes: int
    fp32_scale_bytes: int
    payload_bytes: int
    bits_per_original_weight: float
    storage_ratio_vs_bf16: float
    bf16_forward_flops: int
    bf16_input_grad_flops: int
    sparse_forward_nonzero_flops: int
    sparse_input_grad_nonzero_flops: int
    staged_dense_decode_bytes: int
    ideal_two_orientation_sparse_cache_bytes: int
    ideal_payload_plus_cache_bytes: int


def linear_cost(tokens: int, in_features: int, out_features: int, *, group_size=128):
    NMConfig(group_size=group_size).validate_shape(out_features, in_features)
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    count = in_features * out_features
    values, indices, scales = count // 4, count // 8, count // group_size * 4
    payload = values + indices + scales
    # Each staged BF16 orientation: 8 value bits + 1 index bit per weight.
    # Actual cuSPARSELt allocation can be larger due to padding/opaque layout.
    two_caches = count * 9 // 4
    flops = 2 * tokens * count
    return LinearCost(
        tokens, in_features, out_features, count * 2, values, indices, scales,
        payload, payload * 8 / count, count * 2 / payload,
        flops, flops, flops // 2, flops // 2, count * 2,
        two_caches, payload + two_caches,
    )


def comparison_report(tokens, in_features, out_features, *, group_size=128):
    return {
        "status": "analytic_only_not_a_benchmark",
        "baseline": "dense_bf16",
        "candidate": "2:4_uniform_int4_bf16",
        "cost": asdict(linear_cost(tokens, in_features, out_features, group_size=group_size)),
        "limitations": [
            "FLOP reduction is not wall-clock speedup",
            "staged cusparselt decodes and compresses BF16; it is not a fused W4 kernel",
            "cache bytes exclude allocator padding, workspace, bias, and retained original bases",
            "checkpoint recomputation, adapter, attention, and offload need separate accounting",
        ],
    }
