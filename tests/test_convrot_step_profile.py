"""Unit tests for ConvRot step-profile bucketing helpers."""

from __future__ import annotations

from scripts.experiments.convrot_step_profile_probe import (
    _bucket_for_event,
    _decision_from_results,
)


def test_bucket_for_event_markers() -> None:
    assert _bucket_for_event("convrot::rht") == "convrot_rht"
    assert _bucket_for_event("convrot::dequant") == "convrot_dequant"
    assert _bucket_for_event("convrot::act_quant") == "convrot_act_quant"
    assert _bucket_for_event("convrot::gemm_int8") == "convrot_gemm"
    assert _bucket_for_event("aten::mm") == "gemm_generic"
    assert _bucket_for_event("aten::scaled_dot_product_attention") == "attention"
    assert _bucket_for_event("Optimizer.step#AdamW.step") == "optimizer"


def test_decision_triton_candidate_when_convrot_tax_high() -> None:
    results = [
        {
            "label": "w8a16_free",
            "mode": "w8a16",
            "sec_per_step_no_profiler": 3.0,
            "profile": {
                "buckets_pct": {
                    "convrot_rht": 30.0,
                    "convrot_dequant": 15.0,
                    "convrot_gemm": 10.0,
                    "attention": 20.0,
                    "other": 25.0,
                }
            },
        }
    ]
    d = _decision_from_results(results)
    assert d["branch"] == "P2-K_triton_candidate"
    assert d["convrot_tax_pct"] >= 50.0


def test_decision_stop_when_model_ops_dominate() -> None:
    results = [
        {
            "label": "w8a8_auto",
            "mode": "w8a8",
            "profile": {
                "buckets_pct": {
                    "attention": 40.0,
                    "other": 20.0,
                    "norm_act": 5.0,
                    "convrot_rht": 5.0,
                    "gemm_generic": 30.0,
                }
            },
        }
    ]
    d = _decision_from_results(results)
    assert d["branch"] == "stop_kernel_chase_do_prequant_or_product"


def test_decision_w8a16_fp32_dequant_linear() -> None:
    results = [
        {
            "label": "bf16",
            "mode": "bf16",
            "sec_per_step_no_profiler": 1.7,
            "profile": {"buckets_pct": {"gemm_generic": 50.0}},
        },
        {
            "label": "w8a16_free",
            "mode": "w8a16",
            "sec_per_step_no_profiler": 3.0,
            "profile": {
                "buckets_pct": {
                    "convrot_gemm": 12.0,
                    "convrot_rht": 1.5,
                    "gemm_generic": 52.0,
                    "other": 20.0,
                    "attention": 5.0,
                }
            },
        },
    ]
    d = _decision_from_results(results)
    assert d["branch"] == "fix_w8a16_keep_bf16_compute"
