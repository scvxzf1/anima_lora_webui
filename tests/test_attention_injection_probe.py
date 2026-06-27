from __future__ import annotations

import argparse

import torch

from bench.attention_injection import probe


def test_tensor_stats_reports_finite_values() -> None:
    x = torch.tensor([[1.0, -2.0], [float("nan"), 4.0]])

    stats = probe._tensor_stats(x)

    assert stats["shape"] == [2, 2]
    assert stats["numel"] == 4
    assert stats["finite"] == 3
    assert stats["finite_fraction"] == 0.75
    assert stats["max_abs"] == 4.0
    assert stats["rms"] is not None


def test_attention_stats_samples_large_sequences() -> None:
    q = torch.randn(1, 8, 2, 4)
    k = torch.randn(1, 9, 2, 4)
    v = torch.randn(1, 9, 2, 4)

    stats = probe._attention_stats(q, k, v, max_tokens=3)

    assert stats["q_tokens"] == 8
    assert stats["k_tokens"] == 9
    assert stats["sampled_q_tokens"] <= 3
    assert stats["sampled_k_tokens"] <= 3
    assert stats["sampled"] is True
    assert stats["logits"]["present"] is True
    assert stats["entropy"]["mean"] is not None
    assert stats["max_prob"]["max_abs"] <= 1.0


def test_summarize_events_groups_attention_and_adapter() -> None:
    attention_events = [
        {
            "arm": "base",
            "attn_kind": "self",
            "q": {"rms": 1.0},
            "k": {"rms": 1.0},
            "v": {"rms": 2.0},
            "attention": {
                "logits": {"p95_abs": 3.0, "max_abs": 4.0},
                "entropy": {"mean": 1.5},
                "max_prob": {"max_abs": 0.4},
            },
            "attn_out": {"rms": 2.0},
            "projected": {"rms": 2.0},
            "ratios": {"projected_to_input_rms": 2.0, "attn_out_to_v_rms": 1.0},
        },
        {
            "arm": "adapted",
            "attn_kind": "self",
            "q": {"rms": 1.0},
            "k": {"rms": 1.0},
            "v": {"rms": 2.0},
            "attention": {
                "logits": {"p95_abs": 6.0, "max_abs": 8.0},
                "entropy": {"mean": 1.2},
                "max_prob": {"max_abs": 0.8},
            },
            "attn_out": {"rms": 5.0},
            "projected": {"rms": 6.0},
            "ratios": {"projected_to_input_rms": 6.0, "attn_out_to_v_rms": 2.5},
        },
    ]
    adapter_events = [
        {
            "arm": "adapted",
            "base": {"rms": 4.0},
            "delta": {"rms": 1.0},
            "output": {"rms": 5.0},
            "ratios": {"delta_to_base_rms": 0.25, "output_to_base_rms": 1.25},
        }
    ]

    summary = probe._summarize_events(attention_events, adapter_events)

    assert summary["attention_event_count"] == 2
    assert summary["adapter_event_count"] == 1
    assert summary["attention"]["base/self"]["event_count"] == 1
    assert (
        summary["comparisons"]["self_projected_to_input_rms_p95_adapted_over_base"]
        == 3.0
    )
    assert summary["comparisons"]["adapter_delta_to_base_p95"] == 0.25


def test_parser_defaults_to_torch_attention() -> None:
    args = probe.parse_args(["--dit", "model.safetensors"])

    assert isinstance(args, argparse.Namespace)
    assert args.attn_mode == "torch"
    assert args.num_samples == 2
    assert args.max_logit_tokens == 512
