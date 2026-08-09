"""Unit tests for ConvRot fusion microbench helpers (no full GPU sweep)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "experiments" / "convrot_fusion_microbench.py"


def _load_mod():
    spec = importlib.util.spec_from_file_location("convrot_fusion_microbench", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_shapes_default_has_anima_mlp():
    mod = _load_mod()
    shapes = mod.parse_shapes("default")
    names = {n for n, *_ in shapes}
    assert "mlp_l1_tokens512" in names
    assert "mlp_l2_tokens512" in names
    # layer1 weight is [8192, 2048] → N=8192 K=2048
    l1 = next(s for s in shapes if s[0] == "mlp_l1_tokens512")
    assert l1[1:] == (512, 2048, 8192)


def test_parse_shapes_custom():
    mod = _load_mod()
    shapes = mod.parse_shapes("tiny:8,16,32;other:4,8,8")
    assert shapes == [("tiny", 8, 16, 32), ("other", 4, 8, 8)]


def test_recommend_gates():
    mod = _load_mod()
    # High scale tax alone is NOT enough if int_mm body still >> bf16.
    slow_body = {
        "meta": {"m": 4032, "k": 2048, "n": 8192},
        "upper_bounds": {
            "w8a16_dequant_tax_pct_of_dequant_path": 17.0,
            "w8a8_post_scale_tax_pct": 24.0,
            "w8a16_vs_bf16_ratio": 1.21,
            "w8a8_vs_bf16_ratio": 1.77,
            "w8a8_int_mm_only_vs_bf16_ratio": 1.34,
            "w8a16_predequant_vs_bf16_ratio": 1.00,
            "bwd_chunked_vs_full_ratio": 1.2,
            "bwd_peak_chunk_vs_full": 0.5,
        },
    }
    d = mod._recommend([slow_body])
    assert d["recommend_w8a8_epilogue_fusion_impl"] is False
    assert d["recommend_w8a16_kloop_fusion_impl"] is False
    assert d["recommend_p2_triton_now"] is False
    assert d["recommend_bwd_chunk_for_peak"] is True
    assert d["recommend_bwd_chunk_for_speed"] is False

    # Epilogue only if tax high AND free body near bf16.
    fast_body = {
        "meta": {"m": 4032, "k": 2048, "n": 8192},
        "upper_bounds": {
            "w8a16_dequant_tax_pct_of_dequant_path": 30.0,
            "w8a8_post_scale_tax_pct": 40.0,
            "w8a16_vs_bf16_ratio": 1.3,
            "w8a8_vs_bf16_ratio": 1.1,
            "w8a8_int_mm_only_vs_bf16_ratio": 1.05,
            "w8a16_predequant_vs_bf16_ratio": 1.00,
            "bwd_chunked_vs_full_ratio": 0.9,
            "bwd_peak_chunk_vs_full": 0.5,
        },
    }
    d2 = mod._recommend([fast_body])
    assert d2["recommend_w8a8_epilogue_fusion_impl"] is True
    assert d2["recommend_w8a16_cheap_dequant_elim"] is True
    assert d2["recommend_w8a16_kloop_fusion_impl"] is False
    assert d2["recommend_bwd_chunk_for_speed"] is True
