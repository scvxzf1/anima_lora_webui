from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "experiments" / "int8_base_linear_audit.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("int8_base_linear_audit", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_candidate_filter_defaults_to_mlp_and_skips_sensitive_paths() -> None:
    audit = _load_module()

    assert audit.classify_candidate_key("net.blocks.0.mlp.layer1.weight").family == "mlp"
    assert audit.classify_candidate_key("blocks.1.mlp.layer2.weight").family == "mlp"
    assert audit.classify_candidate_key("net.blocks.0.self_attn.qkv_proj.weight") is None
    assert (
        audit.classify_candidate_key("net.blocks.0.self_attn.qkv_proj.weight", scope="attention").family
        == "attention"
    )
    assert (
        audit.classify_candidate_key("net.blocks.0.self_attn.qkv_proj.weight", scope="self_attn_qkv").family
        == "attention"
    )
    assert audit.classify_candidate_key(
        "net.blocks.0.self_attn.output_proj.weight",
        scope="self_attn_qkv",
    ) is None
    assert (
        audit.classify_candidate_key("net.blocks.0.self_attn.output_proj.weight", scope="self_attn_out").family
        == "attention"
    )
    assert (
        audit.classify_candidate_key("net.blocks.0.cross_attn.kv_proj.weight", scope="cross_attn_kv").family
        == "attention"
    )
    assert audit.classify_candidate_key(
        "net.blocks.0.cross_attn.q_proj.weight",
        scope="cross_attn_kv",
    ) is None

    skipped = [
        "net.blocks.0.adaln_fused_down.1.weight",
        "net.blocks.0.adaln_up_mlp.weight",
        "net.final_layer.linear.weight",
        "net.t_embedder.1.linear_1.weight",
        "net.pooled_text_proj.0.weight",
        "net.blocks.0.layer_norm_mlp.weight",
        "net.blocks.0.mlp.layer1.bias",
        "net.blocks.0.mlp.layer1.lora_down.weight",
        "net.llm_adapter.layers.0.mlp.0.weight",
    ]
    for key in skipped:
        assert audit.classify_candidate_key(key, scope="all") is None


def test_per_channel_int8_quantization_keeps_shape_and_reports_error() -> None:
    audit = _load_module()
    weight = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0],
            [-0.25, 0.5, 0.75],
        ],
        dtype=torch.bfloat16,
    )
    candidate = audit.Candidate(
        key="net.blocks.0.mlp.layer1.weight",
        block_idx=0,
        module_name="mlp.layer1",
        family="mlp",
    )

    quantized, scale = audit.quantize_per_channel_int8(weight)
    dequantized = audit.dequantize_per_channel_int8(quantized, scale)
    stats = audit.audit_tensor(candidate, weight)

    assert quantized.shape == weight.shape
    assert scale.shape == (3,)
    assert dequantized.shape == weight.shape
    assert stats.zero_rows == 1
    assert stats.payload_bytes == weight.numel() + weight.shape[0] * 4
    assert 0.0 <= stats.relative_l2 < 0.01
    assert stats.cosine > 0.999


def test_cli_audits_safetensors_and_writes_summary(tmp_path: Path) -> None:
    model = tmp_path / "tiny.safetensors"
    save_file(
        {
            "net.blocks.0.mlp.layer1.weight": torch.randn(4, 3, dtype=torch.bfloat16),
            "net.blocks.0.mlp.layer2.weight": torch.randn(3, 4, dtype=torch.bfloat16),
            "net.blocks.0.self_attn.qkv_proj.weight": torch.randn(6, 3, dtype=torch.bfloat16),
            "net.blocks.0.adaln_up_mlp.weight": torch.randn(3, 2, dtype=torch.bfloat16),
            "net.final_layer.linear.weight": torch.randn(3, 3, dtype=torch.bfloat16),
        },
        str(model),
    )
    out_dir = tmp_path / "audit"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--model",
            str(model),
            "--out-dir",
            str(out_dir),
            "--scope",
            "all",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )

    summary_path = out_dir / "int8_base_linear_audit_summary.json"
    detail_path = out_dir / "int8_base_linear_audit.jsonl"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    details = [json.loads(line) for line in detail_path.read_text(encoding="utf-8").splitlines()]

    assert "relative L2" in result.stdout
    assert summary["tensor_count"] == 3
    assert summary["families"]["mlp"]["tensor_count"] == 2
    assert summary["families"]["attention"]["tensor_count"] == 1
    assert all("adaln" not in item["key"] for item in details)
    assert all("final_layer" not in item["key"] for item in details)


def test_cli_audits_projection_subset_scope(tmp_path: Path) -> None:
    model = tmp_path / "tiny.safetensors"
    save_file(
        {
            "net.blocks.0.mlp.layer1.weight": torch.randn(4, 3, dtype=torch.bfloat16),
            "net.blocks.0.self_attn.qkv_proj.weight": torch.randn(6, 3, dtype=torch.bfloat16),
            "net.blocks.0.self_attn.output_proj.weight": torch.randn(3, 3, dtype=torch.bfloat16),
            "net.blocks.0.cross_attn.kv_proj.weight": torch.randn(6, 3, dtype=torch.bfloat16),
            "net.blocks.0.cross_attn.output_proj.weight": torch.randn(3, 3, dtype=torch.bfloat16),
        },
        str(model),
    )
    out_dir = tmp_path / "audit"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--model",
            str(model),
            "--out-dir",
            str(out_dir),
            "--scope",
            "mlp,self_attn_out,cross_attn_kv",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )

    summary = json.loads((out_dir / "int8_base_linear_audit_summary.json").read_text(encoding="utf-8"))
    details = [
        json.loads(line)
        for line in (out_dir / "int8_base_linear_audit.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert summary["tensor_count"] == 3
    assert {item["module_name"] for item in details} == {
        "mlp.layer1",
        "self_attn.output_proj",
        "cross_attn.kv_proj",
    }
