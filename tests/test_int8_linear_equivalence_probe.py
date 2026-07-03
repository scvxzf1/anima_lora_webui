from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "experiments" / "int8_linear_equivalence_probe.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("int8_linear_equivalence_probe", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_probe_compares_bf16_loss_grad_and_output_for_mlp_scope() -> None:
    probe = _load_module()

    result = probe.run_probe(scope="mlp")

    assert result["dtype"] == "bfloat16"
    assert result["gate_pass"] is True
    assert result["replacement_count"] == 4
    assert {item["family"] for item in result["replacements"]} == {"mlp"}
    assert result["payload_ratio_vs_bf16"] < 0.6
    assert result["output_rel_l2"] < 0.01
    assert result["loss_rel_delta"] < 0.01
    assert result["grad_norm_rel_delta"] < 0.01
    assert result["output_cosine"] > 0.999
    assert result["baseline_grad_norm"] > 0
    assert result["int8_grad_norm"] > 0


def test_anima_probe_compares_real_tiny_dit_block_outputs() -> None:
    probe = _load_module()

    result = probe.run_anima_probe(scope="mlp")

    assert result["model_kind"] == "anima"
    assert result["dtype"] == "bfloat16"
    assert result["gate_pass"] is True
    assert result["replacement_count"] == 4
    assert {item["family"] for item in result["replacements"]} == {"mlp"}
    assert result["payload_ratio_vs_bf16"] < 0.6
    assert result["output_rel_l2"] < 0.01
    assert result["block_output_rel_l2_max"] < 0.01
    assert set(result["block_output_deltas"]) == {"blocks.0", "blocks.1"}
    assert result["loss_rel_delta"] < 0.01
    assert result["grad_norm_rel_delta"] < 0.01
    assert result["baseline_grad_norm"] > 0
    assert result["int8_grad_norm"] > 0


def test_anima_probe_all_scope_includes_self_and_cross_attention() -> None:
    probe = _load_module()

    result = probe.run_anima_probe(scope="all")
    names = {item["name"] for item in result["replacements"]}

    assert result["model_kind"] == "anima"
    assert result["gate_pass"] is True
    assert result["replacement_count"] == 14
    assert {item["family"] for item in result["replacements"]} == {"attention", "mlp"}
    assert "blocks.0.self_attn.qkv_proj" in names
    assert "blocks.0.cross_attn.kv_proj" in names
    assert "blocks.0.mlp.layer1" in names
    assert result["output_rel_l2"] < 0.01
    assert result["block_output_rel_l2_max"] < 0.01
    assert result["loss_rel_delta"] < 0.01
    assert result["grad_norm_rel_delta"] < 0.01


def test_probe_all_scope_includes_attention_and_writes_json(tmp_path: Path) -> None:
    out_path = tmp_path / "probe.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--scope",
            "all",
            "--out",
            str(out_path),
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=60,
    )

    stdout = json.loads(completed.stdout)
    saved = json.loads(out_path.read_text(encoding="utf-8"))

    assert stdout == saved
    assert saved["gate_pass"] is True
    assert saved["replacement_count"] == 8
    assert {item["family"] for item in saved["replacements"]} == {"attention", "mlp"}
    assert saved["output_rel_l2"] < 0.01
    assert saved["loss_rel_delta"] < 0.01
    assert saved["grad_norm_rel_delta"] < 0.01
