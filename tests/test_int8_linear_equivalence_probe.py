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


def test_cache_pair_discovery_strips_latent_resolution_suffix(tmp_path: Path) -> None:
    probe = _load_module()
    latent = tmp_path / "sample_a_0896x1200_anima.npz"
    text = tmp_path / "sample_a_anima_te.safetensors"
    orphan = tmp_path / "orphan_0896x1200_anima.npz"
    latent.touch()
    text.touch()
    orphan.touch()

    pairs = probe.discover_cached_batch_pairs(tmp_path)

    assert len(pairs) == 1
    assert pairs[0].latent_path == latent
    assert pairs[0].text_path == text
    assert pairs[0].base_stem == "sample_a"
    assert probe.select_cached_batch_pair(tmp_path, 0) == pairs[0]


def test_checkpoint_probe_cli_exposes_real_cache_options(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--model-kind",
            "checkpoint",
            "--dit-path",
            "models/diffusion_models/anima-preview3-base.safetensors",
            "--data-dir",
            "post_image_dataset/rokkotsu_goddess",
            "--device",
            "cpu",
            "--cache-index",
            "2",
            "--repeat-caches",
            "3",
            "--adapter-kind",
            "lora",
            "--lora-rank",
            "8",
            "--lora-alpha",
            "16",
            "--forward-only",
            "--no-capture-blocks",
            "--no-gradient-checkpointing",
        ],
    )
    probe = _load_module()

    args = probe.parse_args()

    assert args.model_kind == "checkpoint"
    assert args.batch_size is None
    assert args.device == "cpu"
    assert args.cache_index == 2
    assert args.repeat_caches == 3
    assert args.adapter_kind == "lora"
    assert args.lora_rank == 8
    assert args.lora_alpha == 16
    assert args.forward_only is True
    assert args.capture_blocks is False
    assert args.gradient_checkpointing is False


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


def test_probe_repeat_seeds_writes_aggregate_json(tmp_path: Path) -> None:
    out_path = tmp_path / "repeat-probe.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--scope",
            "mlp",
            "--repeat-seeds",
            "2",
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
    assert saved["repeat_seeds"] == 2
    assert len(saved["results"]) == 2
    assert saved["summary"]["gate_pass_all"] is True
    assert saved["summary"]["gate_pass_count"] == 2
    assert saved["summary"]["run_count"] == 2
    assert saved["summary"]["grad_norm_rel_delta"]["max"] < 0.01


def test_checkpoint_repeat_caches_summary_records_cache_indices(monkeypatch, capsys) -> None:
    probe = _load_module()
    calls: list[tuple[int, int, str]] = []

    def fake_run_checkpoint_probe(**kwargs):
        calls.append((kwargs["seed"], kwargs["cache_index"], kwargs["adapter_kind"]))
        return {
            "model_kind": "checkpoint",
            "scope": kwargs["scope"],
            "seed": kwargs["seed"],
            "cache_index": kwargs["cache_index"],
            "adapter_kind": kwargs["adapter_kind"],
            "replacement_count": 2,
            "payload_ratio_vs_bf16": 0.5,
            "gate_pass": True,
            "output_rel_l2": 0.001,
            "loss_rel_delta": 0.002,
            "grad_norm_rel_delta": 0.003,
        }

    monkeypatch.setattr(probe, "run_checkpoint_probe", fake_run_checkpoint_probe)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            "--model-kind",
            "checkpoint",
            "--scope",
            "mlp",
            "--seed",
            "4",
            "--repeat-seeds",
            "2",
            "--cache-index",
            "3",
            "--repeat-caches",
            "2",
            "--adapter-kind",
            "lora",
        ],
    )

    assert probe.main() == 0
    result = json.loads(capsys.readouterr().out)

    assert calls == [(4, 3, "lora"), (5, 3, "lora"), (4, 4, "lora"), (5, 4, "lora")]
    assert result["repeat_caches"] == 2
    assert result["adapter_kind"] == "lora"
    assert result["summary"]["cache_indices"] == [3, 4]
    assert result["summary"]["seed_values"] == [4, 5]
    assert result["summary"]["gate_pass_count"] == 4
