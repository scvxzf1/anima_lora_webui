from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "experiments" / "int8_blockswap_equivalence_probe.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "int8_blockswap_equivalence_probe",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_blockswap_probe_compares_bf16_and_int8_cpu_masters() -> None:
    probe = _load_module()

    result = probe.run_probe()

    assert result["model_kind"] == "blockswap_toy"
    assert result["dtype"] == "bfloat16"
    assert result["device"] == "cpu"
    assert result["baseline_transfer_dtype"] == "bf16"
    assert result["candidate_transfer_dtype"] == "int8"
    assert result["candidate_int8_restore_mode"] == "copy"
    assert result["candidate_int8_restore_chunk_rows"] == 0
    assert result["candidate_int8_scope"] == "all"
    assert result["repeat_steps"] == 1
    assert result["gate_pass"] is True
    assert result["int8_quantized_tensors"] == 28
    assert result["int8_master_bytes"] > 0
    assert result["int8_master_ratio_vs_bf16"] < 0.6
    assert result["output_rel_l2"] < 0.01
    assert result["block_output_rel_l2_max"] < 0.01
    assert result["loss_rel_delta"] < 0.01
    assert result["grad_norm_rel_delta"] < 0.01
    assert result["baseline_grad_norm"] > 0
    assert result["int8_grad_norm"] > 0
    assert result["baseline_profile"] is None
    assert result["int8_profile"] is None
    assert result["profile_ratios"] is None
    assert result["baseline_memory"] is None
    assert result["int8_memory"] is None
    assert result["memory_ratios"] is None
    assert set(result["block_output_deltas"]) == {
        "blocks.0",
        "blocks.1",
        "blocks.2",
        "blocks.3",
    }


def test_blockswap_probe_cli_writes_json(tmp_path: Path) -> None:
    out_path = tmp_path / "blockswap-probe.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
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
    assert saved["candidate_transfer_dtype"] == "int8"
    assert saved["candidate_int8_restore_mode"] == "copy"
    assert saved["candidate_int8_scope"] == "all"
    assert saved["int8_master_ratio_vs_bf16"] < 0.6


def test_blockswap_probe_cli_accepts_chunked_reuse_storage(tmp_path: Path) -> None:
    out_path = tmp_path / "blockswap-probe-chunked.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--int8-restore-mode",
            "reuse_storage",
            "--int8-restore-chunk-rows",
            "4",
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
    assert saved["candidate_int8_restore_mode"] == "reuse_storage"
    assert saved["candidate_int8_restore_chunk_rows"] == 4


def test_blockswap_probe_cli_accepts_int8_scope(tmp_path: Path) -> None:
    out_path = tmp_path / "blockswap-probe-mlp.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--int8-scope",
            "mlp",
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
    assert saved["candidate_int8_scope"] == "mlp"
    assert saved["int8_quantized_tensors"] == 8


def test_blockswap_probe_profile_dir_writes_bf16_and_int8_jsonl(tmp_path: Path) -> None:
    probe = _load_module()

    result = probe.run_probe(profile_dir=tmp_path, repeat_steps=2)

    bf16_profile = tmp_path / "bf16_block_swap_profile.jsonl"
    int8_profile = tmp_path / "int8_block_swap_profile.jsonl"
    assert bf16_profile.exists()
    assert int8_profile.exists()
    assert result["baseline_profile"]["path"] == str(bf16_profile)
    assert result["int8_profile"]["path"] == str(int8_profile)
    assert result["baseline_profile"]["config"]["transfer_dtype"] == "bf16"
    assert result["int8_profile"]["config"]["transfer_dtype"] == "int8"
    assert result["int8_profile"]["config"]["int8_restore_mode"] == "copy"
    assert result["int8_profile"]["config"]["int8_restore_chunk_rows"] == 0
    assert result["int8_profile"]["config"]["int8_scope"] == "all"
    assert result["repeat_steps"] == 2
    assert result["baseline_profile"]["wait_event_count"] >= 8
    assert result["int8_profile"]["wait_event_count"] >= 8
    assert result["int8_profile"]["config"]["int8_quantized_tensors"] == 28
    assert result["profile_ratios"] is not None
    assert result["baseline_memory"] is None
    assert result["int8_memory"] is None
    assert result["memory_ratios"] is None
    assert set(result["profile_ratios"]) == {
        "h2d_ms_mean",
        "h2d_ms_p95",
        "h2d_ms_max",
        "wait_ms_mean",
        "wait_ms_p95",
        "wait_ms_max",
    }


def test_blockswap_probe_scope_subset_is_recorded_in_profile(tmp_path: Path) -> None:
    probe = _load_module()

    result = probe.run_probe(profile_dir=tmp_path, int8_scope="mlp")

    assert result["gate_pass"] is True
    assert result["candidate_int8_scope"] == "mlp"
    assert result["int8_quantized_tensors"] == 8
    assert result["int8_profile"]["config"]["int8_scope"] == "mlp"
    assert result["int8_profile"]["config"]["int8_quantized_tensors"] == 8


def test_blockswap_probe_accepts_direct_bind_int8_restore_mode() -> None:
    probe = _load_module()

    result = probe.run_probe(int8_restore_mode="direct_bind")

    assert result["gate_pass"] is True
    assert result["candidate_int8_restore_mode"] == "direct_bind"
    assert result["output_rel_l2"] < 0.01


def test_blockswap_probe_accepts_reuse_storage_int8_restore_mode() -> None:
    probe = _load_module()

    result = probe.run_probe(int8_restore_mode="reuse_storage")

    assert result["gate_pass"] is True
    assert result["candidate_int8_restore_mode"] == "reuse_storage"
    assert result["output_rel_l2"] < 0.01


def test_blockswap_probe_accepts_chunked_reuse_storage() -> None:
    probe = _load_module()

    result = probe.run_probe(
        int8_restore_mode="reuse_storage",
        int8_restore_chunk_rows=4,
    )

    assert result["gate_pass"] is True
    assert result["candidate_int8_restore_mode"] == "reuse_storage"
    assert result["candidate_int8_restore_chunk_rows"] == 4
    assert result["output_rel_l2"] < 0.01
