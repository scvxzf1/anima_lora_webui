"""CPU-only contracts for local K masks, endpoint selection and source rejection."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bench.adaptive_runtime import anima_k_local as local
from bench.adaptive_runtime import diagnose_anima_k_local as driver
from networks.attention_dispatch import AttentionParams, dispatch_attention


def test_case_masks_stable_and_combination_not_additive():
    head_rank = [13, *[x for x in range(16) if x != 13]]
    token_rank = [3343, *[x for x in range(4200) if x != 3343]]
    masks = local.masks(head_rank, token_rank)
    assert set(masks) == set(local.CASES)
    assert masks["baseline"].sum() == 0
    assert masks["full"].sum() == 4200 * 16
    assert masks["head13"].sum() == 4200
    assert masks["token3343"].sum() == 16
    assert masks["activation_top42"].sum() == 42 * 16
    assert masks["low42"].sum() == 42 * 16
    assert masks["low1"].sum() == 16
    assert masks["low_head"].sum() == 4200
    assert torch.where(masks["low42"].any(1))[0].tolist() == sorted(token_rank[-42:])
    assert torch.where(masks["low1"].any(1))[0].tolist() == [token_rank[-1]]
    assert torch.where(masks["low_head"].any(0))[0].tolist() == [head_rank[-1]]
    assert masks["head13_token3343"].sum() == 1
    assert masks["head13_token3343"][3343, 13]
    assert not masks["head13_token3343"][3343, 12]
    with pytest.raises(ValueError, match="ranks"):
        local.masks(head_rank, token_rank[:-1])


def test_real_dispatch_fixed_q_v_endpoints_and_response():
    torch.manual_seed(7)
    inputs = {"reference": tuple(torch.randn(1, 5, 2, 4) for _ in range(3))}
    inputs["candidate"] = tuple(x + 0.1 for x in inputs["reference"])
    mask = torch.zeros(5, 2, dtype=torch.bool)
    mask[3, 1] = True
    params = AttentionParams.create_attention_params("torch")
    du = torch.randn(1, 5, 8)
    outputs = {}
    for side in ("R", "C"):
        for case in ("baseline", "full", "head13"):
            selected = local.select_inputs(case, side, inputs, mask)
            assert selected[0] is inputs["reference"][0]
            assert selected[2] is inputs["reference" if side == "R" else "candidate"][2]
            if case == "baseline":
                assert selected[1] is inputs["reference"][1]
            elif case == "full":
                assert selected[1] is inputs["candidate"][1]
            selected = tuple(x.detach().requires_grad_(True) for x in selected)
            u = dispatch_attention(
                list(selected), attn_params=params.for_attention_kind(is_selfattn=True)
            )
            outputs[case, side] = torch.cat(
                [x.reshape(1, 5, -1) for x in torch.autograd.grad(u, selected, du)],
                dim=-1,
            )
        assert not torch.equal(outputs["baseline", side], outputs["full", side])
    assert not torch.equal(outputs["full", "R"], outputs["full", "C"])


def test_same_total_wrong_axis_drift_rejected():
    drift = torch.arange(1, 13, dtype=torch.float64).reshape(4, 3)
    recorded = local.activation.distribution(drift.numpy(), np.ones((4, 3)))
    local.validate_drift_axes(drift, recorded)
    with pytest.raises(ValueError, match="distribution"):
        local.validate_drift_axes(drift.flip(0), recorded)
    bad = {**recorded, "token_rank": list(reversed(recorded["token_rank"]))}
    with pytest.raises(ValueError, match="rank"):
        local.validate_drift_axes(drift, bad)


def test_input_hashing_preserves_unfused_shapes():
    values = tuple(
        torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4) + i for i in range(3)
    )
    du = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    report = local.input_hashes({"reference": values, "candidate": values}, du)
    assert report["reference_k"] == local.tensor_sha(values[1])
    assert report["candidate_v"] == local.tensor_sha(values[2])
    assert report["du"] == local.tensor_sha(du)
    assert len(report) == 7


def test_mask_energy_and_absolute_response():
    drift = torch.arange(1, 11, dtype=torch.float64).reshape(5, 2)
    mask = torch.zeros(5, 2, dtype=torch.bool)
    mask[3, 1] = True
    record = local.mask_record(mask, drift)
    assert record["head_token_pairs"] == 1
    assert record["token_indices"] == [3]
    assert record["head_indices"] == [1]
    assert record["actual_k_drift_square_share"] == pytest.approx(8 / 55)
    assert len(record["mask_sha256"]) == 64
    baseline = torch.ones(1, 1, 1, 1, 6144)
    full = baseline + 2
    middle = baseline + 1
    result = local.response(middle, baseline, full)
    assert result["fused"]["distance_to_baseline_l2"] == pytest.approx(6144**0.5)
    assert result["fused"]["distance_to_full_l2"] == pytest.approx(6144**0.5)
    assert result["fused"]["full_response_l2"] == pytest.approx(2 * 6144**0.5)
    assert set(result) == {"q", "k", "v", "fused"}


def test_activation_source_rejects_wrong_rank_and_sha(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    chain = {}
    env = {}
    for key, field in (
        ("matrix", "matrix_run"),
        ("qknorm", "qknorm_run"),
        ("core", "core_run"),
        ("upstream", "source_run"),
        ("stage", "stage_run"),
    ):
        path = tmp_path / key
        path.mkdir()
        (path / "summary.json").write_text("{}")
        env[field] = str(path)
        chain[key] = local.sha256_file(path / "summary.json")
    (run / "environment.json").write_text(json.dumps(env))
    distribution = tmp_path / "distribution"
    distribution.mkdir()
    activation_run = (
        tmp_path / "anima-adaptive-precision-90hx-20261001-k-activation-distribution-r2"
    )
    activation_run.mkdir()
    weights = tmp_path / "weights"
    weights.write_bytes(b"weights")
    monkeypatch.setattr(
        local.activation,
        "validate_distribution",
        lambda *_: ({}, {}, {"distribution.json": "d" * 64}),
    )
    shares_h = np.zeros(16)
    shares_h[13] = 1
    shares_t = np.zeros(4200)
    shares_t[3343] = 1
    report = {
        "status": "analyzed",
        "production_ready": False,
        "scope": local.activation.SCOPE,
        "code_sha256": local.sha256_file(local.Path(local.activation.__file__)),
        "source_hashes": {"distribution.json": "d" * 64},
        "source_code_sha256": {
            **local.interpolation.code_hashes(),
            "qknorm_math_sha256": local.sha256_file(
                local.Path(local.interpolation.matrix.norm.norm_math.__file__)
            ),
        },
        "chain_summary_sha256": chain,
        "weights_sha256": local.sha256_file(weights),
        "shape": [4200, 16, 128],
        "activation": {
            "error_square": 1.0,
            "head_rank": np.argsort(-shares_h, kind="stable").tolist(),
            "token_rank": np.argsort(-shares_t, kind="stable").tolist(),
            "head_error_share": shares_h.tolist(),
            "token_error_share": shares_t.tolist(),
        },
    }
    target = activation_run / "activation-distribution.json"
    target.write_text(json.dumps(report))
    assert (
        local.validate_activation(run, distribution, activation_run, weights)[0]
        == report
    )
    report["activation"]["token_rank"][0] = 9
    target.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="ranks/energy"):
        local.validate_activation(run, distribution, activation_run, weights)
    report["activation"]["token_rank"] = np.argsort(-shares_t, kind="stable").tolist()
    report["weights_sha256"] = "0" * 64
    target.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="source binding"):
        local.validate_activation(run, distribution, activation_run, weights)


def test_driver_rejects_short_timeout_without_creating_output(tmp_path):
    args = SimpleNamespace(
        output_dir=tmp_path / "new",
        weights=tmp_path / "missing",
        analysis_timeout=300,
        gpu_uuid="GPU-test",
    )
    for key in driver.SOURCE_KEYS:
        setattr(args, key, tmp_path)
    with pytest.raises(ValueError, match="timeout"):
        driver.run(args)
    assert not args.output_dir.exists()


def test_driver_rejects_existing_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    weights = tmp_path / "weights"
    weights.write_bytes(b"weights")
    args = SimpleNamespace(
        output_dir=output, weights=weights, analysis_timeout=1200, gpu_uuid="GPU-test"
    )
    for key in driver.SOURCE_KEYS:
        setattr(args, key, tmp_path)
    with pytest.raises(ValueError, match="Fresh disjoint output"):
        driver.run(args)
    assert list(output.iterdir()) == []


def test_missing_source_file_does_not_create_output(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    weights = tmp_path / "weights"
    weights.write_bytes(b"weights")
    args = SimpleNamespace(
        output_dir=tmp_path / "new",
        weights=weights,
        analysis_timeout=480,
        gpu_uuid="GPU-test",
    )
    for key in driver.SOURCE_KEYS:
        setattr(args, key, source)
    with pytest.raises(FileNotFoundError):
        driver.run(args, gpu_check=lambda _: {})
    assert not args.output_dir.exists()
