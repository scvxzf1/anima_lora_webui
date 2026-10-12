"""Bench-only local post-RoPE K substitutions at fixed Q, du and V backgrounds."""

import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from bench.adaptive_runtime import anima_k_activation_distribution as activation
from bench.adaptive_runtime import anima_k_interpolation as interpolation
from bench.adaptive_runtime.anima_inputs import sha256_file


SCOPE = "block17_local_postrope_k_fixed_reference_q_du_not_production"
CASES = (
    "baseline",
    "full",
    "head13",
    "token3343",
    "activation_top42",
    "low42",
    "low1",
    "low_head",
    "head13_token3343",
)
SHAPE = (1, 4200, 16, 128)


def code_hashes():
    return {
        "analysis_sha256": sha256_file(Path(__file__)),
        **{
            f"interpolation_{key}": value
            for key, value in interpolation.code_hashes().items()
        },
    }


def validate_activation(run, distribution_run, activation_run, weights):
    _, _, source_hashes = activation.validate_distribution(run, distribution_run)
    path = Path(activation_run) / "activation-distribution.json"
    report = json.loads(path.read_text())
    environment = json.loads((Path(run) / "environment.json").read_text())
    chain = {
        key: sha256_file(Path(environment[field]) / "summary.json")
        for key, field in (
            ("matrix", "matrix_run"),
            ("qknorm", "qknorm_run"),
            ("core", "core_run"),
            ("upstream", "source_run"),
            ("stage", "stage_run"),
        )
    }
    if (
        Path(activation_run).name
        != "anima-adaptive-precision-90hx-20261001-k-activation-distribution-r2"
        or report.get("status") != "analyzed"
        or report.get("production_ready") is not False
        or report.get("scope") != activation.SCOPE
        or report.get("code_sha256") != sha256_file(Path(activation.__file__))
        or report.get("source_hashes") != source_hashes
        or report.get("source_code_sha256")
        != {
            **interpolation.code_hashes(),
            "qknorm_math_sha256": sha256_file(
                Path(interpolation.matrix.norm.norm_math.__file__)
            ),
        }
        or report.get("chain_summary_sha256") != chain
        or report.get("weights_sha256") != sha256_file(weights)
        or report.get("shape") != list(SHAPE[1:])
    ):
        raise ValueError("Final activation-distribution-r2 source binding invalid")
    data = report.get("activation", {})
    head_rank, token_rank = data.get("head_rank"), data.get("token_rank")
    heads, tokens = data.get("head_error_share"), data.get("token_error_share")
    if (
        not isinstance(head_rank, list)
        or not isinstance(token_rank, list)
        or sorted(head_rank) != list(range(16))
        or sorted(token_rank) != list(range(4200))
        or not isinstance(heads, list)
        or not isinstance(tokens, list)
        or len(heads) != 16
        or len(tokens) != 4200
        or any(
            not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0
            for x in (*heads, *tokens)
        )
        or not math.isclose(sum(heads), 1, abs_tol=1e-9)
        or not math.isclose(sum(tokens), 1, abs_tol=1e-9)
        or head_rank != np.argsort(-np.asarray(heads), kind="stable").tolist()
        or token_rank != np.argsort(-np.asarray(tokens), kind="stable").tolist()
        or head_rank[0] != 13
        or token_rank[0] != 3343
        or not isinstance(data.get("error_square"), (int, float))
        or data["error_square"] <= 0
    ):
        raise ValueError("Final activation distribution ranks/energy invalid")
    return report, {
        "activation_report_sha256": sha256_file(path),
        "distribution_report_sha256": source_hashes["distribution.json"],
        "chain_summary_sha256": chain,
    }


def masks(head_rank, token_rank, *, tokens=4200, heads=16):
    if (
        len(head_rank) != heads
        or len(token_rank) != tokens
        or sorted(head_rank) != list(range(heads))
        or sorted(token_rank) != list(range(tokens))
        or head_rank[0] != 13
        or token_rank[0] != 3343
    ):
        raise ValueError("Complete stable activation ranks required")
    result = {name: torch.zeros(tokens, heads, dtype=torch.bool) for name in CASES}
    result["full"][:] = True
    result["head13"][:, 13] = True
    result["token3343"][3343, :] = True
    result["activation_top42"][token_rank[:42], :] = True
    result["low42"][token_rank[-42:], :] = True
    result["low1"][token_rank[-1], :] = True
    result["low_head"][:, head_rank[-1]] = True
    result["head13_token3343"][3343, 13] = True
    return result


def validate_drift_axes(drift, recorded):
    """Bind mask rankings to actual drift, allowing CPU/CUDA FP32 rounding."""
    total = float(drift.sum())
    if not math.isclose(total, recorded["error_square"], rel_tol=1e-5):
        raise ValueError("Actual K drift differs from final activation report")
    for axis, dimension in (("head", 0), ("token", 1)):
        shares = drift.sum(dimension).numpy() / total
        expected = np.asarray(recorded[f"{axis}_error_share"])
        if shares.shape != expected.shape or not np.allclose(
            shares, expected, rtol=1e-5, atol=1e-10
        ):
            raise ValueError(f"Actual K drift {axis} distribution differs")
        ranked = shares[recorded[f"{axis}_rank"]]
        if np.any(np.diff(ranked) > 1e-10):
            raise ValueError(
                f"Actual K drift {axis} rank differs beyond FP32 tolerance"
            )


def select_inputs(case, side, inputs, mask):
    if case not in CASES or side not in interpolation.SIDES:
        raise ValueError("Unknown case or V background")
    q, reference_k, _ = inputs["reference"]
    candidate_k = inputs["candidate"][1]
    if reference_k.shape != candidate_k.shape or mask.shape != reference_k.shape[1:3]:
        raise ValueError("K/mask shape mismatch")
    if case == "baseline":
        k = reference_k
    elif case == "full":
        k = candidate_k
    else:
        k = torch.where(
            mask.to(reference_k.device)[None, :, :, None], candidate_k, reference_k
        )
    return q, k, inputs["reference" if side == "R" else "candidate"][2]


def mask_record(mask, drift_square):
    if tuple(mask.shape) != tuple(drift_square.shape) or drift_square.sum() <= 0:
        raise ValueError("Nonzero K drift with matching mask required")
    indices = torch.nonzero(mask, as_tuple=False).tolist()
    digest = hashlib.sha256(mask.numpy().tobytes()).hexdigest()
    return {
        "head_token_pairs": len(indices),
        "token_indices": torch.where(mask.any(1))[0].tolist(),
        "head_indices": torch.where(mask.any(0))[0].tolist(),
        "mask_sha256": digest,
        "actual_k_drift_square_share": float(
            drift_square[mask].sum() / drift_square.sum()
        ),
    }


def tensor_sha(tensor):
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def input_hashes(inputs, du):
    cpu = interpolation.matrix.norm.core.stages._cpu
    result = {
        f"{side}_{part}": tensor_sha(cpu(value))
        for side, values in inputs.items()
        for part, value in zip("qkv", values)
    }
    result["du"] = tensor_sha(cpu(du))
    return result


def distance(a, b):
    """Absolute L2 only: no response/contribution fraction is inferred."""
    return float(torch.linalg.vector_norm((a.double() - b.double()).reshape(-1)))


def response(c, baseline, full):
    parts = {}
    for name, sl in (
        ("q", slice(0, 2048)),
        ("k", slice(2048, 4096)),
        ("v", slice(4096, 6144)),
        ("fused", slice(None)),
    ):
        a, b, f = c[..., sl], baseline[..., sl], full[..., sl]
        parts[name] = {
            "distance_to_baseline_l2": distance(a, b),
            "distance_to_full_l2": distance(a, f),
            "full_response_l2": distance(f, b),
            "baseline_norm_l2": float(torch.linalg.vector_norm(b.double().reshape(-1))),
        }
    return parts


def analyze(
    matrix_run,
    qknorm_run,
    core_run,
    upstream,
    stage_run,
    run,
    distribution_run,
    activation_run,
    weights,
    device,
):
    if device.type != "cuda":
        raise ValueError("Monitored CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report, binding = validate_activation(
        run, distribution_run, activation_run, weights
    )
    source, manifests, captured, old, historical, matrix_cells = (
        interpolation.source_evidence(
            matrix_run, qknorm_run, core_run, upstream, stage_run, weights
        )
    )
    interpolation_analysis = json.loads((Path(run) / "summary.json").read_text())[
        "analysis"
    ]
    if source != interpolation_analysis["source"]:
        raise ValueError("Interpolation source changed")
    kwargs = interpolation.matrix.norm.core.stages.capture.conditioning(
        manifests["reference"], captured["reference"], device
    )
    if kwargs["attn_params"].attn_mode != "torch" or kwargs["rope_cos_sin"] is None:
        raise ValueError("FP32 torch attention/RoPE required")
    q_norm, k_norm = interpolation.matrix.make_norms(old["norm_parameters"], device)
    inputs = {
        side: interpolation.matrix.postrope(
            old[side]["p"].reshape(1, 4200, interpolation.matrix.WIDTH).to(device),
            q_norm,
            k_norm,
            kwargs["rope_cos_sin"],
            16,
            128,
        )
        for side in interpolation.matrix.SIDES
    }
    drift = (
        (
            inputs["candidate"][1].detach().cpu().double()
            - inputs["reference"][1].detach().cpu().double()
        )
        .square()
        .sum(-1)
        .squeeze(0)
    )
    ranks = report["activation"]
    actual_square = float(drift.sum())
    validate_drift_axes(drift, ranks)
    selected_masks = masks(ranks["head_rank"], ranks["token_rank"])
    du = old["reference"]["du"].reshape(1, 4200, 2048).to(device)
    input_sha256 = input_hashes(inputs, du)
    replay, endpoint, forward, cases = {}, {}, {}, {}
    try:
        for side in interpolation.SIDES:
            endpoints = {}
            cases[side] = {}
            for name in CASES:
                selected = tuple(
                    x.detach().requires_grad_(True)
                    for x in select_inputs(name, side, inputs, selected_masks[name])
                )
                from networks.attention_dispatch import dispatch_attention

                u = dispatch_attention(
                    list(selected),
                    attn_params=kwargs["attn_params"].for_attention_kind(
                        is_selfattn=True
                    ),
                )
                if u.shape != du.shape or u.dtype != torch.float32:
                    raise ValueError("Attention output shape/dtype changed")
                gradients = torch.autograd.grad(u, selected, du)
                c = interpolation.matrix.norm._pack(
                    *gradients, (1, 4200, 6144)
                ).detach()
                del gradients, selected
                repeated = tuple(
                    x.detach().requires_grad_(True)
                    for x in select_inputs(name, side, inputs, selected_masks[name])
                )
                repeated_u = dispatch_attention(
                    list(repeated),
                    attn_params=kwargs["attn_params"].for_attention_kind(
                        is_selfattn=True
                    ),
                )
                repeated_gradients = torch.autograd.grad(repeated_u, repeated, du)
                repeated_c = interpolation.matrix.norm._pack(
                    *repeated_gradients, (1, 4200, 6144)
                ).detach()
                if not torch.equal(u, repeated_u):
                    raise ValueError(f"{name}/{side} forward repeat not bitwise")
                c_cpu = interpolation.matrix.norm._cpu(c)
                replay[f"{name}_v{side}"] = (
                    interpolation.matrix.norm.core.stages.vjp._fidelity(
                        f"{name}/{side} repeated c",
                        interpolation.matrix.norm._cpu(repeated_c),
                        c_cpu,
                        1e-4,
                    )
                )
                if name in ("baseline", "full"):
                    key = ("RR" if name == "baseline" else "RC") + side
                    endpoint[f"{name}_v{side}_matrix"] = (
                        interpolation.matrix.norm.core.stages.vjp._fidelity(
                            f"{name}/{side} matrix endpoint",
                            c_cpu,
                            matrix_cells[key],
                            1e-4,
                        )
                    )
                    endpoint[f"{name}_v{side}_interpolation"] = (
                        interpolation.matrix.norm.core.stages.vjp._fidelity(
                            f"{name}/{side} interpolation endpoint",
                            c_cpu,
                            torch.from_numpy(
                                activation.gradients.load_cell(
                                    run,
                                    interpolation_analysis,
                                    interpolation.label(
                                        0 if name == "baseline" else 1, side
                                    ),
                                )
                                .reshape(*interpolation.matrix.SHAPE[:-1], 6144)
                                .copy()
                            ),
                            1e-4,
                        )
                    )
                if name == "baseline" and side == "R":
                    if not torch.equal(
                        interpolation.matrix.norm.core.stages._cpu(u).reshape(
                            interpolation.matrix.SHAPE
                        ),
                        old["reference"]["u"],
                    ):
                        raise ValueError("Reference core output not bitwise")
                    forward["reference_core"] = True
                    endpoint["baseline_vR_historical"] = (
                        interpolation.matrix.norm.core.stages.vjp._fidelity(
                            "reference historical c",
                            c_cpu,
                            historical["reference"]["c"],
                            1e-4,
                        )
                    )
                if name in ("baseline", "full"):
                    endpoints[name] = c_cpu
                else:
                    cases[side][name] = {
                        "mask": mask_record(selected_masks[name], drift),
                        "c_sha256": tensor_sha(c_cpu),
                        "response": response(
                            c_cpu, endpoints["baseline"], endpoints["full"]
                        ),
                    }
                del u, c, repeated_u, repeated_c, repeated_gradients, repeated, c_cpu
            for name in ("baseline", "full"):
                value = endpoints[name]
                cases[side][name] = {
                    "mask": mask_record(selected_masks[name], drift),
                    "c_sha256": tensor_sha(value),
                    "response": response(
                        value, endpoints["baseline"], endpoints["full"]
                    ),
                }
    finally:
        del inputs, q_norm, k_norm
        gc.collect()
        torch.cuda.empty_cache()
    return {
        "status": "analyzed",
        "scope": SCOPE,
        "production_ready": False,
        "uuid": manifests["reference"]["uuid"],
        "shape": list(interpolation.matrix.SHAPE),
        "source": source,
        "source_binding": binding,
        "analysis_code_sha256": code_hashes(),
        "actual_k_drift_square": actual_square,
        "forward_bitwise_equal": forward,
        "drift_axis_tolerance": {
            "share_rtol": 1e-5,
            "share_atol": 1e-10,
            "rank_share_inversion_atol": 1e-10,
        },
        "replay": replay,
        "endpoint": endpoint,
        "input_sha256": input_sha256,
        "cases": cases,
        "disk_bytes_c": 0,
        "denominators": "Distances are absolute L2; baseline_norm_l2 and full_response_l2 are explicit norms, not attribution fractions.",
        "audit_limit": "No new c tensors saved. Backward gradients are repeat-checked but not independently reconstructible from this report; source endpoints remain separately archived.",
    }
