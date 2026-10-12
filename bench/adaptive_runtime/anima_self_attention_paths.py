"""Block17 A(N(x)) normalization/cotangent matrix using saved stage evidence."""

import gc
import json
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch

from bench.adaptive_runtime import anima_block17_paths as stages
from bench.adaptive_runtime import diagnose_anima_block17_paths as stage_driver
from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime.anima_self_attention_trace import SelfAttentionTrace
from library.training.auto_block_swap.process import write_result


SCOPE = "block17_reference_policy_self_attention_norm_cotangent_matrix_not_production"
SHAPE = stages.capture.SHAPE
MAX_BYTES = stages.capture.MAX_BYTES
TOTAL_MAX_BYTES = stages.TOTAL_MAX_BYTES
SIDES = ("reference", "candidate")
TRACE_KEYS = {"x", "z", "b", "h", "branch_dx", "gate"}
MATRIX_KEYS = {f"norm.{x}.{h}" for x in SIDES for h in SIDES} | {"terminal"}


def code_hashes():
    return {"analysis_sha256": sha256_file(Path(__file__)),
            "helper_sha256": sha256_file(Path(__file__).with_name("anima_self_attention_trace.py")),
            "stage_helper_sha256": sha256_file(Path(stages.__file__).with_name("anima_block_stages.py"))}


def source_evidence(upstream, stage_run, weights):
    upstream, stage_run = Path(upstream), Path(stage_run)
    summary = json.loads((stage_run / "summary.json").read_text())
    environment = json.loads((stage_run / "environment.json").read_text())
    if (summary.get("status") != "completed" or summary.get("production_ready") is not False
            or Path(environment["source_run"]).resolve() != upstream.resolve()):
        raise ValueError("Completed stage evidence bound to upstream run required")
    analysis = summary["analysis"]
    stage_driver.validate_files(stage_run, analysis)
    manifests, tensors, _ = stages.source_evidence(upstream, weights)
    source = analysis["source"]
    expected = {"capture_sha256": {side: m["sha256"] for side, m in manifests.items()},
                "signature": manifests["reference"]["signature"],
                "replay": manifests["reference"]["replay"], "block_index": 17,
                "policy": "reference", "scale": 1,
                "analysis_sha256": sha256_file(Path(stages.__file__)),
                "helper_sha256": code_hashes()["stage_helper_sha256"],
                "block_source_sha256": sha256_file(Path(stages.__file__).resolve().parents[2]
                                                  / "library/anima/models.py")}
    if source != expected:
        raise ValueError("Stage evidence capture, replay or code binding mismatch")
    prior = {}
    for side in SIDES:
        with safe_open(str(stage_run / f"trace-{side}/evidence.safetensors"), framework="pt") as f:
            prior[side] = {key: f.get_tensor(key) for key in ("cotangent.0", "cotangent.1", "activation.3", "gate.self_attn")}
    with safe_open(str(stage_run / "stage-matrix/evidence.safetensors"), framework="pt") as f:
        for side in SIDES:
            prior[side]["branch_dx"] = f.get_tensor(f"branch.self_attn.{side}")
    if not torch.equal(prior["reference"]["gate.self_attn"], prior["candidate"]["gate.self_attn"]):
        raise ValueError("Self-attention gate must be bitwise identical")
    binding = {"stage_summary_sha256": sha256_file(stage_run / "summary.json"),
               "stage_evidence_sha256": {side: item["sha256"] for side, item in analysis["files"].items()},
               "upstream_source": source, **code_hashes()}
    return binding, manifests, tensors, prior


def _cpu(value):
    return stages._cpu(value).reshape(SHAPE)


def trace_one(block, raw, terminal, block_g, kwargs, baseline, previous_branch, fixed_h, device):
    trace = SelfAttentionTrace(block)
    try:
        x = raw.to(device).detach().requires_grad_(True)
        output = block(x, **kwargs)
        trace.validate()
        if not torch.equal(_cpu(output), baseline[0]):
            raise ValueError("Self-attention trace forward differs from baseline")
        whole = trace.cotangents(output, block_g.to(device))[0]
        checks = {"whole_block": stages.vjp._fidelity("self-attention whole block", _cpu(whole),
                                                     baseline[1], stages.vjp.GRADIENT_RELATIVE_L2_MAX)}
        h = trace.attention_vjp(terminal.to(device))
        branch = trace.branch_vjp(terminal.to(device))
        checks["source_branch"] = stages.vjp._fidelity("source self-attention branch", _cpu(branch),
                                                      previous_branch, stages.vjp.GRADIENT_RELATIVE_L2_MAX)
        own = trace.normalization_vjp(h.detach())
        checks["chain_identity"] = stages.vjp._fidelity("A(N(x)) chain identity", _cpu(own),
                                                       _cpu(branch), stages.vjp.GRADIENT_RELATIVE_L2_MAX)
        evidence = {"x": _cpu(x), "z": _cpu(trace.attention_input),
                    "b": _cpu(trace.attention_output), "h": _cpu(h), "branch_dx": _cpu(branch),
                    "gate": stages._cpu(trace.stages[0]["gate"])}
        cells = {side: _cpu(trace.normalization_vjp(value.to(device))) for side, value in fixed_h.items()}
        return evidence, _cpu(own), cells, {"forward_bitwise_equal": True, **checks}
    finally:
        trace.restore()


def save_evidence(directory, name, tensors, source, *, scope=SCOPE):
    size = sum(value.numel() * value.element_size() for value in tensors.values())
    if size > MAX_BYTES or any(v.dtype != torch.float32 or not bool(torch.isfinite(v).all()) for v in tensors.values()):
        raise ValueError("Self-attention evidence dtype, finite or 512MiB budget failed")
    folder = Path(directory) / name
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / "evidence.safetensors"
    save_file(tensors, str(path))
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Self-attention disk file exceeds 512MiB")
    manifest = {"schema": "anima_self_attention_paths_v1", "scope": scope, "production_ready": False,
                "name": name, "file": path.name, "sha256": sha256_file(path), "source": source,
                "tensor_bytes": size, "disk_bytes": path.stat().st_size,
                "inventory": {key: list(v.shape) for key, v in tensors.items()}}
    write_result(folder / "manifest.json", manifest)
    return manifest


def analyze(upstream, stage_run, weights, directory, device):
    if device.type != "cuda":
        raise ValueError("Monitored CUDA analysis required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source, manifests, tensors, prior = source_evidence(upstream, stage_run, weights)
    block, adapters = stages.vjp._build_block(weights, Path(upstream) / "reference", manifests["reference"],
                                              policy="reference", device=device)
    evidence, baselines, checks, files, matrix = {}, {}, {}, {}, {}
    terminal = prior["reference"]["cotangent.1"] * prior["reference"]["gate.self_attn"]
    matrix["terminal"] = terminal
    block_g = tensors["reference"]["g"]
    try:
        kwargs = stages.capture.conditioning(manifests["reference"], tensors["reference"], device)
        for side in SIDES:
            baselines[side] = stages._baseline(block, tensors[side]["x"], block_g, kwargs, device)
            if not torch.equal(baselines[side][0], prior[side]["activation.3"]):
                raise ValueError("Baseline no longer matches source stage forward")
            baseline_check = stages.vjp._fidelity("source stage whole gradient", baselines[side][1],
                                                prior[side]["cotangent.0"], stages.vjp.GRADIENT_RELATIVE_L2_MAX)
            fixed = {} if side == "reference" else {"reference": evidence["reference"]["h"]}
            saved, own, cells, fidelity = trace_one(
                block, tensors[side]["x"], terminal, block_g, kwargs, baselines[side],
                prior[side]["branch_dx"], fixed, device)
            if not torch.equal(saved["gate"], prior["reference"]["gate.self_attn"]):
                raise ValueError("Gate changed during self-attention trace")
            evidence[side], checks[side] = saved, {"source_whole_block": baseline_check, **fidelity}
            matrix[f"norm.{side}.{side}"] = own
            matrix.update({f"norm.{side}.{h}": value for h, value in cells.items()})
            files[side] = save_evidence(directory, f"trace-{side}", saved, source)
            gc.collect()
        # Replay only reference to obtain the remaining cross cell; do not retain two GPU graphs.
        saved, _, cells, fidelity = trace_one(
            block, tensors["reference"]["x"], terminal, block_g, kwargs, baselines["reference"],
            prior["reference"]["branch_dx"], {"candidate": evidence["candidate"]["h"]}, device)
        for key in ("x", "z", "b", "gate"):
            if not torch.equal(saved[key], evidence["reference"][key]):
                raise ValueError(f"Repeated reference forward is not bitwise reproducible: {key}")
        replay_checks = {key: stages.vjp._fidelity(f"repeated reference {key}", saved[key],
                         evidence["reference"][key], stages.vjp.GRADIENT_RELATIVE_L2_MAX)
                         for key in ("h", "branch_dx")}
        checks["reference_replay"] = {**fidelity, "repeated_gradient": replay_checks}
        files["reference_replay"] = save_evidence(directory, "trace-reference_replay", saved, source)
        matrix["norm.reference.candidate"] = cells["candidate"]
    finally:
        del block, adapters
        gc.collect()
        torch.cuda.empty_cache()
    files["matrix"] = save_evidence(directory, "norm-matrix", matrix, source)
    total = sum(item["disk_bytes"] for item in files.values())
    if total > TOTAL_MAX_BYTES:
        raise ValueError("Self-attention aggregate evidence exceeds 1GiB")
    metric = stages.vjp._metric
    contrasts = {
        "activation_fixed_cotangent": {h: metric(matrix[f"norm.reference.{h}"], matrix[f"norm.candidate.{h}"]) for h in SIDES},
        "cotangent_fixed_activation": {x: metric(matrix[f"norm.{x}.reference"], matrix[f"norm.{x}.candidate"]) for x in SIDES}}
    return {"status": "analyzed", "scope": SCOPE, "production_ready": False,
            "uuid": manifests["reference"]["uuid"], "source": source, "shape": list(SHAPE),
            "files": files, "disk_bytes": total, "fidelity": checks, "normalization_contrasts": contrasts,
            "boundaries": {key: metric(evidence["reference"][key], evidence["candidate"][key])
                           for key in ("x", "z", "b", "h", "branch_dx")},
            "interpretation": "N includes LayerNorm and AdaLN modulation; A includes QKV, QK-norm, RoPE, attention and output projection. Contrasts are not additive causes."}


def validate_files(directory, analysis):
    if (analysis.get("status") != "analyzed" or analysis.get("scope") != SCOPE
            or analysis.get("production_ready") is not False or analysis.get("shape") != list(SHAPE)
            or set(analysis.get("files", {})) != {*SIDES, "matrix", "reference_replay"}
            or set(analysis.get("fidelity", {})) != {*SIDES, "reference_replay"}):
        raise ValueError("Incomplete self-attention analysis")
    total, verified = 0, {}
    for side in (*SIDES, "matrix", "reference_replay"):
        name = "norm-matrix" if side == "matrix" else f"trace-{side}"
        folder = Path(directory) / name
        m = json.loads((folder / "manifest.json").read_text())
        path = folder / "evidence.safetensors"
        if (m != analysis["files"][side] or m.get("schema") != "anima_self_attention_paths_v1"
                or m.get("scope") != SCOPE or m.get("production_ready") is not False
                or m.get("source") != analysis["source"] or m.get("name") != name
                or m.get("file") != path.name or path.stat().st_size != m.get("disk_bytes")
                or path.stat().st_size > MAX_BYTES or sha256_file(path) != m.get("sha256")):
            raise ValueError("Self-attention evidence manifest/hash/budget mismatch")
        tensors = load_file(str(path))
        expected = MATRIX_KEYS if side == "matrix" else TRACE_KEYS
        if (set(tensors) != expected or m.get("inventory") != {key: list(v.shape) for key, v in tensors.items()}
                or sum(v.numel() * v.element_size() for v in tensors.values()) != m.get("tensor_bytes")
                or any(v.dtype != torch.float32 or not bool(torch.isfinite(v).all())
                       or tuple(v.shape) != ((1, 1, 1, 1, SHAPE[-1]) if key == "gate" else SHAPE)
                       for key, v in tensors.items())):
            raise ValueError("Self-attention tensor inventory/shape/dtype/finite mismatch")
        total += path.stat().st_size
        verified[side] = tensors
    if total > TOTAL_MAX_BYTES or total != analysis.get("disk_bytes"):
        raise ValueError("Self-attention aggregate budget mismatch")
    for side, row in analysis["fidelity"].items():
        expected = {"whole_block", "source_branch", "chain_identity", "forward_bitwise_equal"}
        if side != "reference_replay":
            expected.add("source_whole_block")
        else:
            expected.add("repeated_gradient")
        if set(row) != expected or row.get("forward_bitwise_equal") is not True:
            raise ValueError("Incomplete self-attention fidelity")
        if any(m.get("finite") is not True or m.get("relative_l2") is None
               or not 0 <= m["relative_l2"] <= stages.vjp.GRADIENT_RELATIVE_L2_MAX
               for key, m in row.items() if key not in ("forward_bitwise_equal", "repeated_gradient")):
            raise ValueError("Self-attention fidelity failed")
        if side == "reference_replay":
            repeat = row["repeated_gradient"]
            if (set(repeat) != {"h", "branch_dx"}
                    or any(m.get("finite") is not True or m.get("relative_l2") is None
                           or not 0 <= m["relative_l2"] <= stages.vjp.GRADIENT_RELATIVE_L2_MAX for m in repeat.values())):
                raise ValueError("Repeated reference gradient fidelity failed")
    return verified


def validate_statistics(analysis, verified):
    """Bind reported contrasts and diagonal chain identities to actual tensors."""
    matrix = verified["matrix"]
    expected = {
        "activation_fixed_cotangent": {h: stages.vjp._metric(matrix[f"norm.reference.{h}"], matrix[f"norm.candidate.{h}"]) for h in SIDES},
        "cotangent_fixed_activation": {x: stages.vjp._metric(matrix[f"norm.{x}.reference"], matrix[f"norm.{x}.candidate"]) for x in SIDES}}
    boundaries = {key: stages.vjp._metric(verified["reference"][key], verified["candidate"][key])
                  for key in ("x", "z", "b", "h", "branch_dx")}
    if expected != analysis.get("normalization_contrasts") or boundaries != analysis.get("boundaries"):
        raise ValueError("Self-attention statistics not bound to evidence tensors")
    for side in SIDES:
        stages.vjp._fidelity("saved diagonal chain identity", matrix[f"norm.{side}.{side}"],
                             verified[side]["branch_dx"], stages.vjp.GRADIENT_RELATIVE_L2_MAX)
    replay = verified["reference_replay"]
    for key in ("x", "z", "b", "gate"):
        if not torch.equal(replay[key], verified["reference"][key]):
            raise ValueError("Saved repeated reference forward differs")
    for key in ("h", "branch_dx"):
        stages.vjp._fidelity(f"saved repeated reference {key}", replay[key], verified["reference"][key],
                             stages.vjp.GRADIENT_RELATIVE_L2_MAX)
