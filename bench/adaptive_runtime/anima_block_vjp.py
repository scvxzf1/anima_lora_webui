"""Offline, full-token Anima block VJP cross matrices; diagnostic only."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
from types import SimpleNamespace

from safetensors.torch import load_file, save_file
import torch

from bench.adaptive_runtime.anima_inputs import sha256_file
from bench.adaptive_runtime import anima_block_vjp_capture as capture
from bench.adaptive_runtime.compare_training import compare, read_capture as read_training
from bench.adaptive_runtime.probe_anima_train import fp32_complement
from bench.adaptive_runtime.replay import network_tensors, tensor_digest
from bench.adaptive_runtime.search_anima_precision import MLP
from library.anima.weights import load_anima_model
from library.training.adaptive_runtime.precision import vector_metrics
from library.training.adaptive_runtime.training_precision import install_training_precision
from networks.lora_anima.config import LoRANetworkCfg
from networks.lora_anima.network import LoRANetwork
from networks.lora_modules.lora import LoRAModule
from library.training.auto_block_swap.process import write_result


BLOCK_INDEX = 14
FORWARD_RELATIVE_L2_MAX = 1e-5
GRADIENT_RELATIVE_L2_MAX = 1e-4
REPLAY_MAX_BYTES = 512 * 1024**2
SCOPE = "actual_full_token_block14_fixed_state_vjp_not_production_certificate"
UPSTREAM_SCOPE = "actual_full_token_blocks15_17_27_fixed_state_vjp_not_production_certificate"


def scope_for_block(block_index):
    if type(block_index) is not int or block_index not in (14, *capture.MULTI_INDICES):
        raise ValueError("Unsupported diagnostic block index")
    return SCOPE if block_index == 14 else UPSTREAM_SCOPE


def training_root(directory, block_index):
    directory = Path(directory)
    scope_for_block(block_index)
    if block_index == 14:
        if directory.name != "block-vjp":
            raise ValueError("Single-block capture path mismatch")
        return directory.parent
    if directory.name != f"block-{block_index:02d}" or directory.parent.name != "block-vjp":
        raise ValueError("Multi-block capture path/index mismatch")
    return directory.parent.parent


def _metric(a, b):
    if a.shape != b.shape:
        raise ValueError("Block VJP shape mismatch")
    return vector_metrics([(a.double(), b.double())])


def _fidelity(name, actual, captured, limit):
    metrics = _metric(captured, actual)
    if (not metrics["finite"] or metrics["relative_l2"] is None
            or metrics["relative_l2"] > limit):
        raise ValueError(f"{name} capture fidelity failed: {metrics}")
    return metrics


def _contract(signature):
    return {key: value for key, value in signature.items()
            if key not in {"precision", "fp32_modules", "loss_scale"}}


def validate_pair(reference, candidate, comparison, *, block_index=BLOCK_INDEX):
    """Bind both block captures to their complete training/replay evidence."""
    roots = tuple(training_root(path, block_index) for path in (reference, candidate))
    verified = compare(*(root / "training-capture" for root in roots))
    if verified != comparison or verified.get("status") != "compared" or not verified["scope"].startswith(
            "fixed_pre_step"):
        raise ValueError("Block VJP comparison is not bound to training captures")
    blocks = [capture.read_capture(path) for path in (reference, candidate)]
    training = [read_training(root / "training-capture") for root in roots]
    signatures = []
    for index, ((manifest, _), (full, worker)) in enumerate(zip(blocks, training, strict=True)):
        precision, scale = (("fp32-reference", 1), ("fp16-islands", 1024))[index]
        signature = full["signature"]
        indices = capture.validate_block_indices(signature.get("block_vjp_indices", [14]))
        if (manifest["signature"] != signature or manifest["uuid"] != worker.get("uuid")
                or manifest["replay"] != full["cases"][0].get("replay")
                or manifest["loss_scale"] != scale or worker.get("loss_scale") != scale
                or signature.get("precision") != precision or signature.get("loss_scale") != scale
                or block_index not in indices
                or signature.get("block_vjp_scope") != capture.scope_for(indices)
                or signature.get("block_vjp_helper_sha256") != sha256_file(Path(capture.__file__))
                or signature.get("steps") != 1 or signature.get("sigmas") != [0.5]
                or signature.get("backend") != "torch" or signature.get("checkpoint") != "full"
                or signature.get("rank") != 4 or signature.get("alpha") != 4
                or manifest.get("block_index") != block_index
                or manifest.get("fp32_residual") is not bool(index)
                or worker.get("swap") != 20):
            raise ValueError("Block VJP capture/signature/replay identity mismatch")
        if index == 0:
            if worker.get("precision_manifest") is not None:
                raise ValueError("Reference precision must be plain FP32")
        else:
            assignments = (worker.get("precision_manifest") or {}).get("assignments", {})
            expected = {name: "fp16" if name.startswith("blocks.") and ".mlp." in name
                        else "fp32" for name in assignments}
            if (assignments != expected or len(assignments) != 377
                    or sum(value == "fp16" for value in assignments.values()) != 56
                    or sorted(signature.get("fp32_modules", [])) !=
                    sorted(name for name, value in assignments.items() if value == "fp32")):
                raise ValueError("Candidate must be exact 56/321 MLP-only assignment")
        signatures.append(signature)
    if (_contract(signatures[0]) != _contract(signatures[1])
            or training[0][0]["initial_adapter_sha256"] != training[1][0]["initial_adapter_sha256"]
            or blocks[0][0]["replay"] != blocks[1][0]["replay"]
            or blocks[0][0]["uuid"] != blocks[1][0]["uuid"]):
        raise ValueError("Block VJP shared source/initial adapter/replay mismatch")
    for key in ("weights_sha256", "inputs_sha256", "probe_sha256"):
        if not isinstance(signatures[0].get(key), str) or len(signatures[0][key]) != 64:
            raise ValueError(f"Missing source identity: {key}")
    return blocks, training


def _replay_state(reference_root, evidence, network):
    directory = Path(reference_root) / "training-replay"
    manifest = json.loads((directory / "manifest.json").read_text())
    cases = manifest.get("cases", [])
    if (manifest.get("status") != "captured" or manifest.get("steps") != 1 or len(cases) != 1
            or cases[0].get("evidence") != evidence or cases[0].get("file") != "step-000001.safetensors"):
        raise ValueError("Reference replay manifest mismatch")
    path = directory / "step-000001.safetensors"
    if path.stat().st_size > REPLAY_MAX_BYTES or sha256_file(path) != evidence["snapshot_sha256"]:
        raise ValueError("Reference replay snapshot hash/budget mismatch")
    values = load_file(str(path), device="cpu")
    current = network_tensors(network)
    names = {f"{kind}.{name}" for kind, name in
             [("parameter", name) for name, _ in network.named_parameters()] +
             [("buffer", name) for name, _ in network.named_buffers()]}
    if (set(current) != names or {k for k in values if k.startswith(("parameter.", "buffer."))} != names
            or tensor_digest({k: values[k] for k in names}) != evidence["state_sha256"]):
        raise ValueError("Reference replay adapter inventory/digest mismatch")
    with torch.no_grad():
        for name, target in current.items():
            source = values[name]
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(f"Replay adapter tensor contract changed: {name}")
            target.copy_(source)
    if tensor_digest(network_tensors(network)) != evidence["state_sha256"]:
        raise ValueError("Restored LoRA state digest mismatch")


def _build_block(weights, reference_root, manifest, *, policy, device):
    block_index = manifest["block_index"]
    scope_for_block(block_index)
    model = load_anima_model(device="cpu", dit_path=str(weights), attn_mode="torch",
                             loading_device="cpu", dit_weight_dtype=torch.bfloat16)
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("Offline block reconstruction placed full model on GPU")
    model.requires_grad_(False)
    if policy == "reference":
        model.float()
    else:
        names = fp32_complement(model, MLP)
        precision = install_training_precision(
            model, SimpleNamespace(adaptive_fp32_modules=names, adaptive_loss_scale=1024,
                                   model_family="anima"), model_family="anima")
        expected_fp16 = {name for name, module in model.named_modules()
                         if type(module) is torch.nn.Linear
                         and name.startswith("blocks.") and ".mlp." in name}
        actual_fp16 = {name for name, dtype in precision["assignments"].items()
                       if dtype == "fp16"}
        if (len(precision["assignments"]) != 377 or len(expected_fp16) != 56
                or actual_fp16 != expected_fp16
                or sorted(manifest["signature"].get("fp32_modules", [])) != sorted(
                    set(precision["assignments"]) - expected_fp16)):
            raise ValueError("Built candidate assignment names mismatch")
    cfg = LoRANetworkCfg.from_kwargs({}, network_dim=4, network_alpha=4,
                                    neuron_dropout=None, module_class=LoRAModule)
    network = LoRANetwork(text_encoders=[], unet=model, cfg=cfg, multiplier=1.0)
    network.apply_to(text_encoders=[], unet=model, apply_text_encoder=False, apply_unet=True)
    network.to(device="cpu", dtype=torch.float32)
    _replay_state(reference_root, manifest["replay"], network)
    model.enable_gradient_checkpointing()
    if policy == "candidate":
        model.enable_fp32_residual()
    base_hash, adapter_hash = capture.state_digests(model, network, block_index)
    if (base_hash != manifest["base_state_sha256"]
            or adapter_hash != manifest["adapter_state_sha256"]):
        raise ValueError(f"{policy} actual block base/adapter hash differs from capture")
    block = model.blocks[block_index]
    names = {id(module) for module in block.modules() if type(module) is torch.nn.Linear}
    adapters = [adapter for adapter in network.unet_loras
                if id(getattr(getattr(adapter, "org_forward", None), "__self__", None)) in names]
    if not adapters or any(not item.lora_name.startswith(f"lora_unet_blocks_{block_index}_")
                           for item in adapters):
        raise ValueError(f"Cannot isolate block-{block_index} LoRA modules")
    del model, network
    gc.collect()
    block.to(device)
    for adapter in adapters:
        adapter.to(device=device, dtype=torch.float32)
        adapter.train()
    block.train()
    if not block.gradient_checkpointing or block.fp32_residual is not (policy == "candidate"):
        raise ValueError("Block checkpoint/residual contract changed")
    return block, adapters


def _run_matrix(block, manifest, tensors_by_side, *, policy, device):
    kwargs = capture.conditioning(manifest, tensors_by_side[policy], device)
    outputs = {}
    for x_side, x_values in tensors_by_side.items():
        for g_side, g_values in tensors_by_side.items():
            x = x_values["x"].to(device).detach().requires_grad_(True)
            g = g_values["g"].to(device)
            scale = manifest["loss_scale"]
            with torch.autocast(device.type, enabled=False):
                y = block(x, **kwargs)
                if (x_side == policy):
                    forward = _fidelity(f"{policy} forward", y.detach().float().cpu(),
                                        x_values["y"], FORWARD_RELATIVE_L2_MAX)
                else:
                    forward = None
                (gx,) = torch.autograd.grad(y, x, grad_outputs=g * scale)
            gx = (gx.detach().float() / scale).cpu()
            if not bool(torch.isfinite(gx).all()):
                raise ValueError(f"Nonfinite {policy}/{x_side}/{g_side} VJP")
            outputs[(policy, x_side, g_side)] = {"gx": gx, "forward_fidelity": forward}
            del x, y, gx
    return outputs


def matrix_metrics(outputs, captures):
    """Keep each controlled contrast separate; errors are not additive."""
    keys = tuple(outputs)
    if set(keys) != {(p, x, g) for p in ("reference", "candidate")
                    for x in ("reference", "candidate") for g in ("reference", "candidate")}:
        raise ValueError("Expected all eight block VJP cells")
    fidelity = {}
    for policy in ("reference", "candidate"):
        item = outputs[(policy, policy, policy)]
        fidelity[policy] = {
            "forward": item["forward_fidelity"],
            "input_cotangent": _fidelity(f"{policy} VJP", item["gx"],
                                         captures[policy]["gx"], GRADIENT_RELATIVE_L2_MAX),
        }
    contrasts = {"activation_fixed_policy_and_cotangent": {},
                 "cotangent_fixed_policy_and_activation": {},
                 "policy_fixed_activation_and_cotangent": {}}
    for policy in ("reference", "candidate"):
        for side in ("reference", "candidate"):
            contrasts["activation_fixed_policy_and_cotangent"][f"{policy}/g={side}"] = _metric(
                outputs[(policy, "reference", side)]["gx"],
                outputs[(policy, "candidate", side)]["gx"])
            contrasts["cotangent_fixed_policy_and_activation"][f"{policy}/x={side}"] = _metric(
                outputs[(policy, side, "reference")]["gx"],
                outputs[(policy, side, "candidate")]["gx"])
    for x_side in ("reference", "candidate"):
        for g_side in ("reference", "candidate"):
            contrasts["policy_fixed_activation_and_cotangent"][f"x={x_side}/g={g_side}"] = _metric(
                outputs[("reference", x_side, g_side)]["gx"],
                outputs[("candidate", x_side, g_side)]["gx"])
    return fidelity, contrasts


def save_matrix_evidence(directory, outputs, manifests):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    tensors = {"/".join(key): item["gx"].detach().cpu().contiguous()
               for key, item in outputs.items()}
    expected = {"/".join((p, x, g)) for p in ("reference", "candidate")
                for x in ("reference", "candidate") for g in ("reference", "candidate")}
    if (set(tensors) != expected or any(value.dtype != torch.float32
            or tuple(value.shape) != capture.SHAPE or not bool(torch.isfinite(value).all())
            for value in tensors.values())):
        raise ValueError("Invalid full-shape VJP matrix evidence")
    total = sum(value.numel() * value.element_size() for value in tensors.values())
    if total > capture.MAX_BYTES:
        raise ValueError("VJP matrix evidence exceeds independent 512MiB budget")
    path = directory / "matrix.safetensors"
    save_file(tensors, str(path))
    if path.stat().st_size > capture.MAX_BYTES:
        raise ValueError("VJP matrix evidence disk budget exceeded")
    block_index = manifests["reference"].get("block_index", BLOCK_INDEX)
    if manifests["candidate"].get("block_index", BLOCK_INDEX) != block_index:
        raise ValueError("Matrix block identity mismatch")
    manifest = {"schema": "anima_block_vjp_matrix_v1", "scope": scope_for_block(block_index),
                "block_index": block_index,
                "axis_order": ["policy", "activation_source", "cotangent_source"],
                "production_ready": False, "file": path.name, "sha256": sha256_file(path),
                "shape": list(capture.SHAPE), "tensor_bytes": total, "cells": sorted(tensors),
                "capture_sha256": {policy: item["sha256"] for policy, item in manifests.items()},
                "source": manifests["reference"]["signature"],
                "replay": manifests["reference"]["replay"]}
    write_result(directory / "manifest.json", manifest)
    return manifest


def analyze(reference, candidate, comparison, *, weights, device, matrix_directory=None,
            block_index=BLOCK_INDEX):
    if device.type != "cuda":
        raise ValueError("Full-token block analysis requires monitored CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    (ref_manifest, ref_tensors), (candidate_manifest, candidate_tensors) = validate_pair(
        reference, candidate, comparison, block_index=block_index)[0]
    if sha256_file(weights) != ref_manifest["signature"]["weights_sha256"]:
        raise ValueError("Analysis checkpoint hash differs from captures")
    conditioning = {"reference": ref_tensors, "candidate": candidate_tensors}
    if {key for key in ref_tensors if key.startswith("conditioning.")} != {
            key for key in candidate_tensors if key.startswith("conditioning.")}:
        raise ValueError("Conditioning inventory differs across policies")
    if ref_manifest["conditioning"] != candidate_manifest["conditioning"]:
        raise ValueError("Conditioning metadata differs across policies")
    if any(not torch.equal(ref_tensors[key], candidate_tensors[key]) for key in ref_tensors
           if key.startswith("conditioning.")):
        raise ValueError("Conditioning tensors differ across policies")
    manifests = {"reference": ref_manifest, "candidate": candidate_manifest}
    outputs = {}
    for policy in ("reference", "candidate"):
        block, adapters = _build_block(weights, training_root(reference, block_index), manifests[policy],
                                       policy=policy, device=device)
        try:
            outputs.update(_run_matrix(block, manifests[policy], conditioning,
                                       policy=policy, device=device))
        finally:
            del block, adapters
            gc.collect()
            torch.cuda.empty_cache()
    fidelity, contrasts = matrix_metrics(outputs, conditioning)
    matrix_evidence = (save_matrix_evidence(matrix_directory, outputs, manifests)
                       if matrix_directory is not None else None)
    return {"status": "analyzed", "scope": scope_for_block(block_index), "production_ready": False,
            "uuid": ref_manifest["uuid"], "block_index": block_index,
            "tokens": math.prod(ref_tensors["x"].shape[1:-1]),
            "source": {key: ref_manifest["signature"][key] for key in
                       ("weights_sha256", "inputs_sha256", "probe_sha256",
                        "block_vjp_helper_sha256")},
            "replay": ref_manifest["replay"],
            "state": {policy: {"base_state_sha256": manifest["base_state_sha256"],
                                "adapter_state_sha256": manifest["adapter_state_sha256"],
                                "loss_scale": manifest["loss_scale"],
                                "fp32_residual": manifest["fp32_residual"]}
                      for policy, manifest in manifests.items()},
            "fidelity_limits": {"forward_relative_l2": FORWARD_RELATIVE_L2_MAX,
                                "gradient_relative_l2": GRADIENT_RELATIVE_L2_MAX},
            "fidelity": fidelity, "contrasts": contrasts,
            "cell_count": len(outputs), "comparison": comparison,
            "matrix_evidence": matrix_evidence}
