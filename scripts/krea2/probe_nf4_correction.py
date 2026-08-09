"""Fit and benchmark a fixed activation-aware NF4 correction on Krea-2.

The probe runs the complete experiment on one GPU:
  1. BF16 calibration forward: fit rank-r ``B(A(x))`` for all 196 target Linear
     layers using a truncated eigenspace of the observed input covariance.
  2. BF16 reference forwards for calibration and held-out sigma/noise cases.
  3. NF4 end-to-end validation with and without the fixed correction.
  4. NF4+LoRA training benchmark with and without the correction.

The correction is probe-only.  It is attached through forward hooks and never
changes the production training path or NF4 weights.

Environment variables:
  K2_CORR_GPU=1             CUDA device (default: PG199)
  K2_CORR_IMG=1024          image size
  K2_CORR_RANK=16           correction rank
  K2_CORR_INPUT_RANK=128    retained input-covariance rank during fitting
  K2_CORR_NITER=2           randomized SVD power iterations
  K2_CORR_WARMUP=1          training benchmark warmup steps
  K2_CORR_STEPS=5           measured training steps
  K2_CORR_OUT=path          JSON result
  K2_CORR_WEIGHTS=path      correction safetensors
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("K2_CORR_GPU", "1"))

import gc
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from library.models.krea2_raw.family import Krea2TextEmbedding, forward_for_loss  # noqa: E402
from library.models.krea2_raw.weights import load_krea2_dit  # noqa: E402
from probe_nf4_ablation import make_network  # noqa: E402
from probe_nf4_ex_spectrum import (  # noqa: E402
    DEFAULT_DIT,
    DEFAULT_NF4,
    DEFAULT_TE,
    DEFAULT_VAE,
    NF4WeightReader,
    _encode_inputs,
    _target_modules,
)

DEFAULT_OUT = ROOT / "output" / "tests" / "krea2_nf4_correction_rank16.json"
DEFAULT_WEIGHTS = ROOT / "output" / "tests" / "krea2_nf4_correction_rank16.safetensors"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    sigma: float
    noise_seed: int


CASES = (
    CaseSpec("calibration", 0.5, 123),
    CaseSpec("heldout_noise", 0.5, 456),
    CaseSpec("heldout_sigma_02", 0.2, 456),
    CaseSpec("heldout_sigma_08", 0.8, 456),
)


@dataclass(frozen=True)
class ProbeConfig:
    image_size: int
    rank: int
    input_rank: int
    niter: int
    warmup: int
    steps: int
    seed: int
    dit_path: Path
    nf4_path: Path
    te_path: Path
    vae_path: Path
    out_path: Path
    weights_path: Path


@dataclass(frozen=True)
class CorrectionFactors:
    path: str
    down: torch.Tensor
    up: torch.Tensor


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _load_config() -> ProbeConfig:
    config = ProbeConfig(
        image_size=_env_int("K2_CORR_IMG", 1024),
        rank=_env_int("K2_CORR_RANK", 16),
        input_rank=_env_int("K2_CORR_INPUT_RANK", 128),
        niter=_env_int("K2_CORR_NITER", 2),
        warmup=_env_int("K2_CORR_WARMUP", 1),
        steps=_env_int("K2_CORR_STEPS", 5),
        seed=_env_int("K2_CORR_SEED", 123),
        dit_path=Path(os.environ.get("K2_CORR_DIT", str(DEFAULT_DIT))),
        nf4_path=Path(os.environ.get("K2_CORR_NF4", str(DEFAULT_NF4))),
        te_path=Path(os.environ.get("K2_CORR_TE", str(DEFAULT_TE))),
        vae_path=Path(os.environ.get("K2_CORR_VAE", str(DEFAULT_VAE))),
        out_path=Path(os.environ.get("K2_CORR_OUT", str(DEFAULT_OUT))),
        weights_path=Path(os.environ.get("K2_CORR_WEIGHTS", str(DEFAULT_WEIGHTS))),
    )
    for path in (config.dit_path, config.nf4_path, config.te_path, config.vae_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if config.rank <= 0 or config.input_rank < config.rank:
        raise ValueError(f"invalid ranks: rank={config.rank}, input_rank={config.input_rank}")
    return config


def _case_tensors(
    latents_4d: torch.Tensor,
    spec: CaseSpec,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    latents = latents_4d.to(device).unsqueeze(2)
    devices = [device.index or 0]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(spec.noise_seed)
        noise = torch.randn_like(latents)
    sigma = torch.full((latents.shape[0],), spec.sigma, device=device, dtype=dtype)
    x_t = (1.0 - sigma) * latents + sigma * noise
    return x_t, sigma, noise - latents


def _low_rank_svd(
    matrix: torch.Tensor, q: int, niter: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q = min(q, min(matrix.shape))
    if q == min(matrix.shape):
        return torch.linalg.svd(matrix, full_matrices=False)
    devices = [matrix.device.index or 0] if matrix.is_cuda else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        u, s, v = torch.svd_lowrank(matrix, q=q, niter=niter)
    return u, s, v.T


def fit_activation_weighted_factors(
    activation: torch.Tensor,
    bf16_weight: torch.Tensor,
    nf4_weight: torch.Tensor,
    *,
    rank: int,
    input_rank: int,
    niter: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit ``up @ down`` to ``W-Q`` under the empirical input covariance."""
    x = activation.reshape(-1, activation.shape[-1]).float()
    _, input_s, input_vh = _low_rank_svd(x, input_rank, niter, seed)
    input_v = input_vh.T
    sample_count = x.shape[0]
    weighted_basis = input_v * (input_s / math.sqrt(sample_count)).unsqueeze(0)
    weighted_error = (
        bf16_weight.float() @ weighted_basis
        - nf4_weight.float() @ weighted_basis
    )
    error_u, error_s, error_vh = torch.linalg.svd(weighted_error, full_matrices=False)
    used_rank = min(rank, error_s.numel())
    safe_s = input_s.clamp_min(input_s.max() * 1e-7)
    inverse_scale = math.sqrt(sample_count) / safe_s
    down = (
        (error_vh[:used_rank] * inverse_scale.unsqueeze(0)) @ input_v.T
    )
    up = error_u[:, :used_rank] * error_s[:used_rank].unsqueeze(0)
    return down.to(torch.bfloat16), up.to(torch.bfloat16)


def _fit_corrections(
    dit: torch.nn.Module,
    reader: NF4WeightReader,
    x_t: torch.Tensor,
    text_embedding: Krea2TextEmbedding,
    sigma: torch.Tensor,
    config: ProbeConfig,
) -> tuple[torch.Tensor, list[CorrectionFactors], list[dict[str, Any]]]:
    targets = _target_modules(dit)
    factors: list[CorrectionFactors] = []
    metrics: list[dict[str, Any]] = []
    handles = []

    def make_hook(path: str, module: torch.nn.Linear, index: int):
        def hook(_module, inputs, output):
            started = time.time()
            activation = inputs[0]
            q_weight = reader.dequantize(path, activation.device)
            down, up = fit_activation_weighted_factors(
                activation,
                module.weight,
                q_weight,
                rank=config.rank,
                input_rank=config.input_rank,
                niter=config.niter,
                seed=config.seed + index,
            )
            correction = F.linear(F.linear(activation, down), up)
            q_output = F.linear(activation, q_weight.to(activation.dtype), module.bias)
            error = output.float() - q_output.float()
            residual = error - correction.float()
            error_energy = error.square().sum(dtype=torch.float64).item()
            residual_energy = residual.square().sum(dtype=torch.float64).item()
            captured = 1.0 - residual_energy / error_energy if error_energy else 1.0
            factors.append(CorrectionFactors(path, down.cpu(), up.cpu()))
            metrics.append(
                {
                    "path": path,
                    "input_shape": list(activation.shape),
                    "error_energy": error_energy,
                    "residual_energy": residual_energy,
                    "captured_energy": captured,
                    "residual_l2_factor": math.sqrt(residual_energy / error_energy)
                    if error_energy
                    else 0.0,
                    "seconds": time.time() - started,
                }
            )
            print(
                f"[{len(factors):3d}/{len(targets)}] {path:<30} "
                f"capture={100 * captured:6.2f}%",
                flush=True,
            )

        return hook

    for index, (path, module) in enumerate(targets):
        handles.append(module.register_forward_hook(make_hook(path, module, index)))
    try:
        with torch.inference_mode():
            velocity = forward_for_loss(dit, x_t, text_embedding, sigma)
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    if len(factors) != len(targets):
        raise RuntimeError(f"fit {len(factors)} of {len(targets)} correction layers")
    return velocity.detach().cpu(), factors, metrics


def _save_factors(factors: list[CorrectionFactors], config: ProbeConfig) -> int:
    tensors: dict[str, torch.Tensor] = {}
    for factor in factors:
        tensors[f"{factor.path}.correction_down.weight"] = factor.down.contiguous()
        tensors[f"{factor.path}.correction_up.weight"] = factor.up.contiguous()
    config.weights_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        tensors,
        str(config.weights_path),
        metadata={
            "format": "krea2_nf4_fixed_correction_v1",
            "rank": str(config.rank),
            "input_rank": str(config.input_rank),
            "sigma": str(CASES[0].sigma),
            "noise_seed": str(CASES[0].noise_seed),
        },
    )
    return config.weights_path.stat().st_size


class FixedCorrection(torch.nn.Module):
    def __init__(self, down: torch.Tensor, up: torch.Tensor) -> None:
        super().__init__()
        self.down = torch.nn.Linear(down.shape[1], down.shape[0], bias=False)
        self.up = torch.nn.Linear(up.shape[1], up.shape[0], bias=False)
        self.down.weight = torch.nn.Parameter(down, requires_grad=False)
        self.up.weight = torch.nn.Parameter(up, requires_grad=False)

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(activation))


class FixedCorrectionBank(torch.nn.Module):
    def __init__(self, factors: list[CorrectionFactors]) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleDict()
        self.path_to_key: dict[str, str] = {}
        self.handles: list[Any] = []
        self.enabled = True
        for index, factor in enumerate(factors):
            key = f"layer_{index:03d}"
            self.path_to_key[factor.path] = key
            self.layers[key] = FixedCorrection(factor.down, factor.up)

    def attach(self, dit: torch.nn.Module) -> None:
        modules = dict(dit.named_modules())
        for path, key in self.path_to_key.items():
            correction = self.layers[key]

            def hook(_module, inputs, output, correction=correction):
                if not self.enabled:
                    return output
                return output + correction(inputs[0]).to(output.dtype)

            self.handles.append(modules[path].register_forward_hook(hook))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _metric(prediction: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    pred = prediction.float().cpu()
    ref = reference.float().cpu()
    delta = pred - ref
    return {
        "relative_l2": delta.norm().item() / ref.norm().item(),
        "max_delta": delta.abs().max().item(),
        "cosine": F.cosine_similarity(pred.flatten(), ref.flatten(), dim=0).item(),
    }


def _reference_forwards(
    dit: torch.nn.Module,
    latents_4d: torch.Tensor,
    text_embedding: Krea2TextEmbedding,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    references: dict[str, torch.Tensor] = {}
    for spec in CASES[1:]:
        x_t, sigma, _ = _case_tensors(latents_4d, spec, device, dtype)
        with torch.inference_mode():
            references[spec.name] = forward_for_loss(
                dit, x_t, text_embedding, sigma
            ).detach().cpu()
    return references


def _evaluate_cases(
    dit: torch.nn.Module,
    latents_4d: torch.Tensor,
    text_embedding: Krea2TextEmbedding,
    references: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for spec in CASES:
        x_t, sigma, _ = _case_tensors(latents_4d, spec, device, dtype)
        with torch.inference_mode():
            prediction = forward_for_loss(dit, x_t, text_embedding, sigma)
        results[spec.name] = _metric(prediction, references[spec.name])
    return results


def _benchmark_training(
    dit: torch.nn.Module,
    network: torch.nn.Module,
    latents_4d: torch.Tensor,
    text_embedding: Krea2TextEmbedding,
    config: ProbeConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    spec = CASES[0]
    x_t, sigma, target = _case_tensors(latents_4d, spec, device, dtype)
    optimizer = torch.optim.AdamW(network.parameters(), lr=2e-3, weight_decay=0.0)
    losses: list[float] = []
    times: list[float] = []
    for step in range(config.warmup + config.steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.time()
        velocity = forward_for_loss(dit, x_t, text_embedding, sigma)
        loss = F.mse_loss(velocity, target)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        if step == config.warmup - 1:
            torch.cuda.reset_peak_memory_stats()
        if step >= config.warmup:
            losses.append(loss.item())
            times.append(time.time() - started)
    result = {
        "steps": config.steps,
        "warmup": config.warmup,
        "mean_step_seconds": statistics.mean(times),
        "median_step_seconds": statistics.median(times),
        "step_seconds": times,
        "losses": losses,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
    }
    del optimizer
    return result


def _fit_phase(
    config: ProbeConfig,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[CorrectionFactors], dict[str, torch.Tensor], dict[str, Any], list[dict[str, Any]]]:
    hiddens, text_mask, latents_4d = inputs
    text_embedding = Krea2TextEmbedding(hiddens.to(device), text_mask.to(device))
    print("--- BF16 calibration and references ---")
    bf16_dit = load_krea2_dit(config.dit_path, device=device, dtype=dtype, eval=True)
    for parameter in bf16_dit.parameters():
        parameter.requires_grad_(False)
    reader = NF4WeightReader(config.nf4_path)
    calibration = CASES[0]
    x_t, sigma, _ = _case_tensors(latents_4d, calibration, device, dtype)
    calibration_ref, factors, fit_metrics = _fit_corrections(
        bf16_dit, reader, x_t, text_embedding, sigma, config
    )
    references = {calibration.name: calibration_ref}
    references.update(
        _reference_forwards(bf16_dit, latents_4d, text_embedding, device, dtype)
    )
    weights_bytes = _save_factors(factors, config)
    fit_error = sum(item["error_energy"] for item in fit_metrics)
    fit_residual = sum(item["residual_energy"] for item in fit_metrics)
    fit_summary = {
        "captured_energy": 1.0 - fit_residual / fit_error,
        "residual_l2_factor": math.sqrt(fit_residual / fit_error),
        "median_layer_capture": statistics.median(
            item["captured_energy"] for item in fit_metrics
        ),
        "weights_mb": weights_bytes / 1e6,
    }
    del bf16_dit, x_t, sigma, text_embedding
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return factors, references, fit_summary, fit_metrics


def _nf4_phase(
    config: ProbeConfig,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    factors: list[CorrectionFactors],
    references: dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    hiddens, text_mask, latents_4d = inputs
    print("--- NF4 baseline + LoRA benchmark ---")
    nf4_dit = load_krea2_dit(
        config.dit_path,
        device="cpu",
        dtype=dtype,
        eval=False,
        nf4_path=config.nf4_path,
    )
    for parameter in nf4_dit.parameters():
        parameter.requires_grad_(False)
    network = make_network(nf4_dit)
    nf4_dit = nf4_dit.to(device)
    network = network.to(device).to(dtype)
    text_embedding = Krea2TextEmbedding(hiddens.to(device), text_mask.to(device))
    initial_lora = {
        key: value.detach().cpu().clone() for key, value in network.state_dict().items()
    }
    nf4_dit.disable_gradient_checkpointing()
    baseline_eval = _evaluate_cases(
        nf4_dit, latents_4d, text_embedding, references, device, dtype
    )
    nf4_dit.enable_gradient_checkpointing()
    baseline_train = _benchmark_training(
        nf4_dit, network, latents_4d, text_embedding, config, device, dtype
    )

    network.load_state_dict(initial_lora)
    del initial_lora
    gc.collect()
    torch.cuda.empty_cache()
    bank = FixedCorrectionBank(factors).to(device).to(dtype)
    bank.attach(nf4_dit)
    nf4_dit.disable_gradient_checkpointing()
    corrected_eval = _evaluate_cases(
        nf4_dit, latents_4d, text_embedding, references, device, dtype
    )
    nf4_dit.enable_gradient_checkpointing()
    corrected_train = _benchmark_training(
        nf4_dit, network, latents_4d, text_embedding, config, device, dtype
    )
    bank.remove()
    return baseline_eval, corrected_eval, baseline_train, corrected_train


def main() -> int:
    config = _load_config()
    device = torch.device("cuda")
    dtype = torch.bfloat16
    started = time.time()
    print(
        f"=== Krea-2 NF4 correction PG199: image={config.image_size}, "
        f"rank={config.rank}, input_rank={config.input_rank} ==="
    )
    inputs = _encode_inputs(
        image_size=config.image_size,
        device=device,
        dtype=dtype,
        te_path=config.te_path,
        vae_path=config.vae_path,
        te_on_cpu=False,
    )
    factors, references, fit_summary, fit_metrics = _fit_phase(
        config, inputs, device, dtype
    )
    baseline_eval, corrected_eval, baseline_train, corrected_train = _nf4_phase(
        config, inputs, factors, references, device, dtype
    )

    report = {
        "config": {
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "image_size": config.image_size,
            "rank": config.rank,
            "input_rank": config.input_rank,
            "niter": config.niter,
            "cases": [spec.__dict__ for spec in CASES],
        },
        "fit": {"summary": fit_summary, "layers": fit_metrics},
        "evaluation": {"baseline": baseline_eval, "corrected": corrected_eval},
        "training": {"baseline": baseline_train, "corrected": corrected_train},
        "delta": {
            "peak_gpu_gb": corrected_train["peak_gpu_gb"] - baseline_train["peak_gpu_gb"],
            "mean_step_percent": 100
            * (
                corrected_train["mean_step_seconds"]
                / baseline_train["mean_step_seconds"]
                - 1.0
            ),
        },
        "runtime_seconds": time.time() - started,
        "weights_path": str(config.weights_path),
    }
    config.out_path.parent.mkdir(parents=True, exist_ok=True)
    config.out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    print("\n=== result ===")
    print(json.dumps({"fit": fit_summary, "evaluation": report["evaluation"], "training": report["training"], "delta": report["delta"]}, indent=2))
    print(f"result={config.out_path}, weights={config.weights_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
