#!/usr/bin/env python3
"""Fit fixed NF4 correction factors against final Krea-2 velocity error.

The local rank-16 factors are used as the fixed-budget initialization.  This
probe then optimizes exactly the same parameters against BF16 teacher outputs,
which includes downstream Jacobians, cross-layer terms, and trajectory drift.
Teacher references are cached so the memory-heavy BF16 phase can be resumed.
"""
from __future__ import annotations

import gc
import json
import math
import os
import random
import re
import statistics
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("K2_DS_GPU", "1"))

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from library.anima.strategy import AnimaLatentsCachingStrategy  # noqa: E402
from library.models.krea2_raw.attention_backend import (  # noqa: E402
    prepare_krea2_attention,
)
from library.models.krea2_raw.family import (  # noqa: E402
    Krea2TextEmbedding,
    forward_for_loss,
)
from library.models.krea2_raw.weights import load_krea2_dit  # noqa: E402


DEFAULT_DATA = ROOT / "post_image_dataset" / "resized"
DEFAULT_CACHE = ROOT / "post_image_dataset" / "lora"
DEFAULT_DIT = ROOT / "models" / "diffusion_models" / "krea2_raw_bf16.safetensors"
DEFAULT_NF4 = (
    ROOT / "models" / "diffusion_models" / "krea2_raw_nf4_self_contained.safetensors"
)
DEFAULT_LOCAL = (
    ROOT / "output" / "tests" / "krea2_nf4_correction_rank16_pg199_1024.safetensors"
)
DEFAULT_RUN = ROOT / "output" / "tests" / "krea2_nf4_downstream_90hx"
LATENT_RE = re.compile(r"_(\d{4})x(\d{4})_anima\.npz$")


@dataclass(frozen=True)
class ProbeConfig:
    data_dir: Path
    cache_dir: Path
    dit_path: Path
    nf4_path: Path
    local_factors_path: Path
    run_dir: Path
    phase: str
    calibration_count: int
    heldout_count: int
    split_seed: int
    blocks_to_swap_teacher: int
    blocks_to_swap_nf4: int
    epochs: int
    learning_rate: float
    attention_mode: str
    compile_blocks: bool
    max_grad_norm: float


@dataclass(frozen=True)
class SampleSpec:
    index: int
    stem: str
    split: str
    latent_path: str
    te_path: str
    width: int
    height: int
    sigma: float
    noise_seed: int
    reference_path: str


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_config() -> ProbeConfig:
    phase = os.environ.get("K2_DS_PHASE", "all").strip().lower()
    if phase not in {"teacher", "optimize", "control", "all"}:
        raise ValueError(
            f"K2_DS_PHASE must be teacher/optimize/control/all, got {phase!r}"
        )
    config = ProbeConfig(
        data_dir=Path(os.environ.get("K2_DS_DATA", DEFAULT_DATA)),
        cache_dir=Path(os.environ.get("K2_DS_CACHE", DEFAULT_CACHE)),
        dit_path=Path(os.environ.get("K2_DS_DIT", DEFAULT_DIT)),
        nf4_path=Path(os.environ.get("K2_DS_NF4", DEFAULT_NF4)),
        local_factors_path=Path(os.environ.get("K2_DS_LOCAL", DEFAULT_LOCAL)),
        run_dir=Path(os.environ.get("K2_DS_RUN", DEFAULT_RUN)),
        phase=phase,
        calibration_count=_env_int("K2_DS_CALIBRATION", 32),
        heldout_count=_env_int("K2_DS_HELDOUT", 16),
        split_seed=_env_int("K2_DS_SEED", 20260812),
        blocks_to_swap_teacher=_env_int("K2_DS_TEACHER_SWAP", 26),
        blocks_to_swap_nf4=_env_int("K2_DS_NF4_SWAP", 20),
        epochs=_env_int("K2_DS_EPOCHS", 2),
        learning_rate=_env_float("K2_DS_LR", 2e-5),
        attention_mode=os.environ.get("K2_DS_ATTN", "flash"),
        compile_blocks=_env_bool("K2_DS_COMPILE", True),
        max_grad_norm=_env_float("K2_DS_MAX_GRAD_NORM", 1.0),
    )
    if config.calibration_count < 1 or config.heldout_count < 1:
        raise ValueError("calibration and heldout counts must both be positive")
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    for path in (config.data_dir, config.cache_dir, config.dit_path, config.nf4_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if phase != "teacher" and not config.local_factors_path.exists():
        raise FileNotFoundError(config.local_factors_path)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    (config.run_dir / "references").mkdir(parents=True, exist_ok=True)
    return config


def _discover_samples(config: ProbeConfig) -> list[SampleSpec]:
    pairs: list[tuple[str, Path, Path, int, int]] = []
    for te_path in sorted(config.cache_dir.glob("*_krea2_te.safetensors")):
        stem = te_path.name[: -len("_krea2_te.safetensors")]
        latent_paths = sorted(config.cache_dir.glob(f"{stem}_*x*_anima.npz"))
        if len(latent_paths) != 1:
            raise RuntimeError(f"expected one latent cache for {stem!r}, got {latent_paths}")
        match = LATENT_RE.search(latent_paths[0].name)
        if match is None:
            raise RuntimeError(f"cannot parse bucket resolution: {latent_paths[0]}")
        width, height = (int(match.group(1)), int(match.group(2)))
        pairs.append((stem, latent_paths[0], te_path, width, height))

    total = config.calibration_count + config.heldout_count
    if len(pairs) < total:
        raise RuntimeError(f"need {total} paired caches, found {len(pairs)}")
    random.Random(config.split_seed).shuffle(pairs)
    pairs = pairs[:total]

    sigma_generator = torch.Generator(device="cpu").manual_seed(config.split_seed + 1)
    sigmas = torch.sigmoid(torch.randn(total, generator=sigma_generator)).tolist()
    samples: list[SampleSpec] = []
    for index, (stem, latent_path, te_path, width, height) in enumerate(pairs):
        split = "calibration" if index < config.calibration_count else "heldout"
        reference_path = config.run_dir / "references" / f"{index:03d}.safetensors"
        samples.append(
            SampleSpec(
                index=index,
                stem=stem,
                split=split,
                latent_path=str(latent_path),
                te_path=str(te_path),
                width=width,
                height=height,
                sigma=float(sigmas[index]),
                noise_seed=config.split_seed + 10_000 + index,
                reference_path=str(reference_path),
            )
        )
    return samples


def _load_conditions(sample: SampleSpec) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    strategy = AnimaLatentsCachingStrategy(
        cache_to_disk=True,
        batch_size=1,
        skip_disk_cache_validity_check=True,
    )
    latents_np, _, _, _, _ = strategy.load_latents_from_disk(
        sample.latent_path, (sample.width, sample.height)
    )
    if latents_np is None:
        raise RuntimeError(f"missing latents in {sample.latent_path}")
    with safe_open(sample.te_path, framework="pt") as handle:
        hiddens = handle.get_tensor("hiddens")
        mask = handle.get_tensor("mask")
    return torch.from_numpy(np.asarray(latents_np)), hiddens, mask


def _make_model_inputs(
    sample: SampleSpec,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, Krea2TextEmbedding, torch.Tensor]:
    latents, hiddens, mask = _load_conditions(sample)
    latents = latents.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(2)
    generator = torch.Generator(device=device).manual_seed(sample.noise_seed)
    noise = torch.randn(latents.shape, generator=generator, device=device, dtype=dtype)
    sigma = torch.full((1,), sample.sigma, device=device, dtype=dtype)
    noisy = (1.0 - sigma) * latents + sigma * noise
    text = Krea2TextEmbedding(
        hiddens.to(device=device, dtype=dtype).unsqueeze(0),
        mask.to(device=device).unsqueeze(0),
    )
    return noisy, text, sigma


def _reference_is_valid(sample: SampleSpec) -> bool:
    path = Path(sample.reference_path)
    if not path.exists():
        return False
    try:
        with safe_open(path, framework="pt") as handle:
            metadata = handle.metadata() or {}
            return (
                metadata.get("stem") == sample.stem
                and metadata.get("sigma") == repr(sample.sigma)
                and metadata.get("noise_seed") == str(sample.noise_seed)
                and "velocity" in handle.keys()
            )
    except Exception:
        return False


def _disable_exact_teacher_masters(dit: torch.nn.Module) -> None:
    """Keep BF16 block swap exact without duplicating 26 GB of CPU weights."""
    offloader = dit.offloader

    def no_cpu_masters(_self, _blocks) -> None:
        return None

    offloader._ensure_cpu_weight_masters = types.MethodType(no_cpu_masters, offloader)


def _teacher_phase(
    config: ProbeConfig,
    samples: list[SampleSpec],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    missing = [sample for sample in samples if not _reference_is_valid(sample)]
    if not missing:
        return {"generated": 0, "reused": len(samples), "peak_gpu_gb": 0.0}

    print(f"--- BF16 teacher: generating {len(missing)}, reusing {len(samples)-len(missing)} ---")
    dit = load_krea2_dit(config.dit_path, device="cpu", dtype=dtype, eval=True)
    prepare_krea2_attention(
        dit, config.attention_mode, dtype=dtype, compile_enabled=False
    )
    for parameter in dit.parameters():
        parameter.requires_grad_(False)
    dit.enable_block_swap(config.blocks_to_swap_teacher, device)
    _disable_exact_teacher_masters(dit)
    dit.move_to_device_except_swap_blocks(device)
    dit.switch_block_swap_for_inference()

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for position, sample in enumerate(missing, 1):
        noisy, text, sigma = _make_model_inputs(sample, device, dtype)
        dit.prepare_block_swap_before_forward()
        with torch.inference_mode():
            velocity = forward_for_loss(dit, noisy, text, sigma).detach().cpu()
        save_file(
            {"velocity": velocity.contiguous()},
            sample.reference_path,
            metadata={
                "format": "krea2_bf16_velocity_reference_v1",
                "stem": sample.stem,
                "sigma": repr(sample.sigma),
                "noise_seed": str(sample.noise_seed),
                "width": str(sample.width),
                "height": str(sample.height),
            },
        )
        del noisy, text, sigma, velocity
        if position == 1 or position % 4 == 0 or position == len(missing):
            print(f"  teacher {position:3d}/{len(missing)}", flush=True)

    peak = torch.cuda.max_memory_allocated() / 1e9
    elapsed = time.time() - started
    del dit
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "generated": len(missing),
        "reused": len(samples) - len(missing),
        "seconds": elapsed,
        "peak_gpu_gb": peak,
    }


class Correction(torch.nn.Module):
    def __init__(self, down: torch.Tensor, up: torch.Tensor) -> None:
        super().__init__()
        self.down = torch.nn.Parameter(down.clone())
        self.up = torch.nn.Parameter(up.clone())

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        return F.linear(F.linear(activation, self.down), self.up)


class CorrectionBank(torch.nn.Module):
    def __init__(self, factor_path: Path) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleDict()
        self.paths: list[str] = []
        self.handles: list[Any] = []
        self.enabled = True
        with safe_open(factor_path, framework="pt") as handle:
            keys = set(handle.keys())
            suffix = ".correction_down.weight"
            paths = sorted(key[: -len(suffix)] for key in keys if key.endswith(suffix))
            for index, path in enumerate(paths):
                down = handle.get_tensor(f"{path}.correction_down.weight")
                up = handle.get_tensor(f"{path}.correction_up.weight")
                self.layers[f"layer_{index:03d}"] = Correction(down, up)
                self.paths.append(path)
        if len(self.paths) != 196:
            raise RuntimeError(f"expected 196 correction layers, got {len(self.paths)}")

    def attach(self, dit: torch.nn.Module) -> None:
        modules = dict(dit.named_modules())
        for index, path in enumerate(self.paths):
            correction = self.layers[f"layer_{index:03d}"]

            def hook(_module, inputs, output, correction=correction):
                if not self.enabled:
                    return output
                return output + correction(inputs[0]).to(output.dtype)

            self.handles.append(modules[path].register_forward_hook(hook))

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def save(self, path: Path, metadata: dict[str, str]) -> None:
        tensors: dict[str, torch.Tensor] = {}
        for index, layer_path in enumerate(self.paths):
            correction = self.layers[f"layer_{index:03d}"]
            tensors[f"{layer_path}.correction_down.weight"] = correction.down.detach().cpu()
            tensors[f"{layer_path}.correction_up.weight"] = correction.up.detach().cpu()
        save_file(tensors, str(path), metadata=metadata)


def _load_reference(sample: SampleSpec, device: torch.device) -> torch.Tensor:
    with safe_open(sample.reference_path, framework="pt") as handle:
        return handle.get_tensor("velocity").to(device)


def _metric(prediction: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    pred = prediction.float()
    ref = reference.float()
    delta = pred - ref
    return {
        "relative_l2": delta.norm().item() / ref.norm().item(),
        "mse": delta.square().mean().item(),
        "cosine": F.cosine_similarity(pred.flatten(), ref.flatten(), dim=0).item(),
        "max_delta": delta.abs().max().item(),
    }


def _aggregate(metrics: list[dict[str, float]]) -> dict[str, float]:
    return {
        f"{name}_{stat}": value
        for name in ("relative_l2", "mse", "cosine", "max_delta")
        for stat, value in (
            ("mean", statistics.mean(item[name] for item in metrics)),
            ("median", statistics.median(item[name] for item in metrics)),
        )
    }


def _evaluate(
    dit: torch.nn.Module,
    bank: CorrectionBank,
    samples: list[SampleSpec],
    device: torch.device,
    dtype: torch.dtype,
    *,
    enabled: bool,
) -> dict[str, Any]:
    bank.enabled = enabled
    by_split: dict[str, list[dict[str, float]]] = {"calibration": [], "heldout": []}
    per_sample: list[dict[str, Any]] = []
    started = time.time()
    for position, sample in enumerate(samples, 1):
        noisy, text, sigma = _make_model_inputs(sample, device, dtype)
        reference = _load_reference(sample, device)
        dit.prepare_block_swap_before_forward()
        with torch.no_grad():
            prediction = forward_for_loss(dit, noisy, text, sigma)
        metric = _metric(prediction, reference)
        by_split[sample.split].append(metric)
        per_sample.append({"index": sample.index, "split": sample.split, **metric})
        del noisy, text, sigma, reference, prediction
        if position % 8 == 0 or position == len(samples):
            print(f"  evaluate {position:3d}/{len(samples)} enabled={enabled}", flush=True)
    return {
        "calibration": _aggregate(by_split["calibration"]),
        "heldout": _aggregate(by_split["heldout"]),
        "per_sample": per_sample,
        "seconds": time.time() - started,
    }


def _relative_mse(prediction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    pred = prediction.float()
    ref = reference.float()
    return (pred - ref).square().mean() / ref.square().mean().clamp_min(1e-12)


def _optimize(
    dit: torch.nn.Module,
    bank: CorrectionBank,
    calibration: list[SampleSpec],
    config: ProbeConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    bank.enabled = True
    optimizer = torch.optim.AdamW(
        bank.parameters(), lr=config.learning_rate, weight_decay=0.0
    )
    order_generator = random.Random(config.split_seed + 2)
    losses: list[float] = []
    grad_norms: list[float] = []
    times: list[float] = []
    torch.cuda.reset_peak_memory_stats()
    started = time.time()

    for epoch in range(config.epochs):
        epoch_samples = list(calibration)
        order_generator.shuffle(epoch_samples)
        for position, sample in enumerate(epoch_samples, 1):
            noisy, text, sigma = _make_model_inputs(sample, device, dtype)
            noisy.requires_grad_(True)
            reference = _load_reference(sample, device)
            optimizer.zero_grad(set_to_none=True)
            dit.prepare_block_swap_before_forward()
            torch.cuda.synchronize()
            step_started = time.time()
            prediction = forward_for_loss(dit, noisy, text, sigma)
            loss = _relative_mse(prediction, reference)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                bank.parameters(), config.max_grad_norm
            )
            optimizer.step()
            torch.cuda.synchronize()
            losses.append(float(loss.detach()))
            grad_norms.append(float(grad_norm))
            times.append(time.time() - step_started)
            del noisy, text, sigma, reference, prediction, loss
            print(
                f"  train epoch={epoch+1}/{config.epochs} "
                f"step={position:02d}/{len(epoch_samples)} "
                f"loss={losses[-1]:.6f} grad={grad_norms[-1]:.4f} "
                f"sec={times[-1]:.2f}",
                flush=True,
            )

    result = {
        "steps": len(losses),
        "losses": losses,
        "grad_norms": grad_norms,
        "step_seconds": times,
        "mean_step_seconds": statistics.mean(times),
        "median_step_seconds": statistics.median(times),
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
        "seconds": time.time() - started,
    }
    del optimizer
    return result


def _improvement(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for split in ("calibration", "heldout"):
        base = before[split]["relative_l2_mean"]
        corrected = after[split]["relative_l2_mean"]
        result[f"{split}_relative_l2_percent"] = 100.0 * (1.0 - corrected / base)
    return result


def _nf4_phase(
    config: ProbeConfig,
    samples: list[SampleSpec],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    for sample in samples:
        if not _reference_is_valid(sample):
            raise RuntimeError(f"missing or stale teacher reference: {sample.reference_path}")
    print("--- NF4 baseline, local correction, and downstream optimization ---")
    dit = load_krea2_dit(
        config.dit_path,
        device="cpu",
        dtype=dtype,
        eval=False,
        nf4_path=config.nf4_path,
    )
    prepare_krea2_attention(
        dit,
        config.attention_mode,
        dtype=dtype,
        compile_enabled=config.compile_blocks,
    )
    for parameter in dit.parameters():
        parameter.requires_grad_(False)
    bank = CorrectionBank(config.local_factors_path)
    bank.attach(dit)

    dit.enable_block_swap(config.blocks_to_swap_nf4, device)
    dit.move_to_device_except_swap_blocks(device)
    bank = bank.to(device=device, dtype=dtype)
    dit.switch_block_swap_for_training()
    dit.disable_gradient_checkpointing()

    if config.phase == "control":
        dit.enable_gradient_checkpointing()
        if config.compile_blocks:
            dit.compile_blocks(backend="inductor", compile_block_scope="resident")
        compiled_local = _evaluate(dit, bank, samples, device, dtype, enabled=True)
        bank.remove()
        return {"compiled_local": compiled_local}

    baseline = _evaluate(dit, bank, samples, device, dtype, enabled=False)
    local = _evaluate(dit, bank, samples, device, dtype, enabled=True)

    dit.enable_gradient_checkpointing()
    if config.compile_blocks:
        dit.compile_blocks(backend="inductor", compile_block_scope="resident")
    calibration = [sample for sample in samples if sample.split == "calibration"]
    training = _optimize(dit, bank, calibration, config, device, dtype)
    optimized_path = config.run_dir / "downstream_correction.safetensors"
    bank.save(
        optimized_path,
        metadata={
            "format": "krea2_nf4_downstream_correction_v1",
            "source": str(config.local_factors_path),
            "calibration_count": str(config.calibration_count),
            "heldout_count": str(config.heldout_count),
            "epochs": str(config.epochs),
            "learning_rate": repr(config.learning_rate),
        },
    )
    dit.disable_gradient_checkpointing()
    downstream = _evaluate(dit, bank, samples, device, dtype, enabled=True)
    bank.remove()
    return {
        "baseline": baseline,
        "local": local,
        "downstream": downstream,
        "improvement_vs_baseline": {
            "local": _improvement(baseline, local),
            "downstream": _improvement(baseline, downstream),
        },
        "improvement_downstream_vs_local": _improvement(local, downstream),
        "training": training,
        "optimized_path": str(optimized_path),
        "optimized_mb": optimized_path.stat().st_size / 1e6,
    }


def main() -> int:
    config = _load_config()
    samples = _discover_samples(config)
    manifest_path = config.run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps([asdict(sample) for sample in samples], ensure_ascii=False, indent=2)
        + "\n"
    )
    device = torch.device("cuda")
    dtype = torch.bfloat16
    print(
        f"=== Krea-2 downstream correction: GPU={torch.cuda.get_device_name()} "
        f"phase={config.phase} cal={config.calibration_count} "
        f"heldout={config.heldout_count} epochs={config.epochs} ==="
    )
    started = time.time()
    report: dict[str, Any] = {
        "config": {
            **asdict(config),
            "data_dir": str(config.data_dir),
            "cache_dir": str(config.cache_dir),
            "dit_path": str(config.dit_path),
            "nf4_path": str(config.nf4_path),
            "local_factors_path": str(config.local_factors_path),
            "run_dir": str(config.run_dir),
        },
        "gpu": torch.cuda.get_device_name(),
        "samples": [asdict(sample) for sample in samples],
    }
    if config.phase in {"teacher", "all"}:
        report["teacher"] = _teacher_phase(config, samples, device, dtype)
    if config.phase in {"optimize", "control", "all"}:
        report["comparison"] = _nf4_phase(config, samples, device, dtype)
    report["runtime_seconds"] = time.time() - started
    report_path = config.run_dir / f"report_{config.phase}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"samples"}}, indent=2, default=str))
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
