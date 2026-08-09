"""Measure per-layer spectra of the Krea-2 NF4 activation error ``E X``.

For each LoRA-targeted Linear, this probe runs the BF16 DiT once and compares
the layer's normal BF16 output with a dequantized NF4 weight on the exact same
input activation.  This isolates ``(W - Q_nf4(W)) X`` from upstream trajectory
drift.  Only one NF4 layer is materialized at a time.

Environment variables:
  K2_EX_GPU=1                 CUDA device index (default: PG199 at index 1)
  K2_EX_IMG=1024              synthetic calibration image size
  K2_EX_SIGMA=0.5             flow-matching sigma
  K2_EX_RANKS=4,8,16,32,64   reported cumulative energy ranks
  K2_EX_OVERSAMPLE=16         randomized SVD oversampling
  K2_EX_NITER=2               randomized SVD power iterations
  K2_EX_EXACT_MAX=128         exact SVD when min(matrix shape) is at most this
  K2_EX_TE_CPU=0              run Qwen text encoder on CPU when set
  K2_EX_OUT=path              JSON output path
  K2_EX_DIT=path              BF16 DiT safetensors path
  K2_EX_NF4=path              prequantized NF4 safetensors path
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("K2_EX_GPU", "1"))

import json
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from bitsandbytes.functional import dequantize_4bit
from bitsandbytes.nn import Params4bit
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from library.models.krea2_raw.family import Krea2TextEmbedding, forward_for_loss  # noqa: E402
from library.models.krea2_raw.strategy import (  # noqa: E402
    Krea2TextEncodingStrategy,
    Krea2TokenizeStrategy,
    load_krea2_text_encoder,
)
from library.models.krea2_raw.weights import load_krea2_dit  # noqa: E402
from library.models.qwen_vae import load_vae  # noqa: E402
from probe_train import PROMPT, make_test_image  # noqa: E402

DEFAULT_DIT = ROOT / "models" / "diffusion_models" / "krea2_raw_bf16.safetensors"
DEFAULT_NF4 = ROOT / "models" / "diffusion_models" / "krea2_raw_nf4.safetensors"
DEFAULT_TE = ROOT / "models" / "text_encoders" / "qwen3vl_4b_bf16.safetensors"
DEFAULT_VAE = ROOT / "models" / "vae" / "qwen_image_vae.safetensors"
DEFAULT_OUT = ROOT / "output" / "tests" / "krea2_nf4_ex_spectrum.json"

TARGET_RE = re.compile(
    r"^blocks\.(\d+)\.(attn\.(?:wq|wk|wv|wo)|mlp\.(?:up|down|gate))$"
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


def _env_ranks() -> tuple[int, ...]:
    raw = os.environ.get("K2_EX_RANKS", "4,8,16,32,64")
    ranks = tuple(sorted({int(value) for value in raw.split(",") if value.strip()}))
    if not ranks or ranks[0] <= 0:
        raise ValueError(f"invalid K2_EX_RANKS={raw!r}")
    return ranks


def _target_modules(dit: torch.nn.Module) -> list[tuple[str, torch.nn.Linear]]:
    targets = [
        (name, module)
        for name, module in dit.named_modules()
        if TARGET_RE.fullmatch(name) and isinstance(module, torch.nn.Linear)
    ]
    if len(targets) != 196:
        names = [name for name, _ in targets[:10]]
        raise RuntimeError(f"expected 196 Krea-2 target Linear modules, got {len(targets)}: {names}")
    return targets


def _spectrum_summary(
    matrix: torch.Tensor,
    *,
    ranks: tuple[int, ...],
    oversample: int,
    niter: int,
    exact_max: int,
    seed: int,
) -> dict[str, Any]:
    """Return top spectrum and exact Frobenius-energy coverage for a 2D matrix."""
    if matrix.ndim != 2:
        raise ValueError(f"expected a matrix, got shape={tuple(matrix.shape)}")
    matrix = matrix.float()
    min_dim = min(matrix.shape)
    max_rank = min(max(ranks), min_dim)
    total_energy = matrix.square().sum(dtype=torch.float64).item()
    if min_dim <= exact_max:
        singular_values = torch.linalg.svdvals(matrix)
        method = "exact"
        sketch_rank = min_dim
    else:
        sketch_rank = min(min_dim, max_rank + max(oversample, 0))
        devices = [matrix.device.index or 0] if matrix.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            _, singular_values, _ = torch.svd_lowrank(
                matrix, q=sketch_rank, niter=niter
            )
        singular_values = singular_values.sort(descending=True).values
        method = "randomized"

    squared = singular_values.double().square()
    cumulative = squared.cumsum(0)
    captured: dict[str, float] = {}
    residual_factor: dict[str, float] = {}
    for rank in ranks:
        used = min(rank, singular_values.numel())
        energy = cumulative[used - 1].item() if used and total_energy else 0.0
        ratio = min(max(energy / total_energy, 0.0), 1.0) if total_energy else 1.0
        captured[str(rank)] = ratio
        residual_factor[str(rank)] = math.sqrt(max(1.0 - ratio, 0.0))

    return {
        "method": method,
        "sketch_rank": sketch_rank,
        "total_energy": total_energy,
        "captured_energy": captured,
        "residual_l2_factor": residual_factor,
        "singular_values": singular_values.detach().cpu().double().tolist(),
    }


class NF4WeightReader:
    """Read and dequantize one saved Params4bit weight at a time."""

    def __init__(self, path: Path) -> None:
        self._handle = safe_open(str(path), framework="pt", device="cpu")
        self._keys = tuple(self._handle.keys())

    def dequantize(self, module_path: str, device: torch.device) -> torch.Tensor:
        weight_key = f"{module_path}.weight"
        prefix = f"{weight_key}.quant_state."
        stats = {
            key[len(prefix) :]: self._handle.get_tensor(key)
            for key in self._keys
            if key.startswith(prefix)
        }
        if weight_key not in self._keys or not stats:
            raise KeyError(f"missing saved NF4 state for {module_path}")
        params = Params4bit.from_prequantized(
            self._handle.get_tensor(weight_key),
            stats,
            requires_grad=False,
            device=device,
        )
        return dequantize_4bit(params.data, params.quant_state)


def _encode_inputs(
    *,
    image_size: int,
    device: torch.device,
    dtype: torch.dtype,
    te_path: Path,
    vae_path: Path,
    te_on_cpu: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    te_device = "cpu" if te_on_cpu else str(device)
    te_model, _ = load_krea2_text_encoder(str(te_path), dtype=dtype, device=te_device)
    tokenizer = Krea2TokenizeStrategy()
    tokens = tokenizer.tokenize([PROMPT])
    encoder = Krea2TextEncodingStrategy()
    with torch.inference_mode():
        [hiddens, text_mask] = encoder.encode_tokens(tokenizer, [te_model], tokens)
    hiddens = hiddens.cpu()
    text_mask = text_mask.cpu()
    del te_model, tokenizer, tokens, encoder
    torch.cuda.empty_cache()

    vae = load_vae(str(vae_path), device=device, dtype=dtype, eval=True)
    pixels = make_test_image(image_size, device, dtype)
    with torch.inference_mode():
        latents = vae.encode_pixels_to_latents(pixels).cpu()
    del vae, pixels
    torch.cuda.empty_cache()
    return hiddens, text_mask, latents


def _aggregate_layers(layers: list[dict[str, Any]], ranks: tuple[int, ...]) -> dict[str, Any]:
    total_energy = sum(layer["total_energy"] for layer in layers)
    aggregate: dict[str, Any] = {"all": {}, "by_kind": {}}

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        group_energy = sum(layer["total_energy"] for layer in group)
        result: dict[str, Any] = {"layers": len(group), "total_energy": group_energy}
        for rank in ranks:
            key = str(rank)
            captured = sum(
                layer["total_energy"] * layer["captured_energy"][key]
                for layer in group
            )
            values = sorted(layer["captured_energy"][key] for layer in group)
            result[f"rank_{rank}"] = {
                "energy_weighted": captured / group_energy if group_energy else 1.0,
                "median_layer": values[len(values) // 2] if values else 1.0,
                "layers_ge_50pct": sum(value >= 0.5 for value in values),
                "layers_ge_75pct": sum(value >= 0.75 for value in values),
                "layers_ge_90pct": sum(value >= 0.9 for value in values),
            }
        return result

    aggregate["all"] = summarize(layers)
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for layer in layers:
        by_kind[layer["kind"]].append(layer)
    aggregate["by_kind"] = {
        kind: summarize(group) for kind, group in sorted(by_kind.items())
    }
    aggregate["all"]["energy_fraction"] = 1.0 if total_energy else 0.0
    return aggregate


@dataclass(frozen=True)
class ProbeConfig:
    image_size: int
    sigma: float
    ranks: tuple[int, ...]
    oversample: int
    niter: int
    exact_max: int
    seed: int
    dit_path: Path
    nf4_path: Path
    out_path: Path
    te_path: Path
    vae_path: Path
    te_on_cpu: bool


def _load_config() -> ProbeConfig:
    config = ProbeConfig(
        image_size=_env_int("K2_EX_IMG", 1024),
        sigma=_env_float("K2_EX_SIGMA", 0.5),
        ranks=_env_ranks(),
        oversample=_env_int("K2_EX_OVERSAMPLE", 16),
        niter=_env_int("K2_EX_NITER", 2),
        exact_max=_env_int("K2_EX_EXACT_MAX", 128),
        seed=_env_int("K2_EX_SEED", 123),
        dit_path=Path(os.environ.get("K2_EX_DIT", str(DEFAULT_DIT))),
        nf4_path=Path(os.environ.get("K2_EX_NF4", str(DEFAULT_NF4))),
        out_path=Path(os.environ.get("K2_EX_OUT", str(DEFAULT_OUT))),
        te_path=Path(os.environ.get("K2_EX_TE", str(DEFAULT_TE))),
        vae_path=Path(os.environ.get("K2_EX_VAE", str(DEFAULT_VAE))),
        te_on_cpu=_env_bool("K2_EX_TE_CPU", False),
    )
    for path in (config.dit_path, config.nf4_path, config.te_path, config.vae_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    return config


def _make_spectrum_hook(
    *,
    module_path: str,
    module: torch.nn.Linear,
    layer_index: int,
    target_count: int,
    reader: NF4WeightReader,
    config: ProbeConfig,
    layer_results: list[dict[str, Any]],
):
    match = TARGET_RE.fullmatch(module_path)
    assert match is not None
    block = int(match.group(1))
    kind = match.group(2)

    def hook(_module, inputs, output):
        layer_started = time.time()
        activation = inputs[0]
        q_weight = reader.dequantize(module_path, activation.device)
        if q_weight.shape != module.weight.shape:
            raise RuntimeError(
                f"{module_path}: NF4 shape {tuple(q_weight.shape)} != "
                f"BF16 shape {tuple(module.weight.shape)}"
            )
        q_output = F.linear(activation, q_weight.to(activation.dtype), module.bias)
        error = output.float() - q_output.float()
        matrix = error.reshape(-1, error.shape[-1])
        summary = _spectrum_summary(
            matrix,
            ranks=config.ranks,
            oversample=config.oversample,
            niter=config.niter,
            exact_max=config.exact_max,
            seed=config.seed + layer_index,
        )
        output_energy = output.float().square().sum(dtype=torch.float64).item()
        rel_l2 = math.sqrt(summary["total_energy"] / output_energy) if output_energy else 0.0
        result = {
            "path": module_path,
            "block": block,
            "kind": kind,
            "input_shape": list(activation.shape),
            "output_shape": list(output.shape),
            "matrix_shape": list(matrix.shape),
            "output_energy": output_energy,
            "relative_l2": rel_l2,
            "seconds": time.time() - layer_started,
            **summary,
        }
        result["corrected_relative_l2"] = {
            key: rel_l2 * factor
            for key, factor in summary["residual_l2_factor"].items()
        }
        layer_results.append(result)
        rank_text = " ".join(
            f"r{rank}={100 * summary['captured_energy'][str(rank)]:5.1f}%"
            for rank in config.ranks
        )
        print(
            f"[{len(layer_results):3d}/{target_count}] {module_path:<30} "
            f"rel={rel_l2:.4f} {rank_text} {result['seconds']:.2f}s",
            flush=True,
        )

    return hook


def _run_forward(
    dit: torch.nn.Module,
    targets: list[tuple[str, torch.nn.Linear]],
    reader: NF4WeightReader,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    config: ProbeConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[dict[str, Any]], float]:
    hiddens, text_mask, latents_4d = inputs
    layer_results: list[dict[str, Any]] = []
    handles = [
        module.register_forward_hook(
            _make_spectrum_hook(
                module_path=name,
                module=module,
                layer_index=index,
                target_count=len(targets),
                reader=reader,
                config=config,
                layer_results=layer_results,
            )
        )
        for index, (name, module) in enumerate(targets)
    ]

    latents_5d = latents_4d.to(device).unsqueeze(2)
    text_embedding = Krea2TextEmbedding(hiddens.to(device), text_mask.to(device))
    torch.manual_seed(config.seed)
    noise = torch.randn_like(latents_5d)
    sigma = torch.full((latents_5d.shape[0],), config.sigma, device=device, dtype=dtype)
    x_t = (1.0 - sigma) * latents_5d + sigma * noise
    forward_started = time.time()
    try:
        with torch.inference_mode():
            velocity = forward_for_loss(dit, x_t, text_embedding, sigma)
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()

    if len(layer_results) != len(targets):
        raise RuntimeError(f"captured {len(layer_results)} of {len(targets)} target layers")
    return velocity, layer_results, time.time() - forward_started


def _save_report(
    *,
    config: ProbeConfig,
    layers: list[dict[str, Any]],
    velocity: torch.Tensor,
    total_seconds: float,
    forward_seconds: float,
    dtype: torch.dtype,
) -> dict[str, Any]:
    report = {
        "config": {
            "image_size": config.image_size,
            "sigma": config.sigma,
            "ranks": list(config.ranks),
            "oversample": config.oversample,
            "niter": config.niter,
            "exact_max": config.exact_max,
            "seed": config.seed,
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "dtype": str(dtype),
            "prompt": PROMPT,
            "dit_path": str(config.dit_path),
            "nf4_path": str(config.nf4_path),
        },
        "runtime": {
            "total_seconds": total_seconds,
            "forward_seconds": forward_seconds,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
            "velocity_shape": list(velocity.shape),
        },
        "aggregate": _aggregate_layers(layers, config.ranks),
        "layers": layers,
    }
    config.out_path.parent.mkdir(parents=True, exist_ok=True)
    config.out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> int:
    config = _load_config()
    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(
        f"=== Krea-2 NF4 EX spectrum: image={config.image_size}, sigma={config.sigma}, "
        f"ranks={config.ranks}, GPU={os.environ.get('CUDA_VISIBLE_DEVICES')} ==="
    )
    started = time.time()
    inputs = _encode_inputs(
        image_size=config.image_size,
        device=device,
        dtype=dtype,
        te_path=config.te_path,
        vae_path=config.vae_path,
        te_on_cpu=config.te_on_cpu,
    )
    hiddens, _, latents_4d = inputs
    print(
        f"inputs: text={tuple(hiddens.shape)}, latent={tuple(latents_4d.shape)}, "
        f"elapsed={time.time() - started:.1f}s"
    )

    load_started = time.time()
    dit = load_krea2_dit(config.dit_path, device="cpu", dtype=dtype, eval=True)
    for parameter in dit.parameters():
        parameter.requires_grad_(False)
    dit = dit.to(device)
    targets = _target_modules(dit)
    print(f"BF16 DiT loaded: {len(targets)} targets, elapsed={time.time() - load_started:.1f}s")

    reader = NF4WeightReader(config.nf4_path)
    torch.cuda.reset_peak_memory_stats()
    velocity, layer_results, forward_seconds = _run_forward(
        dit, targets, reader, inputs, config, device, dtype
    )
    result = _save_report(
        config=config,
        layers=layer_results,
        velocity=velocity,
        total_seconds=time.time() - started,
        forward_seconds=forward_seconds,
        dtype=dtype,
    )

    print("\n=== aggregate energy captured ===")
    for rank in config.ranks:
        stats = result["aggregate"]["all"][f"rank_{rank}"]
        print(
            f"rank {rank:>2}: weighted={100 * stats['energy_weighted']:.2f}% "
            f"median={100 * stats['median_layer']:.2f}% "
            f"layers>=75% {stats['layers_ge_75pct']}/196"
        )
    print(
        f"result={config.out_path}, total={result['runtime']['total_seconds']:.1f}s, "
        f"peak={result['runtime']['peak_gpu_gb']:.2f}GB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
