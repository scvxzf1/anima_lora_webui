#!/usr/bin/env python3
"""Compare BF16, NF4, and fixed corrections over complete Krea-2 rollouts."""
from __future__ import annotations

import gc
import json
import math
import os
import statistics
import sys
import time
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("K2_ROLLOUT_GPU", "0"))

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from torchvision.utils import save_image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_nf4_downstream_correction import CorrectionBank  # noqa: E402
from library.models.krea2_raw.attention_backend import prepare_krea2_attention  # noqa: E402
from library.models.krea2_raw.family import Krea2TextEmbedding, forward_for_loss  # noqa: E402
from library.models.krea2_raw.inference_runner import _encode_prompt  # noqa: E402
from library.models.krea2_raw.sampling import timesteps  # noqa: E402
from library.models.krea2_raw.strategy import load_krea2_text_encoder  # noqa: E402
from library.models.krea2_raw.weights import load_krea2_dit  # noqa: E402
from library.models.qwen_vae import load_vae  # noqa: E402


DEFAULT_DIT = ROOT / "models" / "diffusion_models" / "krea2_raw_bf16.safetensors"
DEFAULT_NF4 = ROOT / "models" / "diffusion_models" / "krea2_raw_nf4.safetensors"
DEFAULT_TE = ROOT / "models" / "text_encoders" / "qwen3vl_4b_bf16.safetensors"
DEFAULT_VAE = ROOT / "models" / "vae" / "qwen_image_vae.safetensors"
DEFAULT_LOCAL = ROOT / "output" / "tests" / "krea2_nf4_correction_rank16_pg199_1024.safetensors"
DEFAULT_DOWNSTREAM = ROOT / "output" / "tests" / "krea2_nf4_downstream_90hx" / "downstream_correction.safetensors"
DEFAULT_RUN = ROOT / "output" / "tests" / "krea2_nf4_correction_rollout_3080"

PROMPTS = (
    "A cinematic photograph of a glass observatory on a snowy mountain at sunrise, detailed reflections, natural light",
    "An anime illustration of a young astronomer in a blue coat standing beneath a sky full of luminous constellations",
    "A watercolor painting of an old tram crossing a rainy European street, soft pigments, expressive brushwork",
)
METHODS = ("teacher", "nf4", "local", "downstream")


@dataclass(frozen=True)
class Config:
    phase: str
    run_dir: Path
    dit_path: Path
    nf4_path: Path
    te_path: Path
    vae_path: Path
    local_path: Path
    downstream_path: Path
    image_size: int
    steps: int
    cfg: float
    seed: int
    prompt_count: int
    teacher_swap: int
    nf4_swap: int
    attention_mode: str
    compile_blocks: bool
    text_encoder_cpu: bool


@dataclass(frozen=True)
class Case:
    index: int
    name: str
    prompt: str
    seed: int


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


def _load_config() -> Config:
    phase = os.environ.get("K2_ROLLOUT_PHASE", "all").strip().lower()
    valid = {"text", "teacher", "nf4", "local", "downstream", "report", "all"}
    if phase not in valid:
        raise ValueError(f"K2_ROLLOUT_PHASE must be one of {sorted(valid)}, got {phase!r}")
    config = Config(
        phase=phase,
        run_dir=Path(os.environ.get("K2_ROLLOUT_RUN", DEFAULT_RUN)),
        dit_path=Path(os.environ.get("K2_ROLLOUT_DIT", DEFAULT_DIT)),
        nf4_path=Path(os.environ.get("K2_ROLLOUT_NF4", DEFAULT_NF4)),
        te_path=Path(os.environ.get("K2_ROLLOUT_TE", DEFAULT_TE)),
        vae_path=Path(os.environ.get("K2_ROLLOUT_VAE", DEFAULT_VAE)),
        local_path=Path(os.environ.get("K2_ROLLOUT_LOCAL", DEFAULT_LOCAL)),
        downstream_path=Path(os.environ.get("K2_ROLLOUT_DOWNSTREAM", DEFAULT_DOWNSTREAM)),
        image_size=_env_int("K2_ROLLOUT_SIZE", 512),
        steps=_env_int("K2_ROLLOUT_STEPS", 12),
        cfg=_env_float("K2_ROLLOUT_CFG", 4.5),
        seed=_env_int("K2_ROLLOUT_SEED", 20260812),
        prompt_count=_env_int("K2_ROLLOUT_PROMPTS", 3),
        teacher_swap=_env_int("K2_ROLLOUT_TEACHER_SWAP", 26),
        nf4_swap=_env_int("K2_ROLLOUT_NF4_SWAP", 20),
        attention_mode=os.environ.get("K2_ROLLOUT_ATTN", "flash"),
        compile_blocks=_env_bool("K2_ROLLOUT_COMPILE", True),
        text_encoder_cpu=_env_bool("K2_ROLLOUT_TE_CPU", True),
    )
    if config.image_size % 16:
        raise ValueError("image size must be divisible by 16")
    if config.steps < 1 or not 1 <= config.prompt_count <= len(PROMPTS):
        raise ValueError("invalid steps or prompt count")
    for path in (config.dit_path, config.nf4_path, config.te_path, config.vae_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if phase in {"local", "all"} and not config.local_path.exists():
        raise FileNotFoundError(config.local_path)
    if phase in {"downstream", "all"} and not config.downstream_path.exists():
        raise FileNotFoundError(config.downstream_path)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    (config.run_dir / "text").mkdir(exist_ok=True)
    (config.run_dir / "trajectories").mkdir(exist_ok=True)
    (config.run_dir / "images").mkdir(exist_ok=True)
    return config


def _cases(config: Config) -> list[Case]:
    return [
        Case(index=i, name=f"prompt_{i:02d}", prompt=PROMPTS[i], seed=config.seed)
        for i in range(config.prompt_count)
    ]


def _text_path(config: Config, case: Case) -> Path:
    return config.run_dir / "text" / f"{case.name}.safetensors"


def _trajectory_path(config: Config, case: Case, method: str) -> Path:
    return config.run_dir / "trajectories" / f"{case.name}_{method}.safetensors"


def _image_path(config: Config, case: Case, method: str) -> Path:
    return config.run_dir / "images" / f"{case.name}_{method}.png"


def _encode_text(config: Config, cases: list[Case], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    missing = [case for case in cases if not _text_path(config, case).exists()]
    if not missing:
        return {"generated": 0, "reused": len(cases)}
    started = time.time()
    te_device = torch.device("cpu") if config.text_encoder_cpu else device
    te_model, tokenizer = load_krea2_text_encoder(
        str(config.te_path), dtype=dtype, device=str(te_device)
    )
    uncond = _encode_prompt(te_model, tokenizer, "", device, dtype)
    for case in missing:
        cond = _encode_prompt(te_model, tokenizer, case.prompt, device, dtype)
        save_file(
            {
                "cond_hiddens": cond.hiddens.cpu().contiguous(),
                "cond_mask": cond.mask.cpu().contiguous(),
                "uncond_hiddens": uncond.hiddens.cpu().contiguous(),
                "uncond_mask": uncond.mask.cpu().contiguous(),
            },
            str(_text_path(config, case)),
            metadata={"format": "krea2_rollout_text_v1", "prompt": case.prompt},
        )
    del te_model, tokenizer, uncond
    gc.collect()
    torch.cuda.empty_cache()
    return {"generated": len(missing), "reused": len(cases) - len(missing), "seconds": time.time() - started}


def _load_text(config: Config, case: Case, device: torch.device, dtype: torch.dtype) -> tuple[Krea2TextEmbedding, Krea2TextEmbedding]:
    path = _text_path(config, case)
    if not path.exists():
        raise FileNotFoundError(f"missing text cache: {path}; run phase=text")
    with safe_open(path, framework="pt") as handle:
        cond = Krea2TextEmbedding(
            handle.get_tensor("cond_hiddens").to(device=device, dtype=dtype),
            handle.get_tensor("cond_mask").to(device=device),
        )
        uncond = Krea2TextEmbedding(
            handle.get_tensor("uncond_hiddens").to(device=device, dtype=dtype),
            handle.get_tensor("uncond_mask").to(device=device),
        )
    return cond, uncond


def _disable_teacher_masters(dit: torch.nn.Module) -> None:
    def no_cpu_masters(_self, _blocks) -> None:
        return None

    dit.offloader._ensure_cpu_weight_masters = types.MethodType(no_cpu_masters, dit.offloader)


def _setup_dit(config: Config, method: str, device: torch.device, dtype: torch.dtype) -> tuple[torch.nn.Module, CorrectionBank | None]:
    if method == "teacher":
        dit = load_krea2_dit(config.dit_path, device="cpu", dtype=dtype, eval=True)
        swap = config.teacher_swap
    else:
        dit = load_krea2_dit(
            config.dit_path,
            device="cpu",
            dtype=dtype,
            eval=True,
            nf4_path=config.nf4_path,
        )
        swap = config.nf4_swap
    prepare_krea2_attention(dit, config.attention_mode, dtype=dtype, compile_enabled=config.compile_blocks)
    for parameter in dit.parameters():
        parameter.requires_grad_(False)

    bank = None
    if method in {"local", "downstream"}:
        factor_path = config.local_path if method == "local" else config.downstream_path
        bank = CorrectionBank(factor_path)
        for parameter in bank.parameters():
            parameter.requires_grad_(False)
        bank.attach(dit)

    dit.enable_block_swap(swap, device)
    if method == "teacher":
        _disable_teacher_masters(dit)
    dit.move_to_device_except_swap_blocks(device)
    if bank is not None:
        bank.to(device=device, dtype=dtype).eval()
    dit.switch_block_swap_for_inference()
    if config.compile_blocks:
        dit.compile_blocks(backend="inductor", compile_block_scope="resident")
    return dit.eval(), bank


def _rollout(
    config: Config,
    case: Case,
    dit: torch.nn.Module,
    cond: Krea2TextEmbedding,
    uncond: Krea2TextEmbedding,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[float]]:
    latent_size = config.image_size // 8
    patch = dit.config.patch
    img_seq_len = (latent_size // patch) ** 2
    generator = torch.Generator(device=device).manual_seed(case.seed)
    image = torch.randn(
        1,
        dit.config.channels,
        1,
        latent_size,
        latent_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    ts = timesteps(img_seq_len, config.steps, device=device, dtype=dtype)
    trajectory = [image.detach().cpu()]
    step_seconds: list[float] = []
    with torch.inference_mode():
        for index, (current, previous) in enumerate(zip(ts[:-1], ts[1:]), 1):
            started = time.time()
            t = current.reshape(1)
            dit.prepare_block_swap_before_forward()
            cond_v = forward_for_loss(dit, image, cond, t)
            if config.cfg > 0:
                dit.prepare_block_swap_before_forward()
                uncond_v = forward_for_loss(dit, image, uncond, t)
                velocity = cond_v + config.cfg * (cond_v - uncond_v)
            else:
                velocity = cond_v
            image = image + (previous - current) * velocity
            torch.cuda.synchronize()
            step_seconds.append(time.time() - started)
            trajectory.append(image.detach().cpu())
            print(
                f"  {case.name} step={index:02d}/{config.steps} "
                f"sigma={float(current):.4f} sec={step_seconds[-1]:.2f}",
                flush=True,
            )
    return torch.cat(trajectory, dim=0), step_seconds


def _run_method(config: Config, cases: list[Case], method: str, device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    missing = [case for case in cases if not _trajectory_path(config, case, method).exists()]
    if not missing:
        return {"generated": 0, "reused": len(cases)}
    print(f"--- rollout method={method}, missing={len(missing)} ---")
    started = time.time()
    dit, bank = _setup_dit(config, method, device, dtype)
    torch.cuda.reset_peak_memory_stats()
    case_results = []
    for case in missing:
        cond, uncond = _load_text(config, case, device, dtype)
        trajectory, step_seconds = _rollout(config, case, dit, cond, uncond, device, dtype)
        save_file(
            {"trajectory": trajectory.contiguous()},
            str(_trajectory_path(config, case, method)),
            metadata={
                "format": "krea2_rollout_trajectory_v1",
                "method": method,
                "prompt": case.prompt,
                "seed": str(case.seed),
                "steps": str(config.steps),
                "cfg": repr(config.cfg),
                "image_size": str(config.image_size),
            },
        )
        case_results.append(
            {
                "name": case.name,
                "mean_step_seconds": statistics.mean(step_seconds),
                "median_step_seconds": statistics.median(step_seconds),
                "step_seconds": step_seconds,
            }
        )
        del cond, uncond, trajectory
    if bank is not None:
        bank.remove()
    peak = torch.cuda.max_memory_allocated() / 1e9
    del dit, bank
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "generated": len(missing),
        "reused": len(cases) - len(missing),
        "peak_gpu_gb": peak,
        "cases": case_results,
        "seconds": time.time() - started,
    }


def _tensor_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    candidate = candidate.float()
    reference = reference.float()
    delta = candidate - reference
    return {
        "relative_l2": delta.norm().item() / reference.norm().item(),
        "mse": delta.square().mean().item(),
        "mae": delta.abs().mean().item(),
        "cosine": F.cosine_similarity(candidate.flatten(), reference.flatten(), dim=0).item(),
        "max_delta": delta.abs().max().item(),
    }


def _load_trajectory(config: Config, case: Case, method: str) -> torch.Tensor:
    path = _trajectory_path(config, case, method)
    if not path.exists():
        raise FileNotFoundError(path)
    with safe_open(path, framework="pt") as handle:
        return handle.get_tensor("trajectory")


def _ssim(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    x = candidate.float().unsqueeze(0) if candidate.ndim == 3 else candidate.float()
    y = reference.float().unsqueeze(0) if reference.ndim == 3 else reference.float()
    mu_x = F.avg_pool2d(x, 11, stride=1, padding=5)
    mu_y = F.avg_pool2d(y, 11, stride=1, padding=5)
    var_x = F.avg_pool2d(x * x, 11, stride=1, padding=5) - mu_x.square()
    var_y = F.avg_pool2d(y * y, 11, stride=1, padding=5) - mu_y.square()
    covariance = F.avg_pool2d(x * y, 11, stride=1, padding=5) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_x * mu_y + c1) * (2 * covariance + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    )
    return float(score.mean())


def _decode_and_report(config: Config, cases: list[Case], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    trajectories = {
        (case.name, method): _load_trajectory(config, case, method)
        for case in cases
        for method in METHODS
    }
    vae = load_vae(str(config.vae_path), device=device, dtype=dtype, eval=True)
    images: dict[tuple[str, str], torch.Tensor] = {}
    with torch.inference_mode():
        for case in cases:
            for method in METHODS:
                final_latent = trajectories[(case.name, method)][-1:].to(device)
                pixels = vae.decode_to_pixels(final_latent.squeeze(2))
                image = ((pixels.clamp(-1, 1).float() + 1.0) / 2.0).cpu()
                images[(case.name, method)] = image
                save_image(image[0], str(_image_path(config, case, method)))
    del vae
    gc.collect()
    torch.cuda.empty_cache()

    result_cases: list[dict[str, Any]] = []
    aggregates: dict[str, list[float]] = {
        f"{method}_{metric}": []
        for method in METHODS[1:]
        for metric in ("latent_relative_l2", "pixel_relative_l2", "pixel_psnr", "pixel_ssim")
    }
    for case in cases:
        teacher_trajectory = trajectories[(case.name, "teacher")]
        teacher_image = images[(case.name, "teacher")]
        methods: dict[str, Any] = {}
        for method in METHODS[1:]:
            candidate_trajectory = trajectories[(case.name, method)]
            per_step = [
                _tensor_metrics(candidate_trajectory[index], teacher_trajectory[index])
                for index in range(config.steps + 1)
            ]
            latent_metric = per_step[-1]
            pixel_metric = _tensor_metrics(images[(case.name, method)], teacher_image)
            psnr = -10.0 * math.log10(max(pixel_metric["mse"], 1e-12))
            ssim = _ssim(images[(case.name, method)], teacher_image)
            methods[method] = {
                "trajectory": per_step,
                "final_latent": latent_metric,
                "pixels": {**pixel_metric, "psnr": psnr, "ssim": ssim},
            }
            aggregates[f"{method}_latent_relative_l2"].append(latent_metric["relative_l2"])
            aggregates[f"{method}_pixel_relative_l2"].append(pixel_metric["relative_l2"])
            aggregates[f"{method}_pixel_psnr"].append(psnr)
            aggregates[f"{method}_pixel_ssim"].append(ssim)
        result_cases.append({"name": case.name, "prompt": case.prompt, "seed": case.seed, "methods": methods})

    summary = {
        key: {"mean": statistics.mean(values), "median": statistics.median(values)}
        for key, values in aggregates.items()
    }
    baseline = summary["nf4_latent_relative_l2"]["mean"]
    local = summary["local_latent_relative_l2"]["mean"]
    downstream = summary["downstream_latent_relative_l2"]["mean"]
    summary["latent_relative_l2_improvement_percent"] = {
        "local_vs_nf4": 100.0 * (1.0 - local / baseline),
        "downstream_vs_nf4": 100.0 * (1.0 - downstream / baseline),
        "downstream_vs_local": 100.0 * (1.0 - downstream / local),
    }
    return {"summary": summary, "cases": result_cases}


def main() -> int:
    config = _load_config()
    cases = _cases(config)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    manifest = {
        "config": {
            **asdict(config),
            **{
                key: str(getattr(config, key))
                for key in (
                    "run_dir",
                    "dit_path",
                    "nf4_path",
                    "te_path",
                    "vae_path",
                    "local_path",
                    "downstream_path",
                )
            },
        },
        "gpu": torch.cuda.get_device_name(),
        "cases": [asdict(case) for case in cases],
    }
    (config.run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    report: dict[str, Any] = {**manifest, "phase": config.phase}
    phases = ("text", "teacher", "nf4", "local", "downstream", "report") if config.phase == "all" else (config.phase,)
    started = time.time()
    for phase in phases:
        if phase == "text":
            report[phase] = _encode_text(config, cases, device, dtype)
        elif phase in METHODS:
            report[phase] = _run_method(config, cases, phase, device, dtype)
        elif phase == "report":
            report[phase] = _decode_and_report(config, cases, device, dtype)
    report["runtime_seconds"] = time.time() - started
    report_path = config.run_dir / f"report_{config.phase}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"cases"}}, indent=2, default=str))
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
