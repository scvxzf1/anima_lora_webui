"""Probe attention amplification around LoRA injection.

Phase 1 goal: run short, fixed-sigma forward passes and record whether the
adapter arm amplifies Q/K/V, attention logits, attention outputs, or LoRA deltas
relative to the base arm. This is a bench tool, not a training-path change.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from types import MethodType
from typing import Any, Iterable, Mapping

import torch

from bench._anima import add_common_args, build_anima, discover_bucketed_samples
from bench._common import make_run_dir, write_result
from library.io.cache import load_cached_crossattn_emb, load_cached_latents
from library.training.router_conditioning import apply_router_conditioning
from networks import attention_dispatch

EPS = 1e-12
MAX_STAT_ELEMENTS = 1_000_000


def _to_jsonable_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _tensor_stats(
    tensor: torch.Tensor | None,
    *,
    max_elements: int = MAX_STAT_ELEMENTS,
) -> dict[str, Any]:
    """Return compact finite-value stats without keeping tensor references."""
    if tensor is None:
        return {"present": False}

    with torch.no_grad():
        x = tensor.detach()
        shape = [int(dim) for dim in x.shape]
        dtype = str(x.dtype).removeprefix("torch.")
        numel = int(x.numel())
        if numel == 0:
            return {
                "present": True,
                "shape": shape,
                "dtype": dtype,
                "numel": 0,
                "finite": 0,
                "finite_fraction": 0.0,
                "mean": None,
                "rms": None,
                "p95_abs": None,
                "max_abs": None,
            }

        x = x.float().flatten()
        sampled = False
        sample_numel = numel
        if max_elements > 0 and numel > max_elements:
            idx = torch.linspace(
                0,
                numel - 1,
                steps=max_elements,
                device=x.device,
            ).round()
            idx = idx.to(torch.long).unique(sorted=True)
            x = x.index_select(0, idx)
            sampled = True
            sample_numel = int(x.numel())

        finite_mask = torch.isfinite(x)
        finite = int(finite_mask.sum().item())
        if finite == 0:
            return {
                "present": True,
                "shape": shape,
                "dtype": dtype,
                "numel": numel,
                "sampled": sampled,
                "sample_numel": sample_numel,
                "finite": 0,
                "finite_fraction": 0.0,
                "mean": None,
                "rms": None,
                "p95_abs": None,
                "max_abs": None,
            }

        xf = x[finite_mask]
        abs_x = xf.abs()
        return {
            "present": True,
            "shape": shape,
            "dtype": dtype,
            "numel": numel,
            "sampled": sampled,
            "sample_numel": sample_numel,
            "finite": finite,
            "finite_fraction": finite / max(1, sample_numel),
            "mean": _to_jsonable_float(xf.mean().item()),
            "rms": _to_jsonable_float(torch.sqrt(torch.mean(xf * xf)).item()),
            "p95_abs": _to_jsonable_float(torch.quantile(abs_x, 0.95).item()),
            "max_abs": _to_jsonable_float(abs_x.max().item()),
        }


def _safe_ratio(num: float | None, denom: float | None) -> float | None:
    if num is None or denom is None or abs(denom) <= EPS:
        return None
    return _to_jsonable_float(num / denom)


def _take_even_tokens(
    tensor: torch.Tensor,
    *,
    max_tokens: int,
) -> tuple[torch.Tensor, int, bool]:
    seq_len = int(tensor.shape[1])
    if max_tokens <= 0 or seq_len <= max_tokens:
        return tensor, seq_len, False
    idx = torch.linspace(
        0,
        seq_len - 1,
        steps=max_tokens,
        device=tensor.device,
    ).round()
    idx = idx.to(torch.long).unique(sorted=True)
    return tensor.index_select(1, idx), int(idx.numel()), True


def _attention_stats(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    max_tokens: int = 512,
) -> dict[str, Any]:
    """Compute sampled logits/probability stats for BLHD q/k/v tensors."""
    del v  # value stats are reported separately; logits depend only on q/k.
    with torch.no_grad():
        q_sample, q_sampled_len, q_sampled = _take_even_tokens(
            q, max_tokens=max_tokens
        )
        k_sample, k_sampled_len, k_sampled = _take_even_tokens(
            k, max_tokens=max_tokens
        )
        scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(q.shape[-1])
        logits = torch.einsum(
            "bqhd,bkhd->bhqk", q_sample.float(), k_sample.float()
        )
        logits = logits * float(scale)
        softmax_logits = torch.nan_to_num(logits, nan=0.0, posinf=80.0, neginf=-80.0)
        probs = torch.softmax(softmax_logits, dim=-1)
        entropy = -(probs * probs.clamp_min(EPS).log()).sum(dim=-1)
        max_prob = probs.max(dim=-1).values

        return {
            "scale": _to_jsonable_float(scale),
            "q_tokens": int(q.shape[1]),
            "k_tokens": int(k.shape[1]),
            "sampled_q_tokens": q_sampled_len,
            "sampled_k_tokens": k_sampled_len,
            "sampled": bool(q_sampled or k_sampled),
            "logits": _tensor_stats(logits),
            "entropy": _tensor_stats(entropy),
            "max_prob": _tensor_stats(max_prob),
        }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _series_stats(values: Iterable[float | None]) -> dict[str, Any]:
    clean = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not clean:
        return {"count": 0, "mean": None, "p95": None, "max": None}
    return {
        "count": len(clean),
        "mean": _to_jsonable_float(sum(clean) / len(clean)),
        "p95": _to_jsonable_float(_percentile(clean, 0.95)),
        "max": _to_jsonable_float(max(clean)),
    }


def _get_path(row: Mapping[str, Any], path: str) -> Any:
    cur: Any = row
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _summarize_attention_events(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in events:
        key = f"{row.get('arm', 'unknown')}/{row.get('attn_kind', 'unknown')}"
        groups.setdefault(key, []).append(row)

    paths = {
        "q_rms": "q.rms",
        "k_rms": "k.rms",
        "v_rms": "v.rms",
        "logits_p95_abs": "attention.logits.p95_abs",
        "logits_max_abs": "attention.logits.max_abs",
        "entropy_mean": "attention.entropy.mean",
        "max_prob_max": "attention.max_prob.max_abs",
        "attn_out_rms": "attn_out.rms",
        "projected_rms": "projected.rms",
        "projected_to_input_rms": "ratios.projected_to_input_rms",
        "attn_out_to_v_rms": "ratios.attn_out_to_v_rms",
    }
    summary: dict[str, Any] = {}
    for key, rows in sorted(groups.items()):
        summary[key] = {"event_count": len(rows)}
        for metric, path in paths.items():
            summary[key][metric] = _series_stats(_get_path(row, path) for row in rows)
    return summary


def _summarize_adapter_events(events: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in events:
        key = str(row.get("arm", "unknown"))
        groups.setdefault(key, []).append(row)

    paths = {
        "delta_rms": "delta.rms",
        "base_rms": "base.rms",
        "output_rms": "output.rms",
        "delta_to_base_rms": "ratios.delta_to_base_rms",
        "output_to_base_rms": "ratios.output_to_base_rms",
    }
    summary: dict[str, Any] = {}
    for key, rows in sorted(groups.items()):
        summary[key] = {"event_count": len(rows)}
        for metric, path in paths.items():
            summary[key][metric] = _series_stats(_get_path(row, path) for row in rows)
    return summary


def _summarize_events(
    attention_events: list[Mapping[str, Any]],
    adapter_events: list[Mapping[str, Any]],
) -> dict[str, Any]:
    attention = _summarize_attention_events(attention_events)
    adapter = _summarize_adapter_events(adapter_events)
    comparisons: dict[str, Any] = {}
    for kind in ("self", "cross"):
        base = attention.get(f"base/{kind}", {})
        adapted = attention.get(f"adapted/{kind}", {})
        for metric in ("projected_to_input_rms", "logits_p95_abs", "max_prob_max"):
            b = _get_path(base, f"{metric}.p95")
            a = _get_path(adapted, f"{metric}.p95")
            comparisons[f"{kind}_{metric}_p95_adapted_over_base"] = _safe_ratio(a, b)

    comparisons["adapter_delta_to_base_p95"] = _get_path(
        adapter.get("adapted", {}), "delta_to_base_rms.p95"
    )
    return {
        "attention_event_count": len(attention_events),
        "adapter_event_count": len(adapter_events),
        "attention": attention,
        "adapter": adapter,
        "comparisons": comparisons,
    }


def _parse_block_idx(name: str) -> int | None:
    parts = name.split(".")
    if "blocks" not in parts:
        return None
    idx = parts.index("blocks") + 1
    if idx >= len(parts):
        return None
    try:
        return int(parts[idx])
    except ValueError:
        return None


def _is_attention_module(module: Any) -> bool:
    return (
        hasattr(module, "compute_qkv")
        and hasattr(module, "output_proj")
        and hasattr(module, "is_selfattn")
    )


@dataclass
class AttentionInjectionCollector:
    model: torch.nn.Module
    network: torch.nn.Module | None = None
    max_logit_tokens: int = 512
    record_adapter_delta: bool = True
    attention_events: list[dict[str, Any]] = field(default_factory=list)
    adapter_events: list[dict[str, Any]] = field(default_factory=list)
    _patches: list[tuple[Any, str, Any]] = field(default_factory=list)
    _run_ctx: dict[str, Any] = field(default_factory=dict)

    def __enter__(self) -> "AttentionInjectionCollector":
        self._install_attention_hooks()
        if self.record_adapter_delta and self.network is not None:
            self._install_adapter_hooks()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for obj, attr, original in reversed(self._patches):
            setattr(obj, attr, original)
        self._patches.clear()

    def set_run_context(self, *, arm: str, sigma: float) -> None:
        self._run_ctx = {"arm": arm, "sigma": _to_jsonable_float(sigma)}

    def _install_attention_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if not _is_attention_module(module):
                continue
            original = module.forward
            self._patches.append((module, "forward", original))
            module.forward = MethodType(self._make_attention_forward(name), module)

    def _install_adapter_hooks(self) -> None:
        assert self.network is not None
        for name, module in self.network.named_modules():
            refs = getattr(module, "org_module_ref", None)
            if not refs or not hasattr(module, "org_forward"):
                continue
            host = refs[0]
            if not hasattr(host, "forward"):
                continue
            original = host.forward
            self._patches.append((host, "forward", original))
            host.forward = self._make_adapter_forward(name, module, original)

    def _make_attention_forward(self, name: str):
        collector = self
        max_logit_tokens = self.max_logit_tokens
        block_idx = _parse_block_idx(name)

        def wrapped(
            module,
            x: torch.Tensor,
            attn_params: attention_dispatch.AttentionParams,
            context: torch.Tensor,
            rope_cos_sin=None,
        ) -> torch.Tensor:
            q, k, v = module.compute_qkv(x, context, rope_cos_sin=rope_cos_sin)
            if q.dtype != v.dtype:
                if (
                    not attn_params.supports_fp32 or attn_params.requires_same_dtype
                ) and torch.is_autocast_enabled():
                    target_dtype = v.dtype
                    q = q.to(target_dtype)
                    k = k.to(target_dtype)

            attn_out = attention_dispatch.dispatch_attention(
                [q, k, v], attn_params=attn_params
            )
            projected = module.output_dropout(module.output_proj(attn_out))
            q_stats = _tensor_stats(q)
            k_stats = _tensor_stats(k)
            v_stats = _tensor_stats(v)
            input_stats = _tensor_stats(x)
            attn_out_stats = _tensor_stats(attn_out)
            projected_stats = _tensor_stats(projected)
            collector.attention_events.append(
                {
                    **collector._run_ctx,
                    "module": name,
                    "block": block_idx,
                    "attn_kind": "self" if module.is_selfattn else "cross",
                    "input": input_stats,
                    "context": _tensor_stats(context),
                    "q": q_stats,
                    "k": k_stats,
                    "v": v_stats,
                    "attention": _attention_stats(
                        q,
                        k,
                        v,
                        softmax_scale=attn_params.softmax_scale,
                        max_tokens=max_logit_tokens,
                    ),
                    "attn_out": attn_out_stats,
                    "projected": projected_stats,
                    "ratios": {
                        "projected_to_input_rms": _safe_ratio(
                            projected_stats.get("rms"), input_stats.get("rms")
                        ),
                        "attn_out_to_v_rms": _safe_ratio(
                            attn_out_stats.get("rms"), v_stats.get("rms")
                        ),
                    },
                }
            )
            return projected

        return wrapped

    def _make_adapter_forward(self, name: str, module: Any, original):
        collector = self

        def wrapped(x: torch.Tensor, *args, **kwargs):
            out = original(x, *args, **kwargs)
            with torch.no_grad():
                base = module.org_forward(x)
                delta = out - base
                base_stats = _tensor_stats(base)
                delta_stats = _tensor_stats(delta)
                output_stats = _tensor_stats(out)
                collector.adapter_events.append(
                    {
                        **collector._run_ctx,
                        "module": name,
                        "lora_name": getattr(module, "lora_name", name),
                        "input": _tensor_stats(x),
                        "base": base_stats,
                        "delta": delta_stats,
                        "output": output_stats,
                        "ratios": {
                            "delta_to_base_rms": _safe_ratio(
                                delta_stats.get("rms"), base_stats.get("rms")
                            ),
                            "output_to_base_rms": _safe_ratio(
                                output_stats.get("rms"), base_stats.get("rms")
                            ),
                        },
                    }
                )
            return out

        return wrapped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase-1 probe for LoRA attention/injection amplification."
    )
    parser.add_argument("--dit", required=True, help="Anima DiT checkpoint path.")
    parser.add_argument("--adapter", default=None, help="Optional adapter checkpoint.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("post_image_dataset/lora"),
        help="Preprocessed cache directory containing *_anima.npz and TE sidecars.",
    )
    parser.add_argument(
        "--bucket",
        default=None,
        help="Latent bucket string, e.g. 128x192. Default: most populous bucket.",
    )
    parser.add_argument("--num-samples", type=int, default=2)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.1, 0.4, 0.7])
    parser.add_argument("--text-variant", default=0)
    parser.add_argument("--allow-replace", action="store_true")
    parser.add_argument("--max-logit-tokens", type=int, default=512)
    parser.add_argument(
        "--no-adapter-delta",
        action="store_true",
        help="Skip LoRA delta/base hooks; attention events are still recorded.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional result root. Defaults to bench/attention_injection/results.",
    )
    add_common_args(parser, include_checkpointing=False, include_compile=False)
    parser.set_defaults(attn_mode="torch")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_probe_batch(
    picks: list[tuple[str, str, str, str]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    text_variant: int | str,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    latents: list[torch.Tensor] = []
    crossattn: list[torch.Tensor] = []
    stems: list[str] = []
    for stem, _latent_key, npz_path, te_path in picks:
        latent, _resolution, _orig_h, _orig_w = load_cached_latents(npz_path)
        text = load_cached_crossattn_emb(te_path, variant=text_variant)
        if text is None:
            raise SystemExit(f"{te_path} does not contain crossattn_emb")
        latents.append(latent)
        crossattn.append(text)
        stems.append(stem)

    latent_batch = torch.stack(latents).unsqueeze(2).to(device=device, dtype=dtype)
    crossattn_batch = torch.stack(crossattn).to(device=device, dtype=dtype)
    return latent_batch, crossattn_batch, stems


def _noise_like(tensor: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=tensor.device)
    generator.manual_seed(int(seed))
    return torch.randn(
        tensor.shape,
        generator=generator,
        device=tensor.device,
        dtype=tensor.dtype,
    )


def _make_noisy_latents(
    latents: torch.Tensor,
    *,
    sigma: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    noise = _noise_like(latents, seed)
    sigma_tensor = torch.full(
        (latents.shape[0],),
        float(sigma),
        device=latents.device,
        dtype=latents.dtype,
    )
    sigma_view = sigma_tensor.view(-1, 1, 1, 1, 1)
    noisy = (1.0 - sigma_view) * latents + sigma_view * noise
    return noisy, sigma_tensor


def _set_network_multiplier(network: Any, value: float) -> None:
    if network is not None and hasattr(network, "set_multiplier"):
        network.set_multiplier(float(value))


def _run_forward(
    *,
    model: Any,
    network: Any,
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    crossattn_emb: torch.Tensor,
) -> torch.Tensor:
    if network is not None:
        apply_router_conditioning(
            network=network,
            noisy_model_input=noisy_latents,
            timesteps=timesteps,
            is_train=False,
            warmup_step=0,
            max_train_steps=0,
            crossattn_emb=crossattn_emb,
        )
    out = model(noisy_latents, timesteps, crossattn_emb)
    return getattr(out, "sample", out)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be > 0")
    if not args.sigmas:
        raise SystemExit("--sigmas must contain at least one value")

    bucket, picks = discover_bucketed_samples(
        args.data_dir,
        args.bucket,
        args.num_samples,
        args.seed,
        allow_replace=args.allow_replace,
    )
    bundle = build_anima(args, adapter=args.adapter, train_mode=False, multiplier=1.0)
    latents, crossattn_emb, stems = _load_probe_batch(
        picks,
        device=bundle.device,
        dtype=bundle.dtype,
        text_variant=args.text_variant,
    )

    arms = ["base", "adapted"] if bundle.network is not None else ["base"]
    run_dir = make_run_dir(
        "attention_injection", label=args.label, root=args.output_root
    )
    output_events: list[dict[str, Any]] = []

    collector = AttentionInjectionCollector(
        bundle.anima,
        network=bundle.network,
        max_logit_tokens=args.max_logit_tokens,
        record_adapter_delta=not args.no_adapter_delta,
    )
    with collector, torch.no_grad():
        for sigma_idx, sigma in enumerate(args.sigmas):
            noisy, timesteps = _make_noisy_latents(
                latents,
                sigma=float(sigma),
                seed=int(args.seed) + sigma_idx * 1009,
            )
            for arm in arms:
                _set_network_multiplier(bundle.network, 0.0 if arm == "base" else 1.0)
                collector.set_run_context(arm=arm, sigma=float(sigma))
                out = _run_forward(
                    model=bundle.anima,
                    network=bundle.network,
                    noisy_latents=noisy,
                    timesteps=timesteps,
                    crossattn_emb=crossattn_emb,
                )
                output_events.append(
                    {
                        "arm": arm,
                        "sigma": _to_jsonable_float(sigma),
                        "output": _tensor_stats(out),
                    }
                )

    attention_path = run_dir / "attention_events.jsonl"
    adapter_path = run_dir / "adapter_events.jsonl"
    outputs_path = run_dir / "outputs.jsonl"
    _write_jsonl(attention_path, collector.attention_events)
    _write_jsonl(adapter_path, collector.adapter_events)
    _write_jsonl(outputs_path, output_events)

    metrics = _summarize_events(collector.attention_events, collector.adapter_events)
    metrics.update(
        {
            "bucket": bucket,
            "sample_stems": stems,
            "arms": arms,
            "sigmas": [_to_jsonable_float(s) for s in args.sigmas],
            "output": {
                arm: _series_stats(
                    _get_path(row, "output.rms")
                    for row in output_events
                    if row.get("arm") == arm
                )
                for arm in arms
            },
        }
    )
    result_path = write_result(
        run_dir,
        script=__file__,
        args=args,
        metrics=metrics,
        label=args.label,
        artifacts=[attention_path, adapter_path, outputs_path],
        device=bundle.device,
    )
    print(result_path)
    return result_path


if __name__ == "__main__":
    main()
