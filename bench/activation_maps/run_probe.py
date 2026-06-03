"""Run a short LoRA / LoKr activation-map probe.

This script intentionally uses dynamic adapter hooks (`--pgraft`) so adapter
delta activations stay observable. Plain inference normally static-merges
regular LoRA-family weights into the DiT, which erases the adapter forward path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from inference import parse_args as parse_inference_args
from library.inference import (
    check_inputs,
    generate_body,
    get_generation_settings,
    load_dit_model,
    load_text_encoder,
    prepare_text_inputs,
    resolve_seed,
)
from library.inference.text import ensure_text_strategies
from bench.activation_maps.collector import (
    ActivationCollector,
    adapter_events_to_jsonable,
    events_to_jsonable,
)


DEFAULT_PROMPTS = [
    "1girl, red hair, blue eyes, white dress, soft window light",
    "1girl, dynamic running pose, city street, motion blur, dramatic rim light",
    "close-up portrait, detailed eyes, watercolor style, pastel colors",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture compact activation-map summaries for Anima LoRA/LoKr."
    )
    parser.add_argument("--dit", required=True, help="Anima DiT checkpoint path")
    parser.add_argument("--text_encoder", required=True, help="Qwen3 text encoder path")
    parser.add_argument("--lora_weight", nargs="+", required=True, help="Adapter weight(s)")
    parser.add_argument(
        "--lora_multiplier",
        type=float,
        nargs="+",
        default=[1.0],
        help="Adapter multiplier(s), forwarded to inference",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt to probe. Repeat for multiple prompts.",
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Optional UTF-8 prompt file; blank/comment lines skipped.",
    )
    parser.add_argument("--negative_prompt", default="", help="Negative prompt")
    parser.add_argument("--image_size", type=int, nargs=2, default=[1024, 1024])
    parser.add_argument("--infer_steps", type=int, default=4)
    parser.add_argument("--flow_shift", type=float, default=3.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--attn_mode",
        default="torch",
        choices=["torch", "sdpa", "flash", "sageattn", "flex", "xformers"],
    )
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument(
        "--output_dir",
        default="output/activation_maps",
        help="Directory for JSON summaries and PNG heatmaps.",
    )
    parser.add_argument(
        "--no_text_adapter",
        action="store_true",
        help="Skip LLMAdapter block hooks; crossattn_emb is still summarized.",
    )
    parser.add_argument(
        "--skip_cfg",
        action="store_true",
        help="Force guidance_scale=1.0 even if a different value was passed.",
    )
    parser.add_argument(
        "--text_encoder_cpu",
        action="store_true",
        help="Run Qwen3 text encoding on CPU to leave more VRAM for DiT probes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = _load_prompts(args)
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    if not prompts:
        raise SystemExit("No prompts to probe.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    inf_args = _build_inference_args(args)
    check_inputs(inf_args)
    ensure_text_strategies(inf_args.text_encoder)

    gen_settings = get_generation_settings(inf_args)
    device = gen_settings.device
    model = load_dit_model(inf_args, device, torch.bfloat16)
    text_encoder_device = torch.device("cpu") if inf_args.text_encoder_cpu else device
    shared_models: dict[str, Any] = {
        "model": model,
        "text_encoder": load_text_encoder(
            inf_args,
            dtype=torch.bfloat16,
            device=text_encoder_device,
        ),
        "conds_cache": {},
    }

    all_runs = []
    collector = ActivationCollector(
        model,
        record_blocks=True,
        record_text_adapter=not args.no_text_adapter,
        record_adapter_delta=True,
        topk=args.topk,
    )
    with collector:
        for prompt_idx, prompt in enumerate(prompts):
            run_args = _clone_namespace(inf_args)
            run_args.prompt = prompt
            run_args.seed = int(args.seed) + prompt_idx
            context, context_null = prepare_text_inputs(
                run_args, device, model, shared_models
            )

            before_events = len(collector.events)
            before_adapter_events = len(collector.adapter_events)

            seed = resolve_seed(run_args)
            with torch.no_grad():
                latents = generate_body(
                    run_args,
                    model,
                    context,
                    context_null,
                    device,
                    seed,
                )

            prompt_events = collector.events[before_events:]
            prompt_adapter_events = collector.adapter_events[before_adapter_events:]
            run = {
                "prompt_index": prompt_idx,
                "prompt": prompt,
                "seed": seed,
                "latent_shape": list(latents.shape),
                "crossattn_summary": _summarize_crossattn(context),
                "events": events_to_jsonable(prompt_events),
                "adapter_events": adapter_events_to_jsonable(prompt_adapter_events),
                "layer_heatmap": _layer_heatmap_from_events(prompt_adapter_events),
            }
            all_runs.append(run)

    payload = {
        "config": {
            "dit": args.dit,
            "text_encoder": args.text_encoder,
            "lora_weight": args.lora_weight,
            "lora_multiplier": args.lora_multiplier,
            "image_size": args.image_size,
            "infer_steps": args.infer_steps,
            "guidance_scale": inf_args.guidance_scale,
            "attn_mode": inf_args.attn_mode,
            "dynamic_adapter_hooks": True,
        },
        "runs": all_runs,
    }
    json_path = output_dir / "activation_summary.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    heatmap_path = output_dir / "adapter_delta_heatmap.png"
    _plot_adapter_heatmap(all_runs, heatmap_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {heatmap_path}")


def _load_prompts(args: argparse.Namespace) -> list[str]:
    prompts = list(args.prompt or [])
    if args.prompt_file:
        path = Path(args.prompt_file)
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts or list(DEFAULT_PROMPTS)


def _build_inference_args(args: argparse.Namespace) -> argparse.Namespace:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    guidance_scale = 1.0 if args.skip_cfg else float(args.guidance_scale)
    argv = [
        "--dit",
        args.dit,
        "--text_encoder",
        args.text_encoder,
        "--lora_weight",
        *args.lora_weight,
        "--lora_multiplier",
        *(str(v) for v in args.lora_multiplier),
        "--prompt",
        args.prompt[0] if args.prompt else DEFAULT_PROMPTS[0],
        "--negative_prompt",
        args.negative_prompt,
        "--image_size",
        str(args.image_size[0]),
        str(args.image_size[1]),
        "--infer_steps",
        str(args.infer_steps),
        "--flow_shift",
        str(args.flow_shift),
        "--guidance_scale",
        str(guidance_scale),
        "--seed",
        str(args.seed),
        "--save_path",
        str(Path(args.output_dir) / "_unused"),
        "--device",
        device,
        "--attn_mode",
        args.attn_mode,
        "--output_type",
        "latent",
        "--pgraft",
    ]
    inf_args = parse_inference_args(argv)
    inf_args.compile = False
    inf_args.compile_blocks = False
    inf_args.text_encoder_cpu = bool(args.text_encoder_cpu)
    inf_args.tiled_diffusion = False
    inf_args.spectrum = False
    inf_args.spd = False
    inf_args.dcw = False
    inf_args.smc_cfg = False
    return inf_args


def _clone_namespace(ns: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(**vars(ns))


def _summarize_crossattn(context: dict[str, Any]) -> dict[str, Any]:
    from bench.activation_maps.collector import summarize_tensor

    embed = context["embed"][0]
    return asdict(summarize_tensor(embed, token_axis=1, channel_axis=-1, topk=16))


def _layer_heatmap_from_events(events) -> list[dict[str, Any]]:
    buckets: dict[tuple[int | None, str, str], list[float]] = {}
    for event in events:
        key = (
            event.meta.get("block_idx"),
            str(event.meta.get("component")),
            event.adapter_type,
        )
        buckets.setdefault(key, []).append(float(event.delta_to_base))

    rows = []
    for (block_idx, component, adapter_type), values in sorted(
        buckets.items(),
        key=lambda item: (
            -1 if item[0][0] is None else item[0][0],
            item[0][1],
            item[0][2],
        ),
    ):
        t = torch.tensor(values, dtype=torch.float32)
        rows.append(
            {
                "block_idx": block_idx,
                "component": component,
                "adapter_type": adapter_type,
                "count": len(values),
                "delta_to_base_mean": float(t.mean().item()),
                "delta_to_base_max": float(t.max().item()),
            }
        )
    return rows


def _plot_adapter_heatmap(runs: list[dict[str, Any]], path: Path) -> None:
    rows: dict[str, dict[int, float]] = {}
    for run in runs:
        for item in run["layer_heatmap"]:
            block_idx = item["block_idx"]
            if block_idx is None:
                continue
            label = f"{item['adapter_type']}:{item['component']}"
            rows.setdefault(label, {})
            rows[label][int(block_idx)] = max(
                rows[label].get(int(block_idx), 0.0),
                float(item["delta_to_base_mean"]),
            )

    if not rows:
        fig, ax = plt.subplots(figsize=(6, 2), constrained_layout=True)
        ax.text(0.5, 0.5, "No adapter events captured", ha="center", va="center")
        ax.set_axis_off()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return

    labels = sorted(rows)
    block_count = max(max(v) for v in (row.keys() for row in rows.values())) + 1
    matrix = torch.zeros(len(labels), block_count, dtype=torch.float32)
    for r, label in enumerate(labels):
        for block_idx, value in rows[label].items():
            matrix[r, block_idx] = value

    fig_w = max(8.0, block_count * 0.35)
    fig_h = max(4.0, len(labels) * 0.28)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(matrix.numpy(), aspect="auto", interpolation="nearest", cmap="magma")
    ax.set_xlabel("DiT block")
    ax.set_ylabel("adapter component")
    ax.set_xticks(range(block_count))
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Adapter delta/base RMS by layer")
    fig.colorbar(im, ax=ax, label="delta_to_base_mean")
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
