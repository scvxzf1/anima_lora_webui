"""Row definitions and per-row asset generation for the MOSS tier hot test.

Protocol: docs/findings/moss_image_token_budget_tiers_20261017.md section 7.
The seven rows below are the documented matrix; fixed conditions mirror the
256x256 BF16 CMP-50HX baseline so tier numbers stay comparable (plain LoRA
rank/alpha 32, BF16 autocast, swap20, full gradient checkpointing, torch
attention, compile off, AdamW lr 5e-5, seed 42, batch 1, accumulation 1).
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

MODEL_DIR = "models/moss_image"
METHODS_DIR = "configs/methods"

# Bump only with the protocol doc: the filler repeats a fixed clause so a long
# caption is deterministic and hits the documented 3500-4096 token window.
LONG_CAPTION_TARGET_TOKENS = 3800
SHORT_CAPTION = ("1girl, solo, silver hair, blue eyes, white shirt, city street, "
                 "daylight, looking at viewer, upper body")


@dataclass(frozen=True)
class TierRow:
    name: str
    method: str
    resolution: int
    min_reso: int
    max_reso: int
    max_token_length: int
    caps: str
    src_size: tuple[int, int]
    tokens: int
    blocks_to_swap: int = 20
    # Transfer/restore knobs. bf16 is the protocol default; int8 with the fused
    # restore is the only experimental combination this harness exercises, and
    # it is added as an explicit A/B row rather than changing the baseline rows.
    transfer_dtype: str = "bf16"
    int8_restore_mode: str | None = None
    note: str = ""


ROWS: dict[str, TierRow] = {
    row.name: row
    for row in (
        TierRow("1024-short", "moss_image_lora", 1024, 672, 1536, 4096, "short",
                (1200, 1600), 4200, note="heaviest tier, short text side"),
        TierRow("1024-long", "moss_image_lora", 1024, 672, 1536, 4096, "long",
                (1200, 1600), 4200, note="heaviest tier, long text side"),
        TierRow("tecap1024-long", "moss_image_lora_tecap1024", 1024, 672, 1536, 1024,
                "long", (1200, 1600), 4200, note="same image tokens, text capped at 1024"),
        TierRow("768-short", "moss_image_lora_t768", 768, 504, 1152, 4096, "short",
                (900, 1200), 2352, note="0.75x area tier, short text side"),
        TierRow("768-long", "moss_image_lora_t768", 768, 504, 1152, 4096, "long",
                (900, 1200), 2352, note="0.75x area tier, long text side"),
        TierRow("512-short", "moss_image_lora_t512", 512, 336, 768, 4096, "short",
                (600, 800), 1064, note="0.5x area tier, short text side"),
        TierRow("512-long", "moss_image_lora_t512", 512, 336, 768, 4096, "long",
                (600, 800), 1064, note="0.5x area tier, long text side"),
        # Supplementary 1024-tier short probes (2026-10-18 follow-up). Same 4200
        # image tokens as the heaviest protocol row; only the text cap or the swap
        # budget changes, so they bracket the cap-1024 ok / cap-4096 OOM boundary.
        TierRow("1024-cap2048-long", "moss_image_lora", 1024, 672, 1536, 2048, "long",
                (1200, 1600), 4200, note="1024 tier, text cap 2048"),
        TierRow("1024-cap3072-long", "moss_image_lora", 1024, 672, 1536, 3072, "long",
                (1200, 1600), 4200, note="1024 tier, text cap 3072"),
        TierRow("1024-swap28-long", "moss_image_lora", 1024, 672, 1536, 4096, "long",
                (1200, 1600), 4200, blocks_to_swap=28,
                note="1024 tier, cap 4096, 28 swapped blocks"),
        # int8 transfer + fused restore A/B rows (2026-10-18). Everything except
        # block_swap_transfer_dtype / block_swap_int8_restore_mode is identical to
        # the bf16 row they are paired with, so the delta is the transfer path.
        TierRow("1024-short-int8", "moss_image_lora", 1024, 672, 1536, 4096, "short",
                (1200, 1600), 4200, transfer_dtype="int8", int8_restore_mode="fused",
                note="A/B vs 1024-short: int8 transfer + fused restore"),
        TierRow("1024-cap2048-long-int8", "moss_image_lora", 1024, 672, 1536, 2048, "long",
                (1200, 1600), 4200, transfer_dtype="int8", int8_restore_mode="fused",
                note="A/B vs 1024-cap2048-long: int8 transfer + fused restore"),
    )
}
PROTOCOL_ORDER = tuple(ROWS)


def select_rows(names: str | None) -> list[TierRow]:
    if not names:
        return [ROWS[name] for name in PROTOCOL_ORDER]
    picked = [item.strip() for item in names.split(",") if item.strip()]
    unknown = [item for item in picked if item not in ROWS]
    if unknown:
        raise SystemExit(f"unknown rows {unknown}; known: {list(ROWS)}")
    return [ROWS[name] for name in picked]


def long_caption() -> str:
    """Deterministic filler long enough to reach the protocol token window."""
    clause = ("detailed illustration of a character standing on a rain-slick rooftop "
              "at dusk, cinematic rim light, layered background architecture, "
              "carefully rendered fabric folds, subtle atmospheric haze, ")
    repeats = max(1, LONG_CAPTION_TARGET_TOKENS // 18)
    return (clause * repeats).strip().rstrip(",")


def caption_for(row: TierRow) -> str:
    return SHORT_CAPTION if row.caps == "short" else long_caption()


def _write_source_image(path: Path, size: tuple[int, int]) -> None:
    import numpy as np
    from PIL import Image

    width, height = size
    ys = np.linspace(0.0, 255.0, height, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 255.0, width, dtype=np.float32)[None, :]
    rng = np.random.default_rng(1234)
    noise = rng.integers(0, 24, size=(height, width, 1), dtype=np.int16)
    base = np.stack([ys + xs * 0.25, xs + ys * 0.5, ys * 0.75 + xs * 0.5], axis=-1)
    array = np.clip(base + noise, 0, 255).astype("uint8")
    Image.fromarray(array, mode="RGB").save(path)


def _dataset_blueprint(row: TierRow, row_root: Path, caption_extension: str = ".txt") -> str:
    return "\n".join([
        "[[datasets]]",
        f"resolution = {row.resolution}",
        "batch_size = 1",
        "enable_bucket = true",
        f"min_bucket_reso = {row.min_reso}",
        f"max_bucket_reso = {row.max_reso}",
        "bucket_reso_steps = 64",
        "bucket_no_upscale = true",
        "caption_dropout_rate = 0.0",
        "[[datasets.subsets]]",
        f'image_dir = "{row_root / "data"}"',
        f'cache_dir = "{row_root / "cache"}"',
        f'text_cache_dir = "{row_root / "cache"}"',
        f'caption_extension = "{caption_extension}"',
        "num_repeats = 1",
        "caption_dropout_rate = 0.0",
        "",
        "[general]",
        "caption_dropout_rate = 0.0",
        "",
    ])


def _method_scalars(method_path: Path) -> dict:
    data = tomllib.loads(method_path.read_text(encoding="utf-8"))
    return {key: value for key, value in data.items() if not isinstance(value, (dict, list))}


def _runtime_scalars(row: TierRow, row_root: Path, steps: int, repo_root: Path) -> dict:
    scalars = _method_scalars(repo_root / METHODS_DIR / f"{row.method}.toml")
    # max_train_epochs would override max_train_steps (0 means "zero steps", not
    # "unset"), so the protocol row is driven purely by the step budget.
    scalars.pop("max_train_epochs", None)
    scalars.update({
        "pretrained_model_name_or_path": f"{MODEL_DIR}/MOSS-Image-DiT",
        "qwen3": f"{MODEL_DIR}/MOSS-Image-TextEncoder-BF16",
        "vae": f"{MODEL_DIR}/MOSS-Image-VAE",
        "dataset_config": str(row_root / "dataset.toml"),
        "output_dir": str(row_root / "adapter"),
        "output_name": f"moss_hot_{row.name}",
        "progress_jsonl": str(row_root / "progress.jsonl"),
        "memory_probe_jsonl": str(row_root / "memory_probe.jsonl"),
        "memory_probe_max_steps": steps,
        "max_train_steps": steps,
        # No mid-run saves: a per-step checkpoint write would inflate the
        # measured step time relative to the 256x256 baseline.
        "checkpointing_epochs": 0,
        "checkpointing_last_n_epochs": 0,
        "gradient_accumulation_steps": 1,
        "max_data_loader_n_workers": 0,
        "seed": 42,
        "save_state": False,
        "save_state_on_train_end": False,
        "sample_at_first": False,
        "sample_every_n_steps": 0,
        "sample_every_n_epochs": 0,
        "validation_split_num": 0,
        "log_every_n_steps": 1,
        "optimizer_type": "AdamW",
        "optimizer_args": ["fused=True"],
        "max_grad_norm": 1.0,
        "full_fp16": False,
        "adaptive_precision": "off",
        "max_token_length": row.max_token_length,
        "blocks_to_swap": row.blocks_to_swap,
        "block_swap_transfer_dtype": row.transfer_dtype,
    })
    if row.int8_restore_mode is not None:
        scalars["block_swap_int8_restore_mode"] = row.int8_restore_mode
    return scalars


def _toml_value(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return "\"" + str(value).replace("\\", "\\\\").replace("\"", "\\\"") + "\""


def write_row_assets(repo_root: Path, root: Path, row: TierRow, steps: int) -> dict:
    """Create src image + caption + dataset blueprint + runtime config for one row."""
    row_root = root / row.name
    (row_root / "src").mkdir(parents=True, exist_ok=True)
    (row_root / "data").mkdir(parents=True, exist_ok=True)
    (row_root / "cache").mkdir(parents=True, exist_ok=True)
    stem = f"hot_{row.name}"
    image_path = row_root / "src" / f"{stem}.png"
    _write_source_image(image_path, row.src_size)
    (row_root / "src" / f"{stem}.txt").write_text(caption_for(row), encoding="utf-8")
    dataset_path = row_root / "dataset.toml"
    dataset_path.write_text(_dataset_blueprint(row, row_root), encoding="utf-8")
    runtime_path = row_root / "config.runtime.toml"
    lines = [f"{key} = {_toml_value(value)}"
             for key, value in _runtime_scalars(row, row_root, steps, repo_root).items()]
    runtime_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"row": row, "root": row_root, "src_dir": row_root / "src",
            "data_dir": row_root / "data", "cache_dir": row_root / "cache",
            "dataset_config": dataset_path, "runtime_config": runtime_path}
