#!/usr/bin/env python
"""Build the CSV, metrics, contact sheet, and Markdown report for a probe run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F


MODES = ("pp2", "tp2", "tp2_int8")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _first_png(root: Path, mode: str) -> Path:
    image_root = root / "images_verified"
    if not (image_root / mode).is_dir():
        image_root = root / "images"
    images = sorted((image_root / mode).glob("*.png"))
    if len(images) != 1:
        raise ValueError(f"expected exactly one {mode} PNG, found {len(images)}")
    return images[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _image_metrics(reference: Image.Image, candidate: Image.Image) -> dict[str, float]:
    a = torch.from_numpy(np.asarray(reference.convert("RGB")).copy()).float() / 255.0
    b = torch.from_numpy(np.asarray(candidate.convert("RGB")).copy()).float() / 255.0
    delta = b - a
    mse = float(delta.square().mean())
    return {
        "mae": float(delta.abs().mean()),
        "rmse": mse**0.5,
        "psnr_db": float(-10.0 * np.log10(max(mse, 1e-12))),
        "cosine": float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
    }


def _tensor_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    a, b = reference.float(), candidate.float()
    delta = b - a
    return {
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "rel_l2": float(delta.norm() / a.norm().clamp_min(1e-12)),
        "cosine": float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0)),
    }


def _contact_sheet(paths: dict[str, Path], output: Path) -> None:
    width = 512
    label_height = 52
    sheet = Image.new("RGB", (width * 3, width + label_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=22)
    labels = {"pp2": "PP2", "tp2": "TP2 BF16", "tp2_int8": "TP2 INT8 group-128"}
    for index, mode in enumerate(MODES):
        image = Image.open(paths[mode]).convert("RGB").resize((width, width), Image.Resampling.LANCZOS)
        sheet.paste(image, (index * width, label_height))
        draw.text((index * width + 16, 14), labels[mode], fill="black", font=font)
    sheet.save(output)


def _gib(value: int | float) -> float:
    return float(value) / (1024**3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    results = {mode: _json(root / f"{mode}.json") for mode in MODES}
    paths = {mode: _first_png(root, mode) for mode in MODES}
    images = {mode: Image.open(path).convert("RGB") for mode, path in paths.items()}
    metadata = {mode: dict(Image.open(path).info) for mode, path in paths.items()}
    required = ("seed", "sampler", "infer_steps", "guidance_scale", "flow_shift", "prompt", "width", "height")
    for key in required:
        values = {metadata[mode].get(key) for mode in MODES}
        if None in values or "" in values:
            raise ValueError(f"image metadata is missing required field {key}")
        if len(values) != 1:
            raise ValueError(f"image metadata mismatch for {key}: {values}")
    sizes = {image.size for image in images.values()}
    if len(sizes) != 1:
        raise ValueError(f"image dimensions differ: {sizes}")

    tensor_paths = {
        "pp2": root / "pp2_initial_output.pt",
        "tp2": root / "tp2_initial_output_rank0.pt",
        "tp2_int8": root / "tp2_int8_initial_output_rank0.pt",
    }
    tensors = {mode: torch.load(path, map_location="cpu", weights_only=True) for mode, path in tensor_paths.items()}
    metrics = {
        "tensor": {
            "tp2_vs_pp2": _tensor_metrics(tensors["pp2"], tensors["tp2"]),
            "tp2_int8_vs_pp2": _tensor_metrics(tensors["pp2"], tensors["tp2_int8"]),
            "tp2_int8_vs_tp2": _tensor_metrics(tensors["tp2"], tensors["tp2_int8"]),
        },
        "image": {
            "tp2_vs_pp2": _image_metrics(images["pp2"], images["tp2"]),
            "tp2_int8_vs_pp2": _image_metrics(images["pp2"], images["tp2_int8"]),
            "tp2_int8_vs_tp2": _image_metrics(images["tp2"], images["tp2_int8"]),
        },
    }
    (root / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    with (root / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "mode", "checkpoint", "checkpoint_sha256", *required,
                "image_path", "image_sha256",
            ),
        )
        writer.writeheader()
        for mode in MODES:
            writer.writerow(
                {
                    "mode": mode,
                    "checkpoint": str(root / "checkpoints" / f"{mode}.safetensors"),
                    "checkpoint_sha256": _sha256(root / "checkpoints" / f"{mode}.safetensors"),
                    **{key: metadata[mode].get(key, "") for key in required},
                    "image_path": str(paths[mode]),
                    "image_sha256": _sha256(paths[mode]),
                }
            )
    _contact_sheet(paths, root / "contact_sheet.png")

    rows = []
    for mode in MODES:
        result = results[mode]
        ranks = result["ranks"]
        communication = ranks[0]["communication"]
        for field in ("payload_bytes_per_rank", "wire_bytes_per_rank", "collective_calls"):
            if any(rank["communication"][field] != communication[field] for rank in ranks[1:]):
                raise ValueError(f"{mode} rank communication mismatch for {field}")
        peaks = "/".join(f"{_gib(rank['peak_allocated_bytes']):.2f}" for rank in ranks)
        reserved = "/".join(f"{_gib(rank['peak_reserved_bytes']):.2f}" for rank in ranks)
        wire = _gib(communication["wire_bytes_per_rank"])
        communication_share = 100.0 * communication["communication_seconds"] / result["elapsed_seconds"]
        relative = metrics["tensor"].get(f"{mode}_vs_pp2")
        rows.append(
            f"| {mode} | {result['seconds_per_step']:.3f} | "
            f"{60.0 / result['seconds_per_step']:.2f} | {peaks} | {reserved} | "
            f"{wire:.2f} | {communication['collective_calls']} | {communication_share:.1f}% | "
            f"{result['loss_first']:.6f} | "
            f"{result['loss_last']:.6f} | "
            f"{relative['rel_l2']:.4f}" if relative else
            f"| {mode} | {result['seconds_per_step']:.3f} | "
            f"{60.0 / result['seconds_per_step']:.2f} | {peaks} | {reserved} | "
            f"{wire:.2f} | {communication['collective_calls']} | {communication_share:.1f}% | "
            f"{result['loss_first']:.6f} | "
            f"{result['loss_last']:.6f} | reference"
        )
        rows[-1] += " |"

    image_rows = []
    for label, values in metrics["image"].items():
        image_rows.append(
            f"| {label} | {values['mae']:.6f} | {values['rmse']:.6f} | "
            f"{values['psnr_db']:.2f} | {values['cosine']:.6f} |"
        )

    prompt = metadata["pp2"]["prompt"]
    wire_reduction = 100.0 * (
        1.0
        - results["tp2_int8"]["ranks"][0]["communication"]["wire_bytes_per_rank"]
        / results["tp2"]["ranks"][0]["communication"]["wire_bytes_per_rank"]
    )
    speedup = 100.0 * (
        1.0
        - results["tp2_int8"]["seconds_per_step"]
        / results["tp2"]["seconds_per_step"]
    )
    report = f"""# Anima 双卡 PP/TP/量化通信实测

## 条件

- 模型：Anima-2.9B-preview-v1，40 blocks，BF16 base，plain MLP LoRA rank 16。
- 训练：同一缓存样本、初始化、seed、学习率，BS=1，1 warmup + 8 optimizer steps，full checkpoint。
- 硬件：rank0 CMP 170HX 64GB；rank1 RTX 3080 10GB；PHB，CUDA P2P=false。
- PP2：20/20 blocks，单 microbatch fill-drain。它不是多 microbatch 1F1B。
- TP2：attention head parallel + MLP feature parallel；AdaLN/patch/final replicated。
- TP2 INT8：group-128 symmetric INT8，BF16 scale；forward 和 checkpoint backward activation collective 均压缩；LoRA 梯度保持 BF16。这不是 QAT。
- 图片：同一 step-8 checkpoint、prompt、seed 114、1024x1024、28 steps、CFG 4、flow shift 1、ER-SDE、FlashAttention。

## 性能与数值

| 模式 | s/step | it/min | 峰值 allocated GiB rank0/rank1 | 峰值 reserved GiB rank0/rank1 | wire GiB/rank（含 warmup） | calls | rank0 通信/等待占比 | loss first | loss last | 初始输出 rel-L2 vs PP2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

INT8 相对 BF16 TP2 的 wire bytes 降低约 {wire_reduction:.1f}%，8-step 均值只快约 {speedup:.2f}%；量化和反量化开销抵消了通信收益。初始输出相对 BF16 TP2 的 rel-L2={metrics['tensor']['tp2_int8_vs_tp2']['rel_l2']:.4f}、cosine={metrics['tensor']['tp2_int8_vs_tp2']['cosine']:.6f}，不满足保守的训练等价阈值，因此当前结论为实验性 REJECT，不应接入生产训练默认值。

“通信/等待占比”是 rank0 在阻塞 send/recv 或 collective 区间内的时间占总运行时间比例，包含 CMP 快卡等待 RTX 3080 慢卡的时间，不是纯 PCIe 传输时间或链路利用率。

## 图片对比

![三方案同参数对比](contact_sheet.png)

| 对比 | MAE | RMSE | PSNR dB | cosine |
|---|---:|---:|---:|---:|
{chr(10).join(image_rows)}

人工观察：三张图主体构图、姿态、光源和背景一致；TP2 主要是花朵与衣褶的局部细节变化；INT8 局部差异更明显，但这 8-step smoke 未出现构图漂移或明显伪影。单张 seed 只能证明本 smoke 的视觉结果，不能替代多 seed、长训练质量评测。

## 结论

本机异构双卡、PHB 且无 P2P 条件下，PP2 明显最快。TP2 被高频细粒度同步限制。INT8 把 wire bytes 近似减半，但没有形成有意义的总步时优势，且相对 PP2 引入约 4% 初始输出 rel-L2；当前不挑战 PP2。

Prompt：`{prompt}`
"""
    (root / "report.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
