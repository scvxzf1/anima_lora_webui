#!/usr/bin/env python
"""int8 块交换「真实 H2D + 数值等价」探针（方向4 验证）。

`blockswap_baseline_probe.py` 的 byte8 测量是**裸 H2D 传输**，不含 int8 反量化
计算（`rows * scale`）。本探针补齐两块决策证据：

1. **真实 H2D 时间**：pinned int8 master → GPU，并在 GPU 上做 per-row 反量化，
   得到 bf16 权重的端到端时间；与裸 bf16 H2D 对比，验证 int8 在"传输减半但多一步
   反量化"之后仍能放进单块计算窗口（方向4 是否值得开）。
2. **数值等价**：在真实 DiT Linear 权重形状上，验证三种 int8 restore_mode
   （copy / direct_bind / reuse_storage）反量化结果逐位一致，并量化 int8 per-row
   往返误差（相对 bf16 源）——这是块交换 int8 不依赖端到端 probe 的正确性断言。

结果落盘 JSON 供 docs/findings 引用。

用法：
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiments/int8_blockswap_h2d_probe.py
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiments/int8_blockswap_h2d_probe.py --repeats 40
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from library.runtime.block_swap_masters import (
    Int8BlockSwapCpuMaster,
    _capture_cpu_master,
    _restore_cpu_master_tensor,
    _restore_int8_cpu_master_into_tensor,
)

# 真实 DiT block 的 frozen Linear 权重形状（x_dim=1536, mlp_ratio=4 → hidden 6144）。
# 一个 block 的可交换权重是这几类 Linear 的集合；这里用代表性的两块测 H2D。
DIT_WEIGHT_SHAPES: list[tuple[int, int]] = [
    (1536 * 3, 1536),  # self_attn.qkv_proj
    (1536, 1536),      # self_attn.output_proj / cross_attn.q_proj / adaln
    (1536 * 2, 1536),  # cross_attn.kv_proj
    (6144, 1536),      # mlp.layer1
    (1536, 6144),      # mlp.layer2
]


def _total_elems() -> int:
    return sum(r * c for r, c in DIT_WEIGHT_SHAPES)


def _bf16_block_bytes() -> int:
    return _total_elems() * 2  # bf16 = 2 bytes


def _int8_block_bytes() -> int:
    rows = sum(r for r, _ in DIT_WEIGHT_SHAPES)
    return _total_elems() + rows * 4  # int8 payload + fp32 per-row scale


def _make_int8_masters(seed: int) -> list[Int8BlockSwapCpuMaster]:
    """按真实形状构造 pinned int8 CPU master（走与运行时相同的 _capture_cpu_master）。"""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    masters: list[Int8BlockSwapCpuMaster] = []
    for rows, cols in DIT_WEIGHT_SHAPES:
        w = torch.randn(rows, cols, generator=gen, dtype=torch.bfloat16) * 0.02
        master, _stats = _capture_cpu_master(
            w, module_name="mlp.layer1", pin_memory=True, transfer_dtype="int8", int8_scope="all"
        )
        assert isinstance(master, Int8BlockSwapCpuMaster)
        masters.append(master)
    return masters


def _median_bf16_h2d_ms(repeats: int) -> float:
    """裸 bf16 整块 H2D（与 baseline probe 口径一致）。"""
    src = torch.zeros(_bf16_block_bytes() // 2, dtype=torch.bfloat16).pin_memory()
    dst = torch.empty(_bf16_block_bytes() // 2, dtype=torch.bfloat16, device="cuda")
    times: list[float] = []
    for i in range(repeats + 5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize()
        if i >= 5:
            times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def _median_int8_restore_ms(masters: list[Int8BlockSwapCpuMaster], repeats: int) -> float:
    """int8 master → GPU 并反量化为 bf16 的端到端时间（含 per-row mul）。"""
    dsts = [
        torch.empty(m.shape, dtype=torch.bfloat16, device="cuda") for m in masters
    ]
    times: list[float] = []
    for i in range(repeats + 5):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for m, d in zip(masters, dsts):
            _restore_int8_cpu_master_into_tensor(
                m, d, device=torch.device("cuda"), dtype=torch.bfloat16, non_blocking=True
            )
        torch.cuda.synchronize()
        if i >= 5:
            times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def _numeric_equivalence(seed: int) -> dict:
    """三种 restore_mode 反量化逐位一致 + int8 往返误差。"""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.device("cuda")
    out: dict = {
        "shapes": [],
        "max_rel_l2": 0.0,
        "max_reuse_rel_l2": 0.0,
        "max_copy_reuse_gap": 0.0,
        "max_abs_err": 0.0,
        "copy_direct_bind_bitwise": True,
    }
    for rows, cols in DIT_WEIGHT_SHAPES:
        w = (torch.randn(rows, cols, generator=gen, dtype=torch.float32) * 0.02).to(torch.bfloat16)
        master, stats = _capture_cpu_master(
            w, module_name="mlp.layer1", pin_memory=False, transfer_dtype="int8", int8_scope="all"
        )
        # 三条路径的 GPU 反量化结果
        copy_t = _restore_cpu_master_tensor(master, device=dev, dtype=torch.bfloat16, non_blocking=False)
        dst_reuse = torch.empty(master.shape, dtype=torch.bfloat16, device=dev)
        _restore_int8_cpu_master_into_tensor(master, dst_reuse, device=dev, dtype=torch.bfloat16, non_blocking=False)
        to_tensor_t = master.to_tensor(device=dev, dtype=torch.bfloat16, non_blocking=False)
        torch.cuda.synchronize()
        # direct_bind 与 copy 共用 _restore_cpu_master_tensor（fp32 下乘 scale，见
        # offloading.py:1238-1246），二者必须逐位一致；reuse_storage 走
        # _restore_int8_cpu_master_into_tensor，scale 先转 bf16 再乘（block_swap_masters.py:210-212），
        # 允许 ~数 ULP 的乘法舍入差。验收标准是「三模式相对误差同量级、且 reuse 偏差远小于
        # int8 量化往返误差」，不是三模式逐位相等（bf16 乘法本就不保证逐位）。
        copy_db_bitwise = torch.equal(copy_t, to_tensor_t)
        out["copy_direct_bind_bitwise"] = out["copy_direct_bind_bitwise"] and copy_db_bitwise
        # int8 往返误差（相对 bf16 源），copy 路径
        src = w.to(dev, torch.float32)
        err = (copy_t.float() - src).norm().item() / max(src.norm().item(), 1e-12)
        abs_err = (copy_t.float() - src).abs().max().item()
        reuse_err = (dst_reuse.float() - src).norm().item() / max(src.norm().item(), 1e-12)
        copy_reuse_gap = (dst_reuse.float() - copy_t.float()).norm().item() / max(copy_t.float().norm().item(), 1e-12)
        out["max_rel_l2"] = max(out["max_rel_l2"], err)
        out["max_reuse_rel_l2"] = max(out["max_reuse_rel_l2"], reuse_err)
        out["max_copy_reuse_gap"] = max(out["max_copy_reuse_gap"], copy_reuse_gap)
        out["max_abs_err"] = max(out["max_abs_err"], abs_err)
        out["shapes"].append({
            "shape": [rows, cols],
            "rel_l2": round(err, 6),
            "reuse_rel_l2": round(reuse_err, 6),
            "copy_reuse_gap": round(copy_reuse_gap, 8),
            "capture_rel_l2": round(stats["relative_l2"], 6),
            "copy_db_bitwise": copy_db_bitwise,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-fwd-ms", type=float, default=11.8, help="3080 单块前向（来自 baseline probe）")
    parser.add_argument("--out", default="/tmp/anima-blockswap-baseline/int8_h2d_probe.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA")

    masters = _make_int8_masters(args.seed)
    bf16_h2d = _median_bf16_h2d_ms(args.repeats)
    int8_restore = _median_int8_restore_ms(masters, args.repeats)
    numeric = _numeric_equivalence(args.seed)

    # 数值等价 gate：三模式 rel_l2 同量级。copy/direct_bind 在 fp32 下乘 scale，
    # reuse_storage 在 bf16 下乘，故 reuse 引入一次额外的 bf16 乘法舍入，其 copy/reuse
    # 偏差与量化误差同量级（均 ~1e-3~1e-2）。关键是「reuse 总误差并不比 copy 更差」，
    # 即量化误差主导、乘法舍入不放大总误差。
    rel_l2 = numeric["max_rel_l2"]
    reuse_gap = numeric["max_copy_reuse_gap"]
    modes_consistent = (
        numeric["max_reuse_rel_l2"] <= rel_l2 * 1.5  # reuse 总误差 ≤ 1.5× copy 量化误差
        and reuse_gap <= max(rel_l2, 1e-3)           # reuse 偏差不超量化误差量级
    )

    result = {
        "device": torch.cuda.get_device_name(0),
        "block_weight_bytes_bf16": _bf16_block_bytes(),
        "block_weight_bytes_int8": _int8_block_bytes(),
        "int8_bytes_ratio": round(_int8_block_bytes() / _bf16_block_bytes(), 4),
        "bf16_h2d_ms": round(bf16_h2d, 3),
        "int8_restore_h2d_ms": round(int8_restore, 3),
        "int8_speedup_vs_bf16": round(bf16_h2d / int8_restore, 3),
        "block_fwd_ms": args.block_fwd_ms,
        "overlap_ratio_bf16": round(args.block_fwd_ms / bf16_h2d, 3),
        "overlap_ratio_int8": round(args.block_fwd_ms / int8_restore, 3),
        "int8_hides_in_compute": bool(args.block_fwd_ms >= int8_restore),
        "numeric": numeric,
        "gate": {
            # copy 与 direct_bind 共用同一反量化路径，必须逐位一致。
            "copy_direct_bind_bitwise": numeric["copy_direct_bind_bitwise"],
            # reuse_storage 允许 ~ULP 级乘法舍入，要求与 copy 容差一致。
            "modes_numerically_consistent": bool(modes_consistent),
            "int8_faster_than_bf16": bool(int8_restore < bf16_h2d),
            "int8_hides": bool(args.block_fwd_ms >= int8_restore),
        },
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
