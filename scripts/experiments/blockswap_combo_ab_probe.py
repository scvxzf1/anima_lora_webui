#!/usr/bin/env python
"""块交换优化方向「组合叠加」合成 A/B 探针（真实 Block × 真实 ModelOffloader）。

在真实 Anima Block（x_dim=1536, mlp_ratio=4）× 28、真实 ModelOffloader（含训练
backward hook 预取）下，量五个方向的组合对 **训练 step time（前向+反向）** 的影响。
不加载 checkpoint、不需要数据集——纯合成权重与输入，只测交换/调度开销，数值不影响计时。

每个 matrix 配置在独立子进程里跑（通过本脚本 `--mode <name>` 自调用），以便用
环境变量精确控制 `ANIMA_BLOCK_SWAP_PREFETCH_DEPTH` / `ANIMA_BLOCK_SWAP_RESTORE_MODE`，
避免同进程内 CPU master 缓存与全局状态的交叉污染。

矩阵（blocks_to_swap=12，seq=4096，与 balanced_16g 档一致）：
  base        基线：K=1 + foreach + bf16
  k2          方向1：请求 K=2 —— 现已被钳制为 K=1（见下），应等价于 base 且不崩
  slab        方向3：slab
  k2slab      方向1+3：请求 K=2 + slab —— K 钳 1，应等价于 slab
  int8        方向4：K=1 + foreach + int8
  int8_k2     方向4+1：int8 + 请求 K=2 —— K 钳 1，应等价于 int8
  all         方向1+2+3+4+5：请求 K=2 + slab + int8（int8 时 slab 自动回退 foreach）

注意（2026-08-07 修复）：K≥2 的预取深度与块交换的「resident slot 数恒定 =
num_blocks - blocks_to_swap、to_cuda 由 to_cpu 唯一决定」设计根本冲突——更深的
lead 必须把「尚未运行的块」当退役块 park 到 CPU，导致 forward 读到 CPU 权重
（mat2 is on cpu）或静默覆写其 storage。零显存代价下不存在正确的 K≥2；要正确需
为超前 job 配独立 GPU staging buffer（+1 块常驻显存），抵消 block swap 的省显存
目的。因此 ``submit_move_blocks`` 已把训练/推理的 prefetch lead 统一钳到 1，
``ANIMA_BLOCK_SWAP_PREFETCH_DEPTH`` 仅作兼容旋钮（>1 被忽略）。本矩阵中的 K=2
模式用于验证钳制生效：它们应不再崩溃、且 step time 与对应 K=1 模式一致。

用法：
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiments/blockswap_combo_ab_probe.py
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiments/blockswap_combo_ab_probe.py --modes base,k2slab,all
结果落盘 JSON 供 docs/findings 引用。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

# 每个 mode -> (prefetch_depth, restore_mode, transfer_dtype)。prefetch_depth/restore_mode
# 经环境变量传入（子进程内生效），transfer_dtype 传给 ModelOffloader。
MODES: dict[str, dict] = {
    "base":    {"depth": "1", "restore": "foreach", "dtype": "bf16"},
    "k2":      {"depth": "2", "restore": "foreach", "dtype": "bf16"},
    "slab":    {"depth": "1", "restore": "slab",    "dtype": "bf16"},
    "k2slab":  {"depth": "2", "restore": "slab",    "dtype": "bf16"},
    "int8":    {"depth": "1", "restore": "foreach", "dtype": "int8"},
    "int8_k2": {"depth": "2", "restore": "foreach", "dtype": "int8"},
    "all":     {"depth": "2", "restore": "slab",    "dtype": "int8"},
}

NUM_BLOCKS = 28
BLOCKS_TO_SWAP = 12
X_DIM = 1536
NUM_HEADS = 12
MLP_RATIO = 4.0
SEQ = 4096
CTX_LEN = 512


def _run_one(mode: str, steps: int, warmup: int) -> dict:
    """单个子进程内：构造真实 Block×28 + offloader，量前向+反向 step time。"""
    from library.anima.models import Block
    from library.runtime.offloading import ModelOffloader
    from networks.attention_dispatch import AttentionParams

    cfg = MODES[mode]
    dev = torch.device("cuda")
    dt = torch.bfloat16

    blocks = torch.nn.ModuleList(
        [Block(x_dim=X_DIM, context_dim=X_DIM, num_heads=NUM_HEADS, mlp_ratio=MLP_RATIO) for _ in range(NUM_BLOCKS)]
    ).to(dev, dt)
    for p in blocks.parameters():
        p.requires_grad_(False)  # frozen base，与块交换语义一致

    offloader = ModelOffloader(
        blocks,
        blocks_to_swap=BLOCKS_TO_SWAP,
        device=dev,
        supports_backward=True,
        transfer_dtype=cfg["dtype"],
    )

    attn = AttentionParams.create_attention_params("torch")
    x = torch.randn(1, 1, SEQ, 1, X_DIM, device=dev, dtype=dt)
    emb = torch.randn(1, 1, X_DIM, device=dev, dtype=dt)
    ctx = torch.randn(1, CTX_LEN, X_DIM, device=dev, dtype=dt)
    target = torch.randn(1, 1, SEQ, 1, X_DIM, device=dev, dtype=dt)

    # 无 checkpoint 时 28 块 × seq=4096 的反向激活在 3080(10GB) 上 OOM。真实 balanced_16g
    # 同样靠 checkpoint 控制激活。这里对每块用 checkpoint 重算，只保留块边界激活，
    # 把显存压到可测范围；反向重算不改变「wait/submit 传输调度」这个被测对象。
    use_ckpt = os.environ.get("ANIMA_COMBO_PROBE_CKPT", "1") != "0"
    ckpt = torch.utils.checkpoint.checkpoint

    def one_step() -> None:
        offloader.prepare_block_devices_before_forward(blocks, free_cache=False)
        h = x.requires_grad_(True)
        for i, blk in enumerate(blocks):
            offloader.wait_for_block(i)
            if use_ckpt:
                h = ckpt(lambda _h, _b=blk: _b(_h, emb, ctx, attn, None, None, False), h, use_reentrant=False)
            else:
                h = blk(h, emb, ctx, attn, None, None, False)
            offloader.submit_move_blocks(blocks, i)
        loss = torch.nn.functional.mse_loss(h.float(), target.float())
        loss.backward()

    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        one_step()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    offloader.thread_pool.shutdown(wait=False)
    return {
        "mode": mode,
        "depth": cfg["depth"],
        "restore": cfg["restore"],
        "transfer_dtype": cfg["dtype"],
        "step_ms_median": round(statistics.median(times), 2),
        "step_ms_min": round(min(times), 2),
        "peak_alloc_mb": round(peak_mb, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default=",".join(MODES.keys()))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--mode", default=None, help=argparse.SUPPRESS)  # 子进程内部入口
    parser.add_argument("--out", default="/tmp/anima-blockswap-baseline/combo_ab_rtx3080.json")
    args = parser.parse_args()

    if args.mode is not None:
        print(json.dumps(_run_one(args.mode, args.steps, args.warmup)))
        return

    if not torch.cuda.is_available():
        raise SystemExit("需要 CUDA")

    results: list[dict] = []
    for mode in [m for m in args.modes.split(",") if m in MODES]:
        env = dict(os.environ)
        env["ANIMA_BLOCK_SWAP_PREFETCH_DEPTH"] = MODES[mode]["depth"]
        env["ANIMA_BLOCK_SWAP_RESTORE_MODE"] = MODES[mode]["restore"]
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--mode", mode,
             "--steps", str(args.steps), "--warmup", str(args.warmup)],
            env=env, capture_output=True, text=True, timeout=600,
        )
        line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        try:
            rec = json.loads(line)
        except Exception:
            rec = {"mode": mode, "error": (proc.stderr or "no output").strip()[-400:]}
        results.append(rec)
        print(f"[{mode}] {rec.get('step_ms_median', 'ERR')} ms  (peak {rec.get('peak_alloc_mb', '?')} MB)", flush=True)

    base = next((r["step_ms_median"] for r in results if r.get("mode") == "base" and "step_ms_median" in r), None)
    for r in results:
        if base and "step_ms_median" in r:
            r["speedup_vs_base"] = round(base / r["step_ms_median"], 3)
    payload = {"device": torch.cuda.get_device_name(0), "num_blocks": NUM_BLOCKS,
               "blocks_to_swap": BLOCKS_TO_SWAP, "seq": SEQ, "results": results}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
