# 块交换基线测量（标准参考）

状态：基线参考 / 已固化
适用版本：当前 main
日期：2026-08-06

本文是块交换（block swap）优化工作的**基准参照基线**。后续所有改动（预取深度、copy stream、slab、int8、prepare 同步）都应以本表为对照判断收益。测量脚本已固化进仓库，可在任意 GPU 上重跑复现。

## 测量环境

| 项 | RTX 3080（主测） | CMP 90HX（对照） |
| --- | --- | --- |
| GPU | NVIDIA GeForce RTX 3080 | NVIDIA CMP 90HX |
| 显存 | 10 GB | 10 GB |
| PCIe | **Gen3 x16（约 10 GB/s 实测）** | Gen1 x16（约 0.8 GB/s 实测） |
| 角色 | 本次优化目标卡 | 极端低带宽对照 |

复现：

```bash
# 综合「计算 vs 传输」探针（主指标）
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiments/blockswap_baseline_probe.py

# 分项微基准
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiments/fp8_blockswap_h2d_microbench.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiments/blockswap_copy_plan_microbench.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/experiments/blockswap_restore_path_microbench.py
```

原始产物：`/tmp/anima-blockswap-baseline/{baseline_probe,h2d_microbench,copy_plan,restore_path}_rtx3080.json`。

## 核心基线数据（RTX 3080）

单个 DiT block 的 frozen 权重约 **132 MiB**（20 个 tensor）。前向计算与传输的相对量级决定了预取策略是否有效：

| 指标 | RTX 3080 | CMP 90HX |
| --- | ---: | ---: |
| 单块前向计算（seq=4096，无 checkpoint） | **11.8 ms** | — |
| 单块 H2D bf16（132 MiB） | **13.4 ms** | 170.8 ms |
| 单块 H2D int8/fp8（66 MiB） | **6.7 ms** | 85.5 ms |
| `overlap_ratio` = 计算 / bf16 传输 | **0.878** | — |
| `overlap_ratio` = 计算 / int8 传输 | **1.755** | — |

**`overlap_ratio < 1` 表示传输超过单块计算、藏不住；`>= 1` 表示可被单块计算隐藏。**

## 拷贝路径对比（RTX 3080，132 MiB / 块）

来自 `blockswap_copy_plan_microbench.py`（纯 DMA）与 `blockswap_restore_path_microbench.py`（含 host issue 成本）：

| 路径 | 纯 DMA p50 | host_issue p50 | ready p50 |
| --- | ---: | ---: | ---: |
| loop_copy（逐 tensor） | 14.12 ms | 0.381 ms | 14.19 ms |
| foreach_copy（当前默认） | 14.26 ms | 0.346 ms | 14.23 ms |
| slab_copy（整段一次） | 13.70 ms | 0.163 ms | 13.51 ms |

- **slab 比 foreach 快约 0.7 ms / 块（约 5%）**，且 host issue 减半。在高带宽卡上 slab 有温和但真实的收益。
- 在 CMP 90HX（Gen1）上三者几乎相同（均约 170ms）：拷贝已被带宽饱和，小拷贝次数无关紧要。

## 关键结论（驱动后续优化方向）

1. **3080 上 bf16 传输（13.4ms）略大于单块计算（11.8ms）**：当前领先量 K=1 的预取**藏不住传输**，每块约空转 2ms。→ **方向 1（预取深度 K）有效**，把 K 提到 ≥2 即可让传输完全藏进多块计算。
2. **int8 传输（6.7ms）远小于单块计算（11.8ms）**：压缩后 `overlap_ratio=1.755`，传输可被轻松隐藏。→ **方向 4（int8）在 3080 上既降传输又消等待**。
3. **host issue 成本极小（<0.4ms）**：瓶颈在 PCIe 传输与调度窗口，不在 Python 侧。方向 2（多 copy stream）的收益主要体现在配合方向 1 形成多级流水线。
4. **slab（方向 3）** 在 3080 上每块省约 0.7ms，叠加 12 块约 8ms/step，值得在无非 int8 路径默认化。
5. CMP 90HX 这类 Gen1 卡上，唯一有意义的手段是**减少传输量（int8）与尽量重叠**，拷贝路径优化无意义。

## 验收口径（供后续改动对照）

任一方向的改动合入前，应在 RTX 3080 上重跑 `blockswap_baseline_probe.py` 与相关微基准，并满足：

- 数值等价：`tests/test_compile_checkpoint_block_swap_hot.py` 全绿。
- `overlap_ratio_bf16` 改善，或 `gpu_wait_ms` / `forward_wait` p95 在 profile 中下降。
- step time 在 `blocks_to_swap=12`（balanced_16g 档）有可测收益。
