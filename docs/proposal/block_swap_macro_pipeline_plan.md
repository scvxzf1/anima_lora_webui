# Block Swap 宏观流水线开发计划（P0'–P3）

状态：活跃 / 第 1 轮（基线）已完成
适用版本：`codex/block-swap-int8`（基于 dev `d65f4a28`）
相关代码：`library/runtime/offloading.py`、`library/runtime/block_swap_config.py`、`library/runtime/block_swap_masters.py`、`bench/block_swap_pipeline/`
证据基线：[Block Swap 宏观流水线基线（CMP 50HX）](../findings/block_swap_pipeline_baseline_50hx_20261019.md)

## 背景

单张量恢复（fused 单算子 + 环形缓冲 + 双流）已接近微观极限，因此本轮起把视角切到
**宏观调度**（谁在什么时候搬运、被掩盖了多少）与 **1 Byte 编码的信噪比**。
原始输入给的是四个方向（U-Turn Pinning / 交错换出 / 离群裁剪·NF8 / auto swap 水位）。
第 1 轮真机基线对其中的 P0 前提做了实测修正，见下。

## 第 1 轮实测修正（重要）

```
swap=0  稳态步时 ≈ 1.01~1.09 s
swap=12 稳态步时 ≈ 1.48~1.57 s        → swap 代价 ≈ +0.45~0.49 s/步
profiler: 24 任务/步（=2S）、重复恢复 0 次、exposed_wait ≈ 1.6 ms/步
```

- **原 P0"折返点重复搬运"不存在**：训练态前向只在 `block_idx < blocks_to_swap` 提交交换，最后一个换出块
  前向用完仍驻留，反向第一时间直接开算；反向早块恢复也已由尾块钩子形成 lead≈1 的流水。
- 真实机制是**串行 DMA 链**：12 次 H2D 合计 ≈ 1.07 s/步，与 swap=0 的计算（≈1.05 s）同量级；
  单块有效带宽仅 ≈1.3~2.3 GB/s，单线程拷贝池把它们排成一条链，于是表现为步时被 DMA 顶住，
  而不是在等待点暴露停顿。
- 结论：调度类优化（交错、预取、折返点驻留）头部空间有限；**减少搬运字节**与**提高单块搬运效率**才是杠杆。

## 阶段与状态

| 阶段 | 内容 | 状态 | 验收口径 |
| --- | --- | --- | --- |
| **P0'** | 压缩串行 DMA 链：量化单块 H2D 有效带宽（纯拷贝微基准 + 训练内 per-block `h2d_ms`），对比 `restore_mode`（copy / slab / fused / int8 fused）与 packing 方式的每档 GB/s | 待开始 | 每档给出 GB/s 与 12 块/步总时长；同一工作区、同一 swap 下 A/B |
| **P1** | `blocks_to_swap="auto"`：加载后按 `max_memory_reserved` 与可用显存自动定水位（Step 0 探针 → 晋升常驻块），避免"固定 swap=20 慢一倍 / 固定 swap=7 换分辨率就 OOM" | 待开始 | 自动化后落在人工甜点位 ±1 块；给出多分辨率（256/512/1024）自动选档记录 |
| **P2** | 1 Byte 高信噪比编码：per-row 离群裁剪（99.95% 分位缩放）或稀疏残差补偿；NF8 分位查表（可选后做） | 待开始 | 同 swap 同分辨率下 loss 偏移从 MOSS 的 ~1.09% 压到 <0.2%，PCIe 字节数不增 |
| **P3** | 交错换出拓扑 / 跨步预取 | 降级（收益受限） | 仅在 P0'/P1 完成后复查：折返点空窗实测仅 37~43 ms/步、无暴露等待 |

## 每轮工作约定

1. 每轮先做**真机测量**，再改代码；改动必须能在同一工作区 A/B（`bench/block_swap_pipeline` 支持持久工作区）。
2. 每轮结束把结果写进 `docs/findings/`（含复现命令、边界与未证明项），并更新本文件的阶段状态。
3. 文档写完后核对：数字与 `result.json` / 训练日志一致、链接可达、`git diff --check` 干净。
4. 只在本分支（`codex/block-swap-int8`）开发；证据与实现分开提交，便于回滚单轮结论。

## 推进记录

| 轮次 | 日期 | 内容 | 结果 |
| --- | --- | --- | --- |
| 1 | 2026-10-19 | 建立 `bench/block_swap_pipeline` 探针（工作区复用 + 相位切分 + 指标口径），跑 swap=12 / swap=0 A/B | P0 前提被实测否定并修正为 P0'；swap 代价 ≈ +0.45~0.49 s/步；重复搬运 0；exposed_wait ≈ 1.6 ms |
