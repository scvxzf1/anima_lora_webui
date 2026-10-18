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
| **P0'** | 压缩串行 DMA 链：量化单块 H2D 有效带宽，对比 `restore_mode`（foreach / slab） | **已完成（结论：链路饱和）** | 见[第 2 轮证据](../findings/block_swap_restore_link_limit_50hx_20261019.md)：链路天花板 3.14 GB/s，训练内已达 3.02（foreach）/ 3.12（slab）GB/s；建议显式设 `block_swap_restore_mode="slab"`（DMA +3.3%、步时 −4%） |
| **P1** | 进程内水位调优 `--block_swap_auto`：第一个优化器步后按空闲显存与 per-block 字节晋升常驻块（复用 `reconfigure()` 边界事务），替代固定水位 | **已实现 + 512 真机验收（256/1024 待验收）** | 见[第 3 轮证据](../findings/block_swap_auto_tuner_50hx_20261019.md)：512 下 24 → 2，字节/步 6644 → 554 MB，任务窗口 2118 → ~720 ms；但 swap≤5 后步时与固定 swap=5 持平（≈0.90 vs 0.94 s），下一步需加"DMA 已低于计算量即停止晋升"的停止条件 |
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
| 2 | 2026-10-19 | 纯拷贝微基准（`copy_bench.py`）+ profiler 加 `h2d_bytes` 字节计量 + `foreach` vs `slab` A/B | 链路天花板 3.14 GB/s；训练内 3.02 / 3.12 GB/s = 96% / 99%；分块粒度无关；P0' 完成并推荐 slab；下一步 P1 |
| 3 | 2026-10-19 | 实现 `library/runtime/block_swap_auto.py` + CLI（`--block_swap_auto` / `--block_swap_vram_margin_mib`）+ 优化器边界挂点 + 10 项单测；真机跑 auto（24 起点）与固定 swap=5 对照 | 512 下自动 24 → 2（bytes 6644 → 554 MB/步，窗口 2118 → ~720 ms）；但 swap≤5 后与固定 5 持平 → 需加停止条件；256/1024 与其它模型族待验收 |
