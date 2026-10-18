# Block Swap 恢复链已到链路极限（CMP 50HX 第 2 轮）

状态：实验 / 第 2 轮（P0' 结论）
适用版本：`codex/block-swap-int8`（基于 dev `d65f4a28` + 第 1 轮 `0d56e43d`）
入口命令：见"复现"
相关代码：`library/runtime/offloading.py`（`h2d_bytes` 计量）、`bench/block_swap_pipeline/copy_bench.py`、`bench/block_swap_pipeline/run_probe.py`

第 1 轮把 P0 从"折返点重复搬运"修正为"压缩串行 DMA 链"。本轮回答该链**还有多少余量**：
结论是**几乎没有**——恢复链已跑到本机链路吞吐的 96%~99%，唯一可用的小杠杆是 `restore_mode="slab"`（+3.3% DMA / +4% 步时）。

## 方法

1. **纯拷贝微基准**（`copy_bench.py`，不加载模型）：模拟 offloader 的搬运方式（pinned/unpinned CPU 源、独立 copy stream、
   `non_blocking`、CUDA event 计时、取中位数），并改变 payload 分块数（1 / 8 / 24）以暴露逐张量开销。
2. **在真机训练里给 profiler 加字节计量**：`Offloader._estimate_swap_bytes()` 从交换计划读取目标 master 的
   `stored_nbytes()`，事件里新增 `h2d_bytes` / `d2h_bytes`，于是"每步搬了多少字节、有效速率多少"可直接读出。
3. **A/B**：`block_swap_restore_mode=foreach` vs `slab`，同一工作区、同一 swap=12。
   （当时项目默认是 `foreach`；该默认值已在提交 `e8ad25b3 perf(blockswap): default restore_mode to slab`
   改为 `slab`，本轮 A/B 结论不受影响，但"默认值"字样按现状读。）

## 结果 1：链路天花板 3.14 GB/s，逐张量开销可忽略

| payload | 分块 | pinned | 中位耗时 | 有效带宽 |
| --- | --- | --- | --- | --- |
| 100 MB | 1 | 是 | 31.882 ms | **3.14 GB/s** |
| 100 MB | 8 | 是 | 31.965 ms | 3.13 GB/s |
| 100 MB | 24 | 是 | 31.883 ms | 3.14 GB/s |
| 100 MB | 1 | 否 | 36.468 ms | 2.74 GB/s |
| 100 MB | 24 | 否 | 36.630 ms | 2.73 GB/s |
| 50 MB | 1 / 8 / 24 | 是 | 15.93 / 15.95 / 15.97 ms | 3.14 GB/s |

- **分块数对总耗时没有影响**（1 块 vs 24 块同速）→ 不存在"逐张量 launch 开销"可优化。
- pinned 对 unpinned 有 +15% 收益，但两者都远低于常见 x16 链路；本机是窄链路（CMP 50HX），
  **3.14 GB/s 就是这块卡的实际上限**。

## 结果 2：训练里的恢复链已达链路 96%~99%

swap=12、同一工作区、稳态（第 2~4 步）：

| `restore_mode` | 每步任务数 | 每步 H2D 字节 | 每步 H2D 时间 | 有效带宽 | 步时样本（ms） |
| --- | --- | --- | --- | --- | --- |
| `foreach`（当时默认） | 24 | 3321.9 MB | 1098 / 1103 / 1101 ms | 3.03 / 3.01 / 3.02 GB/s | 1610 / 1530 / 1620 |
| **`slab`** | 24 | 3321.9 MB | **1065 / 1066 / 1066 ms** | **3.12 / 3.12 / 3.12 GB/s** | **1550 / 1470 / 1550** |

- 每步 24 个任务 = 2S（前向 12 次换出块恢复 + 反向 12 次早块恢复），**无重复恢复**，与第 1 轮一致。
- 单任务 payload ≈ 138.4 MB（3321.9 MB ÷ 24）。
- `slab` 的有效带宽 3.12 GB/s = 微基准天花板 3.14 GB/s 的 **99.4%**；`foreach` 也已有 96%。
- 结论：**恢复链本身没有可回收的余量**。第 1 轮测到的 swap 代价（+0.477 s/步）就是这 3.32 GB/步
  在 3.0~3.1 GB/s 下必然要花的时间，减去计算能掩盖的部分。

## 结果 3：唯一的小杠杆是 slab，方向仍是"减字节"

| 动作 | 收益（本机实测） | 说明 |
| --- | --- | --- |
| `restore_mode="slab"` | DMA +3.3%（1101→1066 ms），步时 ≈ −60~70 ms（−4%） | 零风险、零精度影响；NF4 场景自动回退逐张量路径 |
| 降 `blocks_to_swap`（P1 auto） | 每减 1 个换出块 ≈ 少 138 MB/步 ≈ 少 ~45 ms DMA | 现值最大 |
| int8 master（P2，1 Byte/元素） | H2D 字节约减半 → DMA 时间约减半 | 需要解决 ~1% loss 偏移 |
| 调度类（折返点驻留 / 交错 / 预取） | ≈0 | 无暴露等待（1.5~1.6 ms/步），链路已饱和 |

## 边界与未证明项

- 只在 CMP 50HX（窄链路 3.14 GB/s）上验证；宽链路卡（x16 Gen4/5）上"链路饱和"结论会反转，
  届时调度类优化可能重新有价值——**本结论不跨硬件外推**。
- `slab` 的 +3.3% 是单机 4 步短跑（吞吐量很小），未做多次重复的统计显著性检验。
  本轮当时只把它记录为**建议的显式设置**，没有改项目默认值；该默认值随后由提交
  `e8ad25b3` 改为 `slab`（本分支与 `dev` 一致），所以这一条现在是**复核结论**而非待办，
  不需要额外的关闭动作。
- `h2d_bytes` 由交换计划推算（`stored_nbytes()`），未与 NVML/PCIe 计数器交叉校验。
- 138.4 MB/块是 Anima-base 在本工作区的实测 payload；不同模型族/精度会不同。

## 复现

```bash
# 链路天花板
.venv/bin/python -m bench.block_swap_pipeline.copy_bench --gpu 0 --sizes-mb 100,50 --chunks 1,8,24

# 训练内字节与有效带宽（foreach 档位）
.venv/bin/python -m bench.block_swap_pipeline.run_probe --gpu 0 --steps 4 --blocks-to-swap 12 --label swap12-bytes

# slab 对照
.venv/bin/python -m bench.block_swap_pipeline.run_probe --gpu 0 --steps 4 --blocks-to-swap 12 --restore-mode slab --label swap12-slab
```
