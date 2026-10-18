# Block Swap 进程内水位调优（CMP 50HX 第 3 轮）

状态：实验 / 第 3 轮（P1 实现 + 真机验收）
适用版本：`codex/block-swap-int8`（第 1 轮 `0d56e43d`、第 2 轮 `08835de4` 之后）
入口命令：见「复现」
相关代码：`library/runtime/block_swap_auto.py`、`library/training/cli_args.py`、`library/training/loop.py`、`tests/test_block_swap_auto.py`

第 1、2 轮确认了单次恢复已到链路极限（3.14 GB/s，训练内 3.02~3.12 GB/s），
所以剩下的杠杆是**减少搬运字节**。本轮把「减字节」自动化：实现进程内水位调优
`--block_swap_auto`，并在真机上验收。

## 与既有 AUTO 的分工

仓库里已有一套 AUTO 块交换，但它是**独立进程启动校准**（多次候选进程实测耗时），
其文档明确说明它不是 `blocks_to_swap = "auto"`。本轮补的是另一种更轻的机制：

| 维度 | 启动校准 AUTO（既有） | 进程内 `--block_swap_auto`（本轮） |
| --- | --- | --- |
| 生效时机 | 训练前，多次独立候选进程 | 第一个优化器步之后，一次边界事务 |
| 决策依据 | 候选实测耗时 + 显存/内存策略 | 本进程 `mem_get_info()` 空闲显存 ÷ per-block 字节 |
| 是否改 `blocks_to_swap` | 只覆盖本次运行有效值 | 只下调（晋升常驻），不会上调 |
| 失败行为 | 候选被拒绝 | 事务自动回滚保留原水位；异常只告警不中断训练 |

实现要点：

- 复用现成的 `ModelOffloader.reconfigure()` 边界事务（`library/runtime/block_swap_reconfigure.py`），
  该事务要求梯度已清零、无存留计算图，因此挂点选在 `optimizer.zero_grad()` 之后。
- per-block 字节取 `_cpu_weight_masters` 的 `stored_nbytes()` 之和，与第 2 轮的 profiler 字节计量同源。
- 晋升数受 offloader 契约限制：至少保留 2 个换出块（`promotable = min(headroom // block, current - 2)`）。
- 只降不升，因此即便估计偏乐观也不会把原本能跑的配置推成 OOM；真发生迁移 OOM 时事务自身回滚。

## 真机验收（Anima 512×512，保守起点 swap=24，margin 400 MiB）

日志：`promoting 22 block(s) to resident, blocks_to_swap 24 -> 2 (free=7.57 GiB, margin=0.39 GiB)`。

| 阶段 | 每步任务数 | H2D 字节/步 | DMA 时间/步 | 任务窗口 span | 有效带宽 |
| --- | --- | --- | --- | --- | --- |
| 晋升前（swap=24，第 1 步） | 48 | 6644 MB | 2322 ms | 2118 ms | 2.86 GB/s |
| 晋升后（swap=2，第 2~5 步） | 4 | 554 MB | 178~180 ms | 705~744 ms | 3.07~3.11 GB/s |

对照（固定 `blocks_to_swap=5`，同一工作区，第 2~5 步稳态）：

| 配置 | 每步任务数 | H2D 字节/步 | DMA 时间/步 | 任务窗口 span | trainer 自身步时 |
| --- | --- | --- | --- | --- | --- |
| 固定 swap=5 | 10 | 1384 MB | 445~446 ms | 745~785 ms | `recent_s_per_step` 0.94~0.95 s |
| auto 晋升后 swap=2 | 4 | 554 MB | 178~180 ms | 705~744 ms | `recent_s_per_step` ≈0.90 s |

## 结论

1. **机制成立**：保守起点 24 的 5 步跑，第 2 步起自动收敛到 2；PCIe 流量降到 1/12
   （6644 → 554 MB/步），任务窗口从 2118 ms 降到 ~720 ms。日志、profiler 字节、逐步任务数
   三路证据一致。
2. **在 512 下 swap 已经不需要 7**：晋升停在 2 不是显存不足（晋升后仍空闲 7.57 GiB），
   而是 offloader 契约的下限（至少 2 块换出）。该卡在 512 上可以只换 2 块。
3. **但低于 swap≈5 后收益消失**：swap=2（178 ms DMA）与 swap=5（445~446 ms DMA）的 trainer 步时
   几乎相同（≈0.90 vs 0.94 s），任务窗口也相当（~720 vs ~760 ms）。此时 DMA 只占步时的
   ~25%~60%，已被计算完全掩盖，步时转为计算/数据受限。
   → 因此「减字节」只在 DMA 链仍是关键路径时有效；`--block_swap_auto` 的**晋升应加停止条件**
   （例如预估 DMA 已低于步时计算量就不再晋升），否则只是白搬显存。这是下一轮的改进项，
   本轮先保留「一路晋升到契约下限」的行为并如实记录。

## 单元测试

`tests/test_block_swap_auto.py`：10 项通过（余量解析、per-block 字节、晋升上限、契约下限、
无 CUDA / 无 master、适配器与直挂两种模型形态、无 offloader 告警路径、`:apply` 只调用一次）。

## 边界与未证明项

- 只在 Anima 512×512 / CMP 50HX 上真机跑过；Z-Image、Qwen-Image 2.1、Krea-2 只做了代码级
  适配器解析（`resolve_offloader_and_blocks`），**未真机验证**。
- 256 档未跑：256 的算力底更低，预计晋升会走到契约下限且同样收益递减，属待验收项。
- `free=7.57 GiB` 是晋升后的瞬时读数，不等于峰值安全余量；长时间训练与分辨率切换下的安全性
  未验证。当前只保证「只降不升 + 事务回滚」，不保证任何分辨率都安全。
- `recent_s_per_step` 是 trainer 自己的窗口指标，与训练日志里累积平均的 `s/it` 不是同一口径，
  本文件只用它做同口径对比，不跨口径换算。
- 本轮未改任何默认值：`--block_swap_auto` 默认关闭。

## 复现

```bash
# 自动晋升（保守起点 24）
.venv/bin/python -m bench.block_swap_pipeline.run_probe --gpu 0 --steps 5 --blocks-to-swap 24 \
    --block-swap-auto --margin-mib 400 --label auto24-512

# 固定对照
.venv/bin/python -m bench.block_swap_pipeline.run_probe --gpu 0 --steps 5 --blocks-to-swap 5 --label fixed5-512

# 单元测试
timeout 120 .venv/bin/python -m pytest -q tests/test_block_swap_auto.py
```
